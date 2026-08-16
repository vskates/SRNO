from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from srno.geometry.gripper import GripperAsset
from srno.geometry.sdf import sample_sdf
from srno.geometry.se3 import apply_left_increment, so3_exp
from srno.types import PoseState, SDFBatch


@dataclass(frozen=True)
class StepAux:
    trial_gap: Tensor
    active: Tensor
    joint_residual: Tensor


class ContactIntegralCell(nn.Module):
    """One nonlocal diagonal-kernel integral layer and a residual state head."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.local = nn.Linear(1, hidden_dim, bias=False)
        self.source = nn.Linear(1, hidden_dim, bias=False)
        self.kernel = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 2, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 12),
        )
        final = self.head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        normalized_gap: Tensor,
        normalized_points: Tensor,
        aperture_previous: Tensor,
        aperture_trial: Tensor,
        *,
        pair_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return dimensionless spatial twist and six joint residuals.

        The two scalar aperture values are derived conditioning diagnostics;
        collision geometry and predicted state are functions of the six joints.
        ``normalized_points`` is shared ``[M,3]`` for schedule commands or
        per-sample ``[B,M,3]`` for arbitrary geometries.
        """

        if normalized_gap.ndim != 2:
            raise ValueError("normalized_gap must have shape [batch, points]")
        batch, points = normalized_gap.shape
        embedded = normalized_gap.unsqueeze(-1)
        local = self.local(embedded)
        source = self.source(embedded)

        if pair_features is None:
            if normalized_points.ndim == 2:
                target = normalized_points[:, None, :].expand(points, points, 3)
                source_points = normalized_points[None, :, :].expand_as(target)
            elif normalized_points.ndim == 3:
                if normalized_points.shape[:2] != (batch, points):
                    raise ValueError("normalized_points shape does not match gaps")
                target = normalized_points[:, :, None, :].expand(
                    batch, points, points, 3
                )
                source_points = normalized_points[:, None, :, :].expand_as(target)
            else:
                raise ValueError("normalized_points must have shape [M,3] or [B,M,3]")
            pair_features = torch.cat((target, source_points), dim=-1)

        kernel = self.kernel(pair_features)
        if kernel.ndim == 3:
            message = torch.einsum("ijd,bjd->bid", kernel, source)
        elif kernel.ndim == 4:
            message = torch.einsum("bijd,bjd->bid", kernel, source)
        else:
            raise ValueError("pair_features has an invalid rank")
        features = torch.nn.functional.silu(local + message / points + self.bias)
        pooled = features.mean(dim=1)
        head_input = torch.cat(
            (pooled, aperture_previous[:, None], aperture_trial[:, None]), dim=-1
        )
        output = self.head(head_input)
        return output[:, :6], output[:, 6:]


class SRNOModel(nn.Module):
    """Shared joint-state contact resolvent with an exact deterministic bypass."""

    def __init__(
        self,
        gripper: GripperAsset,
        *,
        sdf_scale: float,
        delta_gate: float,
        contact_offset_sum: float = 0.0,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if gripper.point_count != 256:
            raise ValueError(
                f"SRNO v1 requires exactly 256 gripper samples, got {gripper.point_count}"
            )
        if not gripper.supports_joint_fk:
            raise ValueError("SRNO-r requires a format-v3 gripper asset with joint FK")
        if sdf_scale <= 0:
            raise ValueError("sdf_scale must be positive")
        if delta_gate <= 0:
            raise ValueError("delta_gate must be positive")
        if contact_offset_sum < 0:
            raise ValueError("contact_offset_sum cannot be negative")
        self.sdf_scale = float(sdf_scale)
        self.delta_gate = float(delta_gate)
        self.contact_offset_sum = float(contact_offset_sum)
        self.length_scale = float(gripper.length_scale)
        self.aperture_min = float(gripper.aperture_min)
        self.aperture_max = float(gripper.aperture_max)
        self.joint_names = gripper.joint_names

        def required(value: Tensor | None, name: str) -> Tensor:
            if value is None:
                raise ValueError(f"gripper asset is missing {name}")
            return value

        self.register_buffer(
            "surface_local_points", required(gripper.local_points, "local_points").float()
        )
        self.register_buffer("surface_link_index", gripper.link_index.long())
        self.register_buffer(
            "link_pivots", required(gripper.link_pivots, "link_pivots").float()
        )
        self.register_buffer(
            "link_open_positions",
            required(gripper.link_open_positions, "link_open_positions").float(),
        )
        self.register_buffer(
            "link_open_rotations",
            required(gripper.link_open_rotations, "link_open_rotations").float(),
        )
        self.register_buffer(
            "link_axes", required(gripper.link_axes, "link_axes").float()
        )
        self.register_buffer(
            "link_position_joint_coefficients",
            required(
                gripper.link_position_joint_coefficients,
                "link_position_joint_coefficients",
            ).float(),
        )
        self.register_buffer(
            "link_rotation_joint_coefficients",
            required(
                gripper.link_rotation_joint_coefficients,
                "link_rotation_joint_coefficients",
            ).float(),
        )
        self.register_buffer(
            "free_aperture_knots",
            required(gripper.aperture_knots, "aperture_knots").float(),
        )
        self.register_buffer(
            "free_joint_knots",
            required(gripper.free_joint_knots, "free_joint_knots").float(),
        )
        travel = gripper.joint_travel_range.float()
        if torch.any(travel <= 1e-8):
            raise ValueError("all six gripper joints must have positive free travel")
        self.register_buffer("joint_travel_range", travel)
        self.contact_cell = ContactIntegralCell(hidden_dim)
        self._pair_feature_cache: dict[tuple[float, str, torch.dtype], Tensor] = {}

    def _apply(self, fn: object, recurse: bool = True) -> "SRNOModel":
        self._pair_feature_cache.clear()
        return super()._apply(fn, recurse)  # type: ignore[arg-type,return-value]

    @property
    def point_count(self) -> int:
        return self.surface_local_points.shape[0]

    def free_joint_configuration(self, aperture: Tensor | float) -> Tensor:
        """Interpolate R_free(command) over the frozen 33-command schedule."""

        value = torch.as_tensor(
            aperture,
            dtype=self.free_aperture_knots.dtype,
            device=self.free_aperture_knots.device,
        )
        if torch.any(value < self.aperture_min - 1e-7) or torch.any(
            value > self.aperture_max + 1e-7
        ):
            raise ValueError("aperture command is outside gripper limits")
        flat = value.reshape(-1).contiguous()
        upper = torch.searchsorted(self.free_aperture_knots, flat).clamp(
            1, len(self.free_aperture_knots) - 1
        )
        lower = upper - 1
        low_aperture = self.free_aperture_knots.index_select(0, lower)
        high_aperture = self.free_aperture_knots.index_select(0, upper)
        alpha = (flat - low_aperture) / (high_aperture - low_aperture)
        low_joint = self.free_joint_knots.index_select(0, lower)
        high_joint = self.free_joint_knots.index_select(0, upper)
        interpolated = low_joint + alpha[:, None] * (high_joint - low_joint)
        distance = torch.abs(flat[:, None] - self.free_aperture_knots[None, :])
        exact_distance, exact_index = distance.min(dim=-1)
        exact = exact_distance <= 4.0 * torch.finfo(flat.dtype).eps
        exact_joint = self.free_joint_knots.index_select(0, exact_index)
        interpolated = torch.where(exact[:, None], exact_joint, interpolated)
        return interpolated.reshape(value.shape + (6,))

    def aperture_from_joints(self, joint_position: Tensor) -> Tensor:
        """Derived scalar diagnostic A(r), never used to select geometry."""

        if joint_position.shape[-1:] != (6,):
            raise ValueError("joint_position must end in (6,)")
        open_joint = self.free_joint_knots[-1]
        close_joint = self.free_joint_knots[0]
        joint_range = close_joint - open_joint
        closure = (
            ((joint_position - open_joint) * joint_range).sum(dim=-1)
            / joint_range.square().sum().clamp_min(1e-8)
        ).clamp(0.0, 1.0)
        schedule_position = closure * float(len(self.free_aperture_knots) - 1)
        lower = torch.floor(schedule_position).long()
        upper = (lower + 1).clamp(max=len(self.free_aperture_knots) - 1)
        alpha = schedule_position - lower.to(schedule_position.dtype)
        open_to_closed = torch.flip(self.free_aperture_knots, dims=(0,))
        return (
            open_to_closed.index_select(0, lower.reshape(-1)).reshape(lower.shape)
            * (1.0 - alpha)
            + open_to_closed.index_select(0, upper.reshape(-1)).reshape(upper.shape)
            * alpha
        )

    def gripper_points(self, joint_position: Tensor) -> Tensor:
        """Transform all 256 local collision points through differentiable FK."""

        if joint_position.shape[-1:] != (6,):
            raise ValueError("joint_position must end in (6,)")
        position_angle = torch.einsum(
            "...j,lj->...l", joint_position, self.link_position_joint_coefficients
        )
        rotation_angle = torch.einsum(
            "...j,lj->...l", joint_position, self.link_rotation_joint_coefficients
        )
        position_rotation = so3_exp(position_angle[..., None] * self.link_axes)
        link_rotation = (
            so3_exp(rotation_angle[..., None] * self.link_axes)
            @ self.link_open_rotations
        )
        link_position = (
            torch.einsum(
                "...lij,lj->...li",
                position_rotation,
                self.link_open_positions - self.link_pivots,
            )
            + self.link_pivots
        )
        point_rotation = link_rotation[..., self.surface_link_index, :, :]
        point_position = link_position[..., self.surface_link_index, :]
        return (
            torch.einsum(
                "...mij,mj->...mi", point_rotation, self.surface_local_points
            )
            + point_position
        )

    def _cached_pair_features(self, command: Tensor) -> tuple[Tensor, Tensor]:
        value = float(command.detach().cpu())
        key = (
            value,
            str(self.surface_local_points.device),
            self.surface_local_points.dtype,
        )
        features = self._pair_feature_cache.get(key)
        joint_position = self.free_joint_configuration(command)
        points = self.gripper_points(joint_position) / self.length_scale
        if features is None:
            target = points[:, None, :].expand(self.point_count, self.point_count, 3)
            source = points[None, :, :].expand_as(target)
            features = torch.cat((target, source), dim=-1)
            self._pair_feature_cache[key] = features
        return points, features

    def query_geometric_gap(self, state: PoseState, sdf: SDFBatch) -> Tensor:
        """Return cooked-collider distance without the PhysX contact offset."""

        points_gripper = self.gripper_points(state.joint_position)
        relative = points_gripper - state.position[..., None, :]
        points_object = torch.einsum(
            "bij,bmj->bmi", state.rotation.transpose(-1, -2), relative
        )
        return sample_sdf(
            sdf.values,
            sdf.origin,
            sdf.voxel_size,
            points_object,
            sample_to_object=sdf.sample_to_object,
            outside_value=sdf.outside_value,
        )

    def query_gap(self, state: PoseState, sdf: SDFBatch) -> Tensor:
        """Return contact signal; zero is the PhysX contact-generation onset."""

        return self.query_geometric_gap(state, sdf) - self.contact_offset_sum

    def forward_step(
        self,
        state: PoseState,
        next_command: Tensor | float,
        sdf: SDFBatch,
        *,
        return_aux: bool = False,
    ) -> PoseState | tuple[PoseState, StepAux]:
        if state.joint_position.ndim != 2:
            raise ValueError("forward_step expects a flat batch")
        batch = state.joint_position.shape[0]
        if sdf.sample_to_object.shape != (batch,):
            raise ValueError("SDF mapping length must equal state batch size")

        command = torch.as_tensor(
            next_command,
            dtype=state.joint_position.dtype,
            device=state.device,
        )
        shared_command = command.ndim == 0
        if shared_command:
            command_batch = command.expand(batch)
            shared_points, pair_features = self._cached_pair_features(command)
            trial_points = (shared_points * self.length_scale).unsqueeze(0).expand(
                batch, self.point_count, 3
            )
            trial_joint = self.free_joint_configuration(command).unsqueeze(0).expand(
                batch, 6
            )
        else:
            if command.shape != (batch,):
                raise ValueError("next_command must be scalar or have shape [batch]")
            command_batch = command
            trial_joint = self.free_joint_configuration(command_batch)
            trial_points = self.gripper_points(trial_joint)
            shared_points = pair_features = None

        trial_state = PoseState(state.rotation, state.position, trial_joint)
        relative = trial_points - state.position[:, None, :]
        points_object = torch.einsum(
            "bij,bmj->bmi", state.rotation.transpose(-1, -2), relative
        )
        trial_gap = sample_sdf(
            sdf.values,
            sdf.origin,
            sdf.voxel_size,
            points_object,
            sample_to_object=sdf.sample_to_object,
            outside_value=sdf.outside_value,
        ) - self.contact_offset_sum
        active = trial_gap.amin(dim=-1) <= self.delta_gate

        next_rotation = trial_state.rotation.clone()
        next_position = trial_state.position.clone()
        next_joint = trial_state.joint_position.clone()
        joint_residual = torch.zeros_like(next_joint)
        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        if active_indices.numel():
            if shared_command:
                groups = [(active_indices, shared_points, pair_features)]
            else:
                groups = []
                active_commands = command_batch.index_select(0, active_indices)
                for unique_command in torch.unique(active_commands):
                    local_group = torch.nonzero(
                        active_commands == unique_command, as_tuple=False
                    ).flatten()
                    group_indices = active_indices.index_select(0, local_group)
                    points, features = self._cached_pair_features(unique_command)
                    groups.append((group_indices, points, features))

            for group_indices, normalized_points, active_pair_features in groups:
                group_gap = trial_gap.index_select(0, group_indices) / self.sdf_scale
                previous_aperture = self.aperture_from_joints(
                    state.joint_position.index_select(0, group_indices)
                ) / self.length_scale
                trial_aperture = (
                    command_batch.index_select(0, group_indices) / self.length_scale
                )
                raw_twist, normalized_joint_residual = self.contact_cell(
                    group_gap,
                    normalized_points,
                    previous_aperture,
                    trial_aperture,
                    pair_features=active_pair_features,
                )
                twist = torch.cat(
                    (raw_twist[:, :3] * self.length_scale, raw_twist[:, 3:]),
                    dim=-1,
                ).float()
                active_rotation, active_position = apply_left_increment(
                    state.rotation.index_select(0, group_indices).float(),
                    state.position.index_select(0, group_indices).float(),
                    twist,
                )
                residual = (
                    normalized_joint_residual.float() * self.joint_travel_range
                )
                active_joint = trial_joint.index_select(0, group_indices).float() + residual
                next_rotation[group_indices] = active_rotation.to(next_rotation.dtype)
                next_position[group_indices] = active_position.to(next_position.dtype)
                next_joint[group_indices] = active_joint.to(next_joint.dtype)
                joint_residual[group_indices] = residual.to(joint_residual.dtype)

        result = PoseState(next_rotation, next_position, next_joint)
        if return_aux:
            return result, StepAux(
                trial_gap=trial_gap,
                active=active,
                joint_residual=joint_residual,
            )
        return result

    def rollout(
        self,
        initial_state: PoseState,
        command_schedule: Tensor,
        sdf: SDFBatch,
    ) -> PoseState:
        """Autoregress over next commands and return states including x0."""

        if command_schedule.ndim not in (1, 2):
            raise ValueError("command_schedule must have shape [steps] or [batch, steps]")
        if (
            command_schedule.ndim == 2
            and command_schedule.shape[0] != initial_state.shape[0]
        ):
            raise ValueError("batched command schedule has the wrong batch size")
        states = [initial_state]
        current = initial_state
        for step in range(command_schedule.shape[-1]):
            command = (
                command_schedule[step]
                if command_schedule.ndim == 1
                else command_schedule[:, step]
            )
            current = self.forward_step(current, command, sdf)
            assert isinstance(current, PoseState)
            states.append(current)
        return PoseState.stack(states, dim=1)
