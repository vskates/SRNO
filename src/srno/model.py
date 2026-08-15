from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from srno.geometry.gripper import GripperAsset
from srno.geometry.sdf import sample_sdf
from srno.geometry.se3 import apply_left_increment
from srno.types import PoseState, SDFBatch


@dataclass(frozen=True)
class StepAux:
    trial_gap: Tensor
    active: Tensor
    alpha: Tensor


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
            nn.Linear(128, 7),
        )
        final = self.head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            final.bias[-1] = -4.0

    def forward(
        self,
        normalized_gap: Tensor,
        normalized_points: Tensor,
        aperture_previous: Tensor,
        aperture_trial: Tensor,
        *,
        pair_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return a dimensionless spatial twist and aperture logit.

        ``normalized_points`` is either shared ``[M,3]`` or per-sample
        ``[B,M,3]``. A shared geometry avoids a batch dimension in the O(M^2)
        kernel tensor.
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
                target = normalized_points[:, :, None, :].expand(batch, points, points, 3)
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
        return output[:, :6], output[:, 6]


class SRNOModel(nn.Module):
    """Shared local contact resolvent with an exact deterministic free bypass."""

    def __init__(
        self,
        gripper: GripperAsset,
        *,
        sdf_scale: float,
        delta_gate: float,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if gripper.point_count != 256:
            raise ValueError(f"SRNO v1 requires exactly 256 gripper samples, got {gripper.point_count}")
        if sdf_scale <= 0:
            raise ValueError("sdf_scale must be positive")
        if delta_gate <= 0:
            raise ValueError("delta_gate must be positive")
        self.sdf_scale = float(sdf_scale)
        self.delta_gate = float(delta_gate)
        self.length_scale = float(gripper.length_scale)
        self.aperture_min = float(gripper.aperture_min)
        self.aperture_max = float(gripper.aperture_max)
        self.register_buffer("surface_intercept", gripper.intercept.float())
        self.register_buffer("surface_slope", gripper.slope.float())
        self.register_buffer("surface_link_index", gripper.link_index.long())
        self.contact_cell = ContactIntegralCell(hidden_dim)
        self._pair_feature_cache: dict[tuple[float, str, torch.dtype], Tensor] = {}

    def _apply(self, fn: object, recurse: bool = True) -> "SRNOModel":
        self._pair_feature_cache.clear()
        return super()._apply(fn, recurse)  # type: ignore[arg-type,return-value]

    @property
    def point_count(self) -> int:
        return self.surface_intercept.shape[0]

    def gripper_points(self, aperture: Tensor | float) -> Tensor:
        aperture_tensor = torch.as_tensor(
            aperture, dtype=self.surface_intercept.dtype, device=self.surface_intercept.device
        )
        return self.surface_intercept + aperture_tensor[..., None, None] * self.surface_slope

    def _cached_pair_features(self, command: Tensor) -> tuple[Tensor, Tensor]:
        value = float(command.detach().cpu())
        key = (value, str(self.surface_intercept.device), self.surface_intercept.dtype)
        features = self._pair_feature_cache.get(key)
        points = self.gripper_points(command) / self.length_scale
        if features is None:
            target = points[:, None, :].expand(self.point_count, self.point_count, 3)
            source = points[None, :, :].expand_as(target)
            features = torch.cat((target, source), dim=-1)
            self._pair_feature_cache[key] = features
        return points, features

    def query_gap(self, state: PoseState, sdf: SDFBatch) -> Tensor:
        points_gripper = self.gripper_points(state.aperture)
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

    def forward_step(
        self,
        state: PoseState,
        next_command: Tensor | float,
        sdf: SDFBatch,
        *,
        return_aux: bool = False,
    ) -> PoseState | tuple[PoseState, StepAux]:
        if state.aperture.ndim != 1:
            raise ValueError("forward_step expects a flat batch")
        batch = state.aperture.shape[0]
        if sdf.sample_to_object.shape != (batch,):
            raise ValueError("SDF mapping length must equal state batch size")

        command = torch.as_tensor(next_command, dtype=state.aperture.dtype, device=state.device)
        shared_command = command.ndim == 0
        if shared_command:
            command_batch = command.expand(batch)
            shared_points, pair_features = self._cached_pair_features(command)
            trial_points = shared_points * self.length_scale
            trial_points = trial_points.unsqueeze(0).expand(batch, self.point_count, 3)
        else:
            if command.shape != (batch,):
                raise ValueError("next_command must be scalar or have shape [batch]")
            command_batch = command
            trial_points = self.gripper_points(command_batch)
            shared_points = pair_features = None

        trial_state = PoseState(state.rotation, state.position, command_batch)
        relative = trial_points - state.position[:, None, :]
        points_object = torch.einsum("bij,bmj->bmi", state.rotation.transpose(-1, -2), relative)
        trial_gap = sample_sdf(
            sdf.values,
            sdf.origin,
            sdf.voxel_size,
            points_object,
            sample_to_object=sdf.sample_to_object,
            outside_value=sdf.outside_value,
        )
        active = trial_gap.amin(dim=-1) <= self.delta_gate

        next_rotation = trial_state.rotation.clone()
        next_position = trial_state.position.clone()
        next_aperture = trial_state.aperture.clone()
        alpha = torch.zeros_like(state.aperture)
        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        if active_indices.numel():
            # Bucket vector commands by aperture. This keeps the O(M^2) kernel
            # geometry shared instead of materializing it once per transition.
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
                previous_normalized = (
                    state.aperture.index_select(0, group_indices) / self.length_scale
                )
                trial_normalized = (
                    command_batch.index_select(0, group_indices) / self.length_scale
                )
                raw_twist, eta = self.contact_cell(
                    group_gap,
                    normalized_points,
                    previous_normalized,
                    trial_normalized,
                    pair_features=active_pair_features,
                )
                twist = torch.cat(
                    (raw_twist[:, :3] * self.length_scale, raw_twist[:, 3:]), dim=-1
                ).float()
                active_rotation, active_position = apply_left_increment(
                    state.rotation.index_select(0, group_indices).float(),
                    state.position.index_select(0, group_indices).float(),
                    twist,
                )
                active_alpha = torch.sigmoid(eta.float())
                active_aperture = command_batch.index_select(0, group_indices) + active_alpha * (
                    state.aperture.index_select(0, group_indices)
                    - command_batch.index_select(0, group_indices)
                )
                next_rotation[group_indices] = active_rotation.to(next_rotation.dtype)
                next_position[group_indices] = active_position.to(next_position.dtype)
                next_aperture[group_indices] = active_aperture.to(next_aperture.dtype)
                alpha[group_indices] = active_alpha.to(alpha.dtype)

        result = PoseState(next_rotation, next_position, next_aperture)
        if return_aux:
            return result, StepAux(trial_gap=trial_gap, active=active, alpha=alpha)
        return result

    def rollout(
        self,
        initial_state: PoseState,
        command_schedule: Tensor,
        sdf: SDFBatch,
    ) -> PoseState:
        """Autoregress over next-command values and return states including x0."""

        if command_schedule.ndim not in (1, 2):
            raise ValueError("command_schedule must have shape [steps] or [batch, steps]")
        if command_schedule.ndim == 2 and command_schedule.shape[0] != initial_state.shape[0]:
            raise ValueError("batched command schedule has the wrong batch size")
        states = [initial_state]
        current = initial_state
        step_count = command_schedule.shape[-1]
        for step in range(step_count):
            command = (
                command_schedule[step]
                if command_schedule.ndim == 1
                else command_schedule[:, step]
            )
            current = self.forward_step(current, command, sdf)
            assert isinstance(current, PoseState)
            states.append(current)
        return PoseState.stack(states, dim=1)
