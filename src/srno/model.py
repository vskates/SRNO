from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from srno.geometry.gripper import GripperAsset
from srno.geometry.sdf import sample_sdf, sample_sdf_with_gradient
from srno.geometry.se3 import apply_left_increment, so3_exp, so3_log_vector
from srno.types import PoseState, SDFBatch


@dataclass(frozen=True)
class StepAux:
    trial_gap: Tensor
    active: Tensor
    joint_residual: Tensor


class ResidualIntegralLayer(nn.Module):
    """Geometry-conditioned residual operator layer on hidden point features."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.local = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.source = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.kernel = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, features: Tensor, pair_coordinates: Tensor) -> Tensor:
        points = features.shape[1]
        kernel = self.kernel(pair_coordinates)
        source = self.source(features)
        if kernel.ndim == 3:
            message = torch.einsum("ijd,bjd->bid", kernel, source)
        elif kernel.ndim == 4:
            message = torch.einsum("bijd,bjd->bid", kernel, source)
        else:
            raise ValueError("pair_coordinates has an invalid rank")
        update = torch.nn.functional.silu(
            self.local(features) + message / points + self.bias
        )
        return features + update


class ContactIntegralCell(nn.Module):
    """A depth-L diagonal-kernel operator followed by the unchanged state head."""

    def __init__(
        self,
        hidden_dim: int = 64,
        *,
        input_dim: int = 1,
        operator_layers: int = 1,
        history_dim: int = 0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if operator_layers <= 0:
            raise ValueError("operator_layers must be positive")
        if history_dim < 0:
            raise ValueError("history_dim cannot be negative")
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.operator_layers = operator_layers
        self.history_dim = history_dim
        self.local = nn.Linear(input_dim, hidden_dim, bias=False)
        self.source = nn.Linear(input_dim, hidden_dim, bias=False)
        self.kernel = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.residual_layers = nn.ModuleList(
            ResidualIntegralLayer(hidden_dim) for _ in range(operator_layers - 1)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 2 + history_dim, 128),
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
        node_features: Tensor,
        normalized_points: Tensor,
        aperture_previous: Tensor,
        aperture_trial: Tensor,
        *,
        pair_features: Tensor | None = None,
        history_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return dimensionless spatial twist and six joint residuals.

        The two scalar aperture values are derived conditioning diagnostics;
        collision geometry and predicted state remain functions of the six
        joints.
        ``normalized_points`` is shared ``[M,3]`` for schedule commands or
        per-sample ``[B,M,3]`` for arbitrary geometries.
        """

        if node_features.ndim == 2 and self.input_dim == 1:
            node_features = node_features.unsqueeze(-1)
        if node_features.ndim != 3 or node_features.shape[-1] != self.input_dim:
            raise ValueError(
                "node_features must have shape [batch, points, input_dim]"
            )
        batch, points, _ = node_features.shape
        local = self.local(node_features)
        source = self.source(node_features)

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
        for layer in self.residual_layers:
            features = layer(features, pair_features)
        pooled = features.mean(dim=1)
        aperture_conditioning = torch.stack(
            (aperture_previous, aperture_trial), dim=-1
        )
        if aperture_conditioning.shape != (batch, 2):
            raise ValueError("aperture conditioning must have shape [batch, 2]")
        if self.history_dim:
            if history_features is None or history_features.shape != (
                batch,
                self.history_dim,
            ):
                raise ValueError(
                    "history_features must match the configured history dimension"
                )
            conditioning = torch.cat(
                (aperture_conditioning, history_features), dim=-1
            )
        else:
            if history_features is not None:
                raise ValueError(
                    "history_features were provided to a history-free cell"
                )
            conditioning = aperture_conditioning
        head_input = torch.cat((pooled, conditioning), dim=-1)
        output = self.head(head_input)
        return output[:, :6], output[:, 6:]


class ContactNormalConeCell(nn.Module):
    """Learn contact pressure, then decode through a contact normal cone.

    In dimensionless product coordinates the decoder is

        z = M_theta(e) J(h)^T lambda_theta(h, J, e),

    where ``lambda >= 0`` and ``M = L L^T`` is positive definite.  Thus the
    network cannot invent an unconstrained twelve-vector: its output is a
    positive-metric image of generalized contact normals.  This structural
    constraint alone does not make the learned, state-dependent map globally
    monotone or firmly nonexpansive.  A two-pass
    DeepSets encoder supplies global contact context at O(M) rather than the
    O(M^2) cost of the legacy direct integral head.
    """

    _LOWER_ROW, _LOWER_COLUMN = torch.tril_indices(12, 12)

    def __init__(
        self,
        hidden_dim: int = 64,
        *,
        input_dim: int = 13,
        drive_dim: int = 0,
        friction_coefficient: float = 0.0,
        history_dim: int = 0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if history_dim < 0:
            raise ValueError("history_dim cannot be negative")
        if input_dim not in {13, 14}:
            raise ValueError("normal-cone input_dim must be 13 or 14")
        if drive_dim not in {0, 6}:
            raise ValueError("normal-cone drive_dim must be 0 or 6")
        if friction_coefficient < 0:
            raise ValueError("friction_coefficient cannot be negative")
        # trial gap + optional current gap + full 12D contact Jacobian,
        # followed by the canonical point coordinate.
        self.local_encoder = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        conditioning_dim = 2 + drive_dim + history_dim
        force_dimensions = 3 if friction_coefficient > 0 else 1
        self.pressure_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + conditioning_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, force_dimensions),
        )
        # A conditioned Cholesky factor represents positive-definite mobility.
        self.mobility_head = nn.Sequential(
            nn.Linear(hidden_dim + conditioning_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 12 * 13 // 2),
        )
        pressure_final = self.pressure_head[-1]
        mobility_final = self.mobility_head[-1]
        assert isinstance(pressure_final, nn.Linear)
        assert isinstance(mobility_final, nn.Linear)
        nn.init.normal_(pressure_final.weight, std=1e-3)
        nn.init.zeros_(pressure_final.bias)
        with torch.no_grad():
            pressure_final.bias[0] = -4.0
        nn.init.zeros_(mobility_final.weight)
        nn.init.zeros_(mobility_final.bias)
        self.history_dim = history_dim
        self.input_dim = input_dim
        self.drive_dim = drive_dim
        self.friction_coefficient = float(friction_coefficient)

    def _mobility(self, packed: Tensor) -> Tensor:
        batch = packed.shape[0]
        factor = packed.new_zeros(batch, 12, 12)
        row = self._LOWER_ROW.to(device=packed.device)
        column = self._LOWER_COLUMN.to(device=packed.device)
        factor[:, row, column] = packed
        diagonal = torch.arange(12, device=packed.device)
        # Unit initialization; bounded off-diagonal coupling avoids an
        # ill-conditioned solve while the exponential keeps the diagonal > 0.
        raw_diagonal = factor[:, diagonal, diagonal]
        factor = 0.25 * torch.tanh(factor)
        factor[:, diagonal, diagonal] = torch.exp(raw_diagonal.clamp(-4.0, 4.0))
        return factor @ factor.transpose(-1, -2)

    def forward(
        self,
        node_features: Tensor,
        normalized_points: Tensor,
        aperture_previous: Tensor,
        aperture_trial: Tensor,
        *,
        contact_jacobian: Tensor,
        contact_support: Tensor,
        contact_directions: Tensor | None = None,
        drive_features: Tensor | None = None,
        history_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if node_features.ndim != 3 or node_features.shape[-1] != self.input_dim:
            raise ValueError(
                "normal-cone node_features have the wrong final dimension"
            )
        batch, points, _ = node_features.shape
        if contact_jacobian.shape != (batch, points, 12):
            raise ValueError("contact_jacobian must have shape [B,M,12]")
        if contact_support.shape != (batch, points):
            raise ValueError("contact_support must have shape [B,M]")
        if normalized_points.ndim == 2:
            coordinates = normalized_points.unsqueeze(0).expand(batch, points, 3)
        elif normalized_points.shape == (batch, points, 3):
            coordinates = normalized_points
        else:
            raise ValueError("normalized_points has the wrong shape")
        conditioning_parts = [
            torch.stack((aperture_previous, aperture_trial), dim=-1)
        ]
        if self.drive_dim:
            if drive_features is None or drive_features.shape != (batch, 6):
                raise ValueError("drive_features must have shape [B,6]")
            conditioning_parts.append(drive_features)
        elif drive_features is not None:
            raise ValueError("drive_features were provided to an aperture-only cell")
        if self.history_dim:
            if history_features is None or history_features.shape != (
                batch,
                self.history_dim,
            ):
                raise ValueError("history_features must match configured history")
            conditioning_parts.append(history_features)
        elif history_features is not None:
            raise ValueError("history_features were provided to a history-free cell")
        conditioning = torch.cat(conditioning_parts, dim=-1)
        local = self.local_encoder(torch.cat((node_features, coordinates), dim=-1))
        pooled = local.mean(dim=1)
        expanded = torch.cat((pooled, conditioning), dim=-1).unsqueeze(1).expand(
            batch, points, pooled.shape[-1] + conditioning.shape[-1]
        )
        pressure_input = torch.cat((local, expanded), dim=-1)
        raw_force = self.pressure_head(pressure_input)
        pressure = torch.nn.functional.softplus(raw_force[..., 0])
        pressure = pressure * contact_support.to(dtype=pressure.dtype)
        if self.friction_coefficient > 0:
            if contact_directions is None or contact_directions.shape != (
                batch,
                points,
                12,
                3,
            ):
                raise ValueError("contact_directions must have shape [B,M,12,3]")
            tangent = torch.tanh(raw_force[..., 1:])
            tangent_norm = torch.linalg.vector_norm(
                tangent, dim=-1, keepdim=True
            )
            tangent = tangent / tangent_norm.clamp_min(1.0)
            tangent = (
                tangent
                * pressure.unsqueeze(-1)
                * self.friction_coefficient
            )
            coefficients = torch.cat((pressure.unsqueeze(-1), tangent), dim=-1)
            generalized_normal = torch.einsum(
                "bmk,bmjk->bj", coefficients, contact_directions
            ) / float(points)
        else:
            if contact_directions is not None:
                raise ValueError("contact_directions require a friction-cone cell")
            generalized_normal = torch.einsum(
                "bm,bmj->bj", pressure, contact_jacobian
            ) / float(points)
        mobility = self._mobility(
            self.mobility_head(torch.cat((pooled, conditioning), dim=-1))
        )
        output = torch.einsum("bij,bj->bi", mobility, generalized_normal)
        return output[:, :6], output[:, 6:]


def solve_metric_contact_resolvent(
    query: Tensor,
    metric_factor: Tensor,
    contact_jacobian: Tensor,
    contact_rhs: Tensor,
    contact_support: Tensor,
    *,
    iterations: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Solve a batched metric half-space projection by dual FISTA.

    The primal problem is

        min_z 0.5 (z-query)^T Q (z-query)
        s.t.  J z >= rhs,

    with ``Q = metric_factor @ metric_factor.T``.  The returned dual is
    non-negative and the fixed point satisfies the KKT complementarity system.
    A fixed iteration count makes solver depth explicit and differentiable.
    """

    if iterations <= 0:
        raise ValueError("resolvent iterations must be positive")
    if query.ndim != 2 or query.shape[-1] != 12:
        raise ValueError("query must have shape [B,12]")
    batch = query.shape[0]
    if metric_factor.shape != (batch, 12, 12):
        raise ValueError("metric_factor must have shape [B,12,12]")
    if contact_jacobian.ndim != 3 or contact_jacobian.shape[:1] != (batch,):
        raise ValueError("contact_jacobian must have shape [B,M,12]")
    if contact_jacobian.shape[-1] != 12:
        raise ValueError("contact_jacobian must have shape [B,M,12]")
    points = contact_jacobian.shape[1]
    if contact_rhs.shape != (batch, points):
        raise ValueError("contact_rhs must have shape [B,M]")
    if contact_support.shape != (batch, points):
        raise ValueError("contact_support must have shape [B,M]")

    support = contact_support.to(dtype=query.dtype)
    jacobian = contact_jacobian * support.unsqueeze(-1)
    transpose = jacobian.transpose(-1, -2)
    query_slack = torch.einsum("bmi,bi->bm", jacobian, query) - contact_rhs

    # Q=C C^T and B=Q^-1=C^-T C^-1.  The nonzero eigenvalues of
    # J B J^T equal those of (J C^-T)^T(J C^-T), a 12x12 matrix.
    identity = torch.eye(12, dtype=query.dtype, device=query.device).expand(
        batch, 12, 12
    )
    inverse_transpose = torch.linalg.solve_triangular(
        metric_factor.transpose(-1, -2), identity, upper=True
    )
    whitened = jacobian @ inverse_transpose
    small_gram = whitened.transpose(-1, -2) @ whitened
    with torch.no_grad():
        lipschitz = torch.linalg.eigvalsh(small_gram.detach())[:, -1].clamp_min(
            1e-6
        )
        step = (0.95 / lipschitz).unsqueeze(-1)

    dual = query.new_zeros(batch, points)
    momentum = dual
    acceleration = 1.0
    for _ in range(iterations):
        generalized = torch.einsum("bim,bm->bi", transpose, momentum)
        mobility_generalized = torch.cholesky_solve(
            generalized.unsqueeze(-1), metric_factor, upper=False
        ).squeeze(-1)
        gradient = (
            torch.einsum("bmi,bi->bm", jacobian, mobility_generalized)
            + query_slack
        )
        next_dual = torch.relu(momentum - step * gradient) * support
        next_acceleration = 0.5 * (
            1.0 + (1.0 + 4.0 * acceleration * acceleration) ** 0.5
        )
        coefficient = (acceleration - 1.0) / next_acceleration
        momentum = next_dual + coefficient * (next_dual - dual)
        dual = next_dual
        acceleration = next_acceleration

    generalized = torch.einsum("bim,bm->bi", transpose, dual)
    correction = torch.cholesky_solve(
        generalized.unsqueeze(-1), metric_factor, upper=False
    ).squeeze(-1)
    solution = query + correction
    slack = torch.einsum("bmi,bi->bm", contact_jacobian, solution) - contact_rhs
    return solution, dual, slack


class ContactImplicitResolventCell(nn.Module):
    """Learn a resistance metric; obtain motion from a convex contact VI.

    The encoder never predicts contact pressures or state corrections.  It
    parameterizes only ``Q_theta > 0``.  Multipliers and the coupled 12D motion
    are produced by solving the normal-contact KKT system.
    """

    _LOWER_ROW, _LOWER_COLUMN = torch.tril_indices(12, 12)

    def __init__(
        self,
        hidden_dim: int = 64,
        *,
        pose_weight: float = 100.0,
        iterations: int = 32,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if pose_weight <= 0:
            raise ValueError("resolvent pose_weight must be positive")
        if iterations <= 0:
            raise ValueError("resolvent iterations must be positive")
        self.local_encoder = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Two aperture scalars plus the six-dimensional commanded joint step.
        self.metric_head = nn.Sequential(
            nn.Linear(hidden_dim + 8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 12 * 13 // 2),
        )
        final = self.metric_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        base = torch.full((12,), (1.0 / 6.0) ** 0.5)
        base[:6] = pose_weight**0.5
        self.register_buffer("base_cholesky_diagonal", base)
        self.iterations = int(iterations)

    def _metric_factor(self, packed: Tensor) -> Tensor:
        batch = packed.shape[0]
        factor = packed.new_zeros(batch, 12, 12)
        row = self._LOWER_ROW.to(device=packed.device)
        column = self._LOWER_COLUMN.to(device=packed.device)
        base = self.base_cholesky_diagonal.to(dtype=packed.dtype)
        # Bounded coupling prevents early ill-conditioning.  Diagonal changes
        # are also bounded because a global scale of Q is unidentifiable.
        factor[:, row, column] = (
            0.15
            * torch.tanh(packed)
            * torch.sqrt(base[row] * base[column])
        )
        diagonal = torch.arange(12, device=packed.device)
        diagonal_mask = (self._LOWER_ROW == self._LOWER_COLUMN).to(
            device=packed.device
        )
        diagonal_raw = packed[:, diagonal_mask]
        factor[:, diagonal, diagonal] = base * torch.exp(
            0.5 * torch.tanh(diagonal_raw)
        )
        return factor

    def forward(
        self,
        node_features: Tensor,
        normalized_points: Tensor,
        aperture_previous: Tensor,
        aperture_trial: Tensor,
        *,
        contact_jacobian: Tensor,
        contact_rhs: Tensor,
        contact_support: Tensor,
        drive_features: Tensor,
        pose_query: Tensor,
        joint_lower: Tensor,
        joint_upper: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if node_features.ndim != 3 or node_features.shape[-1] != 13:
            raise ValueError("implicit node_features must have shape [B,M,13]")
        batch, points, _ = node_features.shape
        if normalized_points.shape != (batch, points, 3):
            raise ValueError("implicit normalized_points must have shape [B,M,3]")
        if drive_features.shape != (batch, 6):
            raise ValueError("drive_features must have shape [B,6]")
        if pose_query.shape != (batch, 6):
            raise ValueError("pose_query must have shape [B,6]")
        if joint_lower.shape != (batch, 6) or joint_upper.shape != (batch, 6):
            raise ValueError("joint bounds must have shape [B,6]")
        local = self.local_encoder(
            torch.cat((node_features, normalized_points), dim=-1)
        )
        pooled = local.mean(dim=1)
        conditioning = torch.cat(
            (
                torch.stack((aperture_previous, aperture_trial), dim=-1),
                drive_features,
            ),
            dim=-1,
        )
        metric_factor = self._metric_factor(
            self.metric_head(torch.cat((pooled, conditioning), dim=-1))
        )

        box_jacobian = node_features.new_zeros(batch, 12, 12)
        diagonal = torch.arange(6, device=node_features.device)
        box_jacobian[:, diagonal, diagonal + 6] = 1.0
        box_jacobian[:, diagonal + 6, diagonal + 6] = -1.0
        box_rhs = torch.cat((joint_lower, -joint_upper), dim=-1)
        all_jacobian = torch.cat((contact_jacobian, box_jacobian), dim=1)
        all_rhs = torch.cat((contact_rhs, box_rhs), dim=1)
        all_support = torch.cat(
            (
                contact_support,
                torch.ones(batch, 12, dtype=torch.bool, device=node_features.device),
            ),
            dim=1,
        )
        query = torch.cat((pose_query, drive_features), dim=-1)
        solution, _, _ = solve_metric_contact_resolvent(
            query,
            metric_factor,
            all_jacobian,
            all_rhs,
            all_support,
            iterations=self.iterations,
        )
        return solution[:, :6], solution[:, 6:]


class ContactDualPotentialCell(nn.Module):
    """Gradient of a conditional convex dual dissipation potential.

    For fixed geometry/history context the returned map ``force -> motion`` is
    the gradient of a convex 1-smooth scalar potential.  Its Jacobian is
    positive semidefinite with spectral norm below one, hence the map is firmly
    nonexpansive and is the resolvent of a maximal monotone operator.  The
    command enters only through the generalized force; otherwise conditioning
    on it would invalidate that operator statement.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        *,
        history_dim: int = 0,
        ridge_count: int = 24,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or ridge_count <= 0:
            raise ValueError("hidden_dim and ridge_count must be positive")
        if history_dim not in {0, 6}:
            raise ValueError("dual-potential history_dim must be zero or six")
        self.local_encoder = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.ridge_count = int(ridge_count)
        output_dim = 12 + ridge_count * 14 + 1
        self.potential_head = nn.Sequential(
            nn.Linear(hidden_dim + 1 + history_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        final = self.potential_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        # Start close to the known free incremental response on joint forces,
        # while pose motion must arise from learned cross-coordinate ridges.
        diagonal_fraction = 0.8 / 0.9
        diagonal_bias = torch.logit(torch.tensor(diagonal_fraction))
        with torch.no_grad():
            final.bias[:12].fill_(diagonal_bias)
            final.bias[-1] = -2.0
        # A DCT basis mixes pose and joint dual coordinates from the first
        # update.  Coordinate-axis ridges cannot initially transmit a joint-
        # only actuator force to pose and were empirically trapped near the
        # identity-pose solution despite a decreasing joint loss.
        row = torch.arange(12, dtype=torch.float32)[:, None]
        column = torch.arange(12, dtype=torch.float32)[None, :]
        orthogonal = torch.cos(torch.pi * (column + 0.5) * row / 12.0)
        orthogonal[0] *= 1.0 / 12.0**0.5
        orthogonal[1:] *= (2.0 / 12.0) ** 0.5
        signs = torch.where(
            torch.arange(ridge_count)[:, None] < 12,
            torch.ones(ridge_count, 1),
            -torch.ones(ridge_count, 1),
        )
        base = signs * orthogonal[torch.arange(ridge_count) % 12]
        self.register_buffer("base_directions", base)
        self.history_dim = int(history_dim)

    def forward(
        self,
        node_features: Tensor,
        normalized_points: Tensor,
        aperture_previous: Tensor,
        *,
        drive_features: Tensor,
        pose_features: Tensor,
        history_features: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if node_features.ndim != 3 or node_features.shape[-1] != 13:
            raise ValueError("dual-potential node_features must have shape [B,M,13]")
        batch, points, _ = node_features.shape
        if normalized_points.shape != (batch, points, 3):
            raise ValueError("normalized_points must have shape [B,M,3]")
        if drive_features.shape != (batch, 6):
            raise ValueError("drive_features must have shape [B,6]")
        if pose_features.shape != (batch, 6):
            raise ValueError("pose_features must have shape [B,6]")
        if self.history_dim:
            if history_features is None or history_features.shape != (batch, 6):
                raise ValueError("dual-potential cell requires six history features")
            history = history_features
        else:
            if history_features is not None:
                raise ValueError("history features supplied to a stateless cell")
            history = node_features.new_zeros(batch, 0)

        local = self.local_encoder(
            torch.cat((node_features, normalized_points), dim=-1)
        )
        pooled = local.mean(dim=1)
        packed = self.potential_head(
            torch.cat((pooled, aperture_previous[:, None], history), dim=-1)
        )
        offset = 0
        diagonal = 0.9 * torch.sigmoid(packed[:, offset : offset + 12])
        offset += 12
        direction_delta = packed[
            :, offset : offset + self.ridge_count * 12
        ].reshape(batch, self.ridge_count, 12)
        offset += self.ridge_count * 12
        weight_logits = packed[:, offset : offset + self.ridge_count]
        offset += self.ridge_count
        biases = packed[:, offset : offset + self.ridge_count]
        offset += self.ridge_count
        capacity_raw = packed[:, offset]

        directions = self.base_directions.to(dtype=packed.dtype) + 0.5 * torch.tanh(
            direction_delta
        )
        directions = directions / torch.linalg.vector_norm(
            directions, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        # sigmoid' <= 1/4 and ||a_j a_j^T||=1.  This budget gives
        # ||Jacobian|| <= max(diagonal) + sum(weights)/4 < 0.98.
        remaining = (0.98 - diagonal.amax(dim=-1)).clamp_min(1e-4)
        total_weight = 4.0 * remaining * torch.sigmoid(capacity_raw)
        weights = total_weight[:, None] * torch.softmax(weight_logits, dim=-1)

        generalized_force = torch.cat((pose_features, drive_features), dim=-1)
        ridge_argument = torch.einsum(
            "bri,bi->br", directions, generalized_force
        )
        activation_change = torch.sigmoid(ridge_argument + biases) - torch.sigmoid(
            biases
        )
        motion = diagonal * generalized_force + torch.einsum(
            "br,br,bri->bi", weights, activation_change, directions
        )
        return motion[:, :6], motion[:, 6:]


class ContactMonotoneResolventCell(nn.Module):
    """Resolvent of a conditional, generally non-cyclic monotone operator.

    For a fixed geometry/history context this cell constructs

        A_theta = S_theta + W_theta,   S_theta = L_theta L_theta^T >= 0,
        W_theta^T = -W_theta,

    and returns ``(I + A_theta)^-1 force``.  The skew part permits directional
    pose--actuator coupling that no scalar potential can express, while A is
    still maximal monotone.  Consequently its resolvent remains firmly
    nonexpansive.  As in :class:`ContactDualPotentialCell`, the command is the
    operator query rather than part of the context.
    """

    _LOWER_ROW, _LOWER_COLUMN = torch.tril_indices(12, 12)
    _UPPER_ROW, _UPPER_COLUMN = torch.triu_indices(12, 12, offset=1)

    def __init__(
        self,
        hidden_dim: int = 64,
        *,
        history_dim: int = 0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if history_dim not in {0, 6}:
            raise ValueError("monotone-resolvent history_dim must be zero or six")
        self.local_encoder = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        packed_size = 12 * 13 // 2 + 12 * 11 // 2
        self.operator_head = nn.Sequential(
            nn.Linear(hidden_dim + 1 + history_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, packed_size),
        )
        final = self.operator_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        # A=0.25 I initializes the joint response at 0.8 times free motion.
        # Nonzero Cholesky diagonals also give off-diagonal S entries a
        # first-order training signal, unlike an initialization L=0.
        self.register_buffer(
            "base_cholesky_diagonal", torch.full((12,), 0.5)
        )
        self.history_dim = int(history_dim)

    def _operator(self, packed: Tensor) -> Tensor:
        batch = packed.shape[0]
        lower_size = 12 * 13 // 2
        lower_raw = packed[:, :lower_size]
        skew_raw = packed[:, lower_size:]
        row = self._LOWER_ROW.to(device=packed.device)
        column = self._LOWER_COLUMN.to(device=packed.device)
        factor = packed.new_zeros(batch, 12, 12)
        base = self.base_cholesky_diagonal.to(dtype=packed.dtype)
        factor[:, row, column] = 0.2 * torch.tanh(lower_raw)
        diagonal_mask = (self._LOWER_ROW == self._LOWER_COLUMN).to(
            device=packed.device
        )
        diagonal_raw = lower_raw[:, diagonal_mask]
        diagonal = torch.arange(12, device=packed.device)
        factor[:, diagonal, diagonal] = base * torch.exp(
            0.5 * torch.tanh(diagonal_raw)
        )
        symmetric = factor @ factor.transpose(-1, -2)

        upper_row = self._UPPER_ROW.to(device=packed.device)
        upper_column = self._UPPER_COLUMN.to(device=packed.device)
        skew = packed.new_zeros(batch, 12, 12)
        skew_value = 0.75 * torch.tanh(skew_raw)
        skew[:, upper_row, upper_column] = skew_value
        skew[:, upper_column, upper_row] = -skew_value
        return symmetric + skew

    def forward(
        self,
        node_features: Tensor,
        normalized_points: Tensor,
        aperture_previous: Tensor,
        *,
        drive_features: Tensor,
        pose_features: Tensor,
        history_features: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if node_features.ndim != 3 or node_features.shape[-1] != 13:
            raise ValueError(
                "monotone-resolvent node_features must have shape [B,M,13]"
            )
        batch, points, _ = node_features.shape
        if normalized_points.shape != (batch, points, 3):
            raise ValueError("normalized_points must have shape [B,M,3]")
        if drive_features.shape != (batch, 6):
            raise ValueError("drive_features must have shape [B,6]")
        if pose_features.shape != (batch, 6):
            raise ValueError("pose_features must have shape [B,6]")
        if self.history_dim:
            if history_features is None or history_features.shape != (batch, 6):
                raise ValueError(
                    "monotone-resolvent cell requires six history features"
                )
            history = history_features
        else:
            if history_features is not None:
                raise ValueError("history features supplied to a stateless cell")
            history = node_features.new_zeros(batch, 0)

        local = self.local_encoder(
            torch.cat((node_features, normalized_points), dim=-1)
        )
        pooled = local.mean(dim=1)
        packed = self.operator_head(
            torch.cat((pooled, aperture_previous[:, None], history), dim=-1)
        )
        operator = self._operator(packed)
        identity = torch.eye(
            12, dtype=operator.dtype, device=operator.device
        ).expand(batch, 12, 12)
        generalized_force = torch.cat((pose_features, drive_features), dim=-1)
        motion = torch.linalg.solve(
            identity + operator, generalized_force.unsqueeze(-1)
        ).squeeze(-1)
        return motion[:, :6], motion[:, 6:]


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
        operator_layers: int = 1,
        contact_features: Literal[
            "gap", "gap_jq", "trial_current_gap"
        ] = "gap",
        contact_head: Literal[
            "direct", "normal_cone", "friction_cone", "implicit_resolvent",
            "dual_potential", "monotone_resolvent",
        ] = "direct",
        friction_coefficient: float = 2.4,
        actuation_conditioning: Literal["aperture", "drive_error"] = "aperture",
        pose_predictor: Literal["identity", "continuation"] = "identity",
        pose_corrector: Literal["cone", "predictor_only"] = "cone",
        continuation_factor: float = 0.75,
        resolvent_iterations: int = 32,
        resolvent_pose_weight: float = 100.0,
        resolvent_constraint_gap_m: float = 0.0,
        resolvent_pose_query_factor: float = 0.0,
        pose_update: Literal["se3_left", "decoupled"] = "se3_left",
        history_conditioning: Literal["none", "pose_delta"] = "none",
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
        if operator_layers <= 0:
            raise ValueError("operator_layers must be positive")
        if contact_features not in {"gap", "gap_jq", "trial_current_gap"}:
            raise ValueError(
                "contact_features must be 'gap', 'gap_jq', or "
                "'trial_current_gap'"
            )
        if contact_head not in {
            "direct",
            "normal_cone",
            "friction_cone",
            "implicit_resolvent",
            "dual_potential",
            "monotone_resolvent",
        }:
            raise ValueError(
                "contact_head must be 'direct', 'normal_cone', "
                "'friction_cone', 'implicit_resolvent', 'dual_potential', or "
                "'monotone_resolvent'"
            )
        if friction_coefficient <= 0:
            raise ValueError("friction_coefficient must be positive")
        if actuation_conditioning not in {"aperture", "drive_error"}:
            raise ValueError(
                "actuation_conditioning must be 'aperture' or 'drive_error'"
            )
        if actuation_conditioning == "drive_error" and contact_head in {
            "direct",
        }:
            raise ValueError(
                "drive_error actuation conditioning is available only for cone heads"
            )
        if pose_predictor not in {"identity", "continuation"}:
            raise ValueError("pose_predictor must be 'identity' or 'continuation'")
        if pose_corrector not in {"cone", "predictor_only"}:
            raise ValueError("pose_corrector must be 'cone' or 'predictor_only'")
        if continuation_factor < 0:
            raise ValueError("continuation_factor cannot be negative")
        if pose_predictor == "continuation" and contact_head in {
            "direct",
        }:
            raise ValueError("continuation pose predictor requires a cone head")
        if pose_corrector == "predictor_only" and pose_predictor != "continuation":
            raise ValueError("predictor_only corrector requires continuation predictor")
        if resolvent_iterations <= 0:
            raise ValueError("resolvent_iterations must be positive")
        if resolvent_pose_weight <= 0:
            raise ValueError("resolvent_pose_weight must be positive")
        if not torch.isfinite(torch.tensor(resolvent_constraint_gap_m)):
            raise ValueError("resolvent_constraint_gap_m must be finite")
        if (
            not torch.isfinite(torch.tensor(resolvent_pose_query_factor))
            or resolvent_pose_query_factor < 0
        ):
            raise ValueError("resolvent_pose_query_factor must be finite and non-negative")
        if contact_head == "implicit_resolvent":
            if contact_features != "gap":
                raise ValueError("implicit_resolvent requires contact_features='gap'")
            if actuation_conditioning != "drive_error":
                raise ValueError("implicit_resolvent requires drive_error actuation")
            if pose_predictor != "identity" or pose_corrector != "cone":
                raise ValueError("implicit_resolvent uses a single coupled state solve")
            if pose_update != "decoupled":
                raise ValueError("implicit_resolvent requires the product retraction")
            if (
                resolvent_pose_query_factor > 0
                and history_conditioning != "pose_delta"
            ):
                raise ValueError(
                    "a nonzero resolvent pose query requires pose_delta history"
                )
        if contact_head in {"dual_potential", "monotone_resolvent"}:
            if contact_features != "gap":
                raise ValueError(
                    f"{contact_head} requires contact_features='gap'"
                )
            if actuation_conditioning != "drive_error":
                raise ValueError(f"{contact_head} requires drive_error actuation")
            if pose_predictor != "identity" or pose_corrector != "cone":
                raise ValueError(
                    f"{contact_head} directly returns the coupled increment"
                )
            if pose_update != "decoupled":
                raise ValueError(
                    f"{contact_head} requires the product retraction"
                )
            if (
                resolvent_pose_query_factor > 0
                and history_conditioning != "pose_delta"
            ):
                raise ValueError(
                    f"a nonzero {contact_head} pose query requires pose_delta history"
                )
        if pose_update not in {"se3_left", "decoupled"}:
            raise ValueError("pose_update must be 'se3_left' or 'decoupled'")
        if history_conditioning not in {"none", "pose_delta"}:
            raise ValueError(
                "history_conditioning must be 'none' or 'pose_delta'"
            )
        self.sdf_scale = float(sdf_scale)
        self.delta_gate = float(delta_gate)
        self.contact_offset_sum = float(contact_offset_sum)
        self.contact_features = contact_features
        self.contact_head = contact_head
        self.actuation_conditioning = actuation_conditioning
        self.pose_predictor = pose_predictor
        self.pose_corrector = pose_corrector
        self.continuation_factor = float(continuation_factor)
        self.resolvent_constraint_gap_m = float(resolvent_constraint_gap_m)
        self.resolvent_pose_query_factor = float(resolvent_pose_query_factor)
        self.pose_update = pose_update
        self.history_conditioning = history_conditioning
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
        history_dim = 6 if history_conditioning == "pose_delta" else 0
        if contact_head == "implicit_resolvent":
            self.contact_cell = ContactImplicitResolventCell(
                hidden_dim,
                pose_weight=resolvent_pose_weight,
                iterations=resolvent_iterations,
            )
        elif contact_head == "dual_potential":
            self.contact_cell = ContactDualPotentialCell(
                hidden_dim,
                history_dim=history_dim,
            )
        elif contact_head == "monotone_resolvent":
            self.contact_cell = ContactMonotoneResolventCell(
                hidden_dim,
                history_dim=history_dim,
            )
        elif contact_head in {"normal_cone", "friction_cone"}:
            self.contact_cell = ContactNormalConeCell(
                hidden_dim,
                input_dim=14 if contact_features == "trial_current_gap" else 13,
                drive_dim=6 if actuation_conditioning == "drive_error" else 0,
                friction_coefficient=(
                    friction_coefficient if contact_head == "friction_cone" else 0.0
                ),
                history_dim=history_dim,
            )
        else:
            input_dimensions = {"gap": 1, "gap_jq": 7, "trial_current_gap": 2}
            self.contact_cell = ContactIntegralCell(
                hidden_dim,
                input_dim=input_dimensions[contact_features],
                operator_layers=operator_layers,
                history_dim=history_dim,
            )
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

    def _gripper_points_and_joint_jacobian(
        self, joint_position: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return FK points and analytic ``dy_m / dr_j`` in gripper frame."""

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
        open_displacement = self.link_open_positions - self.link_pivots
        rotated_displacement = torch.einsum(
            "...lij,lj->...li", position_rotation, open_displacement
        )
        link_position = rotated_displacement + self.link_pivots
        point_rotation = link_rotation[..., self.surface_link_index, :, :]
        rotated_local = torch.einsum(
            "...mij,mj->...mi", point_rotation, self.surface_local_points
        )
        point_position = link_position[..., self.surface_link_index, :]
        points = rotated_local + point_position

        point_axes = self.link_axes[self.surface_link_index].expand_as(rotated_local)
        rotation_derivative = torch.linalg.cross(
            point_axes, rotated_local, dim=-1
        )
        position_derivative = torch.linalg.cross(
            point_axes,
            rotated_displacement[..., self.surface_link_index, :],
            dim=-1,
        )
        rotation_coefficients = self.link_rotation_joint_coefficients[
            self.surface_link_index
        ]
        position_coefficients = self.link_position_joint_coefficients[
            self.surface_link_index
        ]
        joint_jacobian = (
            rotation_derivative[..., :, :, None]
            * rotation_coefficients[:, None, :]
            + position_derivative[..., :, :, None]
            * position_coefficients[:, None, :]
        )
        return points, joint_jacobian

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

    def _contact_gap_and_pose_jacobian(
        self,
        state: PoseState,
        points_gripper: Tensor,
        sdf: SDFBatch,
    ) -> tuple[Tensor, Tensor]:
        """Return contact gap and dimensionless left/spatial pose Jacobian."""

        # The surrounding neural cell may run under BF16 autocast, but voxel
        # coordinates, SDF derivatives, normals and frame transforms are
        # explicitly part of the float32 geometry contract.
        with torch.autocast(device_type=state.device.type, enabled=False):
            rotation = state.rotation.float()
            position = state.position.float()
            points = points_gripper.float()
            relative = points - position[:, None, :]
            points_object = torch.einsum(
                "bij,bmj->bmi", rotation.transpose(-1, -2), relative
            )
            geometric_gap, gradient_object = sample_sdf_with_gradient(
                sdf.values,
                sdf.origin,
                sdf.voxel_size,
                points_object,
                sample_to_object=sdf.sample_to_object,
                outside_value=sdf.outside_value,
            )
            gradient_norm = torch.linalg.vector_norm(
                gradient_object, dim=-1, keepdim=True
            )
            normal_object = gradient_object / gradient_norm.clamp_min(1e-8)
            normal_object = torch.where(
                gradient_norm > 0.0, normal_object, torch.zeros_like(normal_object)
            )
            normal_gripper = torch.einsum(
                "bij,bmj->bmi", rotation, normal_object
            )
            normalized_points = points / self.length_scale
            moment_arm = torch.linalg.cross(
                normalized_points, normal_gripper, dim=-1
            )
            pose_jacobian = torch.cat((-normal_gripper, -moment_arm), dim=-1)
            return geometric_gap - self.contact_offset_sum, pose_jacobian

    def _contact_gap_and_full_jacobian(
        self,
        state: PoseState,
        sdf: SDFBatch,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return contact gap, trial points, and ``d(h/s)/dz`` for all 12 DOFs.

        ``z`` contains translation over ``length_scale``, a spatial rotation,
        and joint residual over joint travel.  Its rotational coordinate is
        made exactly consistent with the selected pose retraction.
        """

        with torch.autocast(device_type=state.device.type, enabled=False):
            rotation = state.rotation.float()
            position = state.position.float()
            joints = state.joint_position.float()
            points, point_joint_jacobian = self._gripper_points_and_joint_jacobian(
                joints
            )
            relative = points - position[:, None, :]
            points_object = torch.einsum(
                "bij,bmj->bmi", rotation.transpose(-1, -2), relative
            )
            geometric_gap, gradient_object = sample_sdf_with_gradient(
                sdf.values,
                sdf.origin,
                sdf.voxel_size,
                points_object,
                sample_to_object=sdf.sample_to_object,
                outside_value=sdf.outside_value,
            )
            gradient_norm = torch.linalg.vector_norm(
                gradient_object, dim=-1, keepdim=True
            )
            normal_object = gradient_object / gradient_norm.clamp_min(1e-8)
            normal_object = torch.where(
                gradient_norm > 0.0,
                normal_object,
                torch.zeros_like(normal_object),
            )
            normal_gripper = torch.einsum(
                "bij,bmj->bmi", rotation, normal_object
            )
            translation_jacobian = -normal_gripper * self.length_scale
            rotation_arm = points if self.pose_update == "se3_left" else relative
            rotation_jacobian = -torch.linalg.cross(
                rotation_arm, normal_gripper, dim=-1
            )
            joint_jacobian = torch.einsum(
                "bmc,bmcj->bmj", normal_gripper, point_joint_jacobian
            ) * self.joint_travel_range
            full_jacobian = torch.cat(
                (translation_jacobian, rotation_jacobian, joint_jacobian), dim=-1
            ) / self.sdf_scale
            return (
                geometric_gap - self.contact_offset_sum,
                points,
                full_jacobian,
            )

    def _friction_cone_directions(
        self,
        state: PoseState,
        points: Tensor,
        normal_jacobian: Tensor,
    ) -> Tensor:
        """Return generalized directions for normal and two tangent forces."""

        with torch.autocast(device_type=state.device.type, enabled=False):
            points = points.float()
            normal = (
                -normal_jacobian[..., :3].float()
                * self.sdf_scale
                / self.length_scale
            )
            normal_norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
            normal = normal / normal_norm.clamp_min(1e-8)
            normal = torch.where(
                normal_norm > 0.0, normal, torch.zeros_like(normal)
            )
            x_axis = torch.tensor(
                [1.0, 0.0, 0.0], dtype=normal.dtype, device=normal.device
            ).expand_as(normal)
            y_axis = torch.tensor(
                [0.0, 1.0, 0.0], dtype=normal.dtype, device=normal.device
            ).expand_as(normal)
            reference = torch.where(
                (normal[..., :1].abs() < 0.9), x_axis, y_axis
            )
            tangent_one = torch.linalg.cross(normal, reference, dim=-1)
            tangent_one = tangent_one / torch.linalg.vector_norm(
                tangent_one, dim=-1, keepdim=True
            ).clamp_min(1e-8)
            tangent_two = torch.linalg.cross(normal, tangent_one, dim=-1)
            directions = torch.stack((normal, tangent_one, tangent_two), dim=-2)

            _, point_joint_jacobian = self._gripper_points_and_joint_jacobian(
                state.joint_position.float()
            )
            relative = points - state.position.float()[:, None, :]
            rotation_arm = points if self.pose_update == "se3_left" else relative
            translation = -directions * self.length_scale
            rotation = -torch.linalg.cross(
                rotation_arm.unsqueeze(-2).expand_as(directions),
                directions,
                dim=-1,
            )
            joints = torch.einsum(
                "bmkc,bmcj->bmkj", directions, point_joint_jacobian
            ) * self.joint_travel_range
            generalized = torch.cat((translation, rotation, joints), dim=-1)
            return generalized.transpose(-1, -2) / self.sdf_scale

    def forward_step(
        self,
        state: PoseState,
        next_command: Tensor | float,
        sdf: SDFBatch,
        *,
        previous_state: PoseState | None = None,
        return_aux: bool = False,
    ) -> PoseState | tuple[PoseState, StepAux]:
        if state.joint_position.ndim != 2:
            raise ValueError("forward_step expects a flat batch")
        batch = state.joint_position.shape[0]
        if sdf.sample_to_object.shape != (batch,):
            raise ValueError("SDF mapping length must equal state batch size")
        if self.pose_predictor == "continuation":
            if previous_state is None:
                raise ValueError("continuation pose predictor requires previous_state")
            if previous_state.shape != state.shape:
                raise ValueError("previous_state shape must match state shape")
        history_features = None
        if self.history_conditioning == "pose_delta":
            if previous_state is None:
                raise ValueError(
                    "pose-delta history conditioning requires previous_state"
                )
            if previous_state.shape != state.shape:
                raise ValueError("previous_state shape must match state shape")
            previous_to_current = (
                state.rotation @ previous_state.rotation.transpose(-1, -2)
            )
            history_features = torch.cat(
                (
                    (state.position - previous_state.position) / self.length_scale,
                    so3_log_vector(previous_to_current),
                ),
                dim=-1,
            )

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

        gate_trial_state = PoseState(state.rotation, state.position, trial_joint)
        if self.pose_predictor == "continuation":
            assert previous_state is not None
            incoming_rotation = (
                state.rotation @ previous_state.rotation.transpose(-1, -2)
            )
            predictor_rotation = (
                so3_exp(
                    self.continuation_factor * so3_log_vector(incoming_rotation)
                )
                @ state.rotation
            )
            predictor_position = state.position + self.continuation_factor * (
                state.position - previous_state.position
            )
            trial_state = PoseState(
                predictor_rotation,
                predictor_position,
                trial_joint,
            )
        else:
            trial_state = gate_trial_state
        pose_jacobian = None
        full_contact_jacobian = None
        friction_directions = None
        gate_gap = None
        if self.contact_head in {
            "implicit_resolvent", "dual_potential", "monotone_resolvent"
        }:
            # Contact activation retains the exact legacy free-motion test.
            # These operator heads are conditioned on the admissible current
            # state rather than on the often deeply penetrating free trial.
            gate_gap = self.query_gap(gate_trial_state, sdf)
            trial_gap, trial_points, full_contact_jacobian = (
                self._contact_gap_and_full_jacobian(state, sdf)
            )
        elif self.contact_head in {"normal_cone", "friction_cone"}:
            contact_trial_state = (
                gate_trial_state
                if self.pose_corrector == "predictor_only"
                else trial_state
            )
            trial_gap, trial_points, full_contact_jacobian = (
                self._contact_gap_and_full_jacobian(contact_trial_state, sdf)
            )
            if self.pose_predictor == "continuation":
                gate_gap = (
                    trial_gap
                    if self.pose_corrector == "predictor_only"
                    else self.query_gap(gate_trial_state, sdf)
                )
            if self.contact_head == "friction_cone":
                friction_directions = self._friction_cone_directions(
                    contact_trial_state,
                    trial_points,
                    full_contact_jacobian,
                )
        elif self.contact_features == "gap_jq":
            trial_gap, pose_jacobian = self._contact_gap_and_pose_jacobian(
                state, trial_points, sdf
            )
        else:
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
        active = (
            trial_gap if gate_gap is None else gate_gap
        ).amin(dim=-1) <= self.delta_gate

        current_gap = None
        if self.contact_features == "trial_current_gap":
            # The formal state is (q, r), but the original gap-only cell saw
            # geometry only at R_free(command_next) and reduced r_k to A(r_k).
            # Preserve canonical point correspondence while exposing which
            # actual finger/link samples support the current equilibrium.
            current_gap = self.query_gap(state, sdf)

        # The exact inactive branch stays at q_k even when the active contact
        # solver uses a continuation predictor.
        next_rotation = gate_trial_state.rotation.clone()
        next_position = gate_trial_state.position.clone()
        next_joint = gate_trial_state.joint_position.clone()
        joint_residual = torch.zeros_like(next_joint)
        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        if active_indices.numel():
            if self.contact_head in {
                "implicit_resolvent", "dual_potential", "monotone_resolvent"
            }:
                # Current-state FK points depend on r_k and therefore cannot
                # use the command-keyed canonical point cache.
                assert trial_points.ndim == 3
                groups = [
                    (
                        active_indices,
                        trial_points.index_select(0, active_indices)
                        / self.length_scale,
                        None,
                    )
                ]
            elif shared_command:
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
                node_features = group_gap
                group_full_jacobian = None
                if full_contact_jacobian is not None:
                    group_full_jacobian = full_contact_jacobian.index_select(
                        0, group_indices
                    )
                    normal_cone_features = [group_gap.unsqueeze(-1)]
                    if self.contact_features == "trial_current_gap":
                        assert current_gap is not None
                        normal_cone_features.append(
                            (
                                current_gap.index_select(0, group_indices)
                                / self.sdf_scale
                            ).unsqueeze(-1)
                        )
                    normal_cone_features.append(group_full_jacobian)
                    node_features = torch.cat(normal_cone_features, dim=-1)
                elif pose_jacobian is not None:
                    node_features = torch.cat(
                        (
                            group_gap.unsqueeze(-1),
                            pose_jacobian.index_select(0, group_indices),
                        ),
                        dim=-1,
                    )
                elif current_gap is not None:
                    node_features = torch.stack(
                        (
                            group_gap,
                            current_gap.index_select(0, group_indices)
                            / self.sdf_scale,
                        ),
                        dim=-1,
                    )
                previous_aperture = self.aperture_from_joints(
                    state.joint_position.index_select(0, group_indices)
                ) / self.length_scale
                trial_aperture = (
                    command_batch.index_select(0, group_indices)
                    / self.length_scale
                )
                group_history = (
                    None
                    if history_features is None
                    else history_features.index_select(0, group_indices)
                )
                if self.contact_head == "implicit_resolvent":
                    assert group_full_jacobian is not None
                    assert normalized_points.ndim == 3
                    current_joint = state.joint_position.index_select(
                        0, group_indices
                    )
                    group_drive = (
                        trial_joint.index_select(0, group_indices) - current_joint
                    ) / self.joint_travel_range
                    group_pose_query = (
                        torch.zeros_like(group_drive)
                        if history_features is None
                        else self.resolvent_pose_query_factor
                        * history_features.index_select(0, group_indices)
                    )
                    joint_min = self.free_joint_knots.amin(dim=0)
                    joint_max = self.free_joint_knots.amax(dim=0)
                    joint_lower = (
                        joint_min - current_joint
                    ) / self.joint_travel_range
                    joint_upper = (
                        joint_max - current_joint
                    ) / self.joint_travel_range
                    geometric_gap = (
                        trial_gap.index_select(0, group_indices)
                        + self.contact_offset_sum
                    )
                    contact_rhs = (
                        self.resolvent_constraint_gap_m - geometric_gap
                    ) / self.sdf_scale
                    contact_support = (
                        group_gap <= self.delta_gate / self.sdf_scale
                    ) & (
                        torch.linalg.vector_norm(group_full_jacobian, dim=-1)
                        > 1e-8
                    )
                    # Degenerate/out-of-grid normals should not silently turn
                    # contact on, while a valid nearest sample guarantees that
                    # every active state has at least one VI constraint.
                    no_support = ~contact_support.any(dim=-1)
                    if no_support.any():
                        nearest = group_gap.argmin(dim=-1)
                        valid_nearest = torch.linalg.vector_norm(
                            group_full_jacobian[
                                torch.arange(
                                    len(group_indices), device=group_indices.device
                                ),
                                nearest,
                            ],
                            dim=-1,
                        ) > 1e-8
                        rows = torch.nonzero(
                            no_support & valid_nearest, as_tuple=False
                        ).flatten()
                        contact_support[rows, nearest.index_select(0, rows)] = True
                    raw_twist, normalized_joint_residual = self.contact_cell(
                        node_features,
                        normalized_points,
                        previous_aperture,
                        trial_aperture,
                        contact_jacobian=group_full_jacobian,
                        contact_rhs=contact_rhs,
                        contact_support=contact_support,
                        drive_features=group_drive,
                        pose_query=group_pose_query,
                        joint_lower=joint_lower,
                        joint_upper=joint_upper,
                    )
                elif self.contact_head in {
                    "dual_potential", "monotone_resolvent"
                }:
                    assert group_full_jacobian is not None
                    assert normalized_points.ndim == 3
                    current_joint = state.joint_position.index_select(
                        0, group_indices
                    )
                    group_drive = (
                        trial_joint.index_select(0, group_indices) - current_joint
                    ) / self.joint_travel_range
                    group_pose_query = (
                        torch.zeros_like(group_drive)
                        if group_history is None
                        else self.resolvent_pose_query_factor * group_history
                    )
                    raw_twist, normalized_joint_residual = self.contact_cell(
                        node_features,
                        normalized_points,
                        previous_aperture,
                        drive_features=group_drive,
                        pose_features=group_pose_query,
                        history_features=group_history,
                    )
                elif self.contact_head in {"normal_cone", "friction_cone"}:
                    assert group_full_jacobian is not None
                    group_drive = None
                    if self.actuation_conditioning == "drive_error":
                        group_drive = (
                            trial_joint.index_select(0, group_indices)
                            - state.joint_position.index_select(0, group_indices)
                        ) / self.joint_travel_range
                    raw_twist, normalized_joint_residual = self.contact_cell(
                        node_features,
                        normalized_points,
                        previous_aperture,
                        trial_aperture,
                        contact_jacobian=group_full_jacobian,
                        contact_support=(
                            group_gap <= self.delta_gate / self.sdf_scale
                        ),
                        contact_directions=(
                            None
                            if friction_directions is None
                            else friction_directions.index_select(0, group_indices)
                        ),
                        drive_features=group_drive,
                        history_features=group_history,
                    )
                else:
                    raw_twist, normalized_joint_residual = self.contact_cell(
                        node_features,
                        normalized_points,
                        previous_aperture,
                        trial_aperture,
                        pair_features=active_pair_features,
                        history_features=group_history,
                    )
                if self.pose_corrector == "predictor_only":
                    raw_twist = torch.zeros_like(raw_twist)
                update_state = (
                    state
                    if self.contact_head in {
                        "implicit_resolvent", "dual_potential", "monotone_resolvent"
                    }
                    else trial_state
                )
                current_rotation = update_state.rotation.index_select(
                    0, group_indices
                ).float()
                current_position = update_state.position.index_select(
                    0, group_indices
                ).float()
                if self.pose_update == "se3_left":
                    twist = torch.cat(
                        (raw_twist[:, :3] * self.length_scale, raw_twist[:, 3:]),
                        dim=-1,
                    ).float()
                    active_rotation, active_position = apply_left_increment(
                        current_rotation,
                        current_position,
                        twist,
                    )
                else:
                    # A product-manifold retraction whose translational output
                    # is the displacement supervised by the state metric.  It
                    # removes the origin-dependent omega x p cancellation of a
                    # spatial SE(3) exponential while retaining a left SO(3)
                    # increment in the fixed gripper frame.
                    active_rotation = (
                        so3_exp(raw_twist[:, 3:].float()) @ current_rotation
                    )
                    active_position = current_position + (
                        raw_twist[:, :3].float() * self.length_scale
                    )
                residual = (
                    normalized_joint_residual.float() * self.joint_travel_range
                )
                joint_base = (
                    state.joint_position.index_select(0, group_indices)
                    if self.contact_head in {
                        "implicit_resolvent", "dual_potential", "monotone_resolvent"
                    }
                    else trial_joint.index_select(0, group_indices)
                )
                active_joint = joint_base.float() + residual
                next_rotation[group_indices] = active_rotation.to(next_rotation.dtype)
                next_position[group_indices] = active_position.to(next_position.dtype)
                next_joint[group_indices] = active_joint.to(next_joint.dtype)
                joint_residual[group_indices] = (
                    active_joint
                    - trial_joint.index_select(0, group_indices).float()
                ).to(joint_residual.dtype)

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
        previous = initial_state
        current = initial_state
        for step in range(command_schedule.shape[-1]):
            command = (
                command_schedule[step]
                if command_schedule.ndim == 1
                else command_schedule[:, step]
            )
            next_state = self.forward_step(
                current,
                command,
                sdf,
                previous_state=(
                    previous
                    if (
                        self.history_conditioning == "pose_delta"
                        or self.pose_predictor == "continuation"
                    )
                    else None
                ),
            )
            assert isinstance(next_state, PoseState)
            states.append(next_state)
            previous, current = current, next_state
        return PoseState.stack(states, dim=1)
