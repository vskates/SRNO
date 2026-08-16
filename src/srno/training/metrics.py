from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from srno.geometry.se3 import rotation_geodesic_angle
from srno.types import PoseState


@dataclass
class MetricAccumulator:
    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    maxima: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, values: Tensor) -> None:
        values = values.detach().float()
        self.totals[name] = self.totals.get(name, 0.0) + float(values.sum().cpu())
        self.counts[name] = self.counts.get(name, 0) + values.numel()

    def add_max(self, name: str, values: Tensor) -> None:
        value = float(values.detach().float().max().cpu())
        self.maxima[name] = max(self.maxima.get(name, float("-inf")), value)

    def compute(self) -> dict[str, float]:
        result = {name: self.totals[name] / self.counts[name] for name in self.totals}
        result.update({f"max_{name}": value for name, value in self.maxima.items()})
        return result


def accumulate_trajectory_metrics(
    accumulator: MetricAccumulator,
    prediction: PoseState,
    target: PoseState,
    command_schedule: Tensor,
    predicted_aperture: Tensor,
    target_aperture: Tensor,
    predicted_gap: Tensor,
    *,
    length_scale: float,
    joint_scale: Tensor,
    lag_threshold: float,
) -> None:
    translation = torch.linalg.vector_norm(prediction.position - target.position, dim=-1)
    rotation = rotation_geodesic_angle(prediction.rotation, target.rotation)
    joint_error = (prediction.joint_position - target.joint_position).abs()
    normalized_joint_error = joint_error / joint_scale
    aperture = (predicted_aperture - target_aperture).abs()
    accumulator.add("translation_m", translation)
    accumulator.add("translation_over_length", translation / length_scale)
    accumulator.add("rotation_rad", rotation)
    accumulator.add("joint_abs_rad", joint_error)
    accumulator.add("joint_abs_over_travel", normalized_joint_error)
    accumulator.add("aperture_m", aperture)
    accumulator.add("aperture_over_length", aperture / length_scale)

    terminal_translation_m = translation[:, -1]
    terminal_translation = terminal_translation_m / length_scale
    terminal_rotation = rotation[:, -1]
    terminal_joint = normalized_joint_error[:, -1].square().mean(dim=-1)
    terminal_aperture_m = aperture[:, -1]
    terminal_aperture = terminal_aperture_m / length_scale
    terminal_dx = torch.sqrt(
        terminal_translation.square()
        + terminal_rotation.square()
        + terminal_joint
    )
    accumulator.add("terminal_dx", terminal_dx)
    accumulator.add("terminal_translation_m", terminal_translation_m)
    accumulator.add("terminal_rotation_rad", terminal_rotation)
    accumulator.add("terminal_joint_rmse_over_travel", terminal_joint.sqrt())
    accumulator.add("terminal_aperture_m", terminal_aperture_m)
    accumulator.add("terminal_aperture_over_length", terminal_aperture)
    accumulator.add("predicted_terminal_aperture_m", predicted_aperture[:, -1])
    accumulator.add("target_terminal_aperture_m", target_aperture[:, -1])

    penetration = torch.relu(-predicted_gap)
    accumulator.add("penetration_m", penetration)
    accumulator.add_max("penetration_m", penetration)

    schedule = command_schedule.view(1, -1)
    predicted_lag = predicted_aperture - schedule
    target_lag = target_aperture - schedule
    accumulator.add("lag_error_m", (predicted_lag - target_lag).abs())
    accumulator.add("lag_error_over_length", (predicted_lag - target_lag).abs() / length_scale)

    def onset(lag: Tensor) -> Tensor:
        significant = lag > lag_threshold
        first = significant.float().argmax(dim=1)
        never = ~significant.any(dim=1)
        return torch.where(never, torch.full_like(first, lag.shape[1]), first)

    accumulator.add(
        "lag_onset_step_error",
        (onset(predicted_lag) - onset(target_lag)).abs().float(),
    )
