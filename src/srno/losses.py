from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from srno.geometry.se3 import rotation_geodesic_angle
from srno.types import PoseState


@dataclass(frozen=True)
class LossTerms:
    total: Tensor
    flow: Tensor
    feasibility: Tensor
    translation: Tensor
    rotation: Tensor
    aperture: Tensor


def state_error(
    prediction: PoseState,
    target: PoseState,
    *,
    length_scale: float,
    lambda_rotation: float = 1.0,
    lambda_aperture: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    translation = ((prediction.position - target.position) / length_scale).square().sum(-1)
    rotation = rotation_geodesic_angle(prediction.rotation, target.rotation).square()
    aperture = ((prediction.aperture - target.aperture) / length_scale).square()
    total = translation + lambda_rotation * rotation + lambda_aperture * aperture
    return total, translation, rotation, aperture


def feasibility_loss(gap: Tensor, *, sdf_scale: float) -> Tensor:
    return torch.relu(-gap / sdf_scale).square().mean()


def combined_loss(
    prediction: PoseState,
    target: PoseState,
    predicted_gap: Tensor,
    *,
    length_scale: float,
    sdf_scale: float,
    lambda_rotation: float = 1.0,
    lambda_aperture: float = 1.0,
    lambda_feasibility: float = 1.0,
) -> LossTerms:
    state, translation, rotation, aperture = state_error(
        prediction,
        target,
        length_scale=length_scale,
        lambda_rotation=lambda_rotation,
        lambda_aperture=lambda_aperture,
    )
    flow = state.mean()
    feasibility = feasibility_loss(predicted_gap, sdf_scale=sdf_scale)
    return LossTerms(
        total=flow + lambda_feasibility * feasibility,
        flow=flow,
        feasibility=feasibility,
        translation=translation.mean(),
        rotation=rotation.mean(),
        aperture=aperture.mean(),
    )
