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
    joints: Tensor


def state_error(
    prediction: PoseState,
    target: PoseState,
    *,
    length_scale: float,
    joint_scale: Tensor,
    lambda_rotation: float = 1.0,
    lambda_joints: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    translation = ((prediction.position - target.position) / length_scale).square().sum(-1)
    rotation = rotation_geodesic_angle(prediction.rotation, target.rotation).square()
    joints = (
        (prediction.joint_position - target.joint_position) / joint_scale
    ).square().mean(dim=-1)
    total = translation + lambda_rotation * rotation + lambda_joints * joints
    return total, translation, rotation, joints


def feasibility_loss(
    gap: Tensor,
    *,
    sdf_scale: float,
    admissible_gap: float = 0.0,
) -> Tensor:
    """Penalize gaps below the configured simulator-consistent boundary."""

    return torch.relu((admissible_gap - gap) / sdf_scale).square().mean()


def huberize_squared_norm(squared_norm: Tensor, *, delta: float) -> Tensor:
    """Huber penalty with the same local scale as a squared vector norm.

    Below ``delta`` this returns ``||e||^2`` exactly.  Above it the radial
    derivative is bounded and the penalty becomes ``2 delta ||e||-delta^2``.
    """

    if delta <= 0:
        raise ValueError("delta must be positive")
    delta_squared = delta * delta
    # Clamp the norm used by the inactive linear branch at the transition.
    # Autograd differentiates both inputs of ``where``; evaluating sqrt at
    # exactly zero would otherwise inject an infinite derivative and NaNs.
    linear_norm = torch.sqrt(squared_norm.clamp_min(delta_squared))
    return torch.where(
        squared_norm <= delta_squared,
        squared_norm,
        2.0 * delta * linear_norm - delta_squared,
    )


def combined_loss(
    prediction: PoseState,
    target: PoseState,
    predicted_gap: Tensor,
    *,
    length_scale: float,
    joint_scale: Tensor,
    sdf_scale: float,
    lambda_rotation: float = 1.0,
    lambda_joints: float = 1.0,
    lambda_feasibility: float = 1.0,
    admissible_gap: float = 0.0,
    pose_penalty: str = "squared",
    pose_huber_delta: float = 0.02,
) -> LossTerms:
    _, translation_squared, rotation_squared, joints = state_error(
        prediction,
        target,
        length_scale=length_scale,
        joint_scale=joint_scale,
        lambda_rotation=lambda_rotation,
        lambda_joints=lambda_joints,
    )
    if pose_penalty == "squared":
        translation = translation_squared
        rotation = rotation_squared
    elif pose_penalty == "huber":
        translation = huberize_squared_norm(
            translation_squared, delta=pose_huber_delta
        )
        rotation = huberize_squared_norm(
            rotation_squared, delta=pose_huber_delta
        )
    else:
        raise ValueError("pose_penalty must be 'squared' or 'huber'")
    state = translation + lambda_rotation * rotation + lambda_joints * joints
    flow = state.mean()
    feasibility = feasibility_loss(
        predicted_gap,
        sdf_scale=sdf_scale,
        admissible_gap=admissible_gap,
    )
    return LossTerms(
        total=flow + lambda_feasibility * feasibility,
        flow=flow,
        feasibility=feasibility,
        translation=translation.mean(),
        rotation=rotation.mean(),
        joints=joints.mean(),
    )
