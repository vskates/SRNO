from __future__ import annotations

import torch
from torch import Tensor


def skew(vector: Tensor) -> Tensor:
    """Return the skew-symmetric matrix for vectors with final dimension 3."""

    if vector.shape[-1] != 3:
        raise ValueError("vector must end in dimension 3")
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(vector.shape[:-1] + (3, 3))


def quaternion_xyzw_to_matrix(quaternion: Tensor, eps: float = 1e-12) -> Tensor:
    """Convert normalized-or-not XYZW quaternions to rotation matrices."""

    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must end in dimension 4")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if torch.any(norm < eps):
        raise ValueError("zero-norm quaternion")
    x, y, z, w = (quaternion / norm).unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        (
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def so3_exp(rotation_vector: Tensor) -> Tensor:
    """Stable Rodrigues exponential for axis-angle rotation vectors."""

    if rotation_vector.shape[-1] != 3:
        raise ValueError("rotation_vector must end in dimension 3")
    theta2 = (rotation_vector * rotation_vector).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-8))
    small = theta2 < 1e-8
    a = torch.where(
        small,
        1 - theta2 / 6 + theta2 * theta2 / 120,
        torch.sin(theta) / theta,
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24 + theta2 * theta2 / 720,
        (1 - torch.cos(theta)) / theta2.clamp_min(1e-8),
    )
    omega = skew(rotation_vector)
    identity = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    identity = identity.expand(rotation_vector.shape[:-1] + (3, 3))
    return identity + a.unsqueeze(-1) * omega + b.unsqueeze(-1) * (omega @ omega)


def so3_log_vector(rotation: Tensor) -> Tensor:
    """Return the axis-angle logarithm for a rotation matrix.

    The quasistatic history increments using this helper are small.  The
    near-zero branch therefore uses ``sin(theta) ~= theta``; the general
    branch retains the exact atan2 angle and is stable throughout the range
    observed in the collected trajectories.
    """

    if rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must end in (3, 3)")
    cosine = (
        (rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
    ).clamp(-1.0, 1.0)
    sine_axis = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    ) * 0.5
    sine = torch.linalg.vector_norm(sine_axis, dim=-1)
    angle = torch.atan2(sine, cosine)
    scale = torch.where(
        sine > 1e-7,
        angle / sine.clamp_min(1e-7),
        torch.ones_like(sine),
    )
    return sine_axis * scale.unsqueeze(-1)


def se3_exp(twist: Tensor) -> tuple[Tensor, Tensor]:
    """Exponential of a spatial twist ordered as ``(v, omega)``.

    Returns the rotation and translation of the incremental transform.
    """

    if twist.shape[-1] != 6:
        raise ValueError("twist must end in dimension 6")
    translation_velocity, rotation_vector = twist[..., :3], twist[..., 3:]
    theta2 = (rotation_vector * rotation_vector).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-8))
    small = theta2 < 1e-8
    b = torch.where(
        small,
        0.5 - theta2 / 24 + theta2 * theta2 / 720,
        (1 - torch.cos(theta)) / theta2.clamp_min(1e-8),
    )
    c = torch.where(
        small,
        1 / 6 - theta2 / 120 + theta2 * theta2 / 5040,
        (theta - torch.sin(theta))
        / (theta2.clamp_min(1e-8) * theta),
    )
    omega = skew(rotation_vector)
    identity = torch.eye(3, dtype=twist.dtype, device=twist.device)
    identity = identity.expand(twist.shape[:-1] + (3, 3))
    jacobian = identity + b.unsqueeze(-1) * omega + c.unsqueeze(-1) * (omega @ omega)
    return so3_exp(rotation_vector), (jacobian @ translation_velocity.unsqueeze(-1)).squeeze(-1)


def apply_left_increment(
    rotation: Tensor, position: Tensor, twist: Tensor
) -> tuple[Tensor, Tensor]:
    """Apply ``Exp(twist)`` on the left to an object-to-gripper pose."""

    delta_rotation, delta_position = se3_exp(twist)
    return (
        delta_rotation @ rotation,
        (delta_rotation @ position.unsqueeze(-1)).squeeze(-1) + delta_position,
    )


def rotation_geodesic_angle(prediction: Tensor, target: Tensor) -> Tensor:
    """Geodesic SO(3) distance in radians, robust at zero and pi."""

    relative = target.transpose(-1, -2) @ prediction
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1)
    skew_vector = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    ) * 0.5
    sine = torch.linalg.vector_norm(skew_vector, dim=-1)
    return torch.atan2(sine, cosine)
