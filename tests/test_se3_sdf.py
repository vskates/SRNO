from __future__ import annotations

import math

import torch

from srno.geometry.sdf import sample_sdf
from srno.geometry.se3 import (
    apply_left_increment,
    quaternion_xyzw_to_matrix,
    rotation_geodesic_angle,
    se3_exp,
    so3_exp,
)


def test_quaternion_and_so3_exp() -> None:
    quaternion = torch.tensor([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
    expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(quaternion_xyzw_to_matrix(quaternion), expected, atol=1e-6)
    assert torch.allclose(so3_exp(torch.tensor([0.0, 0.0, math.pi / 2])), expected, atol=1e-6)


def test_se3_zero_gradient_and_spatial_left_update() -> None:
    twist = torch.zeros(2, 6, requires_grad=True)
    rotation, translation = se3_exp(twist)
    assert torch.allclose(rotation, torch.eye(3).expand(2, 3, 3))
    assert torch.allclose(translation, torch.zeros(2, 3))
    (rotation.sum() + translation.sum()).backward()
    assert torch.isfinite(twist.grad).all()

    pose_rotation = torch.eye(3).unsqueeze(0)
    pose_position = torch.tensor([[1.0, 0.0, 0.0]])
    increment = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2]])
    updated_rotation, updated_position = apply_left_increment(
        pose_rotation, pose_position, increment
    )
    assert torch.allclose(updated_position, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)
    assert torch.allclose(
        rotation_geodesic_angle(updated_rotation, updated_rotation), torch.zeros(1), atol=1e-7
    )


def test_trilinear_sdf_values_gradients_mapping_and_outside() -> None:
    z, y, x = torch.meshgrid(
        torch.arange(4.0), torch.arange(4.0), torch.arange(4.0), indexing="ij"
    )
    linear = x + 2 * y + 3 * z
    values = torch.stack((linear, linear + 10))
    origin = torch.zeros(2, 3)
    voxel = torch.ones(2, 3)
    coordinates = torch.tensor(
        [[[1.25, 0.5, 2.0], [5.0, 0.0, 0.0]], [[0.5, 1.0, 1.5], [1.0, 1.0, 1.0]]],
        requires_grad=True,
    )
    sampled = sample_sdf(
        values,
        origin,
        voxel,
        coordinates,
        sample_to_object=torch.tensor([0, 0]),
        outside_value=7.0,
    )
    assert torch.allclose(sampled[0], torch.tensor([8.25, 7.0]))
    assert torch.allclose(sampled[1], torch.tensor([7.0, 6.0]))
    sampled.sum().backward()
    assert torch.allclose(coordinates.grad[0, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(coordinates.grad[0, 1], torch.zeros(3))

