from __future__ import annotations

import math

import torch

from srno.geometry.sdf import sample_sdf, sample_sdf_with_gradient
from srno.geometry.se3 import (
    apply_left_increment,
    quaternion_xyzw_to_matrix,
    rotation_geodesic_angle,
    se3_exp,
    so3_exp,
    so3_log_vector,
)


def test_quaternion_and_so3_exp() -> None:
    quaternion = torch.tensor([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
    expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(quaternion_xyzw_to_matrix(quaternion), expected, atol=1e-6)
    assert torch.allclose(so3_exp(torch.tensor([0.0, 0.0, math.pi / 2])), expected, atol=1e-6)
    vector = torch.tensor([[0.02, -0.03, 0.04], [-0.1, 0.2, 0.05]])
    assert torch.allclose(so3_log_vector(so3_exp(vector)), vector, atol=1e-6)


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


def test_analytic_metric_sdf_gradient_anisotropic_boundary_and_outside() -> None:
    voxel = torch.tensor([[0.2, 0.3, 0.4]])
    origin = torch.tensor([[-0.4, -0.6, -0.8]])
    z, y, x = torch.meshgrid(
        torch.arange(5.0), torch.arange(5.0), torch.arange(5.0), indexing="ij"
    )
    physical_x = origin[0, 0] + x * voxel[0, 0]
    physical_y = origin[0, 1] + y * voxel[0, 1]
    physical_z = origin[0, 2] + z * voxel[0, 2]
    values = (1.5 * physical_x - 0.75 * physical_y + 0.25 * physical_z)[None]
    coordinates = torch.tensor(
        [[
            [-0.17, 0.11, -0.09],
            [origin[0, 0], -0.15, 0.0],
            [origin[0, 0] + 4 * voxel[0, 0], -0.15, 0.0],
            [2.0, 0.0, 0.0],
        ]]
    )
    sampled, gradient = sample_sdf_with_gradient(
        values,
        origin,
        voxel,
        coordinates,
        outside_value=3.0,
    )
    expected = torch.tensor([1.5, -0.75, 0.25])
    assert torch.allclose(gradient[0, 0], expected, atol=1e-6)
    assert torch.allclose(gradient[0, 1], expected, atol=1e-6)
    # At the upper endpoint the implementation clamps x1=x0, matching the
    # one-sided constant continuation of the trilinear sampler.
    assert torch.allclose(gradient[0, 2], torch.tensor([0.0, -0.75, 0.25]), atol=1e-6)
    assert sampled[0, 3] == 3.0
    assert torch.equal(gradient[0, 3], torch.zeros(3))
