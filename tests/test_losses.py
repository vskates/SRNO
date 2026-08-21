from __future__ import annotations

import torch

from srno.losses import combined_loss, feasibility_loss, huberize_squared_norm, state_error
from srno.types import PoseState


def test_feasibility_loss_uses_admissible_gap_boundary() -> None:
    gap = torch.tensor([-0.006, -0.004, 0.001], requires_grad=True)
    loss = feasibility_loss(
        gap,
        sdf_scale=0.01,
        admissible_gap=-0.005,
    )
    assert torch.allclose(loss, torch.tensor((0.1**2) / 3), atol=1e-7)
    loss.backward()
    assert gap.grad is not None
    assert gap.grad[0] < 0
    assert torch.equal(gap.grad[1:], torch.zeros(2))


def test_zero_admissible_gap_preserves_original_loss() -> None:
    gap = torch.tensor([-0.01, 0.0, 0.01])
    expected = torch.relu(-gap / 0.02).square().mean()
    assert torch.equal(feasibility_loss(gap, sdf_scale=0.02), expected)


def test_state_error_uses_mean_joint_error_normalized_by_travel() -> None:
    rotation = torch.eye(3)[None]
    target = PoseState(rotation, torch.zeros(1, 3), torch.zeros(1, 6))
    prediction = PoseState(
        rotation,
        torch.tensor([[0.08, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    total, translation, rotation_error, joints = state_error(
        prediction,
        target,
        length_scale=0.08,
        joint_scale=torch.full((6,), 0.5),
        lambda_joints=2.0,
    )
    assert torch.allclose(translation, torch.ones(1))
    assert torch.equal(rotation_error, torch.zeros(1))
    assert torch.allclose(joints, torch.tensor([4.0 / 6.0]))
    assert torch.allclose(total, torch.tensor([1.0 + 8.0 / 6.0]))


def test_huberized_squared_norm_is_quadratic_then_linear() -> None:
    squared = torch.tensor([0.0, 0.01, 0.04, 0.16], requires_grad=True)
    actual = huberize_squared_norm(squared, delta=0.2)
    expected = torch.tensor([0.0, 0.01, 0.04, 0.12])
    assert torch.allclose(actual, expected)
    actual.sum().backward()
    assert squared.grad is not None
    assert torch.all(torch.isfinite(squared.grad))
    assert torch.allclose(squared.grad, torch.tensor([1.0, 1.0, 1.0, 0.5]))


def test_combined_loss_huberizes_only_pose_components() -> None:
    rotation = torch.eye(3)[None]
    target = PoseState(rotation, torch.zeros(1, 3), torch.zeros(1, 6))
    prediction = PoseState(
        rotation,
        torch.tensor([[0.4, 0.0, 0.0]]),
        torch.ones(1, 6),
    )
    terms = combined_loss(
        prediction,
        target,
        torch.zeros(1, 1),
        length_scale=1.0,
        joint_scale=torch.ones(6),
        sdf_scale=1.0,
        pose_penalty="huber",
        pose_huber_delta=0.2,
    )
    # Translation: 2 * 0.2 * 0.4 - 0.2^2 = 0.12.  Joints remain MSE=1.
    assert torch.allclose(terms.translation, torch.tensor(0.12))
    assert torch.equal(terms.rotation, torch.tensor(0.0))
    assert torch.equal(terms.joints, torch.tensor(1.0))
    assert torch.allclose(terms.flow, torch.tensor(1.12))
