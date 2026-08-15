from __future__ import annotations

import pytest
import torch

from srno.geometry.gripper import GripperAsset
from srno.model import ContactIntegralCell, SRNOModel
from srno.types import PoseState, SDFBatch


def _state(batch: int = 2) -> PoseState:
    return PoseState(
        torch.eye(3).expand(batch, 3, 3).clone(),
        torch.zeros(batch, 3),
        torch.full((batch,), 0.08),
    )


def _sdf(value: float, batch: int = 2) -> SDFBatch:
    return SDFBatch(
        torch.full((1, 8, 8, 8), value),
        torch.full((1, 3), -0.04),
        torch.full((1, 3), 0.01),
        torch.zeros(batch, dtype=torch.long),
        0.02,
    )


def test_exact_free_bypass_does_not_call_cell(gripper: GripperAsset) -> None:
    model = SRNOModel(gripper, sdf_scale=0.02, delta_gate=0.005)
    calls = 0

    def hook(*_: object) -> None:
        nonlocal calls
        calls += 1

    handle = model.contact_cell.register_forward_hook(hook)
    result, aux = model.forward_step(_state(), 0.07, _sdf(0.02), return_aux=True)
    handle.remove()
    assert calls == 0
    assert not aux.active.any()
    assert torch.equal(result.rotation, _state().rotation)
    assert torch.equal(result.position, _state().position)
    assert torch.equal(result.aperture, torch.full((2,), 0.07))


def test_active_step_aperture_bounds_and_finite_backward(gripper: GripperAsset) -> None:
    model = SRNOModel(gripper, sdf_scale=0.02, delta_gate=0.005)
    state = _state()
    result, aux = model.forward_step(state, torch.tensor([0.07, 0.06]), _sdf(-1e-3), return_aux=True)
    assert aux.active.all()
    assert torch.all(result.aperture >= torch.tensor([0.07, 0.06]))
    assert torch.all(result.aperture <= state.aperture)
    loss = result.position.square().sum() + result.aperture.sum() + result.rotation.sum()
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_integral_cell_is_permutation_invariant_after_pooling() -> None:
    torch.manual_seed(4)
    cell = ContactIntegralCell(16)
    gap = torch.randn(3, 12)
    points = torch.randn(12, 3)
    aperture = torch.full((3,), 0.5)
    twist, eta = cell(gap, points, aperture, aperture)
    permutation = torch.randperm(12)
    permuted_twist, permuted_eta = cell(
        gap[:, permutation], points[permutation], aperture, aperture
    )
    assert torch.allclose(twist, permuted_twist, atol=1e-6)
    assert torch.allclose(eta, permuted_eta, atol=1e-6)


def test_32_step_free_rollout_is_exact(gripper: GripperAsset) -> None:
    model = SRNOModel(gripper, sdf_scale=0.02, delta_gate=0.005)
    schedule = torch.linspace(0.08, 0.0, 33)[1:]
    rollout = model.rollout(_state(), schedule, _sdf(0.02))
    assert rollout.rotation.shape == (2, 33, 3, 3)
    assert torch.equal(rollout.position, torch.zeros(2, 33, 3))
    assert torch.allclose(rollout.aperture[:, 1:], schedule.expand(2, 32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_cuda_contact_outputs_agree(gripper: GripperAsset) -> None:
    torch.manual_seed(7)
    cpu = SRNOModel(gripper, sdf_scale=0.02, delta_gate=0.005).eval()
    gpu = SRNOModel(gripper, sdf_scale=0.02, delta_gate=0.005).cuda().eval()
    gpu.load_state_dict(cpu.state_dict())
    with torch.no_grad():
        cpu_result = cpu.forward_step(_state(1), 0.07, _sdf(-1e-3, 1))
        gpu_result = gpu.forward_step(_state(1).to("cuda"), 0.07, _sdf(-1e-3, 1).to(device="cuda"))
    assert isinstance(cpu_result, PoseState) and isinstance(gpu_result, PoseState)
    assert torch.allclose(cpu_result.position, gpu_result.position.cpu(), atol=2e-5)
    assert torch.allclose(cpu_result.aperture, gpu_result.aperture.cpu(), atol=2e-5)

