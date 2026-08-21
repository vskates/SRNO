from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from srno.data.dataset import LocalTransitionBatch
from srno.data.index import file_sha256
from srno.training.config import ExperimentConfig, PathsConfig, TrustRegionConfig
from srno.training.engine import _evaluate_trust_constraint
from srno.training.trust_region import (
    FrozenLocalReference,
    PhysicalBaseline,
    effective_physical_rho,
    effective_rho,
    functional_drift,
    hpr_inequality_penalty,
    physical_one_step_error,
    update_multiplier,
)
from srno.types import PoseState, SDFBatch


def _state(position: torch.Tensor | None = None) -> PoseState:
    return PoseState(
        torch.eye(3).expand(2, 3, 3).clone(),
        torch.zeros(2, 3) if position is None else position,
        torch.zeros(2, 6),
    )


def test_hpr_inequality_penalty_has_projected_gradient() -> None:
    rho = 10.0
    inactive = torch.tensor(-0.2, requires_grad=True)
    value = hpr_inequality_penalty(inactive, multiplier=1.0, rho=rho)
    value.backward()
    assert inactive.grad == 0.0

    active = torch.tensor(0.2, requires_grad=True)
    value = hpr_inequality_penalty(active, multiplier=1.0, rho=rho)
    value.backward()
    assert active.grad == pytest.approx(3.0)
    assert update_multiplier(1.0, 0.2, rho) == pytest.approx(3.0)
    assert update_multiplier(1.0, -0.2, rho) == 0.0


def test_effective_rho_is_scaled_by_squared_radius() -> None:
    assert effective_rho(0.01, 0.0) == pytest.approx(10_000.0)
    assert effective_rho(0.01, 7.0) == 7.0


def test_effective_physical_rho_is_scaled_by_baseline_loss() -> None:
    assert effective_physical_rho(0.004, 0.0) == pytest.approx(250.0)
    assert effective_physical_rho(0.004, 7.0) == 7.0


def test_functional_drift_uses_squared_state_metric() -> None:
    reference = _state()
    prediction = _state(torch.tensor([[0.08, 0.0, 0.0], [0.0, 0.08, 0.0]]))
    total, translation, rotation, joints = functional_drift(
        prediction,
        reference,
        length_scale=0.08,
        joint_scale=torch.ones(6),
    )
    assert total == pytest.approx(1.0)
    assert translation == pytest.approx(1.0)
    assert rotation == 0.0
    assert joints == 0.0


def test_physical_one_step_error_uses_ground_truth_target() -> None:
    target = _state()
    prediction = _state(torch.tensor([[0.08, 0.0, 0.0], [0.0, 0.08, 0.0]]))
    values = physical_one_step_error(
        prediction,
        target,
        length_scale=0.08,
        joint_scale=torch.ones(6),
    )
    assert [float(value) for value in values] == pytest.approx([1.0, 1.0, 0.0, 0.0])


def test_frozen_reference_lookup_and_contract_roundtrip(tmp_path: Path) -> None:
    size = 64
    states = {
        "a": PoseState(
            torch.eye(3).expand(size, 3, 3).clone(),
            torch.arange(size, dtype=torch.float32)[:, None].expand(size, 3).clone(),
            torch.zeros(size, 6),
        ),
        "b": PoseState(
            torch.eye(3).expand(size, 3, 3).clone(),
            -torch.arange(size, dtype=torch.float32)[:, None].expand(size, 3).clone(),
            torch.zeros(size, 6),
        ),
    }
    masks = {"a": torch.zeros(size, dtype=torch.bool), "b": torch.zeros(size, dtype=torch.bool)}
    masks["a"][1] = True
    masks["b"][34] = True
    reference = FrozenLocalReference("m", "g", "i", "l", states, masks)
    batch = LocalTransitionBatch(
        sdf=SDFBatch(
            torch.zeros(2, 2, 2, 2),
            torch.zeros(2, 3),
            torch.ones(2, 3),
            torch.tensor([1, 0]),
            1.0,
        ),
        current=_state(),
        target=_state(),
        next_command=torch.zeros(2),
        target_aperture=torch.zeros(2),
        object_ids=("a", "b"),
        trajectory_index=torch.tensor([1, 0]),
        step_index=torch.tensor([2, 1]),
    )
    looked_up = reference.lookup(batch, device=torch.device("cpu"))
    assert looked_up.position[:, 0].tolist() == [-34.0, 1.0]

    path = tmp_path / "reference.pt"
    reference.save(path)
    loaded = FrozenLocalReference.load(
        path,
        manifest_sha256="m",
        gripper_sha256="g",
        active_index_sha256="i",
        local_checkpoint_sha256="l",
    )
    assert loaded.transition_count == 2
    assert file_sha256(path)
    with pytest.raises(ValueError, match="manifest_sha256 mismatch"):
        FrozenLocalReference.load(
            path,
            manifest_sha256="wrong",
            gripper_sha256="g",
            active_index_sha256="i",
            local_checkpoint_sha256="l",
        )


def test_physical_baseline_contract_roundtrip_and_hash_validation(tmp_path: Path) -> None:
    baseline = PhysicalBaseline(
        manifest_sha256="m",
        gripper_sha256="g",
        active_index_sha256="i",
        local_checkpoint_sha256="l",
        loss=0.4,
        translation=0.1,
        rotation=0.1,
        joints=0.2,
        transitions=17,
    )
    path = tmp_path / "physical-baseline.json"
    baseline.save(path)
    loaded = PhysicalBaseline.load(
        path,
        manifest_sha256="m",
        gripper_sha256="g",
        active_index_sha256="i",
        local_checkpoint_sha256="l",
    )
    assert loaded == baseline
    with pytest.raises(ValueError, match="active_index_sha256 mismatch"):
        PhysicalBaseline.load(
            path,
            manifest_sha256="m",
            gripper_sha256="g",
            active_index_sha256="wrong",
            local_checkpoint_sha256="l",
        )


class _IdentityStepModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.length_scale = 1.0
        self.register_buffer("joint_travel_range", torch.ones(6))

    def forward_step(self, state, next_command, sdf):
        del next_command, sdf
        return state


def _physical_batch(errors: torch.Tensor) -> LocalTransitionBatch:
    count = len(errors)
    current = PoseState(
        torch.eye(3).expand(count, 3, 3).clone(),
        torch.stack((errors, torch.zeros_like(errors), torch.zeros_like(errors)), dim=-1),
        torch.zeros(count, 6),
    )
    target = PoseState(
        torch.eye(3).expand(count, 3, 3).clone(),
        torch.zeros(count, 3),
        torch.zeros(count, 6),
    )
    return LocalTransitionBatch(
        sdf=SDFBatch(
            torch.zeros(1, 2, 2, 2),
            torch.zeros(1, 3),
            torch.ones(1, 3),
            torch.zeros(count, dtype=torch.long),
            1.0,
        ),
        current=current,
        target=target,
        next_command=torch.zeros(count),
        target_aperture=torch.zeros(count),
        object_ids=("object",),
        trajectory_index=torch.arange(count),
        step_index=torch.zeros(count, dtype=torch.long),
    )


def test_physical_constraint_evaluation_is_sample_weighted(tmp_path: Path) -> None:
    config = ExperimentConfig(
        paths=PathsConfig(
            tmp_path / "manifest.json", tmp_path / "active.npz", tmp_path / "run"
        ),
        trust_region=TrustRegionConfig(constraint_target="physical_one_step"),
    )
    metrics = _evaluate_trust_constraint(
        _IdentityStepModel(),
        [_physical_batch(torch.tensor([1.0])), _physical_batch(torch.zeros(3))],
        None,
        config,
        torch.device("cpu"),
        constraint_baseline=0.5,
    )
    assert metrics["physical_one_step_sq_exact"] == pytest.approx(0.25)
    assert metrics["constraint_exact"] == pytest.approx(-0.25)
    assert metrics["constraint_ratio_exact"] == pytest.approx(0.5)
    assert metrics["constraint_samples"] == 4


def test_trust_region_config_is_backward_compatible(tmp_path: Path) -> None:
    config = ExperimentConfig(
        paths=PathsConfig(
            tmp_path / "manifest.json", tmp_path / "active.npz", tmp_path / "run"
        )
    )
    assert math.isclose(config.trust_region.epsilon_dx, 0.002991000423207879)
    assert config.trust_region.reference_cache is None
    assert config.trust_region.constraint_target == "frozen_output"
    with pytest.raises(ValueError, match="epsilon_dx"):
        ExperimentConfig(
            paths=config.paths,
            trust_region=TrustRegionConfig(epsilon_dx=0.0),
        ).validate()
    with pytest.raises(ValueError, match="reference_cache"):
        ExperimentConfig(
            paths=config.paths,
            trust_region=TrustRegionConfig(
                constraint_target="physical_one_step",
                reference_cache=tmp_path / "unused.pt",
            ),
        ).validate()
