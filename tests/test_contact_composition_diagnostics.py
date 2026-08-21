from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.contact_composition_diagnostics import (
    STEP_BANDS,
    _amplification_ratio,
    _checkpoint_config,
    _choose_stratified_indices,
    _make_perturbation_directions,
    _pose_log_error,
    _se3_log,
    _verify_checkpoint_contract,
    _verify_frozen_contract,
)
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import se3_exp
from srno.sim.config import SimulatorConfig
from srno.training.config import ExperimentConfig
from srno.types import PoseState


def test_se3_log_inverts_exp_for_small_and_pure_twists() -> None:
    twists = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.02, -0.01, 0.03, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.04, -0.02, 0.01],
            [0.02, -0.03, 0.01, 0.08, 0.03, -0.05],
            [1e-9, -2e-9, 3e-9, -4e-9, 5e-9, -6e-9],
        ],
        dtype=torch.float64,
    )
    rotation, translation = se3_exp(twists)
    recovered = _se3_log(rotation, translation)
    assert torch.allclose(recovered, twists, atol=2e-9, rtol=2e-8)


def test_pose_log_error_is_zero_for_identical_pose() -> None:
    rotation = torch.eye(3, dtype=torch.float64).repeat(4, 1, 1)
    position = torch.randn(4, 3, dtype=torch.float64)
    joint = torch.randn(4, 6, dtype=torch.float64)
    state = PoseState(rotation, position, joint)
    assert torch.equal(_pose_log_error(state, state), torch.zeros(4, 6, dtype=torch.float64))


def test_joint_perturbation_has_exact_dx_norm_and_respects_limits() -> None:
    joint = np.asarray(
        [
            [0.001, -0.999, 0.5, 0.5, 0.5, -0.5],
            [0.999, -0.001, 0.5, 0.5, 0.5, -0.5],
        ],
        dtype=np.float64,
    )
    lower = np.asarray([0.0, -1.0, 0.0, 0.0, 0.0, -1.0])
    upper = np.asarray([1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    travel = upper - lower
    directions = _make_perturbation_directions(
        joint,
        joint_lower=lower,
        joint_upper=upper,
        joint_travel=travel,
        max_scale=0.01,
        seed=0,
    )
    perturbed = joint + 0.01 * travel[None] * directions.joints
    distance = np.sqrt(np.mean(((perturbed - joint) / travel[None]) ** 2, axis=1))
    assert np.allclose(distance, 0.01, rtol=0.0, atol=1e-12)
    assert np.all(perturbed >= lower[None] - 1e-12)
    assert np.all(perturbed <= upper[None] + 1e-12)
    assert np.allclose(
        np.linalg.norm(directions.translation, axis=1), 1.0, atol=1e-12
    )
    assert np.allclose(
        np.linalg.norm(directions.rotation, axis=1), 1.0, atol=1e-12
    )


def test_amplification_ratio_identity_and_known_gain() -> None:
    denominator = torch.tensor([0.005, 0.01])
    assert torch.equal(
        _amplification_ratio(denominator, denominator), torch.ones(2)
    )
    assert torch.equal(
        _amplification_ratio(2.5 * denominator, denominator),
        torch.full((2,), 2.5),
    )


def test_four_band_selection_is_reproducible_and_balanced() -> None:
    steps = np.repeat(np.arange(1, 32), 5)
    left = _choose_stratified_indices(steps, count=12, seed=7)
    right = _choose_stratified_indices(steps, count=12, seed=7)
    assert np.array_equal(left, right)
    selected_steps = steps[left]
    counts = [
        int(np.count_nonzero((selected_steps >= lower) & (selected_steps <= upper)))
        for lower, upper in STEP_BANDS
    ]
    assert counts == [3, 3, 3, 3]


def test_frozen_material_v2_contract_and_mismatch_fail_fast() -> None:
    manifest = DatasetManifest.load(ROOT / "data/simulator-r-v1/manifest.json")
    config = SimulatorConfig.load(ROOT / "configs/simulator-r.toml")
    _verify_frozen_contract(manifest, config)
    wrong = replace(config, material=replace(config.material, dynamic_friction=1.9))
    with pytest.raises(ValueError, match="physics metadata"):
        _verify_frozen_contract(manifest, wrong)


def test_checkpoint_hash_and_feature_contract_fail_fast() -> None:
    manifest = DatasetManifest.load(ROOT / "data/simulator-r-v1/manifest.json")
    base = ExperimentConfig.load(ROOT / "configs/srno-r-material-v2.toml")
    config = _checkpoint_config(base, arm="aperture", seed=0)
    checkpoint = {
        "stage": "rollout",
        "horizon": 32,
        "manifest_sha256": manifest.sha256(),
        "gripper_sha256": manifest.gripper_sha256,
        "config": config.to_dict(),
    }
    _verify_checkpoint_contract(
        checkpoint,
        config=config,
        manifest=manifest,
        stage="rollout",
        horizon=32,
    )
    wrong = dict(checkpoint)
    wrong["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest hash"):
        _verify_checkpoint_contract(
            wrong,
            config=config,
            manifest=manifest,
            stage="rollout",
            horizon=32,
        )
