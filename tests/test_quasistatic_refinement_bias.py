from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.quasistatic_refinement_bias import (
    BIAS_STEP_BANDS,
    EXPECTED_SOURCE_POSES,
    OBJECTS,
    _bias_feature_vector,
    _bootstrap_refinement_ratio,
    _classify_refinement,
    _command_schedule,
    _common_command_indices,
    _hierarchical_bias_bootstrap,
    _persistent_bias,
    _refinement_metrics,
    _rotation_distance_xyzw,
    _select_refinement_poses,
    _settling_is_materially_sensitive,
    _state_components,
)
from srno.data.schema import DatasetManifest
from srno.geometry.gripper import GripperAsset


def test_command_paths_match_exactly_at_common_points() -> None:
    knots = np.linspace(0.1115, 0.0123, 33) ** 1.01
    n32 = _command_schedule(knots, 32)
    n64 = _command_schedule(knots, 64)
    n128 = _command_schedule(knots, 128)
    assert np.array_equal(n32, knots)
    assert np.array_equal(n32, n64[_common_command_indices(32, 64)])
    assert np.array_equal(n64, n128[_common_command_indices(64, 128)])
    assert n128[0] == knots[0]
    assert n128[-1] == knots[-1]


def test_state_metric_is_quaternion_sign_invariant() -> None:
    position = np.asarray([[0.01, -0.02, 0.03]])
    quaternion = np.asarray([[0.1, -0.2, 0.3, 0.92736185]])
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    joint = np.asarray([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6]])
    components = _state_components(
        position,
        quaternion,
        joint,
        position,
        -quaternion,
        joint,
        length_scale=0.1,
        joint_scale=np.ones(6),
    )
    assert np.array_equal(components["dx"], np.zeros(1))
    assert np.array_equal(
        _rotation_distance_xyzw(quaternion, -quaternion), np.zeros(1)
    )


def test_deterministic_refinement_pose_selection_matches_frozen_dataset() -> None:
    manifest = DatasetManifest.load(ROOT / "data/simulator-r-v1/manifest.json")
    gripper = GripperAsset.load(manifest.gripper_path)
    for object_id in OBJECTS:
        left = _select_refinement_poses(
            manifest,
            object_id,
            expected_source_poses=EXPECTED_SOURCE_POSES[object_id],
            expected_joint_names=gripper.joint_names,
        )
        right = _select_refinement_poses(
            manifest,
            object_id,
            expected_source_poses=EXPECTED_SOURCE_POSES[object_id],
            expected_joint_names=gripper.joint_names,
        )
        assert tuple(left.source_pose_index.tolist()) == EXPECTED_SOURCE_POSES[object_id]
        assert np.array_equal(left.trajectory, right.trajectory)
        assert len(np.unique(left.trajectory)) == 4


def _synthetic_schedule(level: int, offset: float) -> dict[str, np.ndarray]:
    objects, repeats, poses = 1, 2, 2
    alpha = np.linspace(0.0, 1.0, level + 1)
    position = np.zeros((objects, repeats, poses, level + 1, 3), dtype=np.float64)
    position[..., 0] = alpha + offset
    quaternion = np.zeros((objects, repeats, poses, level + 1, 4), dtype=np.float64)
    quaternion[..., 3] = 1.0
    joint = np.zeros((objects, repeats, poses, level + 1, 6), dtype=np.float64)
    settled = np.ones((objects, repeats, poses, level + 1), dtype=bool)
    return {
        f"n{level}_position": position,
        f"n{level}_quaternion_xyzw": quaternion,
        f"n{level}_joint": joint,
        f"n{level}_settled": settled,
    }


def test_synthetic_first_order_refinement_has_half_ratio() -> None:
    arrays: dict[str, np.ndarray] = {}
    arrays.update(_synthetic_schedule(32, 1.0 / 32.0))
    arrays.update(_synthetic_schedule(64, 1.0 / 64.0))
    arrays.update(_synthetic_schedule(128, 1.0 / 128.0))
    arrays["n32_strict_position"] = arrays["n32_position"][:, :1].copy()
    arrays["n32_strict_quaternion_xyzw"] = arrays[
        "n32_quaternion_xyzw"
    ][:, :1].copy()
    arrays["n32_strict_joint"] = arrays["n32_joint"][:, :1].copy()
    arrays["n32_strict_settled"] = arrays["n32_settled"][:, :1].copy()
    metrics = _refinement_metrics(
        arrays, length_scale=1.0, joint_scale=np.ones(6)
    )
    e32 = metrics["e32_64_dx"][:, 0].reshape(-1)
    e64 = metrics["e64_128_dx"][:, 0].reshape(-1)
    assert np.isclose(e64.mean() / e32.mean(), 0.5, atol=1e-12)
    # Repeatability and strict-settling summaries retain every pose and drop
    # only the initial state along the state axis.
    assert metrics["repeat32_dx"].shape == (1, 2)
    assert metrics["repeat32_dx_curve"].shape == (1, 2, 32)
    assert metrics["settle_dx"].shape == (1, 2)
    assert metrics["settle_dx_curve"].shape == (1, 2, 32)


def test_refinement_bootstrap_and_predeclared_classification() -> None:
    denominator = np.asarray([0.1, 0.2, 0.3, 0.4])
    object_index = np.asarray([0, 0, 1, 1])
    convergent = _bootstrap_refinement_ratio(
        0.5 * denominator,
        denominator,
        object_index,
        replicates=200,
        seed=0,
    )
    assert np.isclose(convergent["ratio"], 0.5)
    assert _classify_refinement(
        convergent, e64_128_mean=0.1, repeatability_floor=0.01
    ) == "converged"
    flat = _bootstrap_refinement_ratio(
        denominator,
        denominator,
        object_index,
        replicates=200,
        seed=0,
    )
    assert _classify_refinement(
        flat, e64_128_mean=0.2, repeatability_floor=0.01
    ) == "non-converged"
    assert _classify_refinement(
        flat, e64_128_mean=0.01, repeatability_floor=0.01
    ) == "noise-limited"
    assert _settling_is_materially_sensitive(
        {"ci95_lower": 0.3}, settling_mean=0.1, repeatability_floor=0.01
    )


def test_bias_features_include_all_32_steps_and_zero_case() -> None:
    tf = np.zeros((5, 32, 6), dtype=np.float64)
    tf[:, 31, 2] = 1.0
    ar = np.zeros_like(tf)
    spatial = np.zeros((5, 32, 3), dtype=np.float64)
    features = _bias_feature_vector(tf, ar, spatial)
    assert BIAS_STEP_BANDS[-1] == (25, 32)
    assert np.allclose(features[:, 20], 1.0 / 8.0)
    assert np.allclose(features[:, 26], 1.0)
    assert np.count_nonzero(features) == 10


def test_hierarchical_bias_bootstrap_zero_and_persistent_sign() -> None:
    features = np.zeros((3, 8, 39), dtype=np.float64)
    object_index = np.repeat(np.arange(2), 4)
    summary = _hierarchical_bias_bootstrap(
        features, object_index, replicates=100, seed=0, chunk_size=25
    )
    assert np.array_equal(summary["mean"], np.zeros(39))
    assert not any(
        _persistent_bias(
            summary["mean"], summary["ci95_lower"], summary["ci95_upper"]
        )
    )

    features[:, :, [0, 6, 12]] = 0.001
    summary = _hierarchical_bias_bootstrap(
        features, object_index, replicates=100, seed=0, chunk_size=25
    )
    persistent = _persistent_bias(
        summary["mean"], summary["ci95_lower"], summary["ci95_upper"]
    )
    assert persistent == [True, False, False, False, False, False]
