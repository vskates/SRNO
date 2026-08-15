from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from srno.sim.pose_seeds import PoseSeeds, import_validation_pose_json
from srno.sim.runner import _candidate_pose_order


def test_validation_pose_import_round_trip_and_selection(tmp_path: Path) -> None:
    angle = np.pi / 2
    transform = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0, 0.1],
            [np.sin(angle), np.cos(angle), 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    source = tmp_path / "poses.json"
    source.write_text(
        json.dumps(
            {
                "grasps": {
                    "transforms": [transform.tolist(), np.eye(4).tolist()],
                    "object_in_gripper": [True, False],
                }
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "poses.npz"
    imported = import_validation_pose_json(source, destination)
    restored = PoseSeeds.load(destination)
    assert np.allclose(restored.position_m, imported.position_m)
    assert np.allclose(restored.quaternion_wxyz[0], [2**-0.5, 0.0, 0.0, 2**-0.5])
    selected = restored.select(3, successful_only=True, seed=2)
    assert selected.position_m.shape == (3, 3)
    assert np.all(selected.validation_success)
    assert np.array_equal(selected.source_index, np.zeros(3, dtype=np.int64))

    explicit = restored.take([0], require_successful=True)
    assert np.array_equal(explicit.position_m, restored.position_m[[0]])
    assert explicit.source_index.tolist() == [0]
    with pytest.raises(ValueError, match="not validation-successful"):
        restored.take([1], require_successful=True)
    with pytest.raises(IndexError, match="outside"):
        restored.take([2], require_successful=False)


def test_candidate_order_preserves_primary_sample_and_adds_unique_reserves() -> None:
    count = 12
    seeds = PoseSeeds(
        np.arange(count * 3, dtype=np.float32).reshape(count, 3),
        np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (count, 1)),
        np.array([True] * 10 + [False, False]),
    )
    primary = seeds.select(4, successful_only=True, seed=17)
    candidates = _candidate_pose_order(
        seeds, 4, successful_only=True, seed=17
    )

    assert candidates.source_index[:4].tolist() == primary.source_index.tolist()
    assert len(candidates.source_index) == 10
    assert len(np.unique(candidates.source_index)) == 10
    assert np.all(candidates.validation_success)
