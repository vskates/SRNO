from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from srno.data.dataset import (
    H5ObjectDataset,
    LocalTransitionBatch,
    TrajectoryBatch,
    make_dataloader,
)
from srno.data.index import ActiveIndex
from srno.data.schema import DatasetManifest, objectwise_split
from srno.data.tools import build_active_index, calibrate_gate, validate_dataset


def test_manifest_dataset_validation_and_lazy_loading(dataset_bundle: Path) -> None:
    manifest = DatasetManifest.load(dataset_bundle)
    report = validate_dataset(manifest, strict_resolution=False)
    assert report.objects == 3
    assert report.trajectories == 9
    assert "contact_count" in report.diagnostics
    dataset = H5ObjectDataset(manifest, split="train")
    try:
        assert not dataset._handles
        record = dataset[0]
        assert record.sdf.shape == (8, 8, 8)
        assert record.position.shape == (3, 33, 3)
        assert dataset._handles
    finally:
        dataset.close()


def test_multiworker_object_loader(dataset_bundle: Path) -> None:
    manifest = DatasetManifest.load(dataset_bundle)
    dataset = H5ObjectDataset(manifest, split="train")
    try:
        loader = make_dataloader(
            dataset,
            mode="rollout",
            objects_per_batch=1,
            samples_per_object=2,
            workers=2,
            seed=11,
            shuffle=False,
        )
        batch = next(iter(loader))
        assert batch.states.shape == (2, 33)
        assert batch.sdf.values.shape == (1, 8, 8, 8)
        assert batch.sdf.sample_to_object.tolist() == [0, 0]
    finally:
        dataset.close()


def test_active_index_local_collation_and_gate_calibration(
    dataset_bundle: Path, tmp_path: Path
) -> None:
    output = tmp_path / "active.npz"
    active = build_active_index(dataset_bundle, output)
    assert output.is_file()
    assert len(active.step_index) == 3 * 3 * 32
    loaded = ActiveIndex.load(output)
    manifest = DatasetManifest.load(dataset_bundle)
    dataset = H5ObjectDataset(
        manifest, split="train", active_index=loaded, active_only=True
    )
    try:
        loader = make_dataloader(
            dataset,
            mode="local",
            objects_per_batch=1,
            samples_per_object=4,
            workers=0,
            seed=3,
            shuffle=False,
        )
        batch = next(iter(loader))
        assert isinstance(batch, LocalTransitionBatch)
        assert batch.current.shape == (4,)
        assert batch.sdf.values.shape[0] == 1
        assert batch.sdf.sample_to_object.tolist() == [0, 0, 0, 0]
    finally:
        dataset.close()
    calibration = calibrate_gate(dataset_bundle, target_recall=0.995)
    assert calibration["contact_recall"] == 1.0
    assert calibration["recommended_delta_gate_m"] >= 0.02009


def test_zero_samples_means_complete_object_coverage(
    dataset_bundle: Path, tmp_path: Path
) -> None:
    active_path = tmp_path / "active-full.npz"
    active = build_active_index(dataset_bundle, active_path)
    manifest = DatasetManifest.load(dataset_bundle)
    local_dataset = H5ObjectDataset(
        manifest, split="train", active_index=active, active_only=True
    )
    rollout_dataset = H5ObjectDataset(manifest, split="train")
    try:
        local = next(
            iter(
                make_dataloader(
                    local_dataset,
                    mode="local",
                    objects_per_batch=1,
                    samples_per_object=0,
                    workers=0,
                    shuffle=False,
                )
            )
        )
        rollout = next(
            iter(
                make_dataloader(
                    rollout_dataset,
                    mode="rollout",
                    objects_per_batch=1,
                    samples_per_object=0,
                    workers=0,
                    shuffle=False,
                )
            )
        )
        assert isinstance(local, LocalTransitionBatch)
        assert isinstance(rollout, TrajectoryBatch)
        assert local.current.shape == (3 * 32,)
        assert len(set(zip(local.trajectory_index.tolist(), local.step_index.tolist()))) == 3 * 32
        assert rollout.states.shape == (3, 33)
        assert sorted(rollout.trajectory_index.tolist()) == [0, 1, 2]
    finally:
        local_dataset.close()
        rollout_dataset.close()


def test_chunked_loader_covers_every_sample_once(dataset_bundle: Path, tmp_path: Path) -> None:
    active = build_active_index(dataset_bundle, tmp_path / "active-chunked.npz")
    manifest = DatasetManifest.load(dataset_bundle)
    local_dataset = H5ObjectDataset(
        manifest, split="train", active_index=active, active_only=True
    )
    rollout_dataset = H5ObjectDataset(manifest, split="train")
    try:
        local_loader = make_dataloader(
            local_dataset,
            mode="local",
            objects_per_batch=1,
            samples_per_object=10,
            workers=0,
            seed=17,
            shuffle=True,
        )
        rollout_loader = make_dataloader(
            rollout_dataset,
            mode="rollout",
            objects_per_batch=1,
            samples_per_object=2,
            workers=0,
            seed=17,
            shuffle=True,
        )
        local_pairs = [
            (trajectory, step)
            for batch in local_loader
            for trajectory, step in zip(
                batch.trajectory_index.tolist(), batch.step_index.tolist()
            )
        ]
        rollout_indices = [
            trajectory
            for batch in rollout_loader
            for trajectory in batch.trajectory_index.tolist()
        ]
        assert len(local_loader) == 10
        assert len(local_pairs) == 3 * 32
        assert len(set(local_pairs)) == len(local_pairs)
        assert len(rollout_loader) == 2
        assert sorted(rollout_indices) == [0, 1, 2]
    finally:
        local_dataset.close()
        rollout_dataset.close()


def test_objectwise_split_is_disjoint_and_reproducible() -> None:
    object_ids = [f"object-{index}" for index in range(20)]
    first = objectwise_split(object_ids, seed=8)
    second = objectwise_split(object_ids, seed=8)
    assert first == second
    assert set(first["train"]).isdisjoint(first["val"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set().union(*map(set, first.values())) == set(object_ids)


def test_manifest_rejects_overlapping_split(dataset_bundle: Path) -> None:
    manifest = DatasetManifest.load(dataset_bundle)
    raw = manifest.to_dict()
    raw["splits"]["val"] = raw["splits"]["train"]
    path = dataset_bundle.parent / "invalid.json"
    import json

    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        DatasetManifest.load(path)
