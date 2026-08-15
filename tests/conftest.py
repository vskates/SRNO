from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from srno.data.schema import DatasetManifest, ShardSpec
from srno.data.writer import H5DatasetWriter
from srno.geometry.gripper import GripperAsset


def synthetic_gripper(*, points: int = 256) -> GripperAsset:
    side = points // 2
    yz = torch.linspace(-0.01, 0.01, side)
    intercept = torch.zeros(points, 3)
    intercept[:side, 1] = yz
    intercept[side:, 1] = yz[: points - side]
    slope = torch.zeros_like(intercept)
    slope[:side, 0] = 0.5
    slope[side:, 0] = -0.5
    return GripperAsset(
        intercept,
        slope,
        torch.cat((torch.zeros(side), torch.ones(points - side))).long(),
        0.0,
        0.08,
        0.08,
        "synthetic",
    )


@pytest.fixture
def gripper() -> GripperAsset:
    return synthetic_gripper()


@pytest.fixture
def dataset_bundle(tmp_path: Path, gripper: GripperAsset) -> Path:
    gripper_path = tmp_path / "gripper.npz"
    gripper.save(gripper_path)
    shard_path = tmp_path / "shard-000.h5"
    schedule = np.linspace(0.08, 0.0, 33, dtype=np.float32)
    with H5DatasetWriter(shard_path) as writer:
        for index in range(3):
            trajectories = 3
            position = np.zeros((trajectories, 33, 3), dtype=np.float32)
            position[:, :, 1] = index * 1e-4
            quaternion = np.zeros((trajectories, 33, 4), dtype=np.float32)
            quaternion[..., 3] = 1
            aperture = np.broadcast_to(schedule, (trajectories, 33)).copy()
            writer.add_object(
                f"object-{index}",
                sdf=np.full((8, 8, 8), -1e-3, dtype=np.float32),
                grid_origin=np.full(3, -0.04, dtype=np.float32),
                voxel_size=np.full(3, 0.01, dtype=np.float32),
                position=position,
                quaternion_xyzw=quaternion,
                actual_aperture=aperture,
                diagnostics={
                    "contact_count": np.ones((trajectories, 32), dtype=np.int16),
                    "actuator_effort": np.ones((trajectories, 32), dtype=np.float32),
                    "max_penetration": np.full((trajectories, 32), 1e-3, dtype=np.float32),
                    "linear_velocity": np.zeros((trajectories, 32, 3), dtype=np.float32),
                    "angular_velocity": np.zeros((trajectories, 32, 3), dtype=np.float32),
                    "settling_substeps": np.ones((trajectories, 32), dtype=np.int32),
                },
            )
    object_ids = ("object-0", "object-1", "object-2")
    manifest = DatasetManifest.create(
        root=tmp_path,
        length_scale_m=0.08,
        sdf_scale_m=0.02,
        delta_gate_m=0.021,
        commanded_aperture_m=schedule.tolist(),
        gripper_asset="gripper.npz",
        gripper_sha256=gripper.sha256(),
        shards=[ShardSpec("shard-000.h5", object_ids)],
        splits={"train": (object_ids[0],), "val": (object_ids[1],), "test": (object_ids[2],)},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    return manifest_path

