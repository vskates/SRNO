#!/usr/bin/env python3
"""Compare settled GT gaps from stored SDF and exported exact cooked hulls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from analyze_sdf_collision_alignment import CookedUnion, _query_stored, _stats
from srno.data.schema import DatasetManifest
from srno.geometry.gripper import GripperAsset
from srno.geometry.se3 import quaternion_xyzw_to_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cooked", type=Path, required=True)
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = DatasetManifest.load(args.manifest)
    gripper = GripperAsset.load(manifest.gripper_path)
    locations = manifest.object_locations()
    object_ids = list(locations if args.objects is None else args.objects)
    rows = []
    for index, object_id in enumerate(object_ids, start=1):
        path, group_name = locations[object_id]
        with h5py.File(path, "r", swmr=True) as handle:
            group = handle[group_name]
            values = np.asarray(group["sdf"], dtype=np.float32)
            origin = np.asarray(group.attrs["grid_origin"], dtype=np.float32)
            voxel = np.asarray(group.attrs["voxel_size"], dtype=np.float32)
            position = torch.from_numpy(np.asarray(group["position"], dtype=np.float32))
            quaternion = torch.from_numpy(
                np.asarray(group["quaternion_xyzw"], dtype=np.float32)
            )
            aperture = torch.from_numpy(
                np.asarray(group["actual_aperture"], dtype=np.float32)
            )
        rotation = quaternion_xyzw_to_matrix(quaternion[:, 1:])
        points_gripper = gripper.points(aperture[:, 1:])
        relative = points_gripper - position[:, 1:, None, :]
        points_object = torch.einsum(
            "tsij,tsmj->tsmi", rotation.transpose(-1, -2), relative
        ).numpy()
        flat = points_object.reshape(-1, 3)
        stored = _query_stored(
            values, origin, voxel, flat, manifest.sdf_scale_m
        ).reshape(points_object.shape[:-1])
        exact = CookedUnion(args.cooked / f"{object_id}.npz").signed_distance(
            flat
        ).reshape(points_object.shape[:-1])
        stored_min = stored.min(axis=-1)
        exact_min = exact.min(axis=-1)
        rows.append(
            {
                "object_id": object_id,
                "stored_min_gap_m": _stats(stored_min.reshape(-1)),
                "exact_min_gap_m": _stats(exact_min.reshape(-1)),
                "stored_infeasible_fraction": float(np.mean(stored_min < 0)),
                "exact_infeasible_fraction": float(np.mean(exact_min < 0)),
                "exact_minus_stored_m": _stats(
                    (exact_min - stored_min).reshape(-1)
                ),
            }
        )
        print(f"[{index:02d}/{len(object_ids):02d}] {object_id}", flush=True)
    result = {"objects": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
