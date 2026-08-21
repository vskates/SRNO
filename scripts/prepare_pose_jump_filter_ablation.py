#!/usr/bin/env python3
"""Build an active index excluding large ground-truth one-step pose jumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from srno.data.index import ActiveIndex, file_sha256
from srno.data.schema import DatasetManifest


SPLITS = ("train", "val", "test")


def _pose_motion(group: h5py.Group, length_scale_m: float) -> np.ndarray:
    position = np.asarray(group["position"], dtype=np.float64)
    quaternion = np.asarray(group["quaternion_xyzw"], dtype=np.float64)
    translation = np.linalg.norm(np.diff(position, axis=1), axis=-1)
    # For unit quaternions, 2 acos(|q1 dot q2|) is the SO(3) geodesic angle.
    dot = np.sum(quaternion[:, 1:] * quaternion[:, :-1], axis=-1)
    rotation = 2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    return np.sqrt((translation / length_scale_m) ** 2 + rotation**2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--active-index", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()
    if not np.isfinite(args.threshold) or args.threshold <= 0.0:
        raise ValueError("threshold must be positive and finite")

    manifest = DatasetManifest.load(args.manifest)
    source = ActiveIndex.load(args.active_index)
    if source.manifest_sha256 != manifest.sha256():
        raise ValueError("active index does not match manifest")

    locations = manifest.object_locations()
    kept_trajectory: list[np.ndarray] = []
    kept_step: list[np.ndarray] = []
    offsets = [0]
    object_stats: dict[str, dict[str, float | int]] = {}
    for object_id in source.object_ids:
        pairs = source.pairs_for(object_id)
        shard, group_name = locations[object_id]
        with h5py.File(shard, "r") as handle:
            motion = _pose_motion(handle[group_name], manifest.length_scale_m)
        selected_motion = motion[pairs[:, 0], pairs[:, 1]]
        keep = selected_motion <= args.threshold
        if not np.any(keep):
            raise ValueError(f"pose-jump filter removed every transition for {object_id}")
        retained = pairs[keep]
        kept_trajectory.append(retained[:, 0].astype(np.int32))
        kept_step.append(retained[:, 1].astype(np.uint8))
        offsets.append(offsets[-1] + len(retained))
        total_energy = float(np.sum(selected_motion**2))
        removed_energy = float(np.sum(selected_motion[~keep] ** 2))
        object_stats[object_id] = {
            "before": int(len(pairs)),
            "after": int(len(retained)),
            "removed": int((~keep).sum()),
            "removed_fraction": float((~keep).mean()),
            "removed_pose_motion_energy_fraction": (
                removed_energy / total_energy if total_energy > 0.0 else 0.0
            ),
            "max_retained_pose_motion": float(selected_motion[keep].max()),
            "min_removed_pose_motion": (
                float(selected_motion[~keep].min()) if np.any(~keep) else float("nan")
            ),
        }

    filtered = ActiveIndex(
        manifest_sha256=source.manifest_sha256,
        shards_sha256=source.shards_sha256,
        gripper_sha256=source.gripper_sha256,
        delta_gate_m=source.delta_gate_m,
        object_ids=source.object_ids,
        offsets=np.asarray(offsets, dtype=np.int64),
        trajectory_index=np.concatenate(kept_trajectory),
        step_index=np.concatenate(kept_step),
    )
    filtered.save(args.output_index)

    split_stats: dict[str, dict[str, float | int]] = {}
    for split in SPLITS:
        before = sum(object_stats[name]["before"] for name in manifest.splits[split])
        after = sum(object_stats[name]["after"] for name in manifest.splits[split])
        # Recompute the energy-weighted split statistic from the same arrays to
        # avoid averaging per-object fractions with unequal energy.
        total_energy = 0.0
        rejected_energy = 0.0
        for object_id in manifest.splits[split]:
            pairs = source.pairs_for(object_id)
            shard, group_name = locations[object_id]
            with h5py.File(shard, "r") as handle:
                motion = _pose_motion(handle[group_name], manifest.length_scale_m)
            selected = motion[pairs[:, 0], pairs[:, 1]]
            rejected = selected > args.threshold
            total_energy += float(np.sum(selected**2))
            rejected_energy += float(np.sum(selected[rejected] ** 2))
        split_stats[split] = {
            "before": int(before),
            "after": int(after),
            "removed": int(before - after),
            "removed_fraction": float((before - after) / before),
            "removed_pose_motion_energy_fraction": (
                rejected_energy / total_energy if total_energy > 0.0 else 0.0
            ),
        }

    payload = {
        "format_version": 1,
        "definition": (
            "remove iff sqrt((||p[k+1]-p[k]||/length_scale_m)^2 + "
            "geodesic(R[k],R[k+1])^2) > threshold"
        ),
        "threshold": args.threshold,
        "length_scale_m": manifest.length_scale_m,
        "source_active_index": str(args.active_index.resolve()),
        "source_active_index_sha256": file_sha256(args.active_index),
        "filtered_active_index": str(args.output_index.resolve()),
        "filtered_active_index_sha256": file_sha256(args.output_index),
        "splits": split_stats,
        "objects": object_stats,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(split_stats, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
