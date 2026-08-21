#!/usr/bin/env python3
"""Compare stored SRNO SDFs with exact PhysX-cooked convex colliders."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
import warp as wp
from scipy.spatial import ConvexHull

from srno.data.schema import DatasetManifest
from srno.geometry.gripper import GripperAsset
from srno.geometry.sdf import sample_sdf
from srno.geometry.se3 import quaternion_xyzw_to_matrix
from srno.sim import SimulatorAssetCatalog
from srno.sim.pxr_bootstrap import ensure_pxr_available


@wp.kernel(enable_backward=False)
def _mesh_unsigned_distance(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=wp.float32),
):
    index = wp.tid()
    point = points[index]
    query = wp.mesh_query_point_no_sign(mesh, point, 1.0e6)
    if query.result:
        closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        distances[index] = wp.length(point - closest)
    else:
        distances[index] = 1.0e6


def _stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


class CookedUnion:
    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as archive:
            count = int(archive["hull_count"])
            self.hulls = [
                trimesh.Trimesh(
                    vertices=np.asarray(archive[f"hull_{index:04d}_vertices"], dtype=np.float64),
                    faces=np.asarray(archive[f"hull_{index:04d}_faces"], dtype=np.int64),
                    process=False,
                )
                for index in range(count)
            ]
        self.mesh = trimesh.util.concatenate(self.hulls)
        self.warp_mesh = wp.Mesh(
            points=wp.array(
                np.asarray(self.mesh.vertices, dtype=np.float32),
                dtype=wp.vec3,
                device="cuda:0",
            ),
            indices=wp.array(
                np.asarray(self.mesh.faces, dtype=np.int32).reshape(-1),
                dtype=wp.int32,
                device="cuda:0",
            ),
            support_winding_number=False,
        )
        self.halfspaces = []
        for hull in self.hulls:
            convex = ConvexHull(hull.vertices)
            self.halfspaces.append(
                (np.asarray(hull.bounds), np.asarray(convex.equations, dtype=np.float64))
            )

    def inside(self, points: np.ndarray, *, strict_tolerance: float = 1e-8) -> np.ndarray:
        inside = np.zeros(len(points), dtype=bool)
        for bounds, equations in self.halfspaces:
            candidate = ~inside & np.all(
                (points >= bounds[0] - strict_tolerance)
                & (points <= bounds[1] + strict_tolerance),
                axis=1,
            )
            indices = np.flatnonzero(candidate)
            if not len(indices):
                continue
            values = points[indices] @ equations[:, :3].T + equations[:, 3]
            inside[indices] = np.all(values <= strict_tolerance, axis=1)
        return inside

    def strict_inside(self, points: np.ndarray, tolerance: float = 1e-5) -> np.ndarray:
        inside = np.zeros(len(points), dtype=bool)
        for bounds, equations in self.halfspaces:
            candidate = np.all(
                (points >= bounds[0] + tolerance)
                & (points <= bounds[1] - tolerance),
                axis=1,
            )
            indices = np.flatnonzero(candidate)
            if not len(indices):
                continue
            values = points[indices] @ equations[:, :3].T + equations[:, 3]
            inside[indices] |= np.all(values < -tolerance, axis=1)
        return inside

    def signed_distance(self, points: np.ndarray, chunk_size: int = 100_000) -> np.ndarray:
        result = np.empty(len(points), dtype=np.float32)
        for start in range(0, len(points), chunk_size):
            stop = min(start + chunk_size, len(points))
            chunk = np.asarray(points[start:stop], dtype=np.float64)
            warp_points = wp.array(
                np.asarray(chunk, dtype=np.float32), dtype=wp.vec3, device="cuda:0"
            )
            warp_distance = wp.empty(len(chunk), dtype=wp.float32, device="cuda:0")
            wp.launch(
                _mesh_unsigned_distance,
                dim=len(chunk),
                inputs=[self.warp_mesh.id, warp_points, warp_distance],
                device="cuda:0",
            )
            distance = warp_distance.numpy().astype(np.float64)
            distance[self.inside(chunk)] *= -1.0
            result[start:stop] = distance.astype(np.float32)
        return result


def _query_stored(
    values: np.ndarray,
    origin: np.ndarray,
    voxel_size: np.ndarray,
    points: np.ndarray,
    outside_value: float,
    chunk_size: int = 250_000,
) -> np.ndarray:
    grid = torch.from_numpy(np.asarray(values)).unsqueeze(0)
    grid_origin = torch.from_numpy(np.asarray(origin, dtype=np.float32)).unsqueeze(0)
    voxel = torch.from_numpy(np.asarray(voxel_size, dtype=np.float32)).unsqueeze(0)
    output = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        coordinates = torch.from_numpy(np.asarray(points[start:stop], dtype=np.float32)).unsqueeze(0)
        output[start:stop] = (
            sample_sdf(
                grid,
                grid_origin,
                voxel,
                coordinates,
                outside_value=outside_value,
            )
            .squeeze(0)
            .numpy()
        )
    return output


def _grid_points(
    shape: tuple[int, int, int], origin: np.ndarray, voxel: np.ndarray, indices_zyx: np.ndarray
) -> np.ndarray:
    return origin + indices_zyx[:, ::-1] * voxel


def _trial_points(
    position: np.ndarray,
    quaternion_xyzw: np.ndarray,
    gripper: GripperAsset,
    schedule: np.ndarray,
) -> np.ndarray:
    rotation = quaternion_xyzw_to_matrix(
        torch.from_numpy(np.asarray(quaternion_xyzw[:, :-1], dtype=np.float32))
    )
    commands = torch.from_numpy(np.asarray(schedule[1:], dtype=np.float32))
    points_gripper = gripper.points(commands)
    relative = points_gripper.unsqueeze(0) - torch.from_numpy(
        np.asarray(position[:, :-1], dtype=np.float32)
    ).unsqueeze(-2)
    points_object = torch.einsum(
        "tsij,tsmj->tsmi", rotation.transpose(-1, -2), relative
    )
    return points_object.numpy()


def _trial_points_from_schedule(
    position: np.ndarray,
    quaternion_xyzw: np.ndarray,
    points_open_to_closed: np.ndarray,
) -> np.ndarray:
    rotation = quaternion_xyzw_to_matrix(
        torch.from_numpy(np.asarray(quaternion_xyzw[:, :-1], dtype=np.float32))
    )
    points_gripper = torch.from_numpy(
        np.asarray(points_open_to_closed[1:], dtype=np.float32)
    )
    relative = points_gripper.unsqueeze(0) - torch.from_numpy(
        np.asarray(position[:, :-1], dtype=np.float32)
    ).unsqueeze(-2)
    return torch.einsum(
        "tsij,tsmj->tsmi", rotation.transpose(-1, -2), relative
    ).numpy()


def _threshold_at_recall(gaps: np.ndarray, contact: np.ndarray, recall: float) -> float:
    contact_gaps = np.asarray(gaps)[np.asarray(contact, dtype=bool)]
    rank = max(0, math.ceil(recall * len(contact_gaps)) - 1)
    return float(np.sort(contact_gaps)[rank])


def _classification(gaps: np.ndarray, contact: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = gaps <= threshold
    contact = np.asarray(contact, dtype=bool)
    return {
        "threshold_m": float(threshold),
        "recall": float(predicted[contact].mean()),
        "false_positive_rate": float(predicted[~contact].mean()),
        "precision": float(contact[predicted].mean()) if np.any(predicted) else 0.0,
        "active_fraction": float(predicted.mean()),
    }


def _gap_comparison(stored: np.ndarray, physx: np.ndarray, gate: float) -> dict[str, Any]:
    delta = np.asarray(physx) - np.asarray(stored)
    relevant = (np.asarray(stored) <= gate) | (np.asarray(physx) <= gate)
    return {
        "count": int(relevant.sum()),
        "fraction": float(relevant.mean()),
        "signed_difference_m": _stats(delta[relevant]),
        "absolute_difference_m": _stats(np.abs(delta[relevant])),
        "pearson": float(np.corrcoef(stored[relevant], physx[relevant])[0, 1]),
    }


def _plot(
    output: Path,
    rows: list[dict[str, Any]],
    stored_trial: np.ndarray,
    physx_trial: np.ndarray,
    contact: np.ndarray,
    gate: float,
    runtime_physx_trial: np.ndarray | None = None,
) -> None:
    ordered = sorted(rows, key=lambda row: row["raw_surface_to_physx_m"]["p95"], reverse=True)
    labels = [row["object_id"][:22] for row in ordered]
    p95 = [row["raw_surface_to_physx_m"]["p95"] * 1000 for row in ordered]
    trial = [row["trial_min_difference_m"]["median"] * 1000 for row in ordered]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    y = np.arange(len(ordered))
    axes[0, 0].barh(y, p95, color="#C44E52")
    axes[0, 0].set_yticks(y, labels, fontsize=7)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("p95 distance, mm")
    axes[0, 0].set_title("Raw SDF surface → PhysX cooked collider")
    axes[0, 0].grid(axis="x", alpha=0.25)

    axes[0, 1].barh(y, trial, color="#4C72B0")
    axes[0, 1].set_yticks(y, labels, fontsize=7)
    axes[0, 1].invert_yaxis()
    axes[0, 1].axvline(0, color="black", linewidth=0.8)
    axes[0, 1].set_xlabel("median (PhysX gap - stored SDF gap), mm")
    axes[0, 1].set_title("Exact model trial points")
    axes[0, 1].grid(axis="x", alpha=0.25)

    rng = np.random.default_rng(0)
    selection = rng.choice(len(stored_trial), min(120_000, len(stored_trial)), replace=False)
    axes[1, 0].hexbin(
        stored_trial[selection] * 1000,
        physx_trial[selection] * 1000,
        gridsize=100,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    low = float(min(stored_trial[selection].min(), physx_trial[selection].min()) * 1000)
    high = float(max(stored_trial[selection].max(), physx_trial[selection].max()) * 1000)
    axes[1, 0].plot([low, high], [low, high], "k--", linewidth=1)
    axes[1, 0].axvline(gate * 1000, color="#C44E52", linestyle=":")
    axes[1, 0].axhline(gate * 1000, color="#C44E52", linestyle=":")
    axes[1, 0].set_xlim(max(low, -20), min(high, 40))
    axes[1, 0].set_ylim(max(low, -20), min(high, 40))
    axes[1, 0].set_xlabel("stored SDF minimum, mm")
    axes[1, 0].set_ylabel("PhysX collider minimum, mm")
    axes[1, 0].set_title("Trial-state gap comparison")
    axes[1, 0].grid(alpha=0.2)

    thresholds = np.linspace(-0.005, 0.025, 241)
    for values, name, color in (
        (stored_trial, "stored raw-mesh SDF", "#4C72B0"),
        (physx_trial, "PhysX cooked collider", "#DD8452"),
        *(
            ((runtime_physx_trial, "runtime gripper + PhysX object", "#55A868"),)
            if runtime_physx_trial is not None
            else ()
        ),
    ):
        recalls = [np.mean(values[contact] <= threshold) for threshold in thresholds]
        false_positives = [np.mean(values[~contact] <= threshold) for threshold in thresholds]
        axes[1, 1].plot(np.asarray(false_positives) * 100, np.asarray(recalls) * 100, label=name, color=color)
    axes[1, 1].axhline(99.5, color="black", linestyle="--", linewidth=0.8)
    axes[1, 1].set_xlabel("false-positive rate, %")
    axes[1, 1].set_ylabel("contact recall, %")
    axes[1, 1].set_title("Gate ROC from simulator contact diagnostics")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()
    fig.suptitle("SRNO stored SDF vs actual PhysX collision geometry", fontsize=15)
    fig.savefig(output / "collision_alignment_dashboard.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/simulator-v1/manifest.json"))
    parser.add_argument("--cooked", type=Path, default=Path("runs/sdf-collision-diagnostic/cooked"))
    parser.add_argument("--output", type=Path, default=Path("runs/sdf-collision-diagnostic"))
    parser.add_argument("--surface-samples", type=int, default=20_000)
    parser.add_argument("--band-samples", type=int, default=40_000)
    parser.add_argument("--max-objects", type=int)
    parser.add_argument(
        "--runtime-gripper-points",
        type=Path,
        default=Path("runs/sdf-collision-diagnostic/runtime_gripper_points.npz"),
    )
    args = parser.parse_args()
    ensure_pxr_available(reexec=True)

    from srno.sim.usd_geometry import load_object_mesh_from_usd

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = DatasetManifest.load(args.manifest)
    catalog = SimulatorAssetCatalog.load()
    gripper = GripperAsset.load(manifest.gripper_path)
    with np.load(args.runtime_gripper_points, allow_pickle=False) as archive:
        runtime_gripper_points = np.asarray(
            archive["points_open_to_closed"], dtype=np.float32
        )
    if runtime_gripper_points.shape != (33, 256, 3):
        raise ValueError(
            "runtime gripper points must have shape [33, 256, 3], got "
            f"{runtime_gripper_points.shape}"
        )
    locations = manifest.object_locations()
    object_ids = [object_id for object_id in catalog.object_ids if object_id in locations]
    if args.max_objects is not None:
        object_ids = object_ids[: args.max_objects]
    split_by_object = {
        object_id: split for split, ids in manifest.splits.items() for object_id in ids
    }
    schedule = np.asarray(manifest.commanded_aperture_m, dtype=np.float32)
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    all_stored_trial = []
    all_physx_trial = []
    all_contact = []
    all_runtime_stored_trial = []
    all_runtime_physx_trial = []
    all_object = []
    all_split = []

    for object_index, object_id in enumerate(object_ids, start=1):
        started = perf_counter()
        record = catalog.object(object_id)
        path, group_name = locations[object_id]
        with h5py.File(path, "r", swmr=True) as handle:
            group = handle[group_name]
            values = np.asarray(group["sdf"], dtype=np.float32)
            origin = np.asarray(group.attrs["grid_origin"], dtype=np.float32)
            voxel = np.asarray(group.attrs["voxel_size"], dtype=np.float32)
            position = np.asarray(group["position"], dtype=np.float32)
            quaternion = np.asarray(group["quaternion_xyzw"], dtype=np.float32)
            contact = np.asarray(group["diagnostics/contact_count"]) > 0

        raw_mesh = load_object_mesh_from_usd(record.usd_path)
        cooked = CookedUnion(args.cooked / f"{object_id}.npz")

        raw_surface, _ = trimesh.sample.sample_surface(
            raw_mesh, args.surface_samples, seed=rng
        )
        raw_to_physx = cooked.signed_distance(raw_surface)
        raw_surface_stored = _query_stored(
            values, origin, voxel, raw_surface, manifest.sdf_scale_m
        )

        cooked_candidates, _ = trimesh.sample.sample_surface(
            cooked.mesh, args.surface_samples * 3, seed=rng
        )
        cooked_boundary = cooked_candidates[~cooked.strict_inside(cooked_candidates)]
        if len(cooked_boundary) > args.surface_samples:
            cooked_boundary = cooked_boundary[
                rng.choice(len(cooked_boundary), args.surface_samples, replace=False)
            ]
        cooked_surface_stored = _query_stored(
            values, origin, voxel, cooked_boundary, manifest.sdf_scale_m
        )

        runtime_path = output / "runtime-rays" / f"{object_id}.npz"
        runtime_metrics: dict[str, Any] | None = None
        if runtime_path.is_file():
            with np.load(runtime_path, allow_pickle=False) as archive:
                runtime_points = np.asarray(archive["points_object"], dtype=np.float64)
            runtime_physx_distance = cooked.signed_distance(runtime_points)
            runtime_stored = _query_stored(
                values, origin, voxel, runtime_points, manifest.sdf_scale_m
            )
            runtime_metrics = {
                "ray_hit_count": len(runtime_points),
                "live_surface_to_exported_cooked_abs_m": _stats(
                    np.abs(runtime_physx_distance)
                ),
                "live_surface_to_stored_sdf_abs_m": _stats(np.abs(runtime_stored)),
                "live_surface_stored_sdf_signed_m": _stats(runtime_stored),
            }

        band_indices = np.argwhere(np.abs(values) <= manifest.delta_gate_m)
        if len(band_indices) > args.band_samples:
            band_indices = band_indices[
                rng.choice(len(band_indices), args.band_samples, replace=False)
            ]
        band_points = _grid_points(values.shape, origin, voxel, band_indices)
        band_stored = values[tuple(band_indices.T)]
        band_physx = cooked.signed_distance(band_points)

        points = _trial_points(position, quaternion, gripper, schedule)
        flat_points = points.reshape(-1, 3)
        stored_gap = _query_stored(
            values, origin, voxel, flat_points, manifest.sdf_scale_m
        ).reshape(points.shape[:-1])
        physx_gap = cooked.signed_distance(flat_points).reshape(points.shape[:-1])
        stored_min = stored_gap.min(axis=-1)
        physx_min = physx_gap.min(axis=-1)

        runtime_points = _trial_points_from_schedule(
            position, quaternion, runtime_gripper_points
        )
        runtime_flat_points = runtime_points.reshape(-1, 3)
        runtime_stored_min = _query_stored(
            values,
            origin,
            voxel,
            runtime_flat_points,
            manifest.sdf_scale_m,
        ).reshape(runtime_points.shape[:-1]).min(axis=-1)
        runtime_physx_min = cooked.signed_distance(runtime_flat_points).reshape(
            runtime_points.shape[:-1]
        ).min(axis=-1)

        all_stored_trial.append(stored_min.reshape(-1))
        all_physx_trial.append(physx_min.reshape(-1))
        all_contact.append(contact.reshape(-1))
        all_runtime_stored_trial.append(runtime_stored_min.reshape(-1))
        all_runtime_physx_trial.append(runtime_physx_min.reshape(-1))
        all_object.extend([object_id] * contact.size)
        all_split.extend([split_by_object[object_id]] * contact.size)
        row = {
            "object_id": object_id,
            "split": split_by_object[object_id],
            "raw_faces": int(len(raw_mesh.faces)),
            "raw_vertices": int(len(raw_mesh.vertices)),
            "physx_hulls": int(len(cooked.hulls)),
            "physx_faces": int(len(cooked.mesh.faces)),
            "raw_watertight": bool(raw_mesh.is_watertight),
            "raw_is_volume": bool(raw_mesh.is_volume),
            "raw_winding_consistent": bool(raw_mesh.is_winding_consistent),
            "raw_body_count": int(raw_mesh.body_count),
            "raw_euler_number": int(raw_mesh.euler_number),
            "raw_extent_m": raw_mesh.extents.tolist(),
            "physx_extent_m": cooked.mesh.extents.tolist(),
            "extent_absolute_difference_m": np.abs(raw_mesh.extents - cooked.mesh.extents).tolist(),
            "voxel_size_m": voxel.tolist(),
            "hdf_vs_raw_surface_abs_m": _stats(np.abs(raw_surface_stored)),
            "raw_surface_to_physx_m": _stats(np.abs(raw_to_physx)),
            "raw_surface_physx_signed_m": _stats(raw_to_physx),
            "physx_boundary_to_stored_sdf_abs_m": _stats(np.abs(cooked_surface_stored)),
            "physx_boundary_stored_sdf_signed_m": _stats(cooked_surface_stored),
            "runtime_physx_raycast": runtime_metrics,
            "contact_band_abs_difference_m": _stats(np.abs(band_physx - band_stored)),
            "contact_band_signed_difference_m": _stats(band_physx - band_stored),
            "contact_band_sign_disagreement_fraction": float(
                np.mean((band_physx <= 0) != (band_stored <= 0))
            ),
            "trial_min_stored_m": _stats(stored_min),
            "trial_min_physx_m": _stats(physx_min),
            "trial_min_difference_m": _stats(physx_min - stored_min),
            "runtime_gripper_trial_min_stored_m": _stats(runtime_stored_min),
            "runtime_gripper_trial_min_physx_m": _stats(runtime_physx_min),
            "runtime_vs_model_gripper_physx_trial_min_difference_m": _stats(
                runtime_physx_min - physx_min
            ),
            "trial_active_disagreement_fraction": float(
                np.mean((physx_min <= manifest.delta_gate_m) != (stored_min <= manifest.delta_gate_m))
            ),
            "contact_count": int(contact.sum()),
            "transition_count": int(contact.size),
            "seconds": perf_counter() - started,
        }
        rows.append(row)
        print(
            f"[{object_index:02d}/{len(object_ids):02d}] {object_id}: "
            f"surface p95={row['raw_surface_to_physx_m']['p95'] * 1000:.3f} mm, "
            f"trial median delta={row['trial_min_difference_m']['median'] * 1000:+.3f} mm, "
            f"active disagreement={row['trial_active_disagreement_fraction'] * 100:.2f}%, "
            f"{row['seconds']:.1f}s",
            flush=True,
        )

    stored_trial = np.concatenate(all_stored_trial)
    physx_trial = np.concatenate(all_physx_trial)
    contact = np.concatenate(all_contact).astype(bool)
    runtime_stored_trial = np.concatenate(all_runtime_stored_trial)
    runtime_physx_trial = np.concatenate(all_runtime_physx_trial)
    split = np.asarray(all_split)
    target_recall = 0.995
    calibration_mask = split != "test"
    stored_calibrated = _threshold_at_recall(
        stored_trial[calibration_mask], contact[calibration_mask], target_recall
    )
    physx_calibrated = _threshold_at_recall(
        physx_trial[calibration_mask], contact[calibration_mask], target_recall
    )
    runtime_physx_calibrated = _threshold_at_recall(
        runtime_physx_trial[calibration_mask], contact[calibration_mask], target_recall
    )
    delta = physx_trial - stored_trial
    split_metrics = {}
    for split_name in ("train", "val", "test"):
        mask = split == split_name
        split_metrics[split_name] = {
            "count": int(mask.sum()),
            "stored_at_manifest_gate": _classification(
                stored_trial[mask], contact[mask], manifest.delta_gate_m
            ),
            "physx_at_manifest_gate": _classification(
                physx_trial[mask], contact[mask], manifest.delta_gate_m
            ),
            "stored_at_train_val_calibrated_gate": _classification(
                stored_trial[mask], contact[mask], stored_calibrated
            ),
            "physx_at_train_val_calibrated_gate": _classification(
                physx_trial[mask], contact[mask], physx_calibrated
            ),
            "runtime_gripper_physx_at_manifest_gate": _classification(
                runtime_physx_trial[mask], contact[mask], manifest.delta_gate_m
            ),
            "runtime_gripper_physx_at_train_val_calibrated_gate": _classification(
                runtime_physx_trial[mask], contact[mask], runtime_physx_calibrated
            ),
        }
    summary = {
        "definitions": {
            "stored_sdf": "trilinear samples of the float16 96^3 SDF built from the authored triangle mesh",
            "physx_sdf": "signed distance to the exact convex hulls returned by PhysX cooking for convexDecomposition",
            "trial_points": "all 256 gripper points at next commanded aperture, transformed by every GT current pose",
            "runtime_gripper_points": "256 deterministic samples of the runtime USD contact-link convex hull proxy, posed from live Isaac body frames at all 33 commands",
        },
        "counts": {
            "objects": len(rows),
            "transitions": int(len(stored_trial)),
            "contacts": int(contact.sum()),
        },
        "object_metrics": rows,
        "aggregate": {
            "trial_min_difference_m": _stats(delta),
            "trial_min_absolute_difference_m": _stats(np.abs(delta)),
            "trial_gap_pearson": float(np.corrcoef(stored_trial, physx_trial)[0, 1]),
            "contact_relevant_trial_gaps": _gap_comparison(
                stored_trial, physx_trial, manifest.delta_gate_m
            ),
            "runtime_vs_model_gripper_physx_trial_min_difference_m": _stats(
                runtime_physx_trial - physx_trial
            ),
            "active_disagreement_at_manifest_gate_fraction": float(
                np.mean((stored_trial <= manifest.delta_gate_m) != (physx_trial <= manifest.delta_gate_m))
            ),
            "stored_at_manifest_gate": _classification(stored_trial, contact, manifest.delta_gate_m),
            "physx_at_manifest_gate": _classification(physx_trial, contact, manifest.delta_gate_m),
            "stored_calibrated_train_val_99_5_recall": _classification(
                stored_trial[calibration_mask], contact[calibration_mask], stored_calibrated
            ),
            "physx_calibrated_train_val_99_5_recall": _classification(
                physx_trial[calibration_mask], contact[calibration_mask], physx_calibrated
            ),
            "runtime_gripper_physx_at_manifest_gate": _classification(
                runtime_physx_trial, contact, manifest.delta_gate_m
            ),
            "runtime_gripper_physx_calibrated_train_val_99_5_recall": _classification(
                runtime_physx_trial[calibration_mask],
                contact[calibration_mask],
                runtime_physx_calibrated,
            ),
            "split_metrics": split_metrics,
            "raw_mesh_topology": {
                "watertight_objects": int(sum(row["raw_watertight"] for row in rows)),
                "volume_objects": int(sum(row["raw_is_volume"] for row in rows)),
                "winding_consistent_objects": int(
                    sum(row["raw_winding_consistent"] for row in rows)
                ),
                "total_objects": len(rows),
            },
            "raw_surface_to_physx_abs_m": _stats(
                np.concatenate([
                    np.asarray([row["raw_surface_to_physx_m"]["mean"]]) for row in rows
                ])
            ),
        },
    }
    (output / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output / "trial_gaps.npz",
        stored_sdf_m=stored_trial,
        physx_sdf_m=physx_trial,
        runtime_gripper_stored_sdf_m=runtime_stored_trial,
        runtime_gripper_physx_sdf_m=runtime_physx_trial,
        contact=contact,
        object_id=np.asarray(all_object),
        split=split,
    )
    _plot(
        output,
        rows,
        stored_trial,
        physx_trial,
        contact,
        manifest.delta_gate_m,
        runtime_physx_trial,
    )
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
