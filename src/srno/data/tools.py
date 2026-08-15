from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from srno.data.dataset import H5ObjectDataset, ObjectRecord
from srno.data.index import ActiveIndex
from srno.data.schema import (
    NUM_STATES,
    NUM_STEPS,
    SDF_RESOLUTION,
    DatasetManifest,
)
from srno.data.writer import DIAGNOSTIC_SPECS
from srno.geometry.gripper import GripperAsset
from srno.geometry.sdf import sample_sdf
from srno.geometry.se3 import quaternion_xyzw_to_matrix


class DatasetValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ValidationReport:
    objects: int
    trajectories: int
    diagnostics: tuple[str, ...]


def validate_dataset(
    manifest_or_path: DatasetManifest | str | Path,
    *,
    strict_resolution: bool = True,
) -> ValidationReport:
    manifest = (
        manifest_or_path
        if isinstance(manifest_or_path, DatasetManifest)
        else DatasetManifest.load(manifest_or_path)
    )
    errors: list[str] = []
    if not manifest.gripper_path.is_file():
        errors.append(f"gripper asset does not exist: {manifest.gripper_path}")
    else:
        try:
            asset = GripperAsset.load(manifest.gripper_path)
            if asset.sha256() != manifest.gripper_sha256:
                errors.append("gripper asset hash does not match manifest")
            if not np.isclose(asset.length_scale, manifest.length_scale_m):
                errors.append("gripper and manifest length scales differ")
            schedule = np.asarray(manifest.commanded_aperture_m)
            if schedule.min() < asset.aperture_min - 1e-7 or schedule.max() > asset.aperture_max + 1e-7:
                errors.append("commanded aperture schedule is outside gripper limits")
        except Exception as error:
            errors.append(f"cannot read gripper asset: {error}")

    total_objects = 0
    total_trajectories = 0
    diagnostic_names: set[str] = set()
    schedule = np.asarray(manifest.commanded_aperture_m, dtype=np.float32)
    for shard in manifest.shards:
        path = manifest.shard_path(shard)
        if not path.is_file():
            errors.append(f"shard does not exist: {path}")
            continue
        try:
            with h5py.File(path, "r") as handle:
                if int(handle.attrs.get("schema_version", -1)) != manifest.schema_version:
                    errors.append(f"{path}: schema version mismatch")
                groups = handle.get("objects")
                if groups is None or len(groups) != len(shard.object_ids):
                    errors.append(f"{path}: manifest object count does not match shard")
                    continue
                for group_index, expected_id in enumerate(shard.object_ids):
                    prefix = f"{path}:objects/{group_index:06d}"
                    group = groups.get(f"{group_index:06d}")
                    if group is None:
                        errors.append(f"{prefix}: missing group")
                        continue
                    total_objects += 1
                    if str(group.attrs.get("object_id", "")) != expected_id:
                        errors.append(f"{prefix}: object_id mismatch")
                    required = {"sdf", "position", "quaternion_xyzw", "actual_aperture"}
                    if not required.issubset(group.keys()):
                        errors.append(f"{prefix}: missing one of {sorted(required)}")
                        continue
                    sdf = group["sdf"]
                    if sdf.ndim != 3 or (strict_resolution and sdf.shape != SDF_RESOLUTION):
                        errors.append(f"{prefix}: SDF shape {sdf.shape}, expected {SDF_RESOLUTION}")
                    if sdf.dtype not in (np.dtype("float16"), np.dtype("float32")):
                        errors.append(f"{prefix}: SDF must be float16 or float32")
                    if not np.isfinite(sdf[...]).all():
                        errors.append(f"{prefix}: SDF contains non-finite values")
                    voxel = np.asarray(group.attrs.get("voxel_size", []), dtype=np.float32)
                    origin = np.asarray(group.attrs.get("grid_origin", []), dtype=np.float32)
                    if voxel.shape != (3,) or np.any(voxel <= 0):
                        errors.append(f"{prefix}: invalid voxel_size")
                    elif np.max(voxel) >= manifest.delta_gate_m / 2:
                        errors.append(f"{prefix}: voxel_size must be < delta_gate/2")
                    if origin.shape != (3,) or not np.isfinite(origin).all():
                        errors.append(f"{prefix}: invalid grid_origin")

                    position = group["position"][...]
                    quaternion = group["quaternion_xyzw"][...]
                    aperture = group["actual_aperture"][...]
                    trajectories = position.shape[0] if position.ndim else 0
                    total_trajectories += trajectories
                    if position.shape != (trajectories, NUM_STATES, 3):
                        errors.append(f"{prefix}: invalid position shape")
                    if quaternion.shape != (trajectories, NUM_STATES, 4):
                        errors.append(f"{prefix}: invalid quaternion shape")
                    if aperture.shape != (trajectories, NUM_STATES):
                        errors.append(f"{prefix}: invalid aperture shape")
                    if not all(np.isfinite(array).all() for array in (position, quaternion, aperture)):
                        errors.append(f"{prefix}: trajectory arrays contain non-finite values")
                    if quaternion.shape[-1:] == (4,):
                        norm_error = np.max(np.abs(np.linalg.norm(quaternion, axis=-1) - 1))
                        if norm_error > 1e-3:
                            errors.append(f"{prefix}: quaternion norm error {norm_error:.3e}")
                    if aperture.shape == (trajectories, NUM_STATES):
                        if np.any(np.diff(aperture, axis=1) > 1e-6):
                            errors.append(f"{prefix}: actual aperture is not monotone")
                        if np.any(aperture < schedule[None, :] - 1e-6):
                            errors.append(f"{prefix}: actual aperture is below commanded aperture")
                    if "source_pose_index" in group:
                        source_pose_index = group["source_pose_index"][...]
                        if source_pose_index.shape != (trajectories,):
                            errors.append(f"{prefix}: invalid source_pose_index shape")
                        elif not np.issubdtype(source_pose_index.dtype, np.integer):
                            errors.append(f"{prefix}: source_pose_index must be integer")
                        elif np.any(source_pose_index < 0):
                            errors.append(f"{prefix}: source_pose_index must be non-negative")

                    if "diagnostics" in group:
                        for name, dataset in group["diagnostics"].items():
                            diagnostic_names.add(name)
                            if name not in DIAGNOSTIC_SPECS:
                                errors.append(f"{prefix}: unknown diagnostic {name!r}")
                                continue
                            expected_shape = (trajectories, NUM_STEPS) + DIAGNOSTIC_SPECS[name]
                            if dataset.shape != expected_shape:
                                errors.append(
                                    f"{prefix}: diagnostic {name!r} shape {dataset.shape}, "
                                    f"expected {expected_shape}"
                                )
        except OSError as error:
            errors.append(f"cannot read shard {path}: {error}")
    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:50])
        suffix = f"\n... {len(errors) - 50} more" if len(errors) > 50 else ""
        raise DatasetValidationError(f"dataset validation failed:\n{preview}{suffix}")
    return ValidationReport(total_objects, total_trajectories, tuple(sorted(diagnostic_names)))


def _trial_minimum_gaps(
    record: ObjectRecord,
    gripper: GripperAsset,
    schedule: torch.Tensor,
    *,
    device: torch.device,
    outside_value: float,
) -> np.ndarray:
    position = record.position.to(device=device, dtype=torch.float32)
    rotation = quaternion_xyzw_to_matrix(
        record.quaternion_xyzw.to(device=device, dtype=torch.float32)
    )
    sdf = record.sdf.to(device=device)
    origin = record.origin.to(device=device)
    voxel = record.voxel_size.to(device=device)
    gripper = gripper.to(device)
    trajectories = position.shape[0]
    result = torch.empty((trajectories, NUM_STEPS), device=device)
    mapping = torch.zeros(trajectories, dtype=torch.long, device=device)
    with torch.no_grad():
        for step in range(NUM_STEPS):
            points = gripper.points(schedule[step + 1]).expand(trajectories, -1, -1)
            relative = points - position[:, step, None, :]
            points_object = torch.einsum(
                "bij,bmj->bmi", rotation[:, step].transpose(-1, -2), relative
            )
            gaps = sample_sdf(
                sdf[None],
                origin[None],
                voxel[None],
                points_object,
                sample_to_object=mapping,
                outside_value=outside_value,
            )
            result[:, step] = gaps.amin(dim=-1)
    return result.cpu().numpy()


def build_active_index(
    manifest_or_path: DatasetManifest | str | Path,
    output_path: str | Path,
    *,
    device: str = "cpu",
) -> ActiveIndex:
    manifest = (
        manifest_or_path
        if isinstance(manifest_or_path, DatasetManifest)
        else DatasetManifest.load(manifest_or_path)
    )
    gripper = GripperAsset.load(manifest.gripper_path)
    if gripper.sha256() != manifest.gripper_sha256:
        raise ValueError("gripper hash does not match manifest")
    schedule = torch.tensor(manifest.commanded_aperture_m, device=device)
    all_ids = tuple(object_id for shard in manifest.shards for object_id in shard.object_ids)
    pairs_by_id: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        dataset = H5ObjectDataset(manifest, split=split)
        try:
            for record in dataset:
                minimum_gap = _trial_minimum_gaps(
                    record,
                    gripper,
                    schedule,
                    device=torch.device(device),
                    outside_value=manifest.sdf_scale_m,
                )
                pairs_by_id[record.object_id] = np.argwhere(
                    minimum_gap <= manifest.delta_gate_m
                ).astype(np.int64)
        finally:
            dataset.close()
    offsets = [0]
    trajectory_indices: list[np.ndarray] = []
    step_indices: list[np.ndarray] = []
    for object_id in all_ids:
        pairs = pairs_by_id[object_id]
        trajectory_indices.append(pairs[:, 0].astype(np.int32))
        step_indices.append(pairs[:, 1].astype(np.uint8))
        offsets.append(offsets[-1] + len(pairs))
    index = ActiveIndex(
        manifest.sha256(),
        gripper.sha256(),
        manifest.delta_gate_m,
        all_ids,
        np.asarray(offsets, dtype=np.int64),
        np.concatenate(trajectory_indices) if trajectory_indices else np.empty(0, np.int32),
        np.concatenate(step_indices) if step_indices else np.empty(0, np.uint8),
    )
    index.save(output_path)
    return index


def calibrate_gate(
    manifest_or_path: DatasetManifest | str | Path,
    *,
    target_recall: float = 0.995,
    device: str = "cpu",
) -> dict[str, float | int]:
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    manifest = (
        manifest_or_path
        if isinstance(manifest_or_path, DatasetManifest)
        else DatasetManifest.load(manifest_or_path)
    )
    gripper = GripperAsset.load(manifest.gripper_path)
    schedule = torch.tensor(manifest.commanded_aperture_m, device=device)
    contact_gaps: list[np.ndarray] = []
    free_gaps: list[np.ndarray] = []
    maximum_voxel = 0.0
    for split in ("train", "val"):
        dataset = H5ObjectDataset(manifest, split=split)
        try:
            for record in dataset:
                if "contact_count" not in record.diagnostics:
                    raise ValueError("gate calibration requires contact_count diagnostics")
                gaps = _trial_minimum_gaps(
                    record,
                    gripper,
                    schedule,
                    device=torch.device(device),
                    outside_value=manifest.sdf_scale_m,
                )
                contacts = record.diagnostics["contact_count"].numpy() > 0
                contact_gaps.append(gaps[contacts])
                free_gaps.append(gaps[~contacts])
                maximum_voxel = max(maximum_voxel, float(record.voxel_size.max()))
        finally:
            dataset.close()
    contact = np.concatenate(contact_gaps)
    free = np.concatenate(free_gaps)
    if not len(contact):
        raise ValueError("calibration data contains no simulator contacts")
    quantile = float(np.quantile(contact, target_recall, method="higher"))
    threshold = max(quantile, 2.01 * maximum_voxel, np.finfo(np.float32).eps)
    true_positive_rate = float(np.mean(contact <= threshold))
    false_positive_rate = float(np.mean(free <= threshold)) if len(free) else 0.0
    return {
        "recommended_delta_gate_m": threshold,
        "contact_recall": true_positive_rate,
        "free_false_positive_rate": false_positive_rate,
        "contact_transitions": int(len(contact)),
        "free_transitions": int(len(free)),
        "maximum_voxel_size_m": maximum_voxel,
    }
