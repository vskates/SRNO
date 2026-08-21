#!/usr/bin/env python3
"""No-retrain quasistatic-refinement and split signed-bias diagnostics.

The runner deliberately does not write dataset shards, train a model, or
change production defaults.  It has three resumable phases:

* ``bias`` evaluates the existing gap+aperture H32 checkpoints on all splits;
* ``refinement`` replays a small, fixed simulator sample at N=32/64/128 and
  at N=32 with a stricter settling rule;
* ``summarize`` combines raw sidecars into JSON/NPZ figures.

Run ``--phase all`` to execute the phases in that order.  Separate phases are
useful because Isaac Sim owns substantial CPU/GPU memory even after a stage is
closed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

try:
    from contact_composition_diagnostics import (
        _checkpoint_config,
        _load_model,
        _pose_log_error,
        _runtime_contract_audit,
        _verify_frozen_contract,
    )
    from contact_memory_diagnostics import _close_stage, _open_stage
except ModuleNotFoundError:  # Imported as ``scripts.*`` by pytest.
    from scripts.contact_composition_diagnostics import (
        _checkpoint_config,
        _load_model,
        _pose_log_error,
        _runtime_contract_audit,
        _verify_frozen_contract,
    )
    from scripts.contact_memory_diagnostics import _close_stage, _open_stage


LEVELS = (32, 64, 128)
MODEL_SEEDS = (0, 1, 2)
BIAS_STEP_BANDS = ((1, 8), (9, 16), (17, 24), (25, 32))
BOOTSTRAP_SEED = 0
PRODUCTION_REPEATS = 2
STRICT_LEVEL = 32
OBJECTS = (
    "kofe-naturalnyy-rastvorimyy-sublimirovannyy-16125",
    "masliny-federici-bez-kostochki-300-g-90215",
    "gerkules-ovsyanye-khlopya-400-g-1248",
)
EXPECTED_SPLITS = ("train", "val", "test")
EXPECTED_SOURCE_POSES = {
    OBJECTS[0]: (4987, 3482, 824, 4431),
    OBJECTS[1]: (3055, 2593, 637, 4239),
    OBJECTS[2]: (1894, 2563, 1004, 778),
}


@dataclass(frozen=True)
class SelectedPoses:
    object_id: str
    split: str
    trajectory: np.ndarray
    source_pose_index: np.ndarray
    contact_onset: np.ndarray
    initial_position: np.ndarray
    initial_quaternion_xyzw: np.ndarray
    initial_joint: np.ndarray


@dataclass(frozen=True)
class ScheduleStates:
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    joint: np.ndarray
    aperture: np.ndarray
    contact_count: np.ndarray
    settling_substeps: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    joint_velocity: np.ndarray
    settled: np.ndarray
    initial_restore_dx: np.ndarray


def _take_selected(selected: SelectedPoses, count: int) -> SelectedPoses:
    if count <= 0 or count > len(selected.trajectory):
        raise ValueError("pose count is outside the selected refinement sample")
    return SelectedPoses(
        selected.object_id,
        selected.split,
        selected.trajectory[:count],
        selected.source_pose_index[:count],
        selected.contact_onset[:count],
        selected.initial_position[:count],
        selected.initial_quaternion_xyzw[:count],
        selected.initial_joint[:count],
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _command_schedule(aperture_knots: np.ndarray, level: int) -> np.ndarray:
    """Interpolate the frozen 33-knot aperture path at k / level."""

    knots = np.asarray(aperture_knots, dtype=np.float64)
    if knots.shape != (33,) or level <= 0:
        raise ValueError("expected 33 aperture knots and a positive level")
    coordinate = np.arange(level + 1, dtype=np.float64) * 32.0 / float(level)
    return np.interp(coordinate, np.arange(33, dtype=np.float64), knots)


def _common_command_indices(coarse: int, fine: int) -> np.ndarray:
    if fine % coarse:
        raise ValueError("fine command count must be divisible by coarse count")
    return np.arange(coarse + 1, dtype=np.int64) * (fine // coarse)


def _rotation_distance_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_value = np.asarray(left, dtype=np.float64)
    right_value = np.asarray(right, dtype=np.float64)
    left_value = left_value / np.clip(
        np.linalg.norm(left_value, axis=-1, keepdims=True), 1e-15, None
    )
    right_value = right_value / np.clip(
        np.linalg.norm(right_value, axis=-1, keepdims=True), 1e-15, None
    )
    dot = np.abs(np.sum(left_value * right_value, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


def _state_components(
    left_position: np.ndarray,
    left_quaternion: np.ndarray,
    left_joint: np.ndarray,
    right_position: np.ndarray,
    right_quaternion: np.ndarray,
    right_joint: np.ndarray,
    *,
    length_scale: float,
    joint_scale: np.ndarray,
) -> dict[str, np.ndarray]:
    translation = np.linalg.norm(
        np.asarray(left_position, dtype=np.float64)
        - np.asarray(right_position, dtype=np.float64),
        axis=-1,
    )
    rotation = _rotation_distance_xyzw(left_quaternion, right_quaternion)
    joints = np.sqrt(
        np.mean(
            (
                (
                    np.asarray(left_joint, dtype=np.float64)
                    - np.asarray(right_joint, dtype=np.float64)
                )
                / np.asarray(joint_scale, dtype=np.float64)
            )
            ** 2,
            axis=-1,
        )
    )
    distance = np.sqrt((translation / float(length_scale)) ** 2 + rotation**2 + joints**2)
    return {
        "dx": distance,
        "translation_m": translation,
        "rotation_rad": rotation,
        "joint_rmse_over_travel": joints,
    }


def _object_split(manifest: Any, object_id: str) -> str:
    matches = [name for name, values in manifest.splits.items() if object_id in values]
    if len(matches) != 1:
        raise ValueError(f"{object_id}: expected exactly one dataset split")
    return matches[0]


def _decode_names(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    )


def _select_refinement_poses(
    manifest: Any,
    object_id: str,
    *,
    expected_source_poses: tuple[int, ...] | None = None,
    expected_joint_names: tuple[str, ...] | None = None,
) -> SelectedPoses:
    """Select one informative, non-pathological trajectory per onset quartile."""

    shard, group_name = manifest.object_locations()[object_id]
    with h5py.File(shard, "r") as handle:
        group = handle[group_name]
        position = np.asarray(group["position"], dtype=np.float32)
        quaternion = np.asarray(group["quaternion_xyzw"], dtype=np.float32)
        joint = np.asarray(group["joint_position"], dtype=np.float32)
        aperture = np.asarray(group["actual_aperture"], dtype=np.float32)
        contact = np.asarray(group["diagnostics/contact_count"], dtype=np.float32)
        settling = np.asarray(group["diagnostics/settling_substeps"], dtype=np.int32)
        source = np.asarray(group["source_pose_index"], dtype=np.int64)
        joint_names = _decode_names(group["joint_position"].attrs["joint_names"])
    if expected_joint_names is not None and joint_names != expected_joint_names:
        raise ValueError(f"{object_id}: HDF5 joint order differs from gripper asset")

    command = np.asarray(manifest.commanded_aperture_m, dtype=np.float64)
    has_contact = np.any(contact > 0.0, axis=1)
    onset = np.where(has_contact, np.argmax(contact > 0.0, axis=1) + 1, 33)
    max_settling = settling.max(axis=1)
    settling_p90 = float(np.quantile(max_settling, 0.90))
    eligible = np.flatnonzero(has_contact & (max_settling <= settling_p90))
    if len(eligible) < 4:
        raise ValueError(f"{object_id}: fewer than four eligible contact trajectories")

    translation = np.linalg.norm(position[:, -1] - position[:, 0], axis=1)
    rotation = _rotation_distance_xyzw(quaternion[:, -1], quaternion[:, 0])
    lag = np.max(
        np.maximum(aperture[:, 1:].astype(np.float64) - command[None, 1:], 0.0),
        axis=1,
    )
    ordered = eligible[np.lexsort((source[eligible], onset[eligible]))]
    quartiles = np.array_split(ordered, 4)

    def safe_median(values: np.ndarray) -> float:
        return max(float(np.median(values)), 1e-12)

    combined = (
        translation / safe_median(translation[eligible])
        + rotation / safe_median(rotation[eligible])
        + lag / safe_median(lag[eligible])
    )
    scores = (rotation, translation, lag, combined)
    chosen: list[int] = []
    for candidates, score in zip(quartiles, scores, strict=True):
        order = np.lexsort((source[candidates], -score[candidates]))
        chosen.append(int(candidates[order[0]]))
    trajectory = np.asarray(chosen, dtype=np.int64)
    selected_sources = tuple(int(value) for value in source[trajectory])
    if expected_source_poses is not None and selected_sources != expected_source_poses:
        raise ValueError(
            f"{object_id}: deterministic selection changed: "
            f"{selected_sources} != {expected_source_poses}"
        )
    return SelectedPoses(
        object_id=object_id,
        split=_object_split(manifest, object_id),
        trajectory=trajectory,
        source_pose_index=source[trajectory],
        contact_onset=onset[trajectory].astype(np.int32),
        initial_position=position[trajectory, 0],
        initial_quaternion_xyzw=quaternion[trajectory, 0],
        initial_joint=joint[trajectory, 0],
    )


def _strict_simulator_config(config: Any) -> Any:
    return replace(
        config,
        settling=replace(
            config.settling,
            min_steps=40,
            consecutive_steps=20,
            position_delta_m=0.00025,
            linear_velocity_m_s=0.005,
            angular_velocity_rad_s=0.05,
            joint_velocity_rad_s=0.005,
        ),
    )


def _write_initial_states(
    collector: Any,
    selected: SelectedPoses,
    source_position: Any,
    source_quaternion: Any,
    *,
    repeats: int,
) -> tuple[Any, Any, np.ndarray]:
    """Write the exact stored q0,r0 into fresh parallel PhysX branches."""

    import torch
    from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms

    base_position, base_quaternion = collector._reset_batch(
        source_position, source_quaternion
    )
    position = torch.from_numpy(np.tile(selected.initial_position, (repeats, 1))).to(
        collector.device
    )
    quaternion_xyzw = np.tile(selected.initial_quaternion_xyzw, (repeats, 1))
    quaternion = torch.from_numpy(quaternion_xyzw[:, (3, 0, 1, 2)]).to(
        collector.device
    )
    joint = torch.from_numpy(np.tile(selected.initial_joint, (repeats, 1))).to(
        collector.device
    )

    object_state = collector.object.data.root_state_w.clone()
    object_state[:, :3] = base_position + quat_apply(base_quaternion, position)
    object_state[:, 3:7] = quat_mul(base_quaternion, quaternion)
    object_state[:, 7:13] = 0.0
    collector.object.write_root_state_to_sim(object_state)
    collector.robot.write_joint_state_to_sim(joint, torch.zeros_like(joint))
    collector._hold_root()

    restored_position, restored_quaternion = subtract_frame_transforms(
        base_position,
        base_quaternion,
        collector.object.data.root_pos_w,
        collector.object.data.root_quat_w,
    )
    restored_quaternion_xyzw = restored_quaternion[:, (1, 2, 3, 0)]
    metrics = _state_components(
        restored_position.detach().cpu().numpy(),
        restored_quaternion_xyzw.detach().cpu().numpy(),
        collector.robot.data.joint_pos.detach().cpu().numpy(),
        position.detach().cpu().numpy(),
        quaternion_xyzw,
        joint.detach().cpu().numpy(),
        length_scale=float(collector.gripper_asset.length_scale),
        joint_scale=collector.gripper_asset.joint_travel_range.detach().cpu().numpy(),
    )
    restore_dx = metrics["dx"]
    print(
        "[REFINE] exact initial restore d_X="
        f"{np.array2string(restore_dx, precision=8)}",
        flush=True,
    )
    if not np.all(np.isfinite(restore_dx)) or float(np.max(restore_dx)) > 1e-6:
        raise RuntimeError(
            f"exact q0,r0 restore failed: max d_X={float(np.max(restore_dx)):.6g}"
        )
    return base_position, base_quaternion, restore_dx.astype(np.float32)


def _run_schedule(
    app: Any,
    config: Any,
    catalog: Any,
    gripper: Any,
    selected: SelectedPoses,
    *,
    level: int,
    repeats: int,
) -> ScheduleStates:
    import torch

    from srno.sim.pose_seeds import PoseSeeds

    pose_count = len(selected.trajectory)
    count = pose_count * repeats
    simulation = scene = collector = None
    try:
        simulation, scene, collector = _open_stage(
            app,
            config,
            catalog,
            gripper,
            selected.object_id,
            env_count=count,
            collision_system="PCM",
        )
        source = PoseSeeds.load(catalog.object(selected.object_id).pose_seed_path)
        source_index = np.tile(selected.source_pose_index, repeats)
        source_position = torch.from_numpy(source.position_m[source_index]).to(
            collector.device
        )
        source_quaternion = torch.from_numpy(
            source.quaternion_wxyz[source_index]
        ).to(collector.device)
        base_position, base_quaternion, restore_dx = _write_initial_states(
            collector,
            selected,
            source_position,
            source_quaternion,
            repeats=repeats,
        )

        position = np.full((count, level + 1, 3), np.nan, dtype=np.float32)
        quaternion = np.full((count, level + 1, 4), np.nan, dtype=np.float32)
        joint = np.full((count, level + 1, 6), np.nan, dtype=np.float32)
        aperture = np.full((count, level + 1), np.nan, dtype=np.float32)
        contact = np.full((count, level), np.nan, dtype=np.float32)
        settling = np.full((count, level), -1, dtype=np.int32)
        linear = np.full((count, level, 3), np.nan, dtype=np.float32)
        angular = np.full((count, level, 3), np.nan, dtype=np.float32)
        joint_velocity = np.full((count, level, 6), np.nan, dtype=np.float32)
        settled = np.zeros((count, level + 1), dtype=np.bool_)

        position[:, 0] = np.tile(selected.initial_position, (repeats, 1))
        quaternion[:, 0] = np.tile(selected.initial_quaternion_xyzw, (repeats, 1))
        joint[:, 0] = np.tile(selected.initial_joint, (repeats, 1))
        aperture[:, 0] = (
            collector._actual_aperture(
                torch.from_numpy(joint[:, 0]).to(collector.device)
            )
            .detach()
            .cpu()
            .numpy()
        )
        settled[:, 0] = True
        alive = torch.ones(count, dtype=torch.bool, device=collector.device)
        previous_quaternion = torch.from_numpy(
            quaternion[:, 0][:, (3, 0, 1, 2)]
        ).to(collector.device)
        for command_index in range(1, level + 1):
            if not bool(torch.any(alive)):
                break
            fraction = float(command_index) / float(level)
            target = collector.open_joint_target + fraction * (
                collector.close_joint_target - collector.open_joint_target
            )
            result = collector._settle_command(
                target, command_index, required_mask=alive
            )
            required_before = alive.clone()
            alive &= result.settled_mask
            relative_position, relative_quaternion, measured_aperture = (
                collector._record_state(base_position, base_quaternion, result)
            )
            flip = torch.sum(previous_quaternion * relative_quaternion, dim=-1) < 0.0
            relative_quaternion = torch.where(
                flip[:, None], -relative_quaternion, relative_quaternion
            )
            previous_quaternion = relative_quaternion
            valid = required_before & result.settled_mask
            ids = torch.nonzero(valid).flatten().cpu().numpy()
            if len(ids):
                position[ids, command_index] = (
                    relative_position[valid].detach().cpu().numpy()
                )
                quaternion[ids, command_index] = (
                    relative_quaternion[valid][:, (1, 2, 3, 0)]
                    .detach()
                    .cpu()
                    .numpy()
                )
                joint[ids, command_index] = (
                    result.joint_position[valid].detach().cpu().numpy()
                )
                aperture[ids, command_index] = (
                    measured_aperture[valid].detach().cpu().numpy()
                )
                contact[ids, command_index - 1] = (
                    result.contact_count[valid].detach().cpu().numpy()
                )
                settling[ids, command_index - 1] = (
                    result.environment_steps[valid].detach().cpu().numpy()
                )
                linear[ids, command_index - 1] = (
                    result.linear_velocity[valid].detach().cpu().numpy()
                )
                angular[ids, command_index - 1] = (
                    result.angular_velocity[valid].detach().cpu().numpy()
                )
                joint_velocity[ids, command_index - 1] = (
                    result.joint_velocity[valid].detach().cpu().numpy()
                )
                settled[ids, command_index] = True
            failed = torch.nonzero(required_before & ~result.settled_mask).flatten()
            if len(failed):
                print(
                    f"[REFINE] {selected.object_id}: N={level} failed envs="
                    f"{failed.cpu().tolist()} at command {command_index}/{level}",
                    flush=True,
                )
        shape = (repeats, pose_count)

        def shaped(value: np.ndarray) -> np.ndarray:
            return value.reshape(shape + value.shape[1:])

        return ScheduleStates(
            shaped(position),
            shaped(quaternion),
            shaped(joint),
            shaped(aperture),
            shaped(contact),
            shaped(settling),
            shaped(linear),
            shaped(angular),
            shaped(joint_velocity),
            shaped(settled),
            restore_dx.reshape(shape),
        )
    finally:
        if simulation is not None:
            _close_stage(app, simulation, scene, collector)


def _schedule_arrays(prefix: str, states: ScheduleStates) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_position": states.position,
        f"{prefix}_quaternion_xyzw": states.quaternion_xyzw,
        f"{prefix}_joint": states.joint,
        f"{prefix}_aperture": states.aperture,
        f"{prefix}_contact_count": states.contact_count,
        f"{prefix}_settling_substeps": states.settling_substeps,
        f"{prefix}_linear_velocity": states.linear_velocity,
        f"{prefix}_angular_velocity": states.angular_velocity,
        f"{prefix}_joint_velocity": states.joint_velocity,
        f"{prefix}_settled": states.settled,
        f"{prefix}_initial_restore_dx": states.initial_restore_dx,
    }


def _run_refinement_phase(args: argparse.Namespace) -> None:
    from isaaclab.app import AppLauncher
    from srno.data.schema import DatasetManifest
    from srno.geometry.gripper import GripperAsset
    from srno.sim.assets import SimulatorAssetCatalog
    from srno.sim.config import SimulatorConfig
    from srno.sim.memory_guard import MemoryWatchdog

    manifest = DatasetManifest.load(args.manifest)
    config = SimulatorConfig.load(args.sim_config)
    _verify_frozen_contract(manifest, config)
    if not config.headless:
        raise ValueError("quasistatic refinement requires headless=true")
    if not np.isclose(config.memory_limit_gib, 14.0, atol=1e-12, rtol=0.0):
        raise ValueError("quasistatic refinement requires the 14 GiB watchdog")
    gripper = GripperAsset.load(manifest.gripper_path)
    if gripper.sha256() != manifest.gripper_sha256:
        raise ValueError("manifest gripper hash mismatch")
    catalog = SimulatorAssetCatalog.load(config.catalog)
    objects = tuple(args.objects or OBJECTS)
    if not set(objects) <= set(OBJECTS) or len(objects) != len(set(objects)):
        raise ValueError("refinement objects must be a unique subset of frozen defaults")
    selected = {
        object_id: _select_refinement_poses(
            manifest,
            object_id,
            expected_source_poses=EXPECTED_SOURCE_POSES[object_id],
            expected_joint_names=gripper.joint_names,
        )
        for object_id in objects
    }
    if args.pose_limit != 4:
        selected = {
            object_id: _take_selected(value, args.pose_limit)
            for object_id, value in selected.items()
        }
    actual_splits = tuple(selected[value].split for value in objects)
    if objects == OBJECTS and actual_splits != EXPECTED_SPLITS:
        raise ValueError(f"refinement objects have wrong splits: {actual_splits}")

    output = args.output / "refinement"
    output.mkdir(parents=True, exist_ok=True)
    failure_path = output / "refinement-failure.txt"
    failure_path.unlink(missing_ok=True)
    watchdog = MemoryWatchdog(14.0, 0.25)
    watchdog.start()
    app = AppLauncher({"headless": True, "device": config.device}).app
    try:
        audit_path = output / "runtime-audit.json"
        if not audit_path.is_file() or not args.resume:
            actuator_audit = _runtime_contract_audit(
                app, config, catalog, gripper, objects[0]
            )
            _write_json(
                audit_path,
                {
                    "manifest_sha256": manifest.sha256(),
                    "gripper_sha256": manifest.gripper_sha256,
                    "sim_config_sha256": _sha256(args.sim_config.resolve()),
                    "runtime_actuator_audit": actuator_audit,
                    "runtime_material_audit": "verified by every _open_stage",
                },
            )
        strict_config = _strict_simulator_config(config)
        for object_id in objects:
            destination = output / f"{object_id}.npz"
            if args.resume and destination.is_file():
                print(f"[REFINE] resume: keeping {destination}", flush=True)
                continue
            print(f"[REFINE] object={object_id}", flush=True)
            arrays: dict[str, np.ndarray] = {
                "object_id": np.asarray(object_id),
                "split": np.asarray(selected[object_id].split),
                "trajectory": selected[object_id].trajectory,
                "source_pose_index": selected[object_id].source_pose_index,
                "contact_onset": selected[object_id].contact_onset,
                "manifest_sha256": np.asarray(manifest.sha256()),
                "gripper_sha256": np.asarray(manifest.gripper_sha256),
                "sim_config_sha256": np.asarray(_sha256(args.sim_config.resolve())),
            }
            for level in args.levels:
                print(
                    f"[REFINE] {object_id}: production N={level}, "
                    f"repeats={PRODUCTION_REPEATS}",
                    flush=True,
                )
                states = _run_schedule(
                    app,
                    config,
                    catalog,
                    gripper,
                    selected[object_id],
                    level=level,
                    repeats=PRODUCTION_REPEATS,
                )
                arrays.update(_schedule_arrays(f"n{level}", states))
            print(f"[REFINE] {object_id}: strict N={args.strict_level}", flush=True)
            strict = _run_schedule(
                app,
                strict_config,
                catalog,
                gripper,
                selected[object_id],
                level=args.strict_level,
                repeats=1,
            )
            arrays.update(_schedule_arrays("n32_strict", strict))
            _atomic_savez(destination, **arrays)
            print(f"[REFINE] saved {destination}", flush=True)
    except BaseException:
        import traceback

        failure = traceback.format_exc()
        print(failure, flush=True)
        failure_path.write_text(failure, encoding="utf-8")
        raise
    finally:
        watchdog.stop()
        app.close()


def _evaluate_bias_split(
    model: Any,
    config: Any,
    manifest: Any,
    *,
    split: str,
    device: Any,
    global_object_index: dict[str, int],
) -> dict[str, np.ndarray]:
    import torch

    from srno.data.dataset import H5ObjectDataset, TrajectoryBatch, make_dataloader
    from srno.training.engine import _autocast
    from srno.types import PoseState

    rows: dict[str, list[np.ndarray]] = {}
    dataset = H5ObjectDataset(manifest, split=split)
    loader = make_dataloader(
        dataset,
        mode="rollout",
        objects_per_batch=1,
        samples_per_object=0,
        workers=0,
        seed=config.seed,
        shuffle=False,
    )
    try:
        with torch.no_grad():
            for raw_batch in loader:
                assert isinstance(raw_batch, TrajectoryBatch)
                object_id = raw_batch.object_ids[0]
                batch = raw_batch.to(device)
                initial = PoseState(
                    batch.states.rotation[:, 0],
                    batch.states.position[:, 0],
                    batch.states.joint_position[:, 0],
                )
                with _autocast(config, device):
                    autoregressive = model.rollout(
                        initial, batch.command_schedule[1:33], batch.sdf
                    )
                tf_log: list[Any] = []
                tf_spatial: list[Any] = []
                for step in range(32):
                    current = PoseState(
                        batch.states.rotation[:, step],
                        batch.states.position[:, step],
                        batch.states.joint_position[:, step],
                    )
                    target = PoseState(
                        batch.states.rotation[:, step + 1],
                        batch.states.position[:, step + 1],
                        batch.states.joint_position[:, step + 1],
                    )
                    with _autocast(config, device):
                        prediction = model.forward_step(
                            current,
                            batch.command_schedule[step + 1],
                            batch.sdf,
                        )
                    tf_log.append(_pose_log_error(target, prediction))
                    tf_spatial.append(prediction.position - target.position)
                target_rollout = PoseState(
                    batch.states.rotation[:, 1:33],
                    batch.states.position[:, 1:33],
                    batch.states.joint_position[:, 1:33],
                )
                prediction_rollout = PoseState(
                    autoregressive.rotation[:, 1:33],
                    autoregressive.position[:, 1:33],
                    autoregressive.joint_position[:, 1:33],
                )
                values = {
                    "tf_log": torch.stack(tf_log, dim=1).cpu().numpy(),
                    "ar_log": _pose_log_error(
                        target_rollout, prediction_rollout
                    ).cpu().numpy(),
                    "tf_spatial": torch.stack(tf_spatial, dim=1).cpu().numpy(),
                    "ar_spatial": (
                        prediction_rollout.position - target_rollout.position
                    ).cpu().numpy(),
                    "object_index": np.full(
                        len(raw_batch.trajectory_index),
                        global_object_index[object_id],
                        dtype=np.int32,
                    ),
                    "trajectory": raw_batch.trajectory_index.cpu().numpy().astype(
                        np.int32
                    ),
                }
                for name, value in values.items():
                    rows.setdefault(name, []).append(np.asarray(value))
                print(
                    f"[BIAS] seed={config.seed} split={split} object={object_id} "
                    f"trajectories={len(raw_batch.trajectory_index)}",
                    flush=True,
                )
    finally:
        dataset.close()
    return {name: np.concatenate(values, axis=0) for name, values in rows.items()}


def _run_bias_phase(args: argparse.Namespace) -> None:
    import torch

    from srno.data.schema import DatasetManifest
    from srno.training.config import ExperimentConfig

    manifest = DatasetManifest.load(args.manifest)
    base = ExperimentConfig.load(args.train_config)
    if base.paths.manifest.resolve() != args.manifest.resolve():
        raise ValueError("train config and diagnostic manifest differ")
    object_order = tuple(
        object_id
        for split in EXPECTED_SPLITS
        for object_id in manifest.splits[split]
    )
    if len(object_order) != 28 or len(set(object_order)) != 28:
        raise ValueError("material-v2 bias diagnostic expects 28 disjoint objects")
    global_index = {object_id: index for index, object_id in enumerate(object_order)}
    output = args.output / "bias"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_metadata: dict[str, Any] = {}
    for seed in MODEL_SEEDS:
        config = _checkpoint_config(base, arm="aperture", seed=seed)
        checkpoint_path = (
            args.checkpoint_root / f"seed-{seed}" / "best-rollout-h32.pt"
        )
        model, checkpoint = _load_model(
            config,
            manifest,
            checkpoint_path,
            device=device,
            stage="rollout",
            horizon=32,
        )
        checkpoint_metadata[str(seed)] = {
            "path": str(checkpoint_path.resolve()),
            "sha256": _sha256(checkpoint_path.resolve()),
            "best_metric": float(checkpoint["best_metric"]),
            "contact_features": config.model.contact_features,
            "global_conditioning": config.model.global_conditioning,
        }
        for split in EXPECTED_SPLITS:
            destination = output / f"seed-{seed}-{split}.npz"
            if args.resume and destination.is_file():
                print(f"[BIAS] resume: keeping {destination}", flush=True)
                continue
            values = _evaluate_bias_split(
                model,
                config,
                manifest,
                split=split,
                device=device,
                global_object_index=global_index,
            )
            _atomic_savez(
                destination,
                **values,
                split=np.asarray(split),
                seed=np.asarray(seed, dtype=np.int32),
                manifest_sha256=np.asarray(manifest.sha256()),
                gripper_sha256=np.asarray(manifest.gripper_sha256),
                checkpoint_sha256=np.asarray(checkpoint_metadata[str(seed)]["sha256"]),
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _write_json(
        output / "checkpoint-contract.json",
        {
            "manifest_sha256": manifest.sha256(),
            "gripper_sha256": manifest.gripper_sha256,
            "objects": list(object_order),
            "checkpoints": checkpoint_metadata,
            "production_model": {
                "contact_features": "gap",
                "global_conditioning": "aperture",
            },
        },
    )


def _mean_by_object(values: np.ndarray, object_index: np.ndarray) -> np.ndarray:
    return np.mean(
        [values[object_index == index].mean(axis=0) for index in np.unique(object_index)],
        axis=0,
    )


def _bias_feature_vector(
    tf_log: np.ndarray, ar_log: np.ndarray, tf_spatial: np.ndarray
) -> np.ndarray:
    bands = np.concatenate(
        [
            tf_log[:, lower - 1 : upper].mean(axis=1)
            for lower, upper in BIAS_STEP_BANDS
        ],
        axis=-1,
    )
    return np.concatenate(
        (
            bands,
            tf_log.sum(axis=1),
            ar_log[:, -1],
            tf_spatial.sum(axis=1),
        ),
        axis=-1,
    )


def _hierarchical_bias_bootstrap(
    features: np.ndarray,
    object_index: np.ndarray,
    *,
    replicates: int,
    seed: int,
    chunk_size: int = 100,
) -> dict[str, np.ndarray]:
    """Vectorized seed -> object -> trajectory bootstrap of compact features."""

    data = np.asarray(features, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError("features must have shape [seed, sample, feature]")
    objects = np.unique(object_index)
    grouped: list[np.ndarray] = []
    for object_id in objects:
        value = data[:, object_index == object_id]
        if not len(value[0]):
            raise ValueError("empty object in hierarchical bootstrap")
        grouped.append(value)
    if len({value.shape[1] for value in grouped}) != 1:
        raise ValueError("bias bootstrap expects equal trajectories per object")
    grouped_data = np.stack(grouped, axis=1)  # [seed, object, trajectory, feature]
    seeds, object_count, trajectories, feature_count = grouped_data.shape
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, feature_count), dtype=np.float64)
    offset = 0
    while offset < replicates:
        batch = min(chunk_size, replicates - offset)
        trajectory_means = np.empty(
            (batch, seeds, object_count, feature_count), dtype=np.float64
        )
        for model_seed in range(seeds):
            for object_id in range(object_count):
                draw = rng.integers(0, trajectories, size=(batch, trajectories))
                trajectory_means[:, model_seed, object_id] = grouped_data[
                    model_seed, object_id
                ][draw].mean(axis=1)
        sampled_seeds = rng.integers(0, seeds, size=(batch, seeds))
        sampled_objects = rng.integers(
            0, object_count, size=(batch, seeds, object_count)
        )
        batch_ids = np.arange(batch)[:, None, None]
        seed_ids = sampled_seeds[:, :, None]
        selected = trajectory_means[batch_ids, seed_ids, sampled_objects]
        samples[offset : offset + batch] = selected.mean(axis=(1, 2))
        offset += batch
    point = np.mean(
        [_mean_by_object(data[model_seed], object_index) for model_seed in range(seeds)],
        axis=0,
    )
    return {
        "mean": point,
        "ci95_lower": np.quantile(samples, 0.025, axis=0),
        "ci95_upper": np.quantile(samples, 0.975, axis=0),
    }


def _persistent_bias(
    mean: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> list[bool]:
    band_mean = np.asarray(mean[:24]).reshape(4, 6)
    band_lower = np.asarray(lower[:24]).reshape(4, 6)
    band_upper = np.asarray(upper[:24]).reshape(4, 6)
    excludes_zero = (band_lower > 0.0) | (band_upper < 0.0)
    persistent: list[bool] = []
    for component in range(6):
        signs = np.sign(band_mean[:, component][excludes_zero[:, component]])
        persistent.append(
            bool(
                len(signs) >= 3
                and max(np.count_nonzero(signs > 0), np.count_nonzero(signs < 0))
                >= 3
            )
        )
    return persistent


def _bootstrap_refinement_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    object_index: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    objects = np.unique(object_index)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0.0)
    if not np.all([np.any(valid & (object_index == value)) for value in objects]):
        raise ValueError("every refinement object must have a valid trajectory")
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_objects = rng.choice(objects, size=len(objects), replace=True)
        numerator_means: list[float] = []
        denominator_means: list[float] = []
        for object_id in sampled_objects:
            candidates = np.flatnonzero(valid & (object_index == object_id))
            selected = rng.choice(candidates, size=len(candidates), replace=True)
            numerator_means.append(float(numerator[selected].mean()))
            denominator_means.append(float(denominator[selected].mean()))
        samples[replicate] = float(
            np.mean(numerator_means) / np.mean(denominator_means)
        )
    point = float(
        np.mean(
            [numerator[valid & (object_index == value)].mean() for value in objects]
        )
        / np.mean(
            [denominator[valid & (object_index == value)].mean() for value in objects]
        )
    )
    return {
        "ratio": point,
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
    }


def _trajectory_mean(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[:-1], np.nan, dtype=np.float64)
    for index in np.ndindex(values.shape[:-1]):
        mask = valid[index]
        if np.all(mask):
            result[index] = float(np.mean(values[index][mask]))
    return result


def _refinement_metrics(
    arrays: dict[str, np.ndarray],
    *,
    length_scale: float,
    joint_scale: np.ndarray,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}

    def comparison(
        name: str,
        left: str,
        right: str,
        left_indices: np.ndarray,
        right_indices: np.ndarray,
    ) -> None:
        components = _state_components(
            arrays[f"{left}_position"][:, :, :, left_indices],
            arrays[f"{left}_quaternion_xyzw"][:, :, :, left_indices],
            arrays[f"{left}_joint"][:, :, :, left_indices],
            arrays[f"{right}_position"][:, :, :, right_indices],
            arrays[f"{right}_quaternion_xyzw"][:, :, :, right_indices],
            arrays[f"{right}_joint"][:, :, :, right_indices],
            length_scale=length_scale,
            joint_scale=joint_scale,
        )
        valid = (
            arrays[f"{left}_settled"][:, :, :, left_indices]
            & arrays[f"{right}_settled"][:, :, :, right_indices]
        )
        for component, values in components.items():
            result[f"{name}_{component}_curve"] = values
            result[f"{name}_{component}"] = _trajectory_mean(values, valid)
            result[f"{name}_{component}_terminal"] = np.where(
                valid[..., -1], values[..., -1], np.nan
            )
        result[f"{name}_valid_curve"] = valid

    comparison(
        "e32_64",
        "n32",
        "n64",
        np.arange(1, 33),
        _common_command_indices(32, 64)[1:],
    )
    comparison(
        "e64_128",
        "n64",
        "n128",
        np.arange(1, 65),
        _common_command_indices(64, 128)[1:],
    )
    for level in LEVELS:
        prefix = f"n{level}"
        components = _state_components(
            arrays[f"{prefix}_position"][:, 0, :, 1:],
            arrays[f"{prefix}_quaternion_xyzw"][:, 0, :, 1:],
            arrays[f"{prefix}_joint"][:, 0, :, 1:],
            arrays[f"{prefix}_position"][:, 1, :, 1:],
            arrays[f"{prefix}_quaternion_xyzw"][:, 1, :, 1:],
            arrays[f"{prefix}_joint"][:, 1, :, 1:],
            length_scale=length_scale,
            joint_scale=joint_scale,
        )
        valid = arrays[f"{prefix}_settled"][:, 0, :, 1:] & arrays[
            f"{prefix}_settled"
        ][:, 1, :, 1:]
        for component, values in components.items():
            result[f"repeat{level}_{component}_curve"] = values
            result[f"repeat{level}_{component}"] = _trajectory_mean(values, valid)
            result[f"repeat{level}_{component}_terminal"] = np.where(
                valid[..., -1], values[..., -1], np.nan
            )
        result[f"repeat{level}_valid_curve"] = valid

    components = _state_components(
        arrays["n32_position"][:, 0, :, 1:],
        arrays["n32_quaternion_xyzw"][:, 0, :, 1:],
        arrays["n32_joint"][:, 0, :, 1:],
        arrays["n32_strict_position"][:, 0, :, 1:],
        arrays["n32_strict_quaternion_xyzw"][:, 0, :, 1:],
        arrays["n32_strict_joint"][:, 0, :, 1:],
        length_scale=length_scale,
        joint_scale=joint_scale,
    )
    valid = arrays["n32_settled"][:, 0, :, 1:] & arrays["n32_strict_settled"][
        :, 0, :, 1:
    ]
    for component, values in components.items():
        result[f"settle_{component}_curve"] = values
        result[f"settle_{component}"] = _trajectory_mean(values, valid)
        result[f"settle_{component}_terminal"] = np.where(
            valid[..., -1], values[..., -1], np.nan
        )
    result["settle_valid_curve"] = valid
    return result


def _stats(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def _classify_refinement(
    rho: dict[str, float], *, e64_128_mean: float, repeatability_floor: float
) -> str:
    if e64_128_mean <= 1.5 * repeatability_floor:
        return "noise-limited"
    if rho["ratio"] <= 0.75 and rho["ci95_upper"] < 1.0:
        return "converged"
    if rho["ratio"] > 0.75 and rho["ci95_lower"] >= 0.75:
        return "non-converged"
    return "inconclusive"


def _settling_is_materially_sensitive(
    ratio: dict[str, float], *, settling_mean: float, repeatability_floor: float
) -> bool:
    return bool(
        settling_mean > 1.5 * repeatability_floor
        and ratio["ci95_lower"] > 0.25
    )


def _summarize_refinement(
    args: argparse.Namespace, manifest: Any, gripper: Any
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    combined: dict[str, list[np.ndarray]] = {}
    labels: list[str] = []
    splits: list[str] = []
    for object_id in OBJECTS:
        path = args.output / "refinement" / f"{object_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["manifest_sha256"].item()) != manifest.sha256():
                raise ValueError(f"{path}: manifest hash mismatch")
            for name in archive.files:
                if name in {
                    "object_id",
                    "split",
                    "manifest_sha256",
                    "gripper_sha256",
                    "sim_config_sha256",
                }:
                    continue
                combined.setdefault(name, []).append(np.asarray(archive[name]))
            labels.append(str(archive["object_id"].item()))
            splits.append(str(archive["split"].item()))
    arrays = {name: np.stack(values, axis=0) for name, values in combined.items()}
    arrays["refinement_object_labels"] = np.asarray(labels)
    arrays["refinement_object_splits"] = np.asarray(splits)
    metrics = _refinement_metrics(
        arrays,
        length_scale=float(gripper.length_scale),
        joint_scale=gripper.joint_travel_range.cpu().numpy(),
    )
    arrays.update({f"refinement_{name}": value for name, value in metrics.items()})
    object_index = np.repeat(np.arange(len(OBJECTS)), 4)
    e32 = np.nanmean(metrics["e32_64_dx"], axis=1).reshape(-1)
    e64 = np.nanmean(metrics["e64_128_dx"], axis=1).reshape(-1)
    repeat32 = metrics["repeat32_dx"].reshape(-1)
    repeat64 = metrics["repeat64_dx"].reshape(-1)
    repeat128 = metrics["repeat128_dx"].reshape(-1)
    settle = metrics["settle_dx"].reshape(-1)
    rho = _bootstrap_refinement_ratio(
        e64,
        e32,
        object_index,
        replicates=args.bootstrap_replicates,
        seed=BOOTSTRAP_SEED,
    )
    settle_ratio = _bootstrap_refinement_ratio(
        settle,
        e32,
        object_index,
        replicates=args.bootstrap_replicates,
        seed=BOOTSTRAP_SEED + 1,
    )
    equal_object = lambda value: float(
        np.mean(
            [
                np.nanmean(value[object_index == index])
                for index in np.unique(object_index)
            ]
        )
    )
    repeat_floor = max(equal_object(repeat64), equal_object(repeat128))
    e64_mean = equal_object(e64)
    classification = _classify_refinement(
        rho, e64_128_mean=e64_mean, repeatability_floor=repeat_floor
    )
    settle_floor = equal_object(repeat32)
    materially_sensitive = _settling_is_materially_sensitive(
        settle_ratio,
        settling_mean=equal_object(settle),
        repeatability_floor=settle_floor,
    )
    components = (
        "dx",
        "translation_m",
        "rotation_rad",
        "joint_rmse_over_travel",
    )
    comparisons = {}
    for name in ("e32_64", "e64_128", "repeat32", "repeat64", "repeat128", "settle"):
        def primary(component: str) -> np.ndarray:
            value = metrics[f"{name}_{component}"]
            return (
                np.nanmean(value, axis=1)
                if name in {"e32_64", "e64_128"}
                else value
            )

        comparisons[name] = {
            component: {
                "all_trajectories": _stats(primary(component)),
                "equal_object_mean": equal_object(primary(component).reshape(-1)),
                "terminal": _stats(
                    np.nanmean(
                        metrics[f"{name}_{component}_terminal"], axis=1
                    )
                    if name in {"e32_64", "e64_128"}
                    else metrics[f"{name}_{component}_terminal"]
                ),
            }
            for component in components
        }
    per_object = {}
    for index, object_id in enumerate(OBJECTS):
        per_object[object_id] = {
            "split": splits[index],
            "source_pose_index": arrays["source_pose_index"][index].tolist(),
            "contact_onset": arrays["contact_onset"][index].tolist(),
            "e32_64_dx": _stats(
                np.nanmean(metrics["e32_64_dx"][index], axis=0)
            ),
            "e64_128_dx": _stats(
                np.nanmean(metrics["e64_128_dx"][index], axis=0)
            ),
            "rho": float(
                np.nanmean(metrics["e64_128_dx"][index])
                / np.nanmean(metrics["e32_64_dx"][index])
            ),
            "settle_dx": _stats(metrics["settle_dx"][index]),
        }
    return (
        {
            "classification": classification,
            "rho": rho,
            "repeatability_floor_dx": repeat_floor,
            "e64_128_equal_object_mean_dx": e64_mean,
            "settling": {
                "ratio_to_e32_64": settle_ratio,
                "repeatability_floor_dx": settle_floor,
                "materially_sensitive": materially_sensitive,
            },
            "comparisons": comparisons,
            "per_object": per_object,
            "validity": {
                name: {
                    "valid_states": int(np.count_nonzero(value)),
                    "total_states": int(value.size),
                }
                for name, value in metrics.items()
                if name.endswith("valid_curve")
            },
        },
        arrays,
    )


def _summarize_bias(
    args: argparse.Namespace, manifest: Any
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    persistent_by_split: dict[str, list[bool]] = {}
    for split in EXPECTED_SPLITS:
        seed_values: list[dict[str, np.ndarray]] = []
        for seed in MODEL_SEEDS:
            path = args.output / "bias" / f"seed-{seed}-{split}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as archive:
                if str(archive["manifest_sha256"].item()) != manifest.sha256():
                    raise ValueError(f"{path}: manifest hash mismatch")
                seed_values.append(
                    {
                        name: np.asarray(archive[name])
                        for name in (
                            "tf_log",
                            "ar_log",
                            "tf_spatial",
                            "ar_spatial",
                            "object_index",
                            "trajectory",
                        )
                    }
                )
        reference_object = seed_values[0]["object_index"]
        reference_trajectory = seed_values[0]["trajectory"]
        for values in seed_values[1:]:
            if not np.array_equal(values["object_index"], reference_object) or not np.array_equal(
                values["trajectory"], reference_trajectory
            ):
                raise ValueError(f"{split}: seed bias sample ordering differs")
        tf = np.stack([value["tf_log"] for value in seed_values])
        ar = np.stack([value["ar_log"] for value in seed_values])
        tf_spatial = np.stack([value["tf_spatial"] for value in seed_values])
        ar_spatial = np.stack([value["ar_spatial"] for value in seed_values])
        features = np.stack(
            [
                _bias_feature_vector(tf[index], ar[index], tf_spatial[index])
                for index in range(len(MODEL_SEEDS))
            ]
        )
        bootstrap = _hierarchical_bias_bootstrap(
            features,
            reference_object,
            replicates=args.bootstrap_replicates,
            seed=BOOTSTRAP_SEED + EXPECTED_SPLITS.index(split),
        )
        mean = bootstrap["mean"]
        lower = bootstrap["ci95_lower"]
        upper = bootstrap["ci95_upper"]
        persistent = _persistent_bias(mean, lower, upper)
        persistent_by_split[split] = persistent
        tf_curve = np.mean(
            [_mean_by_object(tf[index], reference_object) for index in range(3)],
            axis=0,
        )
        ar_curve = np.mean(
            [_mean_by_object(ar[index], reference_object) for index in range(3)],
            axis=0,
        )
        tf_spatial_curve = np.mean(
            [
                _mean_by_object(tf_spatial[index], reference_object)
                for index in range(3)
            ],
            axis=0,
        )
        ar_spatial_curve = np.mean(
            [
                _mean_by_object(ar_spatial[index], reference_object)
                for index in range(3)
            ],
            axis=0,
        )
        result[split] = {
            "trajectories": int(tf.shape[1]),
            "objects": int(len(np.unique(reference_object))),
            "teacher_forced_mean_curve_v_m_w_rad": tf_curve.tolist(),
            "teacher_forced_cumulative_curve_v_m_w_rad": np.cumsum(
                tf_curve, axis=0
            ).tolist(),
            "autoregressive_mean_curve_v_m_w_rad": ar_curve.tolist(),
            "teacher_forced_spatial_translation_mean_curve_m": tf_spatial_curve.tolist(),
            "autoregressive_spatial_translation_mean_curve_m": ar_spatial_curve.tolist(),
            "band_mean_v_m_w_rad": mean[:24].reshape(4, 6).tolist(),
            "band_ci95_lower": lower[:24].reshape(4, 6).tolist(),
            "band_ci95_upper": upper[:24].reshape(4, 6).tolist(),
            "cumulative_tf_b32_mean_v_m_w_rad": mean[24:30].tolist(),
            "cumulative_tf_b32_ci95_lower": lower[24:30].tolist(),
            "cumulative_tf_b32_ci95_upper": upper[24:30].tolist(),
            "terminal_ar_b32_mean_v_m_w_rad": mean[30:36].tolist(),
            "terminal_ar_b32_ci95_lower": lower[30:36].tolist(),
            "terminal_ar_b32_ci95_upper": upper[30:36].tolist(),
            "cumulative_tf_spatial_translation_m": mean[36:39].tolist(),
            "cumulative_tf_spatial_translation_ci95_lower": lower[36:39].tolist(),
            "cumulative_tf_spatial_translation_ci95_upper": upper[36:39].tolist(),
            "persistent_components_vx_vy_vz_wx_wy_wz": persistent,
        }
        prefix = f"bias_{split}"
        arrays.update(
            {
                f"{prefix}_tf_log": tf,
                f"{prefix}_ar_log": ar,
                f"{prefix}_tf_spatial": tf_spatial,
                f"{prefix}_ar_spatial": ar_spatial,
                f"{prefix}_object_index": reference_object,
                f"{prefix}_trajectory": reference_trajectory,
                f"{prefix}_tf_mean_curve": tf_curve,
                f"{prefix}_ar_mean_curve": ar_curve,
            }
        )
    result["persistent_by_split"] = persistent_by_split
    return result, arrays


def _classify_diagnosis(
    refinement: dict[str, Any], bias: dict[str, Any]
) -> dict[str, Any]:
    refinement_class = refinement["classification"]
    settling_sensitive = refinement["settling"]["materially_sensitive"]
    persistent = bias["persistent_by_split"]
    train = np.asarray(persistent["train"], dtype=bool)
    val = np.asarray(persistent["val"], dtype=bool)
    test = np.asarray(persistent["test"], dtype=bool)
    unseen = val | test
    shared = train & unseen
    if refinement_class == "non-converged" or settling_sensitive:
        label = "simulator_or_data_discretization"
    elif refinement_class != "converged":
        label = "inconclusive"
    elif np.any(shared):
        label = "explicit_update_law_candidate"
    elif np.any(unseen) and not np.any(train):
        label = "geometry_generalization_or_data"
    else:
        label = "inconclusive"
    return {
        "classification": label,
        "refinement_classification": refinement_class,
        "settling_materially_sensitive": settling_sensitive,
        "persistent_train": train.tolist(),
        "persistent_val": val.tolist(),
        "persistent_test": test.tolist(),
        "shared_train_and_unseen": shared.tolist(),
        "architecture_changed": False,
        "retrained": False,
        "implicit_resolvent_implemented": False,
        "production_remains": "gap+aperture",
    }


def _plot_results(
    output: Path,
    refinement_arrays: dict[str, np.ndarray],
    bias_arrays: dict[str, np.ndarray],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for name, label, color in (
        ("e32_64", r"$E_{32,64}$", "#4472C4"),
        ("e64_128", r"$E_{64,128}$", "#E67E22"),
        ("settle", "prod vs strict", "#70AD47"),
    ):
        raw_curve = refinement_arrays[f"refinement_{name}_dx_curve"]
        curve = (
            np.nanmean(raw_curve, axis=1)
            if name in {"e32_64", "e64_128"}
            else raw_curve
        )
        x = np.linspace(1.0 / curve.shape[-1], 1.0, curve.shape[-1])
        axes[0].plot(x, np.nanmean(curve, axis=(0, 1)), label=label, color=color)
    axes[0].set(xlabel=r"closure fraction $\alpha$", ylabel=r"$d_X$", title="Refinement curves")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    labels = refinement_arrays["refinement_object_labels"]
    x = np.arange(len(labels))
    e32 = np.nanmean(refinement_arrays["refinement_e32_64_dx"], axis=(1, 2))
    e64 = np.nanmean(refinement_arrays["refinement_e64_128_dx"], axis=(1, 2))
    repeat = np.maximum(
        np.nanmean(refinement_arrays["refinement_repeat64_dx"], axis=1),
        np.nanmean(refinement_arrays["refinement_repeat128_dx"], axis=1),
    )
    width = 0.25
    axes[1].bar(x - width, e32, width, label=r"$E_{32,64}$")
    axes[1].bar(x, e64, width, label=r"$E_{64,128}$")
    axes[1].bar(x + width, repeat, width, label="repeat floor")
    axes[1].set_xticks(x, ["train", "val", "test"])
    axes[1].set(ylabel=r"mean $d_X$", title="Per-object refinement")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(fontsize=8)

    for split, color in zip(EXPECTED_SPLITS, ("#4472C4", "#70AD47", "#C00000"), strict=True):
        cumulative = np.cumsum(bias_arrays[f"bias_{split}_tf_mean_curve"], axis=0)
        axes[2].plot(np.arange(1, 33), cumulative[:, 2] * 1000.0, label=split, color=color)
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set(xlabel="step", ylabel=r"cumulative $v_z$, mm", title="Split signed bias")
    axes[2].grid(alpha=0.2)
    axes[2].legend()
    fig.savefig(output / "quasistatic_refinement_bias.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    labels = ("vx", "vy", "vz", "wx", "wy", "wz")
    for split, color in zip(EXPECTED_SPLITS, ("#4472C4", "#70AD47", "#C00000"), strict=True):
        cumulative = np.cumsum(bias_arrays[f"bias_{split}_tf_mean_curve"], axis=0)
        autoregressive = bias_arrays[f"bias_{split}_ar_mean_curve"]
        for component, ax in enumerate(axes.flat):
            scale = 1000.0
            ax.plot(
                np.arange(1, 33),
                cumulative[:, component] * scale,
                color=color,
                label=f"{split} sum TF",
            )
            ax.plot(
                np.arange(1, 33),
                autoregressive[:, component] * scale,
                color=color,
                linestyle="--",
                label=f"{split} AR",
            )
            ax.axhline(0.0, color="black", linewidth=0.7)
            ax.set_title(labels[component])
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(output / "split_signed_bias.png", dpi=180)
    plt.close(fig)


def _run_summary_phase(args: argparse.Namespace) -> None:
    from srno.data.schema import DatasetManifest
    from srno.geometry.gripper import GripperAsset

    manifest = DatasetManifest.load(args.manifest)
    gripper = GripperAsset.load(manifest.gripper_path)
    refinement, refinement_arrays = _summarize_refinement(args, manifest, gripper)
    bias, bias_arrays = _summarize_bias(args, manifest)
    audit = json.loads(
        (args.output / "refinement" / "runtime-audit.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoint_contract = json.loads(
        (args.output / "bias" / "checkpoint-contract.json").read_text(
            encoding="utf-8"
        )
    )
    result = {
        "definition": {
            "dx": "sqrt((||dp||/L)^2 + theta(R1,R2)^2 + mean(((r1-r2)/travel)^2))",
            "E_32_64": "mean_{k=1..32} dX(x32_k,x64_2k)",
            "E_64_128": "mean_{k=1..64} dX(x64_k,x128_2k)",
            "rho": "E_64_128/E_32_64",
            "signed_tf_bias": "E[Log(inv(q*_{k+1}) qhat^{TF}_{k+1})]",
            "cumulative_tf_bias": "sum_{k=0..31} b_k^{TF}",
        },
        "configuration": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest.sha256(),
            "gripper_sha256": manifest.gripper_sha256,
            "sim_config": str(args.sim_config.resolve()),
            "sim_config_sha256": _sha256(args.sim_config.resolve()),
            "objects": list(OBJECTS),
            "source_pose_indices": {
                key: list(value) for key, value in EXPECTED_SOURCE_POSES.items()
            },
            "levels": list(LEVELS),
            "production_repeats": PRODUCTION_REPEATS,
            "strict_level": STRICT_LEVEL,
            "strict_settling": {
                "min_steps": 40,
                "max_steps": 2400,
                "consecutive_steps": 20,
                "position_delta_m": 0.00025,
                "linear_velocity_m_s": 0.005,
                "angular_velocity_rad_s": 0.05,
                "joint_velocity_rad_s": 0.005,
            },
            "bias_step_bands": [list(value) for value in BIAS_STEP_BANDS],
            "model_seeds": list(MODEL_SEEDS),
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "runtime_audit": audit,
            "checkpoint_contract": checkpoint_contract,
        },
        "quasistatic_refinement": refinement,
        "split_signed_bias": bias,
        "decision": _classify_diagnosis(refinement, bias),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    _write_json(args.output / "results.json", result)
    _atomic_savez(
        args.output / "samples.npz", **refinement_arrays, **bias_arrays
    )
    _plot_results(args.output, refinement_arrays, bias_arrays)
    print(
        f"[DIAGNOSTIC] completed: {args.output / 'results.json'}; "
        f"decision={result['decision']['classification']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("all", "bias", "refinement", "summarize"),
        default="all",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/simulator-r-v1/manifest.json"),
    )
    parser.add_argument(
        "--sim-config", type=Path, default=Path("configs/simulator-r.toml")
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/srno-r-material-v2.toml"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("runs/ablation-actuator-rollout/aperture"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/quasistatic-refinement-bias-v1"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--object",
        action="append",
        dest="objects",
        help="diagnostic smoke override; may be repeated",
    )
    parser.add_argument("--pose-limit", type=int, default=4)
    parser.add_argument("--strict-level", type=int, default=STRICT_LEVEL)
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=list(LEVELS),
        help="diagnostic override for smoke tests; full summary requires 32 64 128",
    )
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.sim_config = args.sim_config.resolve()
    args.train_config = args.train_config.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.output = args.output.resolve()
    args.levels = tuple(args.levels)
    if args.bootstrap_replicates <= 0:
        parser.error("--bootstrap-replicates must be positive")
    if args.pose_limit <= 0 or args.pose_limit > 4 or args.strict_level <= 0:
        parser.error("pose limit must be in 1..4 and strict level must be positive")
    if args.phase in {"all", "summarize"} and (
        args.levels != LEVELS
        or args.strict_level != STRICT_LEVEL
        or args.pose_limit != 4
        or args.objects is not None
    ):
        parser.error("full summary requires frozen objects/poses and N=32,64,128")

    if args.phase in {"all", "bias"}:
        _run_bias_phase(args)
    if args.phase in {"all", "refinement"}:
        _run_refinement_phase(args)
    if args.phase in {"all", "summarize"}:
        _run_summary_phase(args)


if __name__ == "__main__":
    main()
