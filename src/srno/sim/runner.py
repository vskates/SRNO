"""CLI-facing launcher and object-wise collection orchestration."""

from __future__ import annotations

import gc
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import signal
import traceback
from typing import Sequence

import h5py
import numpy as np

from srno.data.schema import DatasetManifest, ShardSpec, objectwise_split
from srno.geometry.gripper import GripperAsset, preprocess_scheduled_urdf
from srno.sim.assets import SimulatorAssetCatalog
from srno.sim.config import SimulatorConfig
from srno.sim.pose_seeds import PoseSeeds
from srno.sim.memory_guard import MemoryWatchdog


def _candidate_pose_order(
    seeds: PoseSeeds,
    desired_count: int,
    *,
    successful_only: bool,
    seed: int,
) -> PoseSeeds:
    """Return the original random sample followed by unique reserve poses."""

    eligible = (
        np.flatnonzero(seeds.validation_success)
        if successful_only
        else np.arange(len(seeds.position_m))
    )
    if desired_count > len(eligible):
        raise ValueError(
            f"only {len(eligible)} unique eligible poses are available, "
            f"cannot collect {desired_count} unique trajectories"
        )
    generator = np.random.default_rng(seed)
    primary = generator.choice(eligible, size=desired_count, replace=False)
    remaining = eligible[~np.isin(eligible, primary, assume_unique=True)]
    generator.shuffle(remaining)
    order = np.concatenate((primary, remaining))
    return seeds.take(order.tolist(), require_successful=successful_only)


def run_simulator_collection(
    config: SimulatorConfig,
    *,
    object_ids: Sequence[str] | None = None,
    max_objects: int | None = None,
    trajectories_per_object: int | None = None,
    pose_indices: Sequence[int] | None = None,
    overwrite: bool | None = None,
    resume: bool = False,
) -> Path | None:
    """Launch Isaac Sim and collect the selected catalog objects."""

    # SIGUSR1 provides an on-demand Python stack dump for long native simulator
    # calls without stopping collection.  This is especially useful on headless
    # multi-environment runs.
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)

    catalog = SimulatorAssetCatalog.load(config.catalog)
    catalog.validate_files(verify_hashes=False)
    selected = list(catalog.object_ids if object_ids is None else object_ids)
    unknown = set(selected) - set(catalog.object_ids)
    if unknown:
        raise ValueError(f"unknown catalog objects: {sorted(unknown)}")
    if len(selected) != len(set(selected)):
        raise ValueError("object selection contains duplicates")
    if max_objects is not None:
        if max_objects <= 0:
            raise ValueError("max_objects must be positive")
        selected = selected[:max_objects]
    explicit_pose_indices = None if pose_indices is None else list(pose_indices)
    if explicit_pose_indices is not None:
        if len(selected) != 1:
            raise ValueError("explicit pose indices require exactly one selected object")
        if trajectories_per_object is not None and trajectories_per_object != len(
            explicit_pose_indices
        ):
            raise ValueError(
                "--trajectories-per-object must equal the number of --pose-index values"
            )
        trajectory_count = len(explicit_pose_indices)
    else:
        trajectory_count = (
            config.trajectories_per_object
            if trajectories_per_object is None
            else trajectories_per_object
        )
    if trajectory_count <= 0:
        raise ValueError("trajectories_per_object must be positive")
    replace_existing = config.overwrite if overwrite is None else overwrite
    if replace_existing and resume:
        raise ValueError("overwrite and resume are mutually exclusive")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "collection-failure.txt").unlink(missing_ok=True)

    with MemoryWatchdog(config.memory_limit_gib, config.memory_check_interval_s):
        _precompute_sdfs(config, catalog, selected)
        from isaaclab.app import AppLauncher

        launcher = AppLauncher({"headless": config.headless, "device": config.device})
        app = launcher.app
        try:
            result = _collect_inside_app(
                app,
                config,
                catalog,
                selected,
                trajectory_count=trajectory_count,
                pose_indices=explicit_pose_indices,
                overwrite=replace_existing,
                resume=resume,
            )
        except BaseException:
            failure = traceback.format_exc()
            config.output_dir.mkdir(parents=True, exist_ok=True)
            (config.output_dir / "collection-failure.txt").write_text(
                failure, encoding="utf-8"
            )
            print(failure, file=sys.stderr, flush=True)
            # Kit's close callback can terminate the interpreter successfully
            # while an exception is unwinding, masking collector failures as exit
            # code 0.  On failure the OS is the reliable resource owner; exit
            # immediately after persisting the traceback.
            os._exit(1)
        else:
            app.close()
            return result


def _collect_inside_app(
    app: object,
    config: SimulatorConfig,
    catalog: SimulatorAssetCatalog,
    object_ids: list[str],
    *,
    trajectory_count: int,
    pose_indices: list[int] | None,
    overwrite: bool,
    resume: bool,
) -> Path | None:
    import carb
    import omni.usd
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

    from srno.data.writer import H5DatasetWriter
    from srno.sim.collector import QuasistaticCollector
    from srno.sim.isaac_scene import (
        apply_contact_materials,
        make_scene_cfg,
        make_simulation_cfg,
    )
    from srno.sim.usd_geometry import load_or_generate_sdf

    settings = carb.settings.get_settings()
    # This workstation has one supported NVIDIA GPU.  Explicitly disable the
    # multi-GPU path so GUI collection does not reserve extra host/rendering
    # buffers while probing the unsupported integrated Intel adapter.
    settings.set("/renderer/multiGPU/enabled", False)
    settings.set("/renderer/multiGPU/autoEnable", False)
    settings.set("/rtx/realtime/mgpu/autoTiling/enabled", False)
    if not config.headless:
        settings.set("/app/window/hideUi", False)

    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    gripper = preprocess_scheduled_urdf(
        catalog.gripper.source_urdf,
        finger_links=tuple(catalog.gripper.contact_links),
        close_joint_positions=catalog.gripper.close_joint_position_rad,
        samples_per_link=128,
        seed=config.seed,
    )
    gripper_path = output / "gripper.npz"
    gripper.save(gripper_path)

    completed: list[str] = []
    for object_id in object_ids:
        record = catalog.object(object_id)
        shard_path = output / "shards" / f"{object_id}.h5"
        if shard_path.exists() and not overwrite:
            try:
                _check_existing_shard(shard_path, object_id, trajectory_count)
            except ValueError:
                if not resume:
                    raise
                print(
                    f"[SRNO] {object_id}: existing shard does not match the requested "
                    "trajectory count; replacing it in resume mode",
                    flush=True,
                )
            else:
                completed.append(object_id)
                print(f"[SRNO] {object_id}: existing shard retained")
                continue

        object_seed = config.seed + int.from_bytes(
            hashlib.sha256(object_id.encode()).digest()[:4], "little"
        )
        available_seeds = PoseSeeds.load(record.pose_seed_path)
        if pose_indices is None:
            # Preserve the original deterministic sample as the first choices,
            # then append a deterministic permutation of all unused poses.
            # Non-settling zero-gravity candidates can consequently be replaced
            # without duplicates or changing already collected valid shards.
            try:
                seeds = _candidate_pose_order(
                    available_seeds,
                    trajectory_count,
                    successful_only=config.successful_seed_poses_only,
                    seed=object_seed,
                )
            except ValueError as error:
                raise ValueError(f"{object_id}: {error}") from error
        else:
            seeds = available_seeds.take(
                pose_indices,
                require_successful=config.successful_seed_poses_only,
            )
        print(
            f"[SRNO] {object_id}: collecting {trajectory_count} trajectories from "
            f"{len(seeds.position_m)} unique candidates in "
            f"{min(config.num_envs, len(seeds.position_m))} environments"
        )
        sdf = load_or_generate_sdf(
            record.usd_path,
            output / ".cache" / "sdf" / f"{object_id}.npz",
            config.sdf,
        )

        omni.usd.get_context().new_stage()
        for _ in range(10):
            app.update()
        print(f"[SRNO] {object_id}: creating SimulationContext", flush=True)
        simulation = SimulationContext(make_simulation_cfg(config.device))
        scene = None
        collector = None
        try:
            print(f"[SRNO] {object_id}: spawning scene", flush=True)
            scene = InteractiveScene(
                make_scene_cfg(
                    catalog,
                    record,
                    num_envs=min(config.num_envs, len(seeds.position_m)),
                    relaxation=config.relaxation,
                )
            )
            print(f"[SRNO] {object_id}: binding contact material", flush=True)
            apply_contact_materials(scene)
            print(f"[SRNO] {object_id}: initializing physics", flush=True)
            simulation.reset()
            if not config.headless:
                # The first environment is centered at the world origin.  Keep
                # both a grocery-sized object and the nearby gripper in frame.
                simulation.set_camera_view(
                    eye=(0.45, -0.45, 0.32),
                    target=(0.0, 0.0, 0.02),
                )
            print(f"[SRNO] {object_id}: initializing collector", flush=True)
            collector = QuasistaticCollector(
                simulation,
                scene,
                catalog,
                record,
                config,
                gripper,
            )
            print(f"[SRNO] {object_id}: stepping trajectories", flush=True)
            trajectories = collector.collect(seeds, desired_count=trajectory_count)
            with H5DatasetWriter(shard_path) as writer:
                writer.add_object(
                    object_id,
                    sdf=sdf.values,
                    grid_origin=sdf.origin_xyz,
                    voxel_size=sdf.voxel_size_xyz,
                    position=trajectories.position,
                    quaternion_xyzw=trajectories.quaternion_xyzw,
                    actual_aperture=trajectories.actual_aperture,
                    source_pose_index=trajectories.source_pose_index,
                    diagnostics=trajectories.diagnostics,
                )
        finally:
            del collector
            del scene
            # ``SimulationContext.stop()`` enters a Kit render callback and can
            # block indefinitely while unwinding a physics exception.  Clearing
            # the singleton and closing the stage is the standalone Isaac Lab
            # teardown path we need here; the app itself is closed by the caller.
            simulation.clear_all_callbacks()
            simulation.clear_instance()
            del simulation
            gc.collect()
            omni.usd.get_context().close_stage()
            for _ in range(5):
                app.update()
        completed.append(object_id)
        print(f"[SRNO] {object_id}: wrote {shard_path}")
        _write_collection_metadata(output, config, catalog, completed)

    if len(completed) < 3:
        _write_collection_metadata(output, config, catalog, completed)
        print("[SRNO] fewer than three object shards: manifest deferred until train/val/test split is possible")
        return None
    manifest = DatasetManifest.create(
        root=output,
        length_scale_m=gripper.length_scale,
        sdf_scale_m=config.dataset.sdf_scale_m,
        delta_gate_m=config.dataset.delta_gate_m,
        commanded_aperture_m=[float(value) for value in reversed(gripper.aperture_knots.tolist())],
        gripper_asset="gripper.npz",
        gripper_sha256=gripper.sha256(),
        shards=[
            ShardSpec(path=f"shards/{object_id}.h5", object_ids=(object_id,))
            for object_id in completed
        ],
        splits=objectwise_split(completed, seed=config.dataset.split_seed),
    )
    manifest_path = output / "manifest.json"
    _atomic_json(manifest_path, manifest.to_dict())
    _write_collection_metadata(output, config, catalog, completed)
    return manifest_path


def _precompute_sdfs(
    config: SimulatorConfig,
    catalog: SimulatorAssetCatalog,
    object_ids: list[str],
) -> None:
    from srno.sim.usd_geometry import sdf_cache_is_current

    for object_id in object_ids:
        record = catalog.object(object_id)
        cache = config.output_dir / ".cache" / "sdf" / f"{object_id}.npz"
        if sdf_cache_is_current(record.usd_path, cache, config.sdf):
            continue
        print(f"[SRNO] {object_id}: precomputing SDF in isolated process")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "srno.sim.sdf_worker",
                "--config",
                str(config.source_path),
                "--object",
                object_id,
            ],
            check=True,
            env=dict(os.environ),
        )


def _check_existing_shard(
    path: Path, expected_object_id: str, expected_trajectory_count: int
) -> None:
    with h5py.File(path, "r") as handle:
        groups = handle.get("objects")
        if groups is None or len(groups) != 1:
            raise ValueError(f"existing shard is not a one-object SRNO shard: {path}")
        actual = str(groups["000000"].attrs.get("object_id", ""))
        if actual != expected_object_id:
            raise ValueError(
                f"existing shard {path} contains {actual!r}, expected {expected_object_id!r}"
            )
        actual_trajectory_count = int(groups["000000/position"].shape[0])
        if actual_trajectory_count != expected_trajectory_count:
            raise ValueError(
                f"existing shard {path} contains {actual_trajectory_count} trajectories, "
                f"expected {expected_trajectory_count}; pass --overwrite to replace it"
            )


def _write_collection_metadata(
    output: Path,
    config: SimulatorConfig,
    catalog: SimulatorAssetCatalog,
    completed: list[str],
) -> None:
    payload = {
        "format_version": 1,
        "config_sha256": config.sha256(),
        "catalog_sha256": hashlib.sha256(catalog.catalog_path.read_bytes()).hexdigest(),
        "validation_commit": catalog.payload["source"]["validation_commit"],
        "completed_object_ids": completed,
    }
    _atomic_json(output / "collection.json", payload)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
