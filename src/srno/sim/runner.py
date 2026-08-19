"""CLI-facing launcher and object-wise collection orchestration."""

from __future__ import annotations

import gc
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import signal
import traceback
from typing import TYPE_CHECKING, Sequence

import h5py
import numpy as np
import torch

from srno.data.schema import DatasetManifest, PhysicsMetadata, ShardSpec, objectwise_split
from srno.geometry.gripper import GripperAsset
from srno.sim.assets import SimulatorAssetCatalog
from srno.sim.config import SimulatorConfig
from srno.sim.pose_seeds import PoseSeeds
from srno.sim.memory_guard import MemoryWatchdog
from srno.sim.physx_material import expected_physics_metadata

if TYPE_CHECKING:
    from srno.sim.usd_geometry import DenseSDF


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
    from srno.sim.actuator_audit import verify_runtime_actuator
    from srno.sim.collector import QuasistaticCollector
    from srno.sim.gripper_geometry import preprocess_runtime_gripper
    from srno.sim.isaac_scene import (
        apply_contact_materials,
        make_scene_cfg,
        make_simulation_cfg,
    )
    from srno.sim.physx_material import PhysxMaterialAudit
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
    gripper_path = output / "gripper.npz"
    previous_gripper = (
        GripperAsset.load(gripper_path) if gripper_path.is_file() else None
    )
    print("[SRNO] extracting runtime USD gripper geometry and kinematics", flush=True)
    gripper = preprocess_runtime_gripper(
        app,
        config,
        catalog,
        samples_per_link=128,
    )
    gripper_hash = gripper.sha256()
    physics = expected_physics_metadata(config)

    completed: list[str] = []
    for object_id in object_ids:
        record = catalog.object(object_id)
        shard_path = output / "shards" / f"{object_id}.h5"
        if shard_path.exists() and not overwrite:
            try:
                _check_existing_shard(
                    shard_path, object_id, trajectory_count, physics=physics
                )
            except ValueError:
                if not resume:
                    raise
                print(
                    f"[SRNO] {object_id}: existing shard does not match the requested "
                    "trajectory count; replacing it in resume mode",
                    flush=True,
                )
            else:
                sdf = load_or_generate_sdf(
                    record.usd_path,
                    output / ".cache" / "sdf" / f"{object_id}.npz",
                    config.sdf,
                )
                changed = _replace_shard_sdf(shard_path, object_id, sdf)
                gripper_changed = _replace_shard_gripper_geometry(
                    shard_path,
                    object_id,
                    previous_gripper,
                    gripper,
                )
                completed.append(object_id)
                sdf_status = "SDF replaced" if changed else "SDF already current"
                gripper_status = (
                    "aperture remapped" if gripper_changed else "gripper already current"
                )
                print(
                    f"[SRNO] {object_id}: existing trajectories retained; "
                    f"{sdf_status}; {gripper_status}"
                )
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
        simulation = SimulationContext(
            make_simulation_cfg(config.device, config.material)
        )
        scene = None
        collector = None
        material_audit = PhysxMaterialAudit(physics)
        try:
            material_audit.start()
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
            apply_contact_materials(scene, config.material)
            material_audit.force_load()
            print(f"[SRNO] {object_id}: initializing physics", flush=True)
            simulation.reset()
            verified_physics = material_audit.verify(app)
            actuator_contract = verify_runtime_actuator(
                scene["robot"],
                expected_joint_names=tuple(gripper.joint_names),
                relaxation=config.relaxation,
                position_target=gripper.free_joint_knots[-1],
            )
            _atomic_json(
                output / "actuator-runtime.json",
                {
                    "format_version": 1,
                    "config_sha256": config.sha256(),
                    "gripper_sha256": gripper_hash,
                    "contract": actuator_contract,
                },
            )
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
            with H5DatasetWriter(shard_path, physics=verified_physics) as writer:
                writer.add_object(
                    object_id,
                    sdf=sdf.values,
                    grid_origin=sdf.origin_xyz,
                    voxel_size=sdf.voxel_size_xyz,
                    sdf_representation=sdf.representation,
                    sdf_geometry_sha256=sdf.geometry_sha256,
                    gripper_geometry_sha256=gripper_hash,
                    position=trajectories.position,
                    quaternion_xyzw=trajectories.quaternion_xyzw,
                    joint_position=trajectories.joint_position,
                    joint_names=trajectories.joint_names,
                    actual_aperture=trajectories.actual_aperture,
                    source_pose_index=trajectories.source_pose_index,
                    diagnostics=trajectories.diagnostics,
                )
        finally:
            material_audit.close()
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

    _atomic_gripper(gripper_path, gripper)
    if len(completed) < 3:
        _write_collection_metadata(output, config, catalog, completed)
        print("[SRNO] fewer than three object shards: manifest deferred until train/val/test split is possible")
        return None
    manifest = DatasetManifest.create(
        root=output,
        length_scale_m=gripper.length_scale,
        sdf_scale_m=config.dataset.sdf_scale_m,
        delta_gate_m=config.dataset.delta_gate_m,
        contact_offset_sum_m=config.dataset.contact_offset_sum_m,
        commanded_aperture_m=[float(value) for value in reversed(gripper.aperture_knots.tolist())],
        gripper_asset="gripper.npz",
        gripper_sha256=gripper_hash,
        physics=physics,
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

    stale: list[str] = []
    for object_id in object_ids:
        record = catalog.object(object_id)
        cache = config.output_dir / ".cache" / "sdf" / f"{object_id}.npz"
        if not sdf_cache_is_current(record.usd_path, cache, config.sdf):
            stale.append(object_id)
    if not stale:
        return
    print(f"[SRNO] precomputing {len(stale)} PhysX-cooked SDF(s) in an isolated process")
    command = [
        sys.executable,
        "-m",
        "srno.sim.sdf_worker",
        "--config",
        str(config.source_path),
    ]
    for object_id in stale:
        command.extend(("--object", object_id))
    subprocess.run(command, check=True, env=dict(os.environ))


def _replace_shard_sdf(path: Path, object_id: str, sdf: "DenseSDF") -> bool:
    """Atomically replace only SDF geometry while retaining all trajectories."""

    with h5py.File(path, "r") as source:
        group = source["objects/000000"]
        if str(group.attrs.get("object_id", "")) != object_id:
            raise ValueError(f"existing shard {path} does not contain {object_id!r}")
        if (
            str(group.attrs.get("sdf_representation", "")) == sdf.representation
            and str(group.attrs.get("sdf_geometry_sha256", "")) == sdf.geometry_sha256
        ):
            return False

    temporary = path.with_suffix(path.suffix + ".sdf.tmp")
    temporary.unlink(missing_ok=True)
    try:
        shutil.copy2(path, temporary)
        with h5py.File(temporary, "r+") as target:
            group = target["objects/000000"]
            del group["sdf"]
            values = np.asarray(sdf.values, dtype=np.float16)
            group.create_dataset(
                "sdf",
                data=values,
                chunks=values.shape,
                compression="lzf",
                shuffle=True,
            )
            group.attrs["grid_origin"] = np.asarray(sdf.origin_xyz, dtype=np.float32)
            group.attrs["voxel_size"] = np.asarray(sdf.voxel_size_xyz, dtype=np.float32)
            group.attrs["sdf_representation"] = sdf.representation
            group.attrs["sdf_geometry_sha256"] = sdf.geometry_sha256
            target.flush()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _replace_shard_gripper_geometry(
    path: Path,
    object_id: str,
    source: GripperAsset | None,
    target: GripperAsset,
) -> bool:
    """Atomically update derived aperture values for the runtime gripper.

    Object poses and diagnostics describe the same simulation and stay intact.
    A format-v3 target derives every value directly from the stored six-joint
    state. Legacy assets retain the monotone knot-to-knot migration path.
    """

    target_hash = target.sha256()
    with h5py.File(path, "r") as handle:
        group = handle["objects/000000"]
        if str(group.attrs.get("object_id", "")) != object_id:
            raise ValueError(f"existing shard {path} does not contain {object_id!r}")
        stored_hash = group.attrs.get("gripper_geometry_sha256")
        if stored_hash is not None and str(stored_hash) == target_hash:
            return False
        if source is None:
            raise ValueError(
                f"cannot migrate {path}: the source gripper asset is missing"
            )
        source_hash = source.sha256()
        if stored_hash is not None and str(stored_hash) != source_hash:
            raise ValueError(
                f"cannot migrate {path}: its gripper hash matches neither source nor target"
            )
        if stored_hash is None and source_hash == target_hash:
            raise ValueError(
                f"cannot migrate unversioned aperture values in {path}: "
                "the previous gripper scale is no longer available"
            )
        source_knots = source.aperture_knots
        target_knots = target.aperture_knots
        if (
            source_knots is None
            or target_knots is None
            or source_knots.shape != target_knots.shape
        ):
            raise ValueError("gripper aperture migration requires matching scheduled assets")
        old_aperture = np.asarray(group["actual_aperture"][...], dtype=np.float32)
        if target.supports_joint_fk:
            if "joint_position" not in group:
                raise ValueError(
                    f"cannot migrate {path}: target gripper requires joint_position"
                )
            joint_dataset = group["joint_position"]
            raw_names = joint_dataset.attrs.get("joint_names", ())
            joint_names = tuple(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in raw_names
            )
            if joint_names != target.joint_names:
                raise ValueError(
                    f"cannot migrate {path}: joint order does not match target gripper"
                )
            remapped = (
                target.aperture_from_joints(
                    torch.from_numpy(joint_dataset[...]).float()
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        else:
            remapped = None

    if remapped is None:
        old_knots = source_knots.detach().cpu().numpy().astype(np.float64)
        new_knots = target_knots.detach().cpu().numpy().astype(np.float64)
        tolerance = 1e-6
        if (
            old_aperture.min() < old_knots[0] - tolerance
            or old_aperture.max() > old_knots[-1] + tolerance
        ):
            raise ValueError(f"cannot migrate {path}: aperture lies outside source limits")
        remapped = np.interp(old_aperture, old_knots, new_knots).astype(np.float32)

    temporary = path.with_suffix(path.suffix + ".gripper.tmp")
    temporary.unlink(missing_ok=True)
    try:
        shutil.copy2(path, temporary)
        with h5py.File(temporary, "r+") as handle:
            group = handle["objects/000000"]
            group["actual_aperture"][...] = remapped
            group.attrs["gripper_geometry_sha256"] = target_hash
            handle.flush()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _atomic_gripper(path: Path, gripper: GripperAsset) -> None:
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        gripper.save(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _check_existing_shard(
    path: Path,
    expected_object_id: str,
    expected_trajectory_count: int,
    *,
    physics: PhysicsMetadata,
) -> None:
    with h5py.File(path, "r") as handle:
        stored_physics = str(handle.attrs.get("physics_metadata_json", ""))
        if stored_physics != physics.canonical_json():
            raise ValueError(
                f"existing shard {path} has a different or missing physics fingerprint"
            )
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
        "format_version": 2,
        "config_sha256": config.sha256(),
        "catalog_sha256": hashlib.sha256(catalog.catalog_path.read_bytes()).hexdigest(),
        "validation_commit": catalog.payload["source"]["validation_commit"],
        "completed_object_ids": completed,
        "physics": expected_physics_metadata(config).to_dict(),
    }
    _atomic_json(output / "collection.json", payload)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
