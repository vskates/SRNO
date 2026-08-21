#!/usr/bin/env python3
"""Factorize the hidden-contact-state effect in the SRNO PhysX collector.

The diagnostic suite implements the tests requested after the Markov-state
sufficiency experiment:

* P0-A: continuous replay pairs, with an explicitly repeated zero-velocity
  write in one member of each pair;
* P0-B: continuous/fresh comparison with PhysX strong friction disabled;
* P0-C: continuous/fresh comparison with the legacy SAT contact generator
  instead of PCM;
* fresh preconditioning: rebuild a fresh contact state at the current command
  before applying the next command.

P0-B uses a diagnostic-only native bridge because Omniverse PhysX 107.3 exposes
the SDK flag but not a Python/USD setter.  It changes exactly
``PxMaterialFlag::eDISABLE_STRONG_FRICTION`` on the material already used by the
collector; coefficients, combine modes and the production collector are intact.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from markov_state_sufficiency import (
    TransitionSet,
    _select_transitions,
    _state_errors,
    _stats,
)


@dataclass(frozen=True)
class ReplayStates:
    current_position: np.ndarray
    current_quaternion_wxyz: np.ndarray
    current_joint: np.ndarray
    successor_position: np.ndarray
    successor_quaternion_wxyz: np.ndarray
    successor_joint: np.ndarray
    current_settled: np.ndarray
    successor_settled: np.ndarray
    post_restore_linear_velocity_m_s: np.ndarray
    post_restore_angular_velocity_rad_s: np.ndarray
    post_restore_joint_velocity_rad_s: np.ndarray


@dataclass(frozen=True)
class ResetStates:
    successor_position: np.ndarray
    successor_quaternion_wxyz: np.ndarray
    successor_joint: np.ndarray
    successor_settled: np.ndarray
    successor_contact_count: np.ndarray
    warm_position: np.ndarray | None
    warm_quaternion_wxyz: np.ndarray | None
    warm_joint: np.ndarray | None
    warm_settled: np.ndarray | None


def _configure_collision_system(simulation: Any, collision_system: str) -> None:
    """Set PCM or SAT before the stage is parsed into PhysX."""

    from pxr import PhysxSchema

    if collision_system not in {"PCM", "SAT"}:
        raise ValueError(f"unsupported collision system {collision_system!r}")
    stage = simulation.get_initial_stage()
    scene_prim = stage.GetPrimAtPath(simulation.cfg.physics_prim_path)
    if not scene_prim.IsValid():
        raise RuntimeError(
            f"physics scene not found at {simulation.cfg.physics_prim_path}"
        )
    api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    attribute = api.CreateCollisionSystemAttr()
    attribute.Set(collision_system)
    if attribute.Get() != collision_system:
        raise RuntimeError("failed to author PhysX collision system")


def _close_stage(app: Any, simulation: Any, scene: Any, collector: Any) -> None:
    import omni.usd

    context = omni.usd.get_context()
    del collector
    del scene
    simulation.clear_all_callbacks()
    simulation.clear_instance()
    del simulation
    gc.collect()
    context.close_stage()
    for _ in range(5):
        app.update()


def _open_stage(
    app: Any,
    config: Any,
    catalog: Any,
    gripper: Any,
    object_id: str,
    *,
    env_count: int,
    collision_system: str,
    disable_strong_friction: bool = False,
    native_material_library: Path | None = None,
) -> tuple[Any, Any, Any]:
    import carb
    import omni.usd
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext
    import omni.physx
    from pxr import PhysicsSchemaTools

    from srno.sim.collector import QuasistaticCollector
    from srno.sim.isaac_scene import (
        apply_contact_materials,
        make_scene_cfg,
        make_simulation_cfg,
    )
    from srno.sim.physx_material import PhysxMaterialAudit, expected_physics_metadata

    settings = carb.settings.get_settings()
    settings.set("/renderer/multiGPU/enabled", False)
    settings.set("/renderer/multiGPU/autoEnable", False)
    settings.set("/rtx/realtime/mgpu/autoTiling/enabled", False)
    context = omni.usd.get_context()
    context.new_stage()
    for _ in range(10):
        app.update()
    simulation = SimulationContext(make_simulation_cfg(config.device, config.material))
    _configure_collision_system(simulation, collision_system)
    record = catalog.object(object_id)
    material_audit = PhysxMaterialAudit(expected_physics_metadata(config))
    material_audit.start()
    material_ids: list[int] = []
    material_records: list[tuple[int, int]] = []
    created_physx_types: list[int] = []
    subscription = None
    physx_interface = None
    if disable_strong_friction:
        if native_material_library is None:
            raise ValueError("native material library is required for P0-B")

        def on_created(path: int, object_id: int, physx_type: int) -> None:
            created_physx_types.append(int(physx_type))
            if physx_type != 2:  # omni::physx::ePTMaterial
                return
            material_records.append((int(path), int(object_id)))

        physx_interface = omni.physx.get_physx_interface()
        subscription = physx_interface.subscribe_object_changed_notifications(
            object_creation_fn=on_created,
            stop_callback_when_sim_stopped=False,
        )
    try:
        scene = InteractiveScene(
            make_scene_cfg(
                catalog,
                record,
                num_envs=env_count,
                relaxation=config.relaxation,
            )
        )
        apply_contact_materials(scene, config.material)
        material_audit.force_load()
        simulation.reset()
        material_audit.verify(app)
    except BaseException:
        material_audit.close()
        raise
    if disable_strong_friction:
        assert physx_interface is not None and subscription is not None
        # Notifications are delivered on the application update after PhysX
        # parsing, as in Omniverse's own object-change tests.
        for _ in range(2):
            app.update()
        physx_interface.unsubscribe_object_change_notifications(subscription)
        decoded_materials = [
            (str(PhysicsSchemaTools.intToSdfPath(path)), object_id)
            for path, object_id in material_records
        ]
        # The custom USD relationship currently does not instantiate
        # /World/SRNOContactMaterial in PhysX.  Apply the requested flag to all
        # actually instantiated scene materials (gripper-authored + default),
        # which are precisely the materials the contact shapes can reference.
        material_ids = sorted({object_id for _, object_id in decoded_materials})
        print(
            "[P0-B] active PhysX material object ids: "
            f"{material_ids}; created types: "
            f"{sorted(set(created_physx_types))}; materials: {decoded_materials}",
            flush=True,
        )
        if not material_ids:
            raise RuntimeError(
                "no instantiated PhysX materials were reported by the runtime"
            )
        library = ctypes.CDLL(str(native_material_library.resolve()))
        setter = library.srno_set_disable_strong_friction
        setter.argtypes = (ctypes.c_size_t, ctypes.c_int)
        setter.restype = ctypes.c_int
        statuses = {object_id: setter(object_id, 1) for object_id in material_ids}
        print(f"[P0-B] native material flag statuses: {statuses}", flush=True)
        failed = {object_id: status for object_id, status in statuses.items() if status != 1}
        if failed:
            raise RuntimeError(
                "failed to set PxMaterialFlag::eDISABLE_STRONG_FRICTION: "
                f"{failed}"
            )
    collector = QuasistaticCollector(
        simulation, scene, catalog, record, config, gripper
    )
    return simulation, scene, collector


def _continuous_replay(
    app: Any,
    config: Any,
    catalog: Any,
    gripper: Any,
    selected: TransitionSet,
    *,
    collision_system: str,
    paired_explicit_zero: bool,
    disable_strong_friction: bool = False,
    native_material_library: Path | None = None,
) -> ReplayStates:
    """Replay source poses continuously up to every selected successor."""

    import torch

    from srno.sim.pose_seeds import PoseSeeds

    repeats = 2 if paired_explicit_zero else 1
    source_index = np.tile(selected.source_pose_index, repeats)
    current_step_np = np.tile(selected.current_step, repeats)
    base_count = len(selected.current_step)
    count = len(source_index)
    simulation = scene = collector = None
    try:
        simulation, scene, collector = _open_stage(
            app,
            config,
            catalog,
            gripper,
            selected.object_id,
            env_count=count,
            collision_system=collision_system,
            disable_strong_friction=disable_strong_friction,
            native_material_library=native_material_library,
        )
        device = torch.device(scene.device)
        record = catalog.object(selected.object_id)
        seeds = PoseSeeds.load(record.pose_seed_path)
        seed_position = torch.from_numpy(seeds.position_m[source_index]).to(device)
        seed_quaternion = torch.from_numpy(seeds.quaternion_wxyz[source_index]).to(
            device
        )
        base_position, base_quaternion = collector._reset_batch(
            seed_position, seed_quaternion
        )
        required = torch.ones(count, dtype=torch.bool, device=device)
        initial = collector._settle_command(
            collector.open_joint_target, 0, required_mask=required
        )
        alive = initial.settled_mask.clone()
        current_step = torch.from_numpy(current_step_np).to(device=device)
        successor_step = current_step + 1

        current_position = torch.full((count, 3), torch.nan, device=device)
        current_quaternion = torch.full((count, 4), torch.nan, device=device)
        current_joint = torch.full_like(collector.robot.data.joint_pos, torch.nan)
        successor_position = torch.full((count, 3), torch.nan, device=device)
        successor_quaternion = torch.full((count, 4), torch.nan, device=device)
        successor_joint = torch.full_like(collector.robot.data.joint_pos, torch.nan)
        current_settled = torch.zeros(count, dtype=torch.bool, device=device)
        successor_settled = torch.zeros(count, dtype=torch.bool, device=device)
        post_linear = torch.full((count,), torch.nan, device=device)
        post_angular = torch.full((count,), torch.nan, device=device)
        post_joint = torch.full((count,), torch.nan, device=device)

        max_step = int(successor_step.max().item())
        for command_index in range(1, max_step + 1):
            required = alive & (successor_step >= command_index)
            if not bool(torch.any(required)):
                break
            target = collector.close_joint_target * (float(command_index) / 32.0)
            result = collector._settle_command(
                target, command_index, required_mask=required
            )
            failed = required & ~result.settled_mask
            alive &= ~failed
            relative_position, relative_quaternion, _ = collector._record_state(
                base_position, base_quaternion, result
            )

            at_current = alive & (current_step == command_index)
            if bool(torch.any(at_current)):
                current_position[at_current] = relative_position[at_current]
                current_quaternion[at_current] = relative_quaternion[at_current]
                current_joint[at_current] = result.joint_position[at_current]
                current_settled[at_current] = True
                root_velocity = collector.object.data.root_state_w[:, 7:13]
                joint_velocity = collector.robot.data.joint_vel
                post_linear[at_current] = torch.linalg.vector_norm(
                    root_velocity[at_current, :3], dim=-1
                )
                post_angular[at_current] = torch.linalg.vector_norm(
                    root_velocity[at_current, 3:6], dim=-1
                )
                post_joint[at_current] = torch.linalg.vector_norm(
                    joint_velocity[at_current], dim=-1
                )

                # Production _settle_command already performed this zero write
                # for every environment.  Repeat velocity-only writes for the
                # second paired branch without touching q, r or contact geometry.
                if paired_explicit_zero:
                    explicit = at_current.clone()
                    explicit[:base_count] = False
                    env_ids = torch.nonzero(explicit).flatten()
                    if len(env_ids):
                        collector.object.write_root_velocity_to_sim(
                            torch.zeros((len(env_ids), 6), device=device),
                            env_ids=env_ids,
                        )
                        collector.robot.write_joint_velocity_to_sim(
                            torch.zeros(
                                (len(env_ids), collector.robot.num_joints),
                                device=device,
                            ),
                            env_ids=env_ids,
                        )

            at_successor = alive & (successor_step == command_index)
            if bool(torch.any(at_successor)):
                successor_position[at_successor] = relative_position[at_successor]
                successor_quaternion[at_successor] = relative_quaternion[at_successor]
                successor_joint[at_successor] = result.joint_position[at_successor]
                successor_settled[at_successor] = True

        def array(value: Any) -> np.ndarray:
            return value.detach().cpu().numpy()

        return ReplayStates(
            current_position=array(current_position),
            current_quaternion_wxyz=array(current_quaternion),
            current_joint=array(current_joint),
            successor_position=array(successor_position),
            successor_quaternion_wxyz=array(successor_quaternion),
            successor_joint=array(successor_joint),
            current_settled=array(current_settled),
            successor_settled=array(successor_settled),
            post_restore_linear_velocity_m_s=array(post_linear),
            post_restore_angular_velocity_rad_s=array(post_angular),
            post_restore_joint_velocity_rad_s=array(post_joint),
        )
    finally:
        if simulation is not None:
            _close_stage(app, simulation, scene, collector)


def _reset_successor(
    app: Any,
    config: Any,
    catalog: Any,
    gripper: Any,
    selected: TransitionSet,
    *,
    current_position: np.ndarray,
    current_quaternion_wxyz: np.ndarray,
    current_joint: np.ndarray,
    collision_system: str,
    precondition: bool,
    disable_strong_friction: bool = False,
    native_material_library: Path | None = None,
) -> ResetStates:
    """Cold or preconditioned fresh solve from supplied observable states."""

    import torch
    from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms

    from srno.sim.pose_seeds import PoseSeeds

    count = len(selected.current_step)
    simulation = scene = collector = None
    try:
        simulation, scene, collector = _open_stage(
            app,
            config,
            catalog,
            gripper,
            selected.object_id,
            env_count=count,
            collision_system=collision_system,
            disable_strong_friction=disable_strong_friction,
            native_material_library=native_material_library,
        )
        device = torch.device(scene.device)
        record = catalog.object(selected.object_id)
        source_seeds = PoseSeeds.load(record.pose_seed_path)
        base_position = scene.env_origins + torch.from_numpy(
            source_seeds.position_m[selected.source_pose_index]
        ).to(device)
        base_quaternion = torch.from_numpy(
            source_seeds.quaternion_wxyz[selected.source_pose_index]
        ).to(device)
        collector._root_position, collector._root_quaternion = collector._base_to_root(
            base_position, base_quaternion
        )

        robot_state = collector.robot.data.root_state_w.clone()
        robot_state[:, :3] = collector._root_position
        robot_state[:, 3:7] = collector._root_quaternion
        robot_state[:, 7:13] = 0.0
        collector.robot.write_root_state_to_sim(robot_state)
        joint = torch.from_numpy(current_joint).to(device)
        collector.robot.write_joint_state_to_sim(joint, torch.zeros_like(joint))

        relative_position = torch.from_numpy(current_position).to(device)
        relative_quaternion = torch.from_numpy(current_quaternion_wxyz).to(device)
        object_state = collector.object.data.root_state_w.clone()
        object_state[:, :3] = base_position + quat_apply(
            base_quaternion, relative_position
        )
        object_state[:, 3:7] = quat_mul(base_quaternion, relative_quaternion)
        object_state[:, 7:13] = 0.0
        collector.object.write_root_state_to_sim(object_state)
        collector._hold_root()

        required = torch.ones(count, dtype=torch.bool, device=device)
        warm_result = None
        warm_position = warm_quaternion = warm_joint = warm_settled = None
        if precondition:
            fraction = torch.from_numpy(selected.current_step.astype(np.float32)).to(
                device
            ) / 32.0
            current_target = fraction[:, None] * collector.close_joint_target[None]
            warm_result = collector._settle_command(
                current_target,
                int(selected.current_step.max()),
                required_mask=required,
            )
            warm_position_t, warm_quaternion_t = subtract_frame_transforms(
                base_position,
                base_quaternion,
                warm_result.object_position,
                warm_result.object_quaternion_wxyz,
            )
            warm_position = warm_position_t.cpu().numpy()
            warm_quaternion = warm_quaternion_t.cpu().numpy()
            warm_joint = warm_result.joint_position.cpu().numpy()
            warm_settled = warm_result.settled_mask.cpu().numpy()
            required = warm_result.settled_mask
            if not bool(torch.any(required)):
                nan_position = np.full((count, 3), np.nan, dtype=np.float32)
                nan_quaternion = np.full((count, 4), np.nan, dtype=np.float32)
                nan_joint = np.full_like(current_joint, np.nan)
                return ResetStates(
                    nan_position,
                    nan_quaternion,
                    nan_joint,
                    np.zeros(count, dtype=bool),
                    np.zeros(count, dtype=np.float32),
                    warm_position,
                    warm_quaternion,
                    warm_joint,
                    warm_settled,
                )

        next_fraction = torch.from_numpy(
            (selected.current_step + 1).astype(np.float32)
        ).to(device) / 32.0
        next_target = next_fraction[:, None] * collector.close_joint_target[None]
        successor = collector._settle_command(
            next_target,
            int(selected.current_step.max()) + 1,
            required_mask=required,
        )
        successor_position, successor_quaternion = subtract_frame_transforms(
            base_position,
            base_quaternion,
            successor.object_position,
            successor.object_quaternion_wxyz,
        )
        settled = successor.settled_mask
        if warm_result is not None:
            settled &= warm_result.settled_mask
        return ResetStates(
            successor_position.cpu().numpy(),
            successor_quaternion.cpu().numpy(),
            successor.joint_position.cpu().numpy(),
            settled.cpu().numpy(),
            successor.contact_count.cpu().numpy(),
            warm_position,
            warm_quaternion,
            warm_joint,
            warm_settled,
        )
    finally:
        if simulation is not None:
            _close_stage(app, simulation, scene, collector)


def _errors(
    left_position: np.ndarray,
    left_quaternion: np.ndarray,
    left_joint: np.ndarray,
    right_position: np.ndarray,
    right_quaternion: np.ndarray,
    right_joint: np.ndarray,
    *,
    length_scale: float,
    joint_scale: Any,
    device: Any,
) -> dict[str, np.ndarray]:
    import torch

    values = _state_errors(
        torch.from_numpy(left_position).to(device),
        torch.from_numpy(left_quaternion).to(device),
        torch.from_numpy(left_joint).to(device),
        torch.from_numpy(right_position).to(device),
        torch.from_numpy(right_quaternion).to(device),
        torch.from_numpy(right_joint).to(device),
        length_scale=length_scale,
        joint_scale=joint_scale.to(device),
    )
    return {name: value.detach().cpu().numpy() for name, value in values.items()}


def _hdf_targets(selected: TransitionSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        selected.preserve_position,
        selected.preserve_quaternion_xyzw[:, (3, 0, 1, 2)],
        selected.preserve_joint,
    )


def _hdf_current(selected: TransitionSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        selected.current_position,
        selected.current_quaternion_xyzw[:, (3, 0, 1, 2)],
        selected.current_joint,
    )


def _take_transitions(selected: TransitionSet, count: int) -> TransitionSet:
    if not 0 < count <= len(selected.current_step):
        raise ValueError("invalid transition subset size")
    values: dict[str, Any] = {}
    for field in fields(TransitionSet):
        value = getattr(selected, field.name)
        values[field.name] = value[:count] if isinstance(value, np.ndarray) else value
    return TransitionSet(**values)


def _append(prefix: str, values: dict[str, np.ndarray], rows: dict[str, list[np.ndarray]]) -> None:
    for name, value in values.items():
        rows.setdefault(f"{prefix}_{name}", []).append(np.asarray(value))


def _summarize_errors(rows: dict[str, np.ndarray], prefix: str, valid: np.ndarray) -> dict[str, Any]:
    return {
        name: _stats(rows[f"{prefix}_{name}"][valid])
        for name in ("dx", "translation_m", "rotation_rad", "joint_rmse_over_travel")
    }


def _make_plot(output: Path, rows: dict[str, np.ndarray], baseline_dx: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = rows["object_labels"].tolist()
    object_index = rows["object_index"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    data = [
        baseline_dx,
        rows["p0a_dx"],
        rows["sf_off_dx"],
        rows["pcm_off_dx"],
        rows["warm_dx"],
    ]
    names = [
        "baseline cold",
        "paired continuous repeat",
        "strong friction off",
        "PCM off (SAT)",
        "fresh warm",
    ]
    axes[0, 0].boxplot(data, tick_labels=names, showfliers=True)
    axes[0, 0].set_ylabel(r"$d_X$")
    axes[0, 0].set_yscale("symlog", linthresh=1e-4)
    axes[0, 0].tick_params(axis="x", rotation=15)

    upper = float(np.quantile(np.concatenate((baseline_dx, rows["warm_dx"])), 0.99))
    axes[0, 1].scatter(baseline_dx, rows["warm_dx"], c=rows["current_step"], cmap="viridis", s=28)
    axes[0, 1].plot((0, upper), (0, upper), "k--", linewidth=1)
    axes[0, 1].set_xlim(0, upper)
    axes[0, 1].set_ylim(0, upper)
    axes[0, 1].set_xlabel(r"cold fresh/preserve $d_X$")
    axes[0, 1].set_ylabel(r"warm fresh/preserve $d_X$")
    axes[0, 1].set_title("colour = command step")

    grouped = [
        rows["warm_dx"][object_index == index] for index in range(len(labels))
    ]
    axes[1, 0].boxplot(
        grouped,
        tick_labels=[str(label)[:22] for label in labels],
        showfliers=True,
    )
    axes[1, 0].set_ylabel(r"fresh-warm $d_X$")
    axes[1, 0].tick_params(axis="x", rotation=20)

    bins = np.linspace(
        0.0,
        float(np.quantile(np.concatenate((baseline_dx, rows["pcm_off_dx"])), 0.99)),
        30,
    )
    axes[1, 1].hist(baseline_dx, bins=bins, alpha=0.65, label="PCM baseline")
    axes[1, 1].hist(rows["sf_off_dx"], bins=bins, alpha=0.65, label="strong friction off")
    axes[1, 1].hist(rows["pcm_off_dx"], bins=bins, alpha=0.65, label="SAT")
    axes[1, 1].set_xlabel(r"continuous/fresh $d_X$")
    axes[1, 1].set_ylabel("transitions")
    axes[1, 1].legend()

    fig.savefig(output / "contact_memory_factorization.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sim-config", type=Path, required=True)
    parser.add_argument("--baseline-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--samples-per-object", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--native-material-library",
        type=Path,
        required=True,
        help="diagnostic bridge that sets eDISABLE_STRONG_FRICTION",
    )
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    from srno.sim.memory_guard import MemoryWatchdog

    watchdog = MemoryWatchdog(14.0, 0.25)
    watchdog.start()
    app = AppLauncher({"headless": True, "device": "cuda:0"}).app
    try:
        import torch

        from srno.data.schema import DatasetManifest
        from srno.geometry.gripper import GripperAsset
        from srno.sim.assets import SimulatorAssetCatalog
        from srno.sim.config import SimulatorConfig

        manifest = DatasetManifest.load(args.manifest)
        config = SimulatorConfig.load(args.sim_config)
        catalog = SimulatorAssetCatalog.load(config.catalog)
        gripper = GripperAsset.load(manifest.gripper_path)
        objects = args.objects or list(manifest.splits["val"] + manifest.splits["test"])
        device = torch.device(config.device)

        baseline = np.load(args.baseline_samples, allow_pickle=False)
        baseline_labels = baseline["object_labels"].tolist()
        if any(object_id not in baseline_labels for object_id in objects):
            raise ValueError(
                "objects must be present in baseline samples: "
                f"{baseline_labels}"
            )

        rows: dict[str, list[np.ndarray]] = {}
        baseline_dx_parts: list[np.ndarray] = []
        for object_index, object_id in enumerate(objects):
            baseline_object_index = baseline_labels.index(object_id)
            baseline_mask = baseline["object_index"] == baseline_object_index
            baseline_count = int(baseline_mask.sum())
            if args.samples_per_object > baseline_count:
                raise ValueError(
                    f"{object_id}: requested {args.samples_per_object} samples, "
                    f"baseline has {baseline_count}"
                )
            selected_full = _select_transitions(
                manifest,
                object_id,
                count=baseline_count,
                seed=args.seed + baseline_object_index,
                joint_names=gripper.joint_names,
            )
            if (
                not np.array_equal(
                    baseline["trajectory"][baseline_mask], selected_full.trajectory
                )
                or not np.array_equal(
                    baseline["current_step"][baseline_mask], selected_full.current_step
                )
            ):
                raise ValueError(
                    f"{object_id}: selected transitions do not match baseline NPZ"
                )
            selected = _take_transitions(selected_full, args.samples_per_object)
            baseline_dx_parts.append(
                baseline["reset_dx"][baseline_mask][: args.samples_per_object]
            )
            count = len(selected.current_step)
            rows.setdefault("object_index", []).append(
                np.full(count, object_index, dtype=np.int16)
            )
            rows.setdefault("trajectory", []).append(selected.trajectory)
            rows.setdefault("current_step", []).append(selected.current_step)

            print(f"[P0-A] {object_id}: paired continuous replay", flush=True)
            paired = _continuous_replay(
                app,
                config,
                catalog,
                gripper,
                selected,
                collision_system="PCM",
                paired_explicit_zero=True,
            )
            left = slice(0, count)
            right = slice(count, 2 * count)
            p0a = _errors(
                paired.successor_position[right],
                paired.successor_quaternion_wxyz[right],
                paired.successor_joint[right],
                paired.successor_position[left],
                paired.successor_quaternion_wxyz[left],
                paired.successor_joint[left],
                length_scale=gripper.length_scale,
                joint_scale=gripper.joint_travel_range,
                device=device,
            )
            _append("p0a", p0a, rows)
            rows.setdefault("p0a_valid", []).append(
                paired.successor_settled[left] & paired.successor_settled[right]
            )
            rows.setdefault("p0a_pre_zero_linear_velocity_m_s", []).append(
                paired.post_restore_linear_velocity_m_s[right]
            )
            rows.setdefault("p0a_pre_zero_angular_velocity_rad_s", []).append(
                paired.post_restore_angular_velocity_rad_s[right]
            )
            rows.setdefault("p0a_pre_zero_joint_velocity_rad_s", []).append(
                paired.post_restore_joint_velocity_rad_s[right]
            )

            print(f"[P0-B] {object_id}: strong-friction-off continuous replay", flush=True)
            sf_continuous = _continuous_replay(
                app,
                config,
                catalog,
                gripper,
                selected,
                collision_system="PCM",
                paired_explicit_zero=False,
                disable_strong_friction=True,
                native_material_library=args.native_material_library,
            )
            print(f"[P0-B] {object_id}: strong-friction-off fresh reset A/B", flush=True)
            sf_reset = _reset_successor(
                app,
                config,
                catalog,
                gripper,
                selected,
                current_position=sf_continuous.current_position,
                current_quaternion_wxyz=sf_continuous.current_quaternion_wxyz,
                current_joint=sf_continuous.current_joint,
                collision_system="PCM",
                precondition=False,
                disable_strong_friction=True,
                native_material_library=args.native_material_library,
            )
            sf_repeat = _reset_successor(
                app,
                config,
                catalog,
                gripper,
                selected,
                current_position=sf_continuous.current_position,
                current_quaternion_wxyz=sf_continuous.current_quaternion_wxyz,
                current_joint=sf_continuous.current_joint,
                collision_system="PCM",
                precondition=False,
                disable_strong_friction=True,
                native_material_library=args.native_material_library,
            )
            sf_off = _errors(
                sf_reset.successor_position,
                sf_reset.successor_quaternion_wxyz,
                sf_reset.successor_joint,
                sf_continuous.successor_position,
                sf_continuous.successor_quaternion_wxyz,
                sf_continuous.successor_joint,
                length_scale=gripper.length_scale,
                joint_scale=gripper.joint_travel_range,
                device=device,
            )
            sf_off_repeat = _errors(
                sf_repeat.successor_position,
                sf_repeat.successor_quaternion_wxyz,
                sf_repeat.successor_joint,
                sf_reset.successor_position,
                sf_reset.successor_quaternion_wxyz,
                sf_reset.successor_joint,
                length_scale=gripper.length_scale,
                joint_scale=gripper.joint_travel_range,
                device=device,
            )
            _append("sf_off", sf_off, rows)
            _append("sf_off_repeat", sf_off_repeat, rows)
            rows.setdefault("sf_off_valid", []).append(
                sf_continuous.current_settled
                & sf_continuous.successor_settled
                & sf_reset.successor_settled
                & sf_repeat.successor_settled
            )

            print(f"[P0-C] {object_id}: continuous SAT replay", flush=True)
            sat_continuous = _continuous_replay(
                app,
                config,
                catalog,
                gripper,
                selected,
                collision_system="SAT",
                paired_explicit_zero=False,
            )
            print(f"[P0-C] {object_id}: fresh SAT reset A/B", flush=True)
            sat_reset = _reset_successor(
                app,
                config,
                catalog,
                gripper,
                selected,
                current_position=sat_continuous.current_position,
                current_quaternion_wxyz=sat_continuous.current_quaternion_wxyz,
                current_joint=sat_continuous.current_joint,
                collision_system="SAT",
                precondition=False,
            )
            sat_repeat = _reset_successor(
                app,
                config,
                catalog,
                gripper,
                selected,
                current_position=sat_continuous.current_position,
                current_quaternion_wxyz=sat_continuous.current_quaternion_wxyz,
                current_joint=sat_continuous.current_joint,
                collision_system="SAT",
                precondition=False,
            )
            pcm_off = _errors(
                sat_reset.successor_position,
                sat_reset.successor_quaternion_wxyz,
                sat_reset.successor_joint,
                sat_continuous.successor_position,
                sat_continuous.successor_quaternion_wxyz,
                sat_continuous.successor_joint,
                length_scale=gripper.length_scale,
                joint_scale=gripper.joint_travel_range,
                device=device,
            )
            pcm_off_repeat = _errors(
                sat_repeat.successor_position,
                sat_repeat.successor_quaternion_wxyz,
                sat_repeat.successor_joint,
                sat_reset.successor_position,
                sat_reset.successor_quaternion_wxyz,
                sat_reset.successor_joint,
                length_scale=gripper.length_scale,
                joint_scale=gripper.joint_travel_range,
                device=device,
            )
            _append("pcm_off", pcm_off, rows)
            _append("pcm_off_repeat", pcm_off_repeat, rows)
            rows.setdefault("pcm_off_valid", []).append(
                sat_continuous.current_settled
                & sat_continuous.successor_settled
                & sat_reset.successor_settled
                & sat_repeat.successor_settled
            )

            print(f"[WARM] {object_id}: fresh preconditioning A/B", flush=True)
            current_position, current_quaternion, current_joint = _hdf_current(selected)
            warm = _reset_successor(
                app,
                config,
                catalog,
                gripper,
                selected,
                current_position=current_position,
                current_quaternion_wxyz=current_quaternion,
                current_joint=current_joint,
                collision_system="PCM",
                precondition=True,
            )
            warm_repeat = _reset_successor(
                app,
                config,
                catalog,
                gripper,
                selected,
                current_position=current_position,
                current_quaternion_wxyz=current_quaternion,
                current_joint=current_joint,
                collision_system="PCM",
                precondition=True,
            )
            target_position, target_quaternion, target_joint = _hdf_targets(selected)
            warm_errors = _errors(
                warm.successor_position,
                warm.successor_quaternion_wxyz,
                warm.successor_joint,
                target_position,
                target_quaternion,
                target_joint,
                length_scale=gripper.length_scale,
                joint_scale=gripper.joint_travel_range,
                device=device,
            )
            warm_repeat_errors = _errors(
                warm_repeat.successor_position,
                warm_repeat.successor_quaternion_wxyz,
                warm_repeat.successor_joint,
                warm.successor_position,
                warm.successor_quaternion_wxyz,
                warm.successor_joint,
                length_scale=gripper.length_scale,
                joint_scale=gripper.joint_travel_range,
                device=device,
            )
            warm_drift = _errors(
                warm.warm_position,
                warm.warm_quaternion_wxyz,
                warm.warm_joint,
                current_position,
                current_quaternion,
                current_joint,
                length_scale=gripper.length_scale,
                joint_scale=gripper.joint_travel_range,
                device=device,
            )
            _append("warm", warm_errors, rows)
            _append("warm_repeat", warm_repeat_errors, rows)
            _append("warm_drift", warm_drift, rows)
            rows.setdefault("warm_valid", []).append(
                warm.successor_settled & warm_repeat.successor_settled
            )

        arrays = {name: np.concatenate(parts) for name, parts in rows.items()}
        arrays["object_labels"] = np.asarray(objects)
        baseline_dx = np.concatenate(baseline_dx_parts)
        p0a_valid = arrays["p0a_valid"].astype(bool)
        sf_valid = arrays["sf_off_valid"].astype(bool)
        pcm_valid = arrays["pcm_off_valid"].astype(bool)
        warm_valid = arrays["warm_valid"].astype(bool)
        summary = {
            "definition": {
                "dx": "sqrt((||dp||/L)^2 + d_SO3^2 + mean((dr/s)^2))",
                "p0a": "paired continuous PCM replay; second branch receives a redundant velocity-only zero write at k",
                "pcm_off": "continuous versus fresh successor with physxScene:collisionSystem=SAT",
                "fresh_warm": "fresh q_k,r_k; settle at current command; then apply next command",
            },
            "dataset": {
                "manifest": str(args.manifest.resolve()),
                "objects": objects,
                "samples_per_object": args.samples_per_object,
                "total_samples": int(len(baseline_dx)),
            },
            "baseline_cold_pcm": {
                "dx": _stats(baseline_dx),
            },
            "p0a_velocity": {
                "conclusion": "ordinary production preserve already restores zero object and joint velocities before every next increment; therefore preserve and preserve-zero-velocity are the same branch and D_v is identically zero",
                "settled_samples": int(p0a_valid.sum()),
                "paired_continuous_repeatability": _summarize_errors(
                    arrays, "p0a", p0a_valid
                ),
                "velocities_before_redundant_zero": {
                    "linear_m_s": _stats(
                        arrays["p0a_pre_zero_linear_velocity_m_s"][p0a_valid]
                    ),
                    "angular_rad_s": _stats(
                        arrays["p0a_pre_zero_angular_velocity_rad_s"][p0a_valid]
                    ),
                    "joint_rad_s": _stats(
                        arrays["p0a_pre_zero_joint_velocity_rad_s"][p0a_valid]
                    ),
                },
            },
            "p0b_strong_friction": {
                "status": "completed_via_diagnostic_native_bridge",
                "isaac_sim_physx_schema": "107.3 / PhysX 5.6.1",
                "changed_flag": "PxMaterialFlag::eDISABLE_STRONG_FRICTION=true",
                "unchanged": "friction coefficients, combine modes, PCM, scene and collector",
                "settled_samples": int(sf_valid.sum()),
                "errors": _summarize_errors(arrays, "sf_off", sf_valid),
                "fresh_repeatability": _summarize_errors(
                    arrays, "sf_off_repeat", sf_valid
                ),
            },
            "p0c_pcm": {
                "settled_samples": int(pcm_valid.sum()),
                "collision_system": "SAT",
                "backend_note": "PhysX 5.6.1 keeps GPU dynamics but moves contact generation to CPU when SAT is selected",
                "errors": _summarize_errors(arrays, "pcm_off", pcm_valid),
                "fresh_repeatability": _summarize_errors(
                    arrays, "pcm_off_repeat", pcm_valid
                ),
            },
            "fresh_preconditioning": {
                "settled_samples": int(warm_valid.sum()),
                "errors": _summarize_errors(arrays, "warm", warm_valid),
                "fresh_repeatability": _summarize_errors(
                    arrays, "warm_repeat", warm_valid
                ),
                "preconditioning_drift": _summarize_errors(
                    arrays, "warm_drift", warm_valid
                ),
                "mean_dx_ratio_warm_over_cold": float(
                    arrays["warm_dx"][warm_valid].mean()
                    / baseline_dx[warm_valid].mean()
                ),
                "fraction_improved": float(
                    np.mean(
                        arrays["warm_dx"][warm_valid]
                        < baseline_dx[warm_valid]
                    )
                ),
            },
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "results.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(
            args.output / "samples.npz",
            **arrays,
            baseline_cold_pcm_dx=baseline_dx,
        )
        _make_plot(args.output, arrays, baseline_dx)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    finally:
        app.close()
        watchdog.stop()


if __name__ == "__main__":
    main()
