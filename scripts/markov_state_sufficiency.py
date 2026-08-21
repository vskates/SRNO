#!/usr/bin/env python3
"""Test whether (object pose, six gripper joints) determines one PhysX step.

The recorded HDF5 successor is the ``preserve`` branch: it was produced by the
continuous collector without rebuilding the PhysX scene between load
increments.  The ``reset`` branch creates a fresh stage, restores only the
recorded object-to-gripper pose and six joint positions with zero velocities,
applies the identical next command, and uses the production settling rule.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class TransitionSet:
    object_id: str
    trajectory: np.ndarray
    source_pose_index: np.ndarray
    current_step: np.ndarray
    current_position: np.ndarray
    current_quaternion_xyzw: np.ndarray
    current_joint: np.ndarray
    preserve_position: np.ndarray
    preserve_quaternion_xyzw: np.ndarray
    preserve_joint: np.ndarray
    preserve_aperture: np.ndarray
    current_contact_count: np.ndarray
    next_contact_count: np.ndarray
    current_lag_m: np.ndarray


def _decode_names(values: object) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    )


def _select_transitions(
    manifest: Any,
    object_id: str,
    *,
    count: int,
    seed: int,
    joint_names: tuple[str, ...],
) -> TransitionSet:
    shard, group_name = manifest.object_locations()[object_id]
    with h5py.File(shard, "r") as handle:
        group = handle[group_name]
        position = np.asarray(group["position"], dtype=np.float32)
        quaternion = np.asarray(group["quaternion_xyzw"], dtype=np.float32)
        joint = np.asarray(group["joint_position"], dtype=np.float32)
        aperture = np.asarray(group["actual_aperture"], dtype=np.float32)
        contact = np.asarray(
            group["diagnostics/contact_count"], dtype=np.float32
        )
        source_pose_index = np.asarray(group["source_pose_index"], dtype=np.int64)
        names = _decode_names(group["joint_position"].attrs["joint_names"])

    schedule = np.asarray(manifest.commanded_aperture_m, dtype=np.float32)
    lag = aperture[:, 1:32] - schedule[None, 1:32]
    sustained = (
        (contact[:, :31] > 0.0)
        & (contact[:, 1:32] > 0.0)
        & (lag > 1e-4)
    )
    candidates = np.argwhere(sustained)
    if len(candidates) < count:
        raise ValueError(
            f"{object_id}: only {len(candidates)} sustained contact transitions, "
            f"requested {count}"
        )
    # The second candidate coordinate indexes states 1..31.  Sample separately
    # from equally spaced step strata so early and late closure are represented.
    generator = np.random.default_rng(seed)
    state = candidates[:, 1] + 1
    strata = np.array_split(np.arange(1, 32), min(count, 8))
    chosen: list[int] = []
    quota = np.full(len(strata), count // len(strata), dtype=np.int64)
    quota[: count % len(strata)] += 1
    for state_values, requested in zip(strata, quota, strict=True):
        eligible = np.flatnonzero(np.isin(state, state_values))
        if len(eligible):
            take = min(int(requested), len(eligible))
            chosen.extend(generator.choice(eligible, size=take, replace=False).tolist())
    remaining = count - len(chosen)
    if remaining:
        pool = np.setdiff1d(np.arange(len(candidates)), np.asarray(chosen), assume_unique=False)
        chosen.extend(generator.choice(pool, size=remaining, replace=False).tolist())
    chosen_array = np.asarray(chosen, dtype=np.int64)
    generator.shuffle(chosen_array)
    trajectory = candidates[chosen_array, 0]
    current_step = candidates[chosen_array, 1] + 1
    if names != joint_names:
        raise ValueError(f"{object_id}: joint order does not match gripper asset")
    return TransitionSet(
        object_id=object_id,
        trajectory=trajectory,
        source_pose_index=source_pose_index[trajectory],
        current_step=current_step,
        current_position=position[trajectory, current_step],
        current_quaternion_xyzw=quaternion[trajectory, current_step],
        current_joint=joint[trajectory, current_step],
        preserve_position=position[trajectory, current_step + 1],
        preserve_quaternion_xyzw=quaternion[trajectory, current_step + 1],
        preserve_joint=joint[trajectory, current_step + 1],
        preserve_aperture=aperture[trajectory, current_step + 1],
        current_contact_count=contact[trajectory, current_step - 1],
        next_contact_count=contact[trajectory, current_step],
        current_lag_m=lag[trajectory, current_step - 1],
    )


def _stats(values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(len(flat)),
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "p90": float(np.quantile(flat, 0.90)),
        "p95": float(np.quantile(flat, 0.95)),
        "p99": float(np.quantile(flat, 0.99)),
        "max": float(np.max(flat)),
    }


def _rotation_error_wxyz(left: Any, right: Any) -> Any:
    from isaaclab.utils.math import quat_inv, quat_mul
    import torch

    relative = quat_mul(quat_inv(right), left)
    relative = torch.where(relative[:, :1] < 0.0, -relative, relative)
    sine = torch.linalg.vector_norm(relative[:, 1:], dim=-1)
    return 2.0 * torch.atan2(sine, relative[:, 0].clamp_min(0.0))


def _state_errors(
    position: Any,
    quaternion_wxyz: Any,
    joint: Any,
    target_position: Any,
    target_quaternion_wxyz: Any,
    target_joint: Any,
    *,
    length_scale: float,
    joint_scale: Any,
) -> dict[str, Any]:
    import torch

    translation = torch.linalg.vector_norm(position - target_position, dim=-1)
    rotation = _rotation_error_wxyz(quaternion_wxyz, target_quaternion_wxyz)
    joint_rmse = torch.sqrt(
        ((joint - target_joint) / joint_scale).square().mean(dim=-1)
    )
    dx = torch.sqrt(
        (translation / length_scale).square()
        + rotation.square()
        + joint_rmse.square()
    )
    return {
        "translation_m": translation,
        "rotation_rad": rotation,
        "joint_rmse_over_travel": joint_rmse,
        "dx": dx,
    }


def _model_errors(
    selected: TransitionSet,
    manifest: Any,
    gripper: Any,
    model: Any,
    *,
    device: Any,
) -> dict[str, np.ndarray]:
    import torch

    from srno.geometry.se3 import quaternion_xyzw_to_matrix
    from srno.types import PoseState, SDFBatch

    shard, group_name = manifest.object_locations()[selected.object_id]
    with h5py.File(shard, "r") as handle:
        group = handle[group_name]
        sdf = torch.from_numpy(np.asarray(group["sdf"], dtype=np.float32)).to(device)
        origin = torch.from_numpy(
            np.asarray(group.attrs["grid_origin"], dtype=np.float32)
        ).to(device)
        voxel = torch.from_numpy(
            np.asarray(group.attrs["voxel_size"], dtype=np.float32)
        ).to(device)
    current_quaternion = torch.from_numpy(selected.current_quaternion_xyzw).to(device)
    target_quaternion = torch.from_numpy(selected.preserve_quaternion_xyzw).to(device)
    current = PoseState(
        quaternion_xyzw_to_matrix(current_quaternion),
        torch.from_numpy(selected.current_position).to(device),
        torch.from_numpy(selected.current_joint).to(device),
    )
    command = torch.as_tensor(
        np.asarray(manifest.commanded_aperture_m, dtype=np.float32)[
            selected.current_step + 1
        ],
        device=device,
    )
    sdf_batch = SDFBatch(
        sdf[None],
        origin[None],
        voxel[None],
        torch.zeros(len(command), dtype=torch.long, device=device),
        manifest.sdf_scale_m,
    )
    with torch.no_grad():
        prediction, aux = model.forward_step(
            current, command, sdf_batch, return_aux=True
        )
    target_position = torch.from_numpy(selected.preserve_position).to(device)
    target_joint = torch.from_numpy(selected.preserve_joint).to(device)
    target_quaternion_wxyz = target_quaternion[:, (3, 0, 1, 2)]
    # Matrix prediction is converted only for reuse of the common quaternion
    # metric below; the direct geodesic implementation is equivalent.
    from srno.geometry.se3 import rotation_geodesic_angle

    translation = torch.linalg.vector_norm(prediction.position - target_position, dim=-1)
    rotation = rotation_geodesic_angle(
        prediction.rotation, quaternion_xyzw_to_matrix(target_quaternion)
    )
    joint_rmse = torch.sqrt(
        (
            (prediction.joint_position - target_joint)
            / model.joint_travel_range
        )
        .square()
        .mean(dim=-1)
    )
    dx = torch.sqrt(
        (translation / model.length_scale).square()
        + rotation.square()
        + joint_rmse.square()
    )
    predicted_aperture = model.aperture_from_joints(prediction.joint_position)
    return {
        "translation_m": translation.cpu().numpy(),
        "rotation_rad": rotation.cpu().numpy(),
        "joint_rmse_over_travel": joint_rmse.cpu().numpy(),
        "dx": dx.cpu().numpy(),
        "aperture_error_m": (
            predicted_aperture
            - torch.from_numpy(selected.preserve_aperture).to(device)
        ).abs().cpu().numpy(),
        "active": aux.active.cpu().numpy(),
        "trial_min_contact_gap_m": aux.trial_gap.amin(dim=-1).cpu().numpy(),
    }


def _fresh_reset_successor(
    app: Any,
    config: Any,
    catalog: Any,
    gripper: Any,
    manifest: Any,
    selected: TransitionSet,
) -> dict[str, np.ndarray]:
    import carb
    import omni.usd
    import torch
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext
    from isaaclab.utils.math import (
        quat_apply,
        quat_mul,
        subtract_frame_transforms,
    )

    from srno.sim.collector import QuasistaticCollector
    from srno.sim.isaac_scene import (
        apply_contact_materials,
        make_scene_cfg,
        make_simulation_cfg,
    )
    from srno.sim.pose_seeds import PoseSeeds

    settings = carb.settings.get_settings()
    settings.set("/renderer/multiGPU/enabled", False)
    settings.set("/renderer/multiGPU/autoEnable", False)
    settings.set("/rtx/realtime/mgpu/autoTiling/enabled", False)
    context = omni.usd.get_context()
    context.new_stage()
    for _ in range(10):
        app.update()
    simulation = SimulationContext(make_simulation_cfg(config.device))
    scene = None
    collector = None
    try:
        record = catalog.object(selected.object_id)
        scene = InteractiveScene(
            make_scene_cfg(
                catalog,
                record,
                num_envs=len(selected.trajectory),
                relaxation=config.relaxation,
            )
        )
        apply_contact_materials(scene)
        simulation.reset()
        collector = QuasistaticCollector(
            simulation, scene, catalog, record, config, gripper
        )
        device = torch.device(scene.device)
        count = len(selected.trajectory)
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
        current_joint = torch.from_numpy(selected.current_joint).to(device)
        collector.robot.write_joint_state_to_sim(
            current_joint, torch.zeros_like(current_joint)
        )

        current_position = torch.from_numpy(selected.current_position).to(device)
        current_quaternion = torch.from_numpy(
            selected.current_quaternion_xyzw[:, (3, 0, 1, 2)]
        ).to(device)
        object_state = collector.object.data.root_state_w.clone()
        object_state[:, :3] = base_position + quat_apply(
            base_quaternion, current_position
        )
        object_state[:, 3:7] = quat_mul(base_quaternion, current_quaternion)
        object_state[:, 7:13] = 0.0
        collector.object.write_root_state_to_sim(object_state)
        collector._hold_root()

        next_command = torch.as_tensor(
            np.asarray(manifest.commanded_aperture_m, dtype=np.float32)[
                selected.current_step + 1
            ],
            device=device,
        )
        next_target = gripper.to(device).free_joint_configuration(next_command)
        required = torch.ones(count, dtype=torch.bool, device=device)
        result = collector._settle_command(
            next_target,
            int(np.max(selected.current_step) + 1),
            required_mask=required,
        )
        reset_position, reset_quaternion = subtract_frame_transforms(
            base_position,
            base_quaternion,
            result.object_position,
            result.object_quaternion_wxyz,
        )
        reset_aperture = collector._actual_aperture(result.joint_position)
        return {
            "settled": result.settled_mask.cpu().numpy(),
            "position": reset_position.cpu().numpy(),
            "quaternion_wxyz": reset_quaternion.cpu().numpy(),
            "joint": result.joint_position.cpu().numpy(),
            "aperture": reset_aperture.cpu().numpy(),
            "contact_count": result.contact_count.cpu().numpy(),
            "settling_control_steps": result.environment_steps.cpu().numpy(),
            "residual_linear_velocity_m_s": result.linear_velocity.cpu().numpy(),
            "residual_angular_velocity_rad_s": result.angular_velocity.cpu().numpy(),
        }
    finally:
        del collector
        del scene
        simulation.clear_all_callbacks()
        simulation.clear_instance()
        del simulation
        gc.collect()
        context.close_stage()
        for _ in range(5):
            app.update()


def _make_plots(output: Path, rows: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reset_dx = rows["reset_dx"]
    repeat_dx = rows["repeat_dx"]
    model_dx = rows["model_dx"]
    reset_translation = rows["reset_translation_m"] * 1000.0
    reset_rotation = rows["reset_rotation_rad"] * 180.0 / np.pi
    object_index = rows["object_index"]
    labels = rows["object_labels"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)
    bins = np.linspace(
        0.0,
        float(np.quantile(np.concatenate((reset_dx, repeat_dx, model_dx)), 0.99)),
        35,
    )
    axes[0, 0].hist(reset_dx, bins=bins, alpha=0.65, label="fresh reset → preserve")
    axes[0, 0].hist(repeat_dx, bins=bins, alpha=0.65, label="fresh reset A → B")
    axes[0, 0].hist(model_dx, bins=bins, alpha=0.65, label="best-local → preserve")
    axes[0, 0].axvline(0.0251530744, color="black", linestyle="--", label="global local val mean")
    axes[0, 0].set_xlabel(r"$d_X$")
    axes[0, 0].set_ylabel("transitions")
    axes[0, 0].legend()

    axes[0, 1].scatter(reset_translation, reset_rotation, c=rows["current_step"], cmap="viridis", s=28)
    axes[0, 1].set_xlabel("reset/preserve translation [mm]")
    axes[0, 1].set_ylabel("reset/preserve rotation [deg]")
    axes[0, 1].set_title("colour = current command step")

    grouped = [reset_dx[object_index == index] for index in range(len(labels))]
    axes[1, 0].boxplot(grouped, tick_labels=[str(x)[:22] for x in labels], showfliers=True)
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].set_ylabel(r"fresh-reset $d_X$")

    axes[1, 1].scatter(model_dx, reset_dx, c=rows["current_step"], cmap="viridis", s=28)
    upper = float(np.quantile(np.concatenate((reset_dx, model_dx)), 0.99))
    axes[1, 1].plot((0, upper), (0, upper), "k--", linewidth=1)
    axes[1, 1].set_xlim(0, upper)
    axes[1, 1].set_ylim(0, upper)
    axes[1, 1].set_xlabel(r"best-local one-step $d_X$")
    axes[1, 1].set_ylabel(r"fresh-reset/preserve $d_X$")
    fig.savefig(output / "markov_state_sufficiency.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sim-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--samples-per-object", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.samples_per_object <= 0:
        parser.error("--samples-per-object must be positive")

    from isaaclab.app import AppLauncher
    from srno.sim.memory_guard import MemoryWatchdog

    watchdog = MemoryWatchdog(14.0, 0.25)
    watchdog.start()
    app = AppLauncher({"headless": True, "device": "cuda:0"}).app
    try:
        import torch

        from srno.data.schema import DatasetManifest
        from srno.geometry.gripper import GripperAsset
        from srno.model import SRNOModel
        from srno.sim.assets import SimulatorAssetCatalog
        from srno.sim.config import SimulatorConfig
        from srno.training.checkpoint import load_checkpoint

        manifest = DatasetManifest.load(args.manifest)
        config = SimulatorConfig.load(args.sim_config)
        catalog = SimulatorAssetCatalog.load(config.catalog)
        gripper = GripperAsset.load(manifest.gripper_path)
        if gripper.sha256() != manifest.gripper_sha256:
            raise ValueError("manifest gripper hash mismatch")
        objects = args.objects or list(manifest.splits["val"] + manifest.splits["test"])
        unknown = set(objects) - set(manifest.object_locations())
        if unknown:
            raise ValueError(f"objects are absent from manifest: {sorted(unknown)}")

        device = torch.device(config.device)
        model = SRNOModel(
            gripper,
            sdf_scale=manifest.sdf_scale_m,
            delta_gate=manifest.delta_gate_m,
            contact_offset_sum=manifest.contact_offset_sum_m,
        ).to(device)
        checkpoint = load_checkpoint(args.checkpoint, model=model, map_location=device)
        if checkpoint["manifest_sha256"] != manifest.sha256():
            raise ValueError("checkpoint was trained on a different manifest")
        model.eval()

        selections: list[TransitionSet] = []
        model_outputs: list[dict[str, np.ndarray]] = []
        reset_outputs: list[dict[str, np.ndarray]] = []
        repeat_outputs: list[dict[str, np.ndarray]] = []
        for object_offset, object_id in enumerate(objects):
            selected = _select_transitions(
                manifest,
                object_id,
                count=args.samples_per_object,
                seed=args.seed + object_offset,
                joint_names=gripper.joint_names,
            )
            selections.append(selected)
            model_outputs.append(
                _model_errors(selected, manifest, gripper, model, device=device)
            )
            print(
                f"[MARKOV] {object_id}: fresh-stage reset of "
                f"{len(selected.trajectory)} contact transitions",
                flush=True,
            )
            reset_outputs.append(
                _fresh_reset_successor(
                    app, config, catalog, gripper, manifest, selected
                )
            )
            print(
                f"[MARKOV] {object_id}: identical fresh-stage repeat control",
                flush=True,
            )
            repeat_outputs.append(
                _fresh_reset_successor(
                    app, config, catalog, gripper, manifest, selected
                )
            )

        aggregate: dict[str, list[np.ndarray]] = {
            key: []
            for key in (
                "object_index", "trajectory", "source_pose_index", "current_step", "current_lag_m",
                "preserve_current_contact_count", "preserve_next_contact_count",
                "reset_settled", "reset_contact_count", "reset_settling_control_steps",
                "reset_translation_m", "reset_rotation_rad",
                "reset_joint_rmse_over_travel", "reset_dx", "reset_aperture_error_m",
                "repeat_settled", "repeat_translation_m", "repeat_rotation_rad",
                "repeat_joint_rmse_over_travel", "repeat_dx", "repeat_aperture_error_m",
                "model_translation_m", "model_rotation_rad",
                "model_joint_rmse_over_travel", "model_dx", "model_aperture_error_m",
                "model_active", "model_trial_min_contact_gap_m",
            )
        }
        per_object: dict[str, Any] = {}
        for object_index, (selected, model_result, reset_result, repeat_result) in enumerate(
            zip(selections, model_outputs, reset_outputs, repeat_outputs, strict=True)
        ):
            import torch

            reset_errors = _state_errors(
                torch.from_numpy(reset_result["position"]).to(device),
                torch.from_numpy(reset_result["quaternion_wxyz"]).to(device),
                torch.from_numpy(reset_result["joint"]).to(device),
                torch.from_numpy(selected.preserve_position).to(device),
                torch.from_numpy(
                    selected.preserve_quaternion_xyzw[:, (3, 0, 1, 2)]
                ).to(device),
                torch.from_numpy(selected.preserve_joint).to(device),
                length_scale=gripper.length_scale,
                joint_scale=gripper.joint_travel_range.to(device),
            )
            reset_numpy = {
                name: value.detach().cpu().numpy()
                for name, value in reset_errors.items()
            }
            reset_aperture_error = np.abs(
                reset_result["aperture"] - selected.preserve_aperture
            )
            repeat_errors = _state_errors(
                torch.from_numpy(repeat_result["position"]).to(device),
                torch.from_numpy(repeat_result["quaternion_wxyz"]).to(device),
                torch.from_numpy(repeat_result["joint"]).to(device),
                torch.from_numpy(reset_result["position"]).to(device),
                torch.from_numpy(reset_result["quaternion_wxyz"]).to(device),
                torch.from_numpy(reset_result["joint"]).to(device),
                length_scale=gripper.length_scale,
                joint_scale=gripper.joint_travel_range.to(device),
            )
            repeat_numpy = {
                name: value.detach().cpu().numpy()
                for name, value in repeat_errors.items()
            }
            repeat_aperture_error = np.abs(
                repeat_result["aperture"] - reset_result["aperture"]
            )
            count = len(selected.trajectory)
            values = {
                "object_index": np.full(count, object_index, dtype=np.int16),
                "trajectory": selected.trajectory,
                "source_pose_index": selected.source_pose_index,
                "current_step": selected.current_step,
                "current_lag_m": selected.current_lag_m,
                "preserve_current_contact_count": selected.current_contact_count,
                "preserve_next_contact_count": selected.next_contact_count,
                "reset_settled": reset_result["settled"],
                "reset_contact_count": reset_result["contact_count"],
                "reset_settling_control_steps": reset_result["settling_control_steps"],
                "reset_translation_m": reset_numpy["translation_m"],
                "reset_rotation_rad": reset_numpy["rotation_rad"],
                "reset_joint_rmse_over_travel": reset_numpy["joint_rmse_over_travel"],
                "reset_dx": reset_numpy["dx"],
                "reset_aperture_error_m": reset_aperture_error,
                "repeat_settled": repeat_result["settled"],
                "repeat_translation_m": repeat_numpy["translation_m"],
                "repeat_rotation_rad": repeat_numpy["rotation_rad"],
                "repeat_joint_rmse_over_travel": repeat_numpy["joint_rmse_over_travel"],
                "repeat_dx": repeat_numpy["dx"],
                "repeat_aperture_error_m": repeat_aperture_error,
                "model_translation_m": model_result["translation_m"],
                "model_rotation_rad": model_result["rotation_rad"],
                "model_joint_rmse_over_travel": model_result["joint_rmse_over_travel"],
                "model_dx": model_result["dx"],
                "model_aperture_error_m": model_result["aperture_error_m"],
                "model_active": model_result["active"],
                "model_trial_min_contact_gap_m": model_result["trial_min_contact_gap_m"],
            }
            for name, value in values.items():
                aggregate[name].append(np.asarray(value))
            valid = (
                reset_result["settled"].astype(bool)
                & repeat_result["settled"].astype(bool)
            )
            per_object[selected.object_id] = {
                "samples": count,
                "settled": int(valid.sum()),
                "current_steps": sorted(set(map(int, selected.current_step))),
                "fresh_reset": {
                    name: _stats(value[valid])
                    for name, value in reset_numpy.items()
                },
                "fresh_reset_aperture_error_m": _stats(reset_aperture_error[valid]),
                "fresh_reset_repeatability": {
                    name: _stats(value[valid])
                    for name, value in repeat_numpy.items()
                },
                "fresh_reset_repeat_aperture_error_m": _stats(
                    repeat_aperture_error[valid]
                ),
                "best_local_model": {
                    name: _stats(value[valid])
                    for name, value in model_result.items()
                    if name not in {"active"}
                },
                "model_active_fraction": float(model_result["active"][valid].mean()),
            }

        arrays = {name: np.concatenate(parts) for name, parts in aggregate.items()}
        valid = (
            arrays["reset_settled"].astype(bool)
            & arrays["repeat_settled"].astype(bool)
        )
        filtered = {name: value[valid] for name, value in arrays.items()}
        reset_dx = filtered["reset_dx"]
        model_dx = filtered["model_dx"]
        benchmark = 0.02515307441353798
        summary = {
            "definition": {
                "preserve": "recorded successor from uninterrupted collector history",
                "reset": "fresh USD stage; restore only q_k,r_k; zero velocities; identical next command and settling",
                "selection": "contact at incoming and outgoing increments, current aperture lag > 0.1 mm",
                "dx": "sqrt((||dp||/L)^2 + d_SO3^2 + mean((dr/s)^2))",
            },
            "dataset": {
                "manifest": str(args.manifest.resolve()),
                "manifest_sha256": manifest.sha256(),
                "checkpoint": str(args.checkpoint.resolve()),
                "objects": objects,
                "samples_per_object": args.samples_per_object,
                "total_samples": int(len(valid)),
                "settled_samples": int(valid.sum()),
            },
            "aggregate": {
                "fresh_reset": {
                    "dx": _stats(filtered["reset_dx"]),
                    "translation_m": _stats(filtered["reset_translation_m"]),
                    "rotation_rad": _stats(filtered["reset_rotation_rad"]),
                    "joint_rmse_over_travel": _stats(filtered["reset_joint_rmse_over_travel"]),
                    "aperture_error_m": _stats(filtered["reset_aperture_error_m"]),
                },
                "fresh_reset_repeatability": {
                    "dx": _stats(filtered["repeat_dx"]),
                    "translation_m": _stats(filtered["repeat_translation_m"]),
                    "rotation_rad": _stats(filtered["repeat_rotation_rad"]),
                    "joint_rmse_over_travel": _stats(filtered["repeat_joint_rmse_over_travel"]),
                    "aperture_error_m": _stats(filtered["repeat_aperture_error_m"]),
                },
                "best_local_model": {
                    "dx": _stats(filtered["model_dx"]),
                    "translation_m": _stats(filtered["model_translation_m"]),
                    "rotation_rad": _stats(filtered["model_rotation_rad"]),
                    "joint_rmse_over_travel": _stats(filtered["model_joint_rmse_over_travel"]),
                    "aperture_error_m": _stats(filtered["model_aperture_error_m"]),
                    "active_fraction": float(filtered["model_active"].mean()),
                },
                "comparison": {
                    "global_local_validation_dx": benchmark,
                    "fresh_reset_dx_over_global_local_validation_mean": float(reset_dx.mean() / benchmark),
                    "fresh_reset_dx_over_same_sample_model_mean": float(reset_dx.mean() / model_dx.mean()),
                    "fresh_reset_dx_over_repeatability_mean": float(reset_dx.mean() / filtered["repeat_dx"].mean()),
                    "fraction_fresh_reset_dx_ge_global_local_validation_mean": float(np.mean(reset_dx >= benchmark)),
                    "fraction_fresh_reset_dx_ge_same_sample_model_dx": float(np.mean(reset_dx >= model_dx)),
                    "pearson_reset_vs_model_dx": float(np.corrcoef(reset_dx, model_dx)[0, 1]),
                },
            },
            "per_object": per_object,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "results.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        np.savez_compressed(
            args.output / "samples.npz",
            **arrays,
            object_labels=np.asarray(objects),
        )
        plot_arrays = dict(filtered)
        plot_arrays["object_labels"] = np.asarray(objects)
        _make_plots(args.output, plot_arrays)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    finally:
        app.close()
        watchdog.stop()


if __name__ == "__main__":
    main()
