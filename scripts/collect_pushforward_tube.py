#!/usr/bin/env python3
"""Prepare and PhysX-label a small train-only pushforward tube.

``prepare`` rolls a checkpoint through recorded training trajectories and
selects model-induced states from four horizon bands.  ``collect`` restores
those states in fresh PhysX scenes and computes their actual one-command
successors.  This is deliberately separate from the production trajectory
collector: it cannot write the canonical dataset and it rejects val/test
objects.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Any

import h5py
import numpy as np


BANDS: tuple[tuple[int, int], ...] = ((1, 8), (9, 16), (17, 24), (25, 31))


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    quaternion = Rotation.from_matrix(matrix).as_quat().astype(np.float32)
    # Give every request a deterministic representative of SO(3)/{+q,-q}.
    flip = quaternion[:, 3] < 0.0
    quaternion[flip] *= -1.0
    return quaternion


def _prepare(args: argparse.Namespace) -> None:
    import torch

    from srno.data.index import file_sha256
    from srno.data.schema import DatasetManifest
    from srno.geometry.se3 import quaternion_xyzw_to_matrix, rotation_geodesic_angle
    from srno.training.checkpoint import load_checkpoint
    from srno.training.config import ExperimentConfig
    from srno.training.engine import _build_model
    from srno.types import PoseState, SDFBatch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = ExperimentConfig.load(args.config)
    manifest = DatasetManifest.load(config.paths.manifest)
    checkpoint_path = args.checkpoint.resolve()
    device = torch.device(args.device)
    model = _build_model(config, manifest, device)
    checkpoint = load_checkpoint(checkpoint_path, model=model, map_location=device)
    if checkpoint["manifest_sha256"] != manifest.sha256():
        raise ValueError("checkpoint and manifest hashes differ")
    model.eval()

    requested: dict[str, list[np.ndarray | int | float | str]] = {
        name: []
        for name in (
            "object_id", "trajectory", "source_pose_index", "current_step",
            "model_position", "model_quaternion_xyzw", "model_joint",
            "nominal_current_position", "nominal_current_quaternion_xyzw",
            "nominal_current_joint", "nominal_successor_position",
            "nominal_successor_quaternion_xyzw", "nominal_successor_joint",
            "nominal_successor_aperture", "pushforward_dx",
        )
    }
    locations = manifest.object_locations()
    schedule = torch.tensor(manifest.commanded_aperture_m, device=device)
    for object_offset, object_id in enumerate(manifest.splits["train"]):
        shard, group_name = locations[object_id]
        with h5py.File(shard, "r", swmr=True) as handle:
            group = handle[group_name]
            positions = np.asarray(group["position"], dtype=np.float32)
            quaternions = np.asarray(group["quaternion_xyzw"], dtype=np.float32)
            joints = np.asarray(group["joint_position"], dtype=np.float32)
            apertures = np.asarray(group["actual_aperture"], dtype=np.float32)
            sources = np.asarray(group["source_pose_index"], dtype=np.int64)
            sdf_values = np.asarray(group["sdf"], dtype=np.float32)
            sdf_origin = np.asarray(group.attrs["grid_origin"], dtype=np.float32)
            sdf_voxel = np.asarray(group.attrs["voxel_size"], dtype=np.float32)
        generator = np.random.default_rng(args.seed + object_offset)
        trajectory_count = len(positions)
        sample_count = min(args.candidate_trajectories, trajectory_count)
        trajectories = np.sort(
            generator.choice(trajectory_count, size=sample_count, replace=False)
        )
        initial = PoseState(
            quaternion_xyzw_to_matrix(
                torch.from_numpy(quaternions[trajectories, 0]).to(device)
            ),
            torch.from_numpy(positions[trajectories, 0]).to(device),
            torch.from_numpy(joints[trajectories, 0]).to(device),
        )
        sdf = SDFBatch(
            torch.from_numpy(sdf_values[None]).to(device),
            torch.from_numpy(sdf_origin[None]).to(device),
            torch.from_numpy(sdf_voxel[None]).to(device),
            torch.zeros(sample_count, dtype=torch.long, device=device),
            manifest.sdf_scale_m,
        )
        with torch.no_grad():
            prediction = model.rollout(initial, schedule[1:], sdf)
        predicted_position = prediction.position.detach().cpu().numpy()
        predicted_rotation = prediction.rotation.detach().cpu().numpy()
        predicted_joint = prediction.joint_position.detach().cpu().numpy()
        predicted_quaternion = _matrix_to_quaternion_xyzw(
            predicted_rotation.reshape(-1, 3, 3)
        ).reshape(sample_count, 33, 4)
        target_rotation = quaternion_xyzw_to_matrix(
            torch.from_numpy(quaternions[trajectories]).to(device)
        )
        with torch.no_grad():
            translation = torch.linalg.vector_norm(
                prediction.position
                - torch.from_numpy(positions[trajectories]).to(device),
                dim=-1,
            ) / model.length_scale
            rotation = rotation_geodesic_angle(prediction.rotation, target_rotation)
            joint = torch.sqrt(
                (
                    (prediction.joint_position
                     - torch.from_numpy(joints[trajectories]).to(device))
                    / model.joint_travel_range
                ).square().mean(dim=-1)
            )
            dx = torch.sqrt(translation.square() + rotation.square() + joint.square())
        dx_numpy = dx.cpu().numpy()

        selected: list[tuple[int, int]] = []
        for first, last in BANDS:
            candidates = [
                (trajectory_local, step)
                for trajectory_local in range(sample_count)
                for step in range(first, last + 1)
            ]
            # The largest drift is the most informative part of the induced
            # measure; stable tie-breaking makes the request reproducible.
            candidates.sort(
                key=lambda item: (-float(dx_numpy[item]), item[0], item[1])
            )
            selected.extend(candidates[: args.states_per_band])
        for trajectory_local, step in selected:
            trajectory = int(trajectories[trajectory_local])
            values: dict[str, Any] = {
                "object_id": object_id,
                "trajectory": trajectory,
                "source_pose_index": int(sources[trajectory]),
                "current_step": step,
                "model_position": predicted_position[trajectory_local, step],
                "model_quaternion_xyzw": predicted_quaternion[trajectory_local, step],
                "model_joint": predicted_joint[trajectory_local, step],
                "nominal_current_position": positions[trajectory, step],
                "nominal_current_quaternion_xyzw": quaternions[trajectory, step],
                "nominal_current_joint": joints[trajectory, step],
                "nominal_successor_position": positions[trajectory, step + 1],
                "nominal_successor_quaternion_xyzw": quaternions[trajectory, step + 1],
                "nominal_successor_joint": joints[trajectory, step + 1],
                "nominal_successor_aperture": float(apertures[trajectory, step + 1]),
                "pushforward_dx": float(dx_numpy[trajectory_local, step]),
            }
            for name, value in values.items():
                requested[name].append(value)
        print(
            f"[TUBE prepare] {object_id}: candidates={sample_count * 31} "
            f"selected={len(selected)} max_dX={max(dx_numpy[:, 1:32].reshape(-1)):.6f}",
            flush=True,
        )

    arrays = {name: np.asarray(values) for name, values in requested.items()}
    arrays.update(
        manifest_sha256=np.asarray(manifest.sha256()),
        checkpoint_sha256=np.asarray(file_sha256(checkpoint_path)),
        checkpoint_path=np.asarray(str(checkpoint_path)),
        split=np.asarray("train"),
        command_schedule=np.asarray(manifest.commanded_aperture_m, dtype=np.float32),
    )
    _atomic_savez(args.requests, **arrays)
    summary = {
        "phase": "prepare",
        "definition": "model-induced x_hat_k; no nominal labels are reused as successors",
        "config": str(args.config.resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "manifest_sha256": manifest.sha256(),
        "split": "train",
        "objects": len(manifest.splits["train"]),
        "candidate_trajectories_per_object": args.candidate_trajectories,
        "states_per_band": args.states_per_band,
        "bands": BANDS,
        "requests": len(arrays["current_step"]),
        "pushforward_dx": {
            "mean": float(arrays["pushforward_dx"].mean()),
            "median": float(np.median(arrays["pushforward_dx"])),
            "min": float(arrays["pushforward_dx"].min()),
            "max": float(arrays["pushforward_dx"].max()),
        },
        "output": str(args.requests.resolve()),
        "output_sha256": _sha256(args.requests),
    }
    _write_json(args.requests.with_suffix(".json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _fresh_successors_for_object(
    app: Any,
    config: Any,
    catalog: Any,
    gripper: Any,
    manifest: Any,
    object_id: str,
    source_pose_index: np.ndarray,
    current_step: np.ndarray,
    position: np.ndarray,
    quaternion_xyzw: np.ndarray,
    joint: np.ndarray,
) -> dict[str, np.ndarray]:
    import carb
    import omni.usd
    import torch
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext
    from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms

    from srno.sim.collector import QuasistaticCollector
    from srno.sim.isaac_scene import apply_contact_materials, make_scene_cfg, make_simulation_cfg
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
        record = catalog.object(object_id)
        scene = InteractiveScene(
            make_scene_cfg(catalog, record, num_envs=len(current_step), relaxation=config.relaxation)
        )
        apply_contact_materials(scene)
        simulation.reset()
        collector = QuasistaticCollector(simulation, scene, catalog, record, config, gripper)
        device = torch.device(scene.device)
        seeds = PoseSeeds.load(record.pose_seed_path)
        base_position = scene.env_origins + torch.from_numpy(
            seeds.position_m[source_pose_index]
        ).to(device)
        base_quaternion = torch.from_numpy(
            seeds.quaternion_wxyz[source_pose_index]
        ).to(device)
        collector._root_position, collector._root_quaternion = collector._base_to_root(
            base_position, base_quaternion
        )
        robot_state = collector.robot.data.root_state_w.clone()
        robot_state[:, :3] = collector._root_position
        robot_state[:, 3:7] = collector._root_quaternion
        robot_state[:, 7:13] = 0.0
        collector.robot.write_root_state_to_sim(robot_state)
        current_joint = torch.from_numpy(joint).to(device)
        collector.robot.write_joint_state_to_sim(
            current_joint, torch.zeros_like(current_joint)
        )
        relative_position = torch.from_numpy(position).to(device)
        relative_quaternion = torch.from_numpy(quaternion_xyzw[:, (3, 0, 1, 2)]).to(device)
        object_state = collector.object.data.root_state_w.clone()
        object_state[:, :3] = base_position + quat_apply(base_quaternion, relative_position)
        object_state[:, 3:7] = quat_mul(base_quaternion, relative_quaternion)
        object_state[:, 7:13] = 0.0
        collector.object.write_root_state_to_sim(object_state)
        collector._hold_root()
        command = torch.as_tensor(
            np.asarray(manifest.commanded_aperture_m, dtype=np.float32)[current_step + 1],
            device=device,
        )
        target = gripper.to(device).free_joint_configuration(command)
        result = collector._settle_command(
            target,
            int(np.max(current_step) + 1),
            required_mask=torch.ones(len(current_step), dtype=torch.bool, device=device),
        )
        successor_position, successor_quaternion = subtract_frame_transforms(
            base_position, base_quaternion,
            result.object_position, result.object_quaternion_wxyz,
        )
        successor_quaternion = successor_quaternion / torch.linalg.vector_norm(
            successor_quaternion, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        return {
            "settled": result.settled_mask.cpu().numpy(),
            "successor_position": successor_position.cpu().numpy(),
            "successor_quaternion_xyzw": successor_quaternion[:, (1, 2, 3, 0)].cpu().numpy(),
            "successor_joint": result.joint_position.cpu().numpy(),
            "successor_aperture": collector._actual_aperture(result.joint_position).cpu().numpy(),
            "contact_count": result.contact_count.cpu().numpy(),
            "settling_control_steps": result.environment_steps.cpu().numpy(),
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


def _collect(args: argparse.Namespace) -> None:
    # Isaac Kit may terminate the process with exit status zero when the GPU
    # foundation cannot be created.  Fail explicitly before AppLauncher so an
    # interrupted experiment can never be mistaken for a completed collection.
    gpu_check = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if gpu_check.returncode != 0:
        detail = (gpu_check.stderr or gpu_check.stdout).strip()
        raise RuntimeError(f"NVIDIA driver preflight failed: {detail}")
    from isaaclab.app import AppLauncher
    from srno.sim.memory_guard import MemoryWatchdog

    watchdog = MemoryWatchdog(14.0, 0.25)
    watchdog.start()
    app = AppLauncher({"headless": True, "device": "cuda:0"}).app
    try:
        from srno.data.schema import DatasetManifest
        from srno.geometry.gripper import GripperAsset
        from srno.sim.assets import SimulatorAssetCatalog
        from srno.sim.config import SimulatorConfig

        requests = np.load(args.requests, allow_pickle=False)
        config = SimulatorConfig.load(args.sim_config)
        manifest = DatasetManifest.load(args.manifest)
        if str(requests["manifest_sha256"]) != manifest.sha256():
            raise ValueError("request and manifest hashes differ")
        if str(requests["split"]) != "train":
            raise ValueError("pushforward collection is restricted to train requests")
        object_ids = requests["object_id"].astype(str)
        unknown = set(object_ids) - set(manifest.splits["train"])
        if unknown:
            raise ValueError(f"non-train objects in request: {sorted(unknown)}")
        catalog = SimulatorAssetCatalog.load(config.catalog)
        gripper = GripperAsset.load(manifest.gripper_path)
        if gripper.sha256() != manifest.gripper_sha256:
            raise ValueError("gripper hash mismatch")
        collected: dict[str, np.ndarray] = {
            name: np.zeros_like(requests["current_step"], dtype=dtype)
            for name, dtype in (
                ("settled", np.bool_),
                ("successor_aperture", np.float32),
                ("contact_count", np.float32),
                ("settling_control_steps", np.int32),
            )
        }
        collected["successor_position"] = np.zeros((len(object_ids), 3), dtype=np.float32)
        collected["successor_quaternion_xyzw"] = np.zeros((len(object_ids), 4), dtype=np.float32)
        collected["successor_joint"] = np.zeros((len(object_ids), 6), dtype=np.float32)
        if args.state_source == "model":
            position_key = "model_position"
            quaternion_key = "model_quaternion_xyzw"
            joint_key = "model_joint"
        else:
            position_key = "nominal_current_position"
            quaternion_key = "nominal_current_quaternion_xyzw"
            joint_key = "nominal_current_joint"
        for object_id in manifest.splits["train"]:
            indices = np.flatnonzero(object_ids == object_id)
            if not len(indices):
                continue
            print(f"[TUBE collect] {object_id}: states={len(indices)}", flush=True)
            result = _fresh_successors_for_object(
                app, config, catalog, gripper, manifest, object_id,
                requests["source_pose_index"][indices],
                requests["current_step"][indices],
                requests[position_key][indices],
                requests[quaternion_key][indices],
                requests[joint_key][indices],
            )
            for name, values in result.items():
                collected[name][indices] = values
        arrays = {name: requests[name] for name in requests.files}
        arrays.update(collected)
        arrays["simulator_config_sha256"] = np.asarray(config.sha256())
        arrays["collection_state_source"] = np.asarray(args.state_source)
        _atomic_savez(args.labels, **arrays)
        settled = collected["settled"]
        summary = {
            "phase": "collect",
            "definition": "fresh PhysX successor of each model-induced train state",
            "state_source": args.state_source,
            "requests": str(args.requests.resolve()),
            "requests_sha256": _sha256(args.requests),
            "sim_config": str(args.sim_config.resolve()),
            "sim_config_sha256": config.sha256(),
            "manifest_sha256": manifest.sha256(),
            "objects": int(len(set(object_ids))),
            "samples": int(len(object_ids)),
            "settled": int(settled.sum()),
            "settled_fraction": float(settled.mean()),
            "output": str(args.labels.resolve()),
            "output_sha256": _sha256(args.labels),
        }
        _write_json(args.labels.with_suffix(".json"), summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    finally:
        app.close()
        watchdog.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--requests", type=Path, required=True)
    prepare.add_argument("--device", default="cpu")
    prepare.add_argument("--candidate-trajectories", type=int, default=8)
    prepare.add_argument("--states-per-band", type=int, default=2)
    prepare.add_argument("--seed", type=int, default=0)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--requests", type=Path, required=True)
    collect.add_argument("--labels", type=Path, required=True)
    collect.add_argument("--sim-config", type=Path, required=True)
    collect.add_argument("--manifest", type=Path, required=True)
    collect.add_argument(
        "--state-source", choices=("model", "nominal"), default="model"
    )
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.candidate_trajectories <= 0 or args.states_per_band <= 0:
            parser.error("candidate counts must be positive")
        _prepare(args)
    else:
        _collect(args)


if __name__ == "__main__":
    main()
