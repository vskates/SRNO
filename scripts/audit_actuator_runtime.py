#!/usr/bin/env python3
"""Audit the live six-joint implicit drive and record one closure trace."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _plot(output: Path, arrays: dict[str, np.ndarray]) -> None:
    step = np.arange(len(arrays["joint_position"]))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].plot(step, arrays["target_joint"], linestyle="--", alpha=0.7)
    axes[0].plot(step, arrays["joint_position"], linewidth=1.2)
    axes[0].set_ylabel("joint position [rad]")
    axes[0].set_title("Dashed: target; solid: settled actual")
    axes[0].grid(alpha=0.25)
    axes[1].plot(step, arrays["approximate_pd_effort"])
    axes[1].set_xlabel("closure command")
    axes[1].set_ylabel("approximate clipped PD effort")
    axes[1].grid(alpha=0.25)
    fig.savefig(output / "actuator_trace.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sim-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    from srno.sim.memory_guard import MemoryWatchdog

    watchdog = MemoryWatchdog(14.0, 0.25)
    watchdog.start()
    app = AppLauncher({"headless": True, "device": "cuda:0"}).app
    try:
        import gc
        import h5py
        import omni.usd
        import torch
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext

        from srno.data.schema import DatasetManifest
        from srno.geometry.gripper import GripperAsset
        from srno.sim.actuator_audit import verify_runtime_actuator
        from srno.sim.assets import SimulatorAssetCatalog
        from srno.sim.collector import QuasistaticCollector
        from srno.sim.config import SimulatorConfig
        from srno.sim.isaac_scene import (
            GRIPPER_EFFORT_LIMIT,
            GRIPPER_STIFFNESS,
            apply_contact_materials,
            make_scene_cfg,
            make_simulation_cfg,
        )
        from srno.sim.physx_material import PhysxMaterialAudit, expected_physics_metadata
        from srno.sim.pose_seeds import PoseSeeds

        manifest = DatasetManifest.load(args.manifest)
        config = replace(
            SimulatorConfig.load(args.sim_config),
            headless=True,
            num_envs=1,
        )
        catalog = SimulatorAssetCatalog.load(config.catalog)
        gripper = GripperAsset.load(manifest.gripper_path)
        if gripper.sha256() != manifest.gripper_sha256:
            raise ValueError("manifest gripper hash mismatch")
        object_id = manifest.splits["train"][0]
        record = catalog.object(object_id)
        shard_path, group_name = manifest.object_locations()[object_id]
        with h5py.File(shard_path, "r") as shard:
            # Reuse a trajectory that is known to have satisfied the frozen
            # material-v2 settling contract, instead of sampling a merely
            # validation-successful pose that the quasistatic collector may
            # legitimately reject.
            source_pose_index = int(shard[group_name]["source_pose_index"][0])
        seeds = PoseSeeds.load(record.pose_seed_path).take(
            [source_pose_index], require_successful=True
        )

        omni.usd.get_context().new_stage()
        for _ in range(10):
            app.update()
        simulation = SimulationContext(
            make_simulation_cfg(config.device, config.material)
        )
        scene = None
        material_audit = PhysxMaterialAudit(expected_physics_metadata(config))
        try:
            material_audit.start()
            scene = InteractiveScene(
                make_scene_cfg(
                    catalog,
                    record,
                    num_envs=1,
                    relaxation=config.relaxation,
                )
            )
            apply_contact_materials(scene, config.material)
            material_audit.force_load()
            simulation.reset()
            material_audit.verify(app)
            contract = verify_runtime_actuator(
                scene["robot"],
                expected_joint_names=tuple(gripper.joint_names),
                relaxation=config.relaxation,
                position_target=gripper.free_joint_knots[-1],
            )
            collector = QuasistaticCollector(
                simulation, scene, catalog, record, config, gripper
            )
            base_position, base_quaternion = collector._reset_batch(
                torch.from_numpy(seeds.position_m).to(collector.device),
                torch.from_numpy(seeds.quaternion_wxyz).to(collector.device),
            )
            del base_position, base_quaternion

            targets: list[np.ndarray] = []
            positions: list[np.ndarray] = []
            velocities: list[np.ndarray] = []
            runtime_computed: list[np.ndarray] = []
            runtime_applied: list[np.ndarray] = []
            settled_formula: list[np.ndarray] = []
            settled_formula_clipped: list[np.ndarray] = []
            substeps: list[int] = []
            settled: list[bool] = []
            apertures = collector.commanded_aperture.detach().cpu().numpy()
            for command_index in range(33):
                fraction = float(command_index) / 32.0
                target = collector.close_joint_target * fraction
                result = collector._settle_command(target, command_index)
                joint_position = result.joint_position[0]
                joint_velocity = result.joint_velocity[0]
                pd = (
                    GRIPPER_STIFFNESS * (target - joint_position)
                    - config.relaxation.gripper_damping * joint_velocity
                )
                pd_clipped = pd.clamp(
                    min=-GRIPPER_EFFORT_LIMIT, max=GRIPPER_EFFORT_LIMIT
                )
                # ``computed_torque`` is evaluated by IsaacLab immediately
                # before the final physics step, whereas the settled r/r_dot
                # below are read immediately after that step.  First verify
                # the exact same-instant clipping identity, then bound this
                # expected one-step diagnostic stagger separately.
                torch.testing.assert_close(
                    result.approximate_pd_effort_clipped[0],
                    result.approximate_pd_effort[0].clamp(
                        min=-GRIPPER_EFFORT_LIMIT,
                        max=GRIPPER_EFFORT_LIMIT,
                    ),
                    rtol=0.0,
                    atol=1e-7,
                )
                targets.append(target.detach().cpu().numpy())
                positions.append(joint_position.cpu().numpy())
                velocities.append(joint_velocity.cpu().numpy())
                runtime_computed.append(
                    result.approximate_pd_effort[0].cpu().numpy()
                )
                runtime_applied.append(
                    result.approximate_pd_effort_clipped[0].cpu().numpy()
                )
                settled_formula.append(pd.cpu().numpy())
                settled_formula_clipped.append(pd_clipped.cpu().numpy())
                substeps.append(int(result.environment_steps[0].item()) * 2)
                settled.append(bool(result.settled_mask[0].item()))
                if not settled[-1]:
                    raise RuntimeError(
                        f"actuator audit trajectory did not settle at command {command_index}"
                    )

            arrays = {
                "commanded_aperture_m": np.asarray(apertures, dtype=np.float32),
                "target_joint": np.asarray(targets, dtype=np.float32),
                "joint_position": np.asarray(positions, dtype=np.float32),
                "joint_velocity": np.asarray(velocities, dtype=np.float32),
                "runtime_computed_pd_effort": np.asarray(
                    runtime_computed, dtype=np.float32
                ),
                # Required semantics: this is robot.data.applied_torque from
                # an implicit drive and therefore only an approximate clipped
                # PD diagnostic, never a measured contact reaction.
                "approximate_pd_effort": np.asarray(
                    runtime_applied, dtype=np.float32
                ),
                "settled_state_pd_effort": np.asarray(
                    settled_formula, dtype=np.float32
                ),
                "settled_state_pd_effort_clipped": np.asarray(
                    settled_formula_clipped, dtype=np.float32
                ),
                "settling_substeps": np.asarray(substeps, dtype=np.int32),
                "settled": np.asarray(settled, dtype=np.bool_),
            }
            max_formula_error = float(
                np.max(
                    np.abs(
                        arrays["runtime_computed_pd_effort"]
                        - arrays["settled_state_pd_effort"]
                    )
                )
            )
            # At 120 Hz and |r_dot|<=0.1 rad/s, even the stiffness part of a
            # one-substep stagger can differ by 14*0.1/120 = 0.0117 N m.
            # The bound remains tiny relative to the 480 N m drive limit.
            pd_stagger_tolerance = 0.02
            if max_formula_error > pd_stagger_tolerance:
                raise RuntimeError(
                    "runtime computed PD effort disagrees with the settled-state "
                    f"formula beyond the one-step bound: {max_formula_error} > "
                    f"{pd_stagger_tolerance}"
                )
            output = args.output.resolve()
            output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output / "actuator_trace.npz", **arrays)
            result_json = {
                "definition": {
                    "tau_pd": "clip(K*(r_bar-r)-D*r_dot, -tau_max, tau_max)",
                    "effort_semantics": (
                        "approximate PD diagnostic; not a measured PhysX "
                        "contact reaction"
                    ),
                    "runtime_timing": (
                        "IsaacLab computed/applied effort is evaluated before the "
                        "physics step; settled r and r_dot are read after it"
                    ),
                },
                "configuration": {
                    "manifest": str(args.manifest.resolve()),
                    "manifest_sha256": manifest.sha256(),
                    "gripper_sha256": manifest.gripper_sha256,
                    "sim_config": str(args.sim_config.resolve()),
                    "sim_config_sha256": config.sha256(),
                    "object_id": object_id,
                    "source_pose_index": int(seeds.source_index[0]),
                },
                "runtime_contract": contract,
                "states": 33,
                "all_settled": bool(np.all(arrays["settled"])),
                "max_runtime_vs_settled_pd_formula_abs_error": max_formula_error,
                "pd_stagger_tolerance": pd_stagger_tolerance,
                "max_abs_approximate_effort": float(
                    np.max(np.abs(arrays["approximate_pd_effort"]))
                ),
                "max_abs_joint_velocity_rad_s": float(
                    np.max(np.abs(arrays["joint_velocity"]))
                ),
            }
            (output / "results.json").write_text(
                json.dumps(result_json, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _plot(output, arrays)
            (output / "audit-failure.txt").unlink(missing_ok=True)
            print(json.dumps(result_json, indent=2, sort_keys=True), flush=True)
        finally:
            material_audit.close()
            del scene
            simulation.clear_all_callbacks()
            simulation.clear_instance()
            del simulation
            gc.collect()
            omni.usd.get_context().close_stage()
            for _ in range(5):
                app.update()
    except BaseException:
        failure = traceback.format_exc()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "audit-failure.txt").write_text(failure, encoding="utf-8")
        print(failure, file=sys.stderr, flush=True)
        # Kit's shutdown path can turn an active exception into exit code 0.
        # Persist the traceback first, then leave cleanup to the OS so the
        # physics-contract audit remains genuinely fail-fast.
        watchdog.stop()
        os._exit(1)
    finally:
        app.close()
        watchdog.stop()


if __name__ == "__main__":
    main()
