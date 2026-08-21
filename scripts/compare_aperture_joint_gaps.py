#!/usr/bin/env python3
"""Compare settled GT gaps from scalar aperture and measured six-joint FK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


OLD_ADMISSIBLE_GAP_M = -0.031241722404956818
QUANTILES = (0.0, 0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995, 1.0)


def _stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(values.size),
        "mean_m": float(values.mean()),
        **{
            f"q{100 * quantile:g}_m": float(np.quantile(values, quantile))
            for quantile in QUANTILES
        },
    }


def _decode_names(values: object) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    )


def _comparison(aperture_gap: np.ndarray, joint_gap: np.ndarray) -> dict[str, object]:
    penetration_a = np.maximum(-aperture_gap, 0.0)
    penetration_r = np.maximum(-joint_gap, 0.0)
    return {
        "aperture_min_gap_m": _stats(aperture_gap),
        "joint_fk_min_gap_m": _stats(joint_gap),
        "joint_minus_aperture_gap_m": _stats(joint_gap - aperture_gap),
        "aperture_penetration_m": _stats(penetration_a),
        "joint_fk_penetration_m": _stats(penetration_r),
        "aperture_zero_surface_violation_fraction": float(np.mean(aperture_gap < 0.0)),
        "joint_fk_zero_surface_violation_fraction": float(np.mean(joint_gap < 0.0)),
        "aperture_penetration_gt_0_5mm_fraction": float(
            np.mean(aperture_gap < -0.0005)
        ),
        "joint_fk_penetration_gt_0_5mm_fraction": float(
            np.mean(joint_gap < -0.0005)
        ),
        "aperture_penetration_gt_1mm_fraction": float(
            np.mean(aperture_gap < -0.001)
        ),
        "joint_fk_penetration_gt_1mm_fraction": float(
            np.mean(joint_gap < -0.001)
        ),
        "aperture_penetration_gt_2mm_fraction": float(
            np.mean(aperture_gap < -0.002)
        ),
        "joint_fk_penetration_gt_2mm_fraction": float(
            np.mean(joint_gap < -0.002)
        ),
        "aperture_old_admissible_violation_fraction": float(
            np.mean(aperture_gap < OLD_ADMISSIBLE_GAP_M)
        ),
        "joint_fk_old_admissible_violation_fraction": float(
            np.mean(joint_gap < OLD_ADMISSIBLE_GAP_M)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/joint-gap-diagnostic")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-limit-gib", type=float, default=14.0)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher
    from srno.sim.memory_guard import MemoryWatchdog

    watchdog = MemoryWatchdog(args.memory_limit_gib, 0.25)
    watchdog.start()
    app = None
    simulation = None
    payloads: list[dict[str, object]] = []
    aperture_parts: list[np.ndarray] = []
    joint_parts: list[np.ndarray] = []
    contact_parts: list[np.ndarray] = []
    free_fk_errors: list[np.ndarray] = []
    max_replay_joint_error = 0.0
    try:
        app = AppLauncher({"headless": True, "device": args.device}).app

        import torch
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext
        from isaaclab.utils.math import quat_apply, subtract_frame_transforms

        from srno.data.schema import DatasetManifest
        from srno.geometry.gripper import GripperAsset
        from srno.geometry.sdf import sample_sdf
        from srno.geometry.se3 import quaternion_xyzw_to_matrix
        from srno.sim import SimulatorAssetCatalog
        from srno.sim.config import RelaxationConfig
        from srno.sim.gripper_geometry import _runtime_link_samples
        from srno.sim.isaac_scene import CONTACT_LINKS, _gripper_cfg, make_simulation_cfg

        manifest = DatasetManifest.load(args.manifest)
        catalog = SimulatorAssetCatalog.load()
        gripper = GripperAsset.load(manifest.gripper_path).to(args.device)
        locations = manifest.object_locations()
        trajectory_counts = []
        for object_id in locations:
            shard, group_name = locations[object_id]
            with h5py.File(shard, "r") as handle:
                group = handle[group_name]
                if "joint_position" not in group:
                    raise ValueError(f"{object_id} has no joint_position dataset")
                trajectory_counts.append(int(group["joint_position"].shape[0]))
        if len(set(trajectory_counts)) != 1:
            raise ValueError("diagnostic currently requires equal trajectories per object")
        environment_count = trajectory_counts[0]

        simulation = SimulationContext(make_simulation_cfg(args.device))
        from isaaclab.scene import InteractiveSceneCfg

        scene_cfg = InteractiveSceneCfg(
            num_envs=environment_count,
            env_spacing=0.25,
            lazy_sensor_update=True,
            replicate_physics=True,
            filter_collisions=True,
        )
        scene_cfg.robot = _gripper_cfg(catalog, RelaxationConfig())
        scene = InteractiveScene(scene_cfg)
        simulation.reset()
        robot = scene["robot"]

        if len(robot.joint_names) != 6:
            raise RuntimeError(f"expected six gripper joints, got {robot.joint_names}")
        body_indices = []
        for link_name in CONTACT_LINKS:
            matches = [
                index for index, name in enumerate(robot.body_names) if name == link_name
            ]
            if len(matches) != 1:
                raise RuntimeError(f"expected one body {link_name}, got {matches}")
            body_indices.append(matches[0])
        local_samples_np = _runtime_link_samples(
            catalog.gripper.runtime_usd,
            CONTACT_LINKS,
            samples_per_link=128,
            seed=0,
        )
        local_samples = [
            torch.from_numpy(points).to(device=args.device, dtype=torch.float32)
            for points in local_samples_np
        ]

        def points_from_joint_position(joint_position: torch.Tensor) -> torch.Tensor:
            nonlocal max_replay_joint_error
            robot.write_joint_state_to_sim(
                joint_position, torch.zeros_like(joint_position)
            )
            robot.set_joint_position_target(joint_position)
            scene.write_data_to_sim()
            simulation.step(render=False)
            scene.update(simulation.get_physics_dt())
            max_replay_joint_error = max(
                max_replay_joint_error,
                float(torch.max(torch.abs(robot.data.joint_pos - joint_position)).item()),
            )
            root_position = robot.data.root_pos_w[:, None, :].expand_as(
                robot.data.body_pos_w
            )
            root_quaternion = robot.data.root_quat_w[:, None, :].expand_as(
                robot.data.body_quat_w
            )
            relative_position, relative_quaternion = subtract_frame_transforms(
                root_position,
                root_quaternion,
                robot.data.body_pos_w,
                robot.data.body_quat_w,
            )
            transformed = []
            for body_index, points in zip(body_indices, local_samples, strict=True):
                quaternion = relative_quaternion[:, body_index]
                position = relative_position[:, body_index]
                count = len(points)
                rotated = quat_apply(
                    quaternion[:, None].expand(-1, count, -1).reshape(-1, 4),
                    points[None].expand(environment_count, -1, -1).reshape(-1, 3),
                ).reshape(environment_count, count, 3)
                transformed.append(rotated + position[:, None])
            return torch.cat(transformed, dim=1)

        open_joint = torch.tensor(
            [catalog.gripper.open_joint_position_rad[name] for name in robot.joint_names],
            device=args.device,
            dtype=torch.float32,
        )
        close_joint = torch.tensor(
            [catalog.gripper.close_joint_position_rad[name] for name in robot.joint_names],
            device=args.device,
            dtype=torch.float32,
        )
        schedule = torch.tensor(
            manifest.commanded_aperture_m, device=args.device, dtype=torch.float32
        )
        with torch.no_grad():
            for state_index, fraction in enumerate(
                torch.linspace(0.0, 1.0, 33, device=args.device)
            ):
                joint = open_joint + fraction * (close_joint - open_joint)
                fk_points = points_from_joint_position(
                    joint[None].expand(environment_count, -1)
                )
                expected = gripper.points(schedule[state_index])
                free_fk_errors.append(
                    torch.linalg.vector_norm(fk_points[0] - expected, dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                )

            for object_id, (shard, group_name) in locations.items():
                with h5py.File(shard, "r") as handle:
                    group = handle[group_name]
                    joint_dataset = group["joint_position"]
                    joint_names = _decode_names(joint_dataset.attrs["joint_names"])
                    if joint_names != tuple(robot.joint_names):
                        raise ValueError(
                            f"{object_id} joint order {joint_names} != runtime {robot.joint_names}"
                        )
                    joint_position = torch.from_numpy(joint_dataset[...]).to(
                        device=args.device, dtype=torch.float32
                    )
                    aperture = torch.from_numpy(group["actual_aperture"][...]).to(
                        device=args.device, dtype=torch.float32
                    )
                    position = torch.from_numpy(group["position"][...]).to(
                        device=args.device, dtype=torch.float32
                    )
                    rotation = quaternion_xyzw_to_matrix(
                        torch.from_numpy(group["quaternion_xyzw"][...]).to(
                            device=args.device, dtype=torch.float32
                        )
                    )
                    sdf = torch.from_numpy(group["sdf"][...]).to(
                        device=args.device, dtype=torch.float32
                    )[None]
                    origin = torch.from_numpy(
                        np.asarray(group.attrs["grid_origin"], dtype=np.float32)
                    ).to(args.device)[None]
                    voxel = torch.from_numpy(
                        np.asarray(group.attrs["voxel_size"], dtype=np.float32)
                    ).to(args.device)[None]
                    contact = np.asarray(
                        group["diagnostics/contact_count"], dtype=np.float32
                    ) > 0.0

                mapping = torch.zeros(
                    environment_count, device=args.device, dtype=torch.long
                )
                aperture_gap = np.empty((environment_count, 32), dtype=np.float32)
                joint_gap = np.empty_like(aperture_gap)
                for state_index in range(1, 33):
                    points_a = gripper.points(aperture[:, state_index])
                    points_r = points_from_joint_position(
                        joint_position[:, state_index]
                    )
                    for output, points in (
                        (aperture_gap, points_a),
                        (joint_gap, points_r),
                    ):
                        relative = points - position[:, state_index, None]
                        points_object = torch.einsum(
                            "bij,bmj->bmi",
                            rotation[:, state_index].transpose(-1, -2),
                            relative,
                        )
                        gaps = sample_sdf(
                            sdf,
                            origin,
                            voxel,
                            points_object,
                            sample_to_object=mapping,
                            outside_value=manifest.sdf_scale_m,
                        )
                        output[:, state_index - 1] = (
                            gaps.amin(dim=-1).detach().cpu().numpy()
                        )
                payloads.append(
                    {
                        "object_id": object_id,
                        "all_states": _comparison(aperture_gap, joint_gap),
                        "contact_states": _comparison(
                            aperture_gap[contact], joint_gap[contact]
                        ),
                        "contact_state_fraction": float(contact.mean()),
                    }
                )
                aperture_parts.append(aperture_gap.reshape(-1))
                joint_parts.append(joint_gap.reshape(-1))
                contact_parts.append(contact.reshape(-1))
    finally:
        if simulation is not None:
            simulation.clear_all_callbacks()
            simulation.clear_instance()
        watchdog.stop()

    aperture_all = np.concatenate(aperture_parts)
    joint_all = np.concatenate(joint_parts)
    contact_all = np.concatenate(contact_parts)
    free_error = np.concatenate(free_fk_errors)
    residual_r = np.maximum(-joint_all, 0.0)
    admissible_r_geometric = -float(
        np.quantile(residual_r, 0.995, method="higher")
    )
    effective_aperture_all = aperture_all - manifest.contact_offset_sum_m
    effective_joint_all = joint_all - manifest.contact_offset_sum_m
    effective_residual_r = np.maximum(-effective_joint_all, 0.0)
    admissible_r_effective = -float(
        np.quantile(effective_residual_r, 0.995, method="higher")
    )
    result = {
        "definitions": {
            "m_a": "min_j SDF_cooked(q^-1 points(actual_aperture)_j)",
            "m_r": "min_j SDF_cooked(q^-1 T_link(actual_joint_position) y_j)",
            "rest_offset_m": 0.0,
            "contact_offset_sum_m": manifest.contact_offset_sum_m,
            "old_admissible_gap_m": OLD_ADMISSIBLE_GAP_M,
        },
        "objects": payloads,
        "aggregate_all_states": _comparison(aperture_all, joint_all),
        "aggregate_contact_states": _comparison(
            aperture_all[contact_all], joint_all[contact_all]
        ),
        "aggregate_effective_all_states": _comparison(
            effective_aperture_all, effective_joint_all
        ),
        "aggregate_effective_contact_states": _comparison(
            effective_aperture_all[contact_all], effective_joint_all[contact_all]
        ),
        "recalibrated_joint_fk_geometric_admissible_gap_q99_5_m": (
            admissible_r_geometric
        ),
        "recalibrated_joint_fk_effective_admissible_gap_q99_5_m": (
            admissible_r_effective
        ),
        "free_schedule_fk_point_error_m": _stats(free_error),
        "max_replayed_joint_position_error_rad": max_replay_joint_error,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output / "arrays.npz",
        aperture_min_gap_m=aperture_all,
        joint_fk_min_gap_m=joint_all,
        contact=contact_all,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8), constrained_layout=True)
    lower = min(np.quantile(aperture_all, 0.001), np.quantile(joint_all, 0.001))
    upper = max(np.quantile(aperture_all, 0.999), np.quantile(joint_all, 0.999))
    axes[0].hist(
        np.clip(aperture_all * 1000, lower * 1000, upper * 1000),
        bins=100,
        alpha=0.62,
        label="$m_a^*$",
    )
    axes[0].hist(
        np.clip(joint_all * 1000, lower * 1000, upper * 1000),
        bins=100,
        alpha=0.62,
        label="$m_r^*$",
    )
    axes[0].axvline(0.0, color="black", linewidth=1)
    axes[0].set(title="Settled GT minimum gaps", xlabel="minimum gap, mm")
    axes[0].legend()

    axes[1].scatter(
        aperture_all * 1000,
        joint_all * 1000,
        s=5,
        alpha=0.2,
        rasterized=True,
    )
    diagonal_min = min(aperture_all.min(), joint_all.min()) * 1000
    diagonal_max = max(aperture_all.max(), joint_all.max()) * 1000
    axes[1].plot(
        [diagonal_min, diagonal_max],
        [diagonal_min, diagonal_max],
        color="black",
        linewidth=1,
    )
    axes[1].set(xlabel="$m_a^*$, mm", ylabel="$m_r^*$, mm", title="Per-state change")

    labels = [row["object_id"][:18] for row in payloads] + ["aggregate"]
    aperture_fraction = [
        row["all_states"]["aperture_zero_surface_violation_fraction"]
        for row in payloads
    ] + [float(np.mean(aperture_all < 0.0))]
    joint_fraction = [
        row["all_states"]["joint_fk_zero_surface_violation_fraction"]
        for row in payloads
    ] + [float(np.mean(joint_all < 0.0))]
    x = np.arange(len(labels))
    axes[2].bar(x - 0.2, aperture_fraction, 0.4, label="aperture")
    axes[2].bar(x + 0.2, joint_fraction, 0.4, label="actual joints FK")
    axes[2].set(
        xticks=x,
        xticklabels=labels,
        ylabel="fraction $m<0$",
        title="GT zero-surface violations",
    )
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].legend()

    aperture_deep_fraction = [
        row["all_states"]["aperture_penetration_gt_1mm_fraction"]
        for row in payloads
    ] + [float(np.mean(aperture_all < -0.001))]
    joint_deep_fraction = [
        row["all_states"]["joint_fk_penetration_gt_1mm_fraction"]
        for row in payloads
    ] + [float(np.mean(joint_all < -0.001))]
    axes[3].bar(x - 0.2, aperture_deep_fraction, 0.4, label="aperture")
    axes[3].bar(x + 0.2, joint_deep_fraction, 0.4, label="actual joints FK")
    axes[3].set(
        xticks=x,
        xticklabels=labels,
        ylabel="fraction $m<-1$ mm",
        title="Material GT penetration",
    )
    axes[3].tick_params(axis="x", rotation=25)
    axes[3].legend()
    fig.savefig(args.output / "aperture-vs-joints.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if app is not None:
        app.close()


if __name__ == "__main__":
    main()
