#!/usr/bin/env python3
"""Validate the runtime gripper convex-hull proxy against live PhysX raycasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def _pose_matrix(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion_wxyz
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix[:3, 3] = position
    return matrix


def _transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (homogeneous @ matrix.T)[:, :3]


def _rays(bounds: np.ndarray, samples: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
    extent = bounds[1] - bounds[0]
    margin = max(0.015, float(extent.max()) * 0.25)
    result = []
    for axis in range(3):
        other = [index for index in range(3) if index != axis]
        coordinates = [
            np.linspace(bounds[0, index], bounds[1, index], samples)
            for index in other
        ]
        for sign in (-1.0, 1.0):
            direction = np.zeros(3)
            direction[axis] = -sign
            for first in coordinates[0]:
                for second in coordinates[1]:
                    origin = np.zeros(3)
                    origin[axis] = bounds[1, axis] + margin if sign > 0 else bounds[0, axis] - margin
                    origin[other] = (first, second)
                    result.append((origin, direction, float(extent[axis] + 2 * margin)))
    return result


def _stats(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proxy",
        type=Path,
        default=Path("runs/sdf-collision-diagnostic/runtime_gripper_points.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/sdf-collision-diagnostic/live_gripper_rays.json"),
    )
    parser.add_argument("--samples-per-side", type=int, default=12)
    parser.add_argument("--memory-limit-gib", type=float, default=14.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from isaaclab.app import AppLauncher
    from srno.sim.memory_guard import MemoryWatchdog

    watchdog = MemoryWatchdog(args.memory_limit_gib, 0.25)
    watchdog.start()
    app = None
    simulation = None
    try:
        launcher = AppLauncher({"headless": True, "device": args.device})
        app = launcher.app

        import carb
        import omni.physx
        import torch
        from isaaclab.assets import Articulation
        from isaaclab.sim import SimulationContext
        from isaaclab.utils.math import subtract_frame_transforms

        from srno.sim import SimulatorAssetCatalog
        from srno.sim.config import RelaxationConfig
        from srno.sim.isaac_scene import CONTACT_LINKS, _gripper_cfg, make_simulation_cfg

        catalog = SimulatorAssetCatalog.load()
        with np.load(args.proxy, allow_pickle=False) as archive:
            local_hulls = [
                trimesh.Trimesh(
                    vertices=archive[f"link_{index}_vertices"],
                    faces=archive[f"link_{index}_faces"],
                    process=False,
                )
                for index in range(2)
            ]

        simulation_cfg = make_simulation_cfg(args.device)
        simulation_cfg.enable_scene_query_support = True
        simulation = SimulationContext(simulation_cfg)
        robot = Articulation(
            _gripper_cfg(catalog, RelaxationConfig()).replace(prim_path="/World/Robot")
        )
        simulation.reset()
        robot.update(simulation.get_physics_dt())
        close_target = torch.zeros_like(robot.data.joint_pos)
        for index, joint_name in enumerate(robot.joint_names):
            close_target[:, index] = catalog.gripper.close_joint_position_rad[joint_name]
        body_indices = [robot.body_names.index(name) for name in CONTACT_LINKS]
        query = omni.physx.get_physx_scene_query_interface()
        rows = []
        all_distances = []
        for command_index in (0, 8, 16, 24, 32):
            position = close_target * (command_index / 32.0)
            robot.write_joint_state_to_sim(position, torch.zeros_like(position))
            robot.set_joint_position_target(position)
            robot.write_data_to_sim()
            for _ in range(4):
                simulation.step(render=False)
                robot.update(simulation.get_physics_dt())
            # Scene-query acceleration structures are serviced by the Kit
            # update loop in headless mode after articulation teleports.
            for _ in range(2):
                app.update()
            root_position = robot.data.root_pos_w[:, None, :].expand_as(robot.data.body_pos_w)
            root_quaternion = robot.data.root_quat_w[:, None, :].expand_as(robot.data.body_quat_w)
            relative_position, relative_quaternion = subtract_frame_transforms(
                root_position,
                root_quaternion,
                robot.data.body_pos_w,
                robot.data.body_quat_w,
            )
            root_to_world = _pose_matrix(
                robot.data.root_pos_w[0].cpu().numpy(),
                robot.data.root_quat_w[0].cpu().numpy(),
            )
            world_to_root = np.linalg.inv(root_to_world)
            for link_index, (link_name, body_index, local_hull) in enumerate(
                zip(CONTACT_LINKS, body_indices, local_hulls, strict=True)
            ):
                link_to_root = _pose_matrix(
                    relative_position[0, body_index].cpu().numpy(),
                    relative_quaternion[0, body_index].cpu().numpy(),
                )
                proxy = local_hull.copy()
                proxy.apply_transform(link_to_root)
                hits = []
                ray_count = 0
                for origin_root, direction_root, distance in _rays(
                    proxy.bounds, args.samples_per_side
                ):
                    ray_count += 1
                    origin_world = _transform(root_to_world, origin_root[None])[0]
                    direction_world = root_to_world[:3, :3] @ direction_root
                    hit = query.raycast_closest(
                        carb.Float3(*origin_world.tolist()),
                        carb.Float3(*direction_world.tolist()),
                        distance,
                        True,
                    )
                    if hit["hit"] and link_name in str(hit["collision"]):
                        hits.append(
                            _transform(
                                world_to_root,
                                np.asarray(hit["position"], dtype=np.float64)[None],
                            )[0]
                        )
                hit_points = np.asarray(hits)
                _, distances, _ = trimesh.proximity.closest_point(proxy, hit_points)
                all_distances.append(distances)
                row = {
                    "command_index": command_index,
                    "link_name": link_name,
                    "rays": ray_count,
                    "hits": len(hits),
                    "live_physx_surface_to_convex_hull_proxy_abs_m": _stats(distances),
                }
                rows.append(row)
                print(
                    f"command {command_index:02d} {link_name}: "
                    f"{len(hits)}/{ray_count} hits, p95={row['live_physx_surface_to_convex_hull_proxy_abs_m']['p95'] * 1000:.4f} mm",
                    flush=True,
                )
        result = {
            "rows": rows,
            "aggregate": {
                "ray_hits": int(sum(len(values) for values in all_distances)),
                "live_physx_surface_to_convex_hull_proxy_abs_m": _stats(
                    np.concatenate(all_distances)
                ),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    finally:
        if simulation is not None:
            simulation.clear_all_callbacks()
            simulation.clear_instance()
        if app is not None:
            app.close()
        watchdog.stop()


if __name__ == "__main__":
    main()
