#!/usr/bin/env python3
"""Probe the live PhysX scene and export object-frame collider hit points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    return (homogeneous @ matrix.T)[:, :3]


def _rigid_transform(translation: tuple[float, float, float], quaternion_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quaternion_wxyz
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _ray_origins(bounds: np.ndarray, samples_per_side: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
    extent = bounds[1] - bounds[0]
    margin = max(0.03, float(extent.max()) * 0.25)
    rays: list[tuple[np.ndarray, np.ndarray, float]] = []
    for axis in range(3):
        transverse = [index for index in range(3) if index != axis]
        coordinates = [
            np.linspace(
                bounds[0, index] + 0.01 * extent[index],
                bounds[1, index] - 0.01 * extent[index],
                samples_per_side,
            )
            for index in transverse
        ]
        for sign in (-1.0, 1.0):
            direction = np.zeros(3, dtype=np.float64)
            direction[axis] = -sign
            start = bounds[1, axis] + margin if sign > 0 else bounds[0, axis] - margin
            for first in coordinates[0]:
                for second in coordinates[1]:
                    origin = np.zeros(3, dtype=np.float64)
                    origin[axis] = start
                    origin[transverse] = (first, second)
                    rays.append((origin, direction, float(extent[axis] + 2.0 * margin)))
    return rays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cooked", type=Path, default=Path("runs/sdf-collision-diagnostic/cooked")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/sdf-collision-diagnostic/runtime-rays")
    )
    parser.add_argument("--samples-per-side", type=int, default=14)
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--memory-limit-gib", type=float, default=12.0)
    parser.add_argument("--memory-check-interval-s", type=float, default=0.25)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher
    from srno.sim.memory_guard import MemoryWatchdog

    watchdog = MemoryWatchdog(args.memory_limit_gib, args.memory_check_interval_s)
    watchdog.start()
    app = None
    simulation = None
    try:
        launcher = AppLauncher({"headless": True, "device": "cuda:0"})
        app = launcher.app

        import carb
        import isaaclab.sim as sim_utils
        import omni.physx
        from isaaclab.sim import SimulationContext
        from pxr import Usd, UsdGeom

        from srno.sim import SimulatorAssetCatalog
        from srno.sim.isaac_scene import make_simulation_cfg

        catalog = SimulatorAssetCatalog.load()
        object_ids = list(catalog.object_ids if args.objects is None else args.objects)
        unknown = set(object_ids) - set(catalog.object_ids)
        if unknown:
            raise ValueError(f"unknown objects: {sorted(unknown)}")

        simulation_cfg = make_simulation_cfg("cuda:0")
        simulation_cfg.enable_scene_query_support = True
        simulation = SimulationContext(simulation_cfg)
        stage = simulation.stage
        paths: dict[str, str] = {}
        spawn_transforms: dict[str, np.ndarray] = {}
        spacing = 0.8
        columns = 6
        for index, object_id in enumerate(object_ids):
            record = catalog.object(object_id)
            path = f"/World/Objects/Object_{index:02d}"
            paths[object_id] = path
            spawn_options = {}
            if record.spawn_scale is not None:
                spawn_options["scale"] = record.spawn_scale
            cfg = sim_utils.UsdFileCfg(
                usd_path=str(record.usd_path),
                **spawn_options,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    disable_gravity=True,
                    max_depenetration_velocity=10.0,
                    max_angular_velocity=10.0,
                    max_linear_velocity=10.0,
                    solver_position_iteration_count=64,
                    solver_velocity_iteration_count=16,
                    stabilization_threshold=0.1,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            )
            translation = (
                spacing * (index % columns),
                spacing * (index // columns),
                0.0,
            )
            spawn_transforms[object_id] = _rigid_transform(
                translation, record.spawn_quaternion_wxyz
            )
            cfg.func(
                path,
                cfg,
                translation=translation,
                orientation=record.spawn_quaternion_wxyz,
            )

        simulation.reset()
        for _ in range(8):
            simulation.step(render=False)
        for _ in range(4):
            app.update()

        args.output.mkdir(parents=True, exist_ok=True)
        query = omni.physx.get_physx_scene_query_interface()
        rows = []
        for object_id in object_ids:
            path = paths[object_id]
            with np.load(args.cooked / f"{object_id}.npz", allow_pickle=False) as archive:
                count = int(archive["hull_count"])
                vertices = np.concatenate(
                    [archive[f"hull_{index:04d}_vertices"] for index in range(count)], axis=0
                ).astype(np.float64)
            bounds = np.asarray((vertices.min(axis=0), vertices.max(axis=0)))
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"spawned prim does not exist: {path}")
            authored_stage_transform = np.asarray(
                UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                dtype=np.float64,
            ).T
            # Cooked vertices already have the source USD default-prim transform
            # (including its normalization scale) baked into metric coordinates.
            # Apply only the validation spawn pose here, otherwise the authored
            # scale would be counted twice.
            object_to_world = spawn_transforms[object_id]
            world_to_object = np.linalg.inv(object_to_world)
            rotation = object_to_world[:3, :3]

            hit_points = []
            hit_normals = []
            ray_count = 0
            rejected_other = 0
            for origin_object, direction_object, distance in _ray_origins(
                bounds, args.samples_per_side
            ):
                ray_count += 1
                origin_world = _transform_points(object_to_world, origin_object[None])[0]
                direction_world = rotation @ direction_object
                result = query.raycast_closest(
                    carb.Float3(*origin_world.tolist()),
                    carb.Float3(*direction_world.tolist()),
                    distance,
                    True,
                )
                if not result["hit"]:
                    continue
                collision_path = str(result["collision"])
                if not collision_path.startswith(path):
                    rejected_other += 1
                    continue
                hit_world = np.asarray(result["position"], dtype=np.float64)
                normal_world = np.asarray(result["normal"], dtype=np.float64)
                hit_points.append(_transform_points(world_to_object, hit_world[None])[0])
                hit_normals.append(rotation.T @ normal_world)

            hit_array = np.asarray(hit_points, dtype=np.float32).reshape(-1, 3)
            normal_array = np.asarray(hit_normals, dtype=np.float32).reshape(-1, 3)
            np.savez_compressed(
                args.output / f"{object_id}.npz",
                points_object=hit_array,
                normals_object=normal_array,
                object_to_world=object_to_world.astype(np.float32),
                authored_stage_transform=authored_stage_transform.astype(np.float32),
            )
            row = {
                "object_id": object_id,
                "rays": ray_count,
                "hits": len(hit_array),
                "hit_fraction": len(hit_array) / ray_count,
                "rejected_other_object_hits": rejected_other,
                "spawn_path": path,
                "object_to_world": object_to_world.tolist(),
                "authored_stage_transform": authored_stage_transform.tolist(),
            }
            rows.append(row)
            print(
                f"{object_id}: {len(hit_array)}/{ray_count} live PhysX ray hits",
                flush=True,
            )
        (args.output / "index.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        if simulation is not None:
            simulation.clear_all_callbacks()
            simulation.clear_instance()
        if app is not None:
            app.close()
        watchdog.stop()


if __name__ == "__main__":
    main()
