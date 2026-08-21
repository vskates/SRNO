#!/usr/bin/env python3
"""Record live Isaac articulation frames for the gripper collision audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _triangulate_convex(convex: object) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    indices = list(convex.indices)
    for polygon in convex.polygons:
        face = indices[polygon.index_base : polygon.index_base + polygon.num_vertices]
        triangles.extend((face[0], face[index], face[index + 1]) for index in range(1, len(face) - 1))
    return np.asarray(triangles, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/sdf-collision-diagnostic/live_gripper_frames.json"),
    )
    parser.add_argument(
        "--cooked-output",
        type=Path,
        default=Path("runs/sdf-collision-diagnostic/gripper_physx_cooked.npz"),
    )
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

        import torch
        from omni.physx import get_physx_cooking_interface
        from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
        from isaaclab.assets import Articulation
        from isaaclab.sim import SimulationContext
        from isaaclab.utils.math import subtract_frame_transforms
        from pxr import PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils

        from srno.sim import SimulatorAssetCatalog
        from srno.sim.config import RelaxationConfig
        from srno.sim.isaac_scene import CONTACT_LINKS, _gripper_cfg, make_simulation_cfg

        catalog = SimulatorAssetCatalog.load()
        print("[gripper-audit] creating simulation context", flush=True)
        simulation = SimulationContext(make_simulation_cfg("cuda:0"))
        robot_cfg = _gripper_cfg(catalog, RelaxationConfig()).replace(prim_path="/World/Robot")
        print("[gripper-audit] spawning articulation", flush=True)
        robot = Articulation(robot_cfg)
        print("[gripper-audit] resetting simulation", flush=True)
        simulation.reset()
        print("[gripper-audit] stepping open state", flush=True)
        for _ in range(10):
            robot.set_joint_position_target(torch.zeros_like(robot.data.joint_pos))
            robot.write_data_to_sim()
            simulation.step(render=False)
            robot.update(simulation.get_physics_dt())

        print("[gripper-audit] reading frames", flush=True)
        selected_names = ("astribot_gripper_right_base_link", *CONTACT_LINKS)
        body_index = {}
        for link_name in selected_names:
            indices = [index for index, name in enumerate(robot.body_names) if name == link_name]
            if len(indices) != 1:
                raise RuntimeError(f"expected one body named {link_name}, got {indices}")
            body_index[link_name] = indices[0]

        close_target = torch.zeros_like(robot.data.joint_pos)
        for index, joint_name in enumerate(robot.joint_names):
            close_target[:, index] = catalog.gripper.close_joint_position_rad[joint_name]
        fractions = np.linspace(0.0, 1.0, 33)
        schedule_bodies = {name: {"position_in_root_m": [], "quaternion_in_root_wxyz": []} for name in selected_names}
        schedule_joint_position = []
        selected = {}
        for fraction_index, fraction in enumerate(fractions):
            position = close_target * float(fraction)
            robot.write_joint_state_to_sim(position, torch.zeros_like(position))
            robot.set_joint_position_target(position)
            robot.write_data_to_sim()
            simulation.step(render=False)
            robot.update(simulation.get_physics_dt())
            root_position = robot.data.root_pos_w[:, None, :].expand_as(robot.data.body_pos_w)
            root_quaternion = robot.data.root_quat_w[:, None, :].expand_as(robot.data.body_quat_w)
            relative_position, relative_quaternion = subtract_frame_transforms(
                root_position,
                root_quaternion,
                robot.data.body_pos_w,
                robot.data.body_quat_w,
            )
            schedule_joint_position.append(robot.data.joint_pos[0].cpu().tolist())
            for link_name in selected_names:
                index = body_index[link_name]
                schedule_bodies[link_name]["position_in_root_m"].append(
                    relative_position[0, index].cpu().tolist()
                )
                schedule_bodies[link_name]["quaternion_in_root_wxyz"].append(
                    relative_quaternion[0, index].cpu().tolist()
                )
                if fraction_index == 0:
                    selected[link_name] = {
                        "body_index": index,
                        "position_in_root_m": relative_position[0, index].cpu().tolist(),
                        "quaternion_in_root_wxyz": relative_quaternion[0, index].cpu().tolist(),
                        "position_world_m": robot.data.body_pos_w[0, index].cpu().tolist(),
                        "quaternion_world_wxyz": robot.data.body_quat_w[0, index].cpu().tolist(),
                    }

        print("[gripper-audit] schedule recorded; cooking contact colliders", flush=True)
        stage = simulation.stage
        stage_cache = UsdUtils.StageCache.Get()
        stage_id = stage_cache.Insert(stage).ToLongInt()
        cooking = get_physx_cooking_interface()
        cooked_payload: dict[str, np.ndarray] = {}
        cooked_index = []
        for link_name in CONTACT_LINKS:
            print(f"[gripper-audit] cooking {link_name}", flush=True)
            link_prims = [prim for prim in stage.Traverse() if prim.GetName() == link_name]
            print(
                f"[gripper-audit] matched links: {[str(prim.GetPath()) for prim in link_prims]}",
                flush=True,
            )
            if len(link_prims) != 1:
                raise RuntimeError(f"expected one spawned link {link_name}, got {len(link_prims)}")
            link_prim = link_prims[0]
            collision_prims = [
                prim
                for prim in Usd.PrimRange(link_prim)
                if prim.HasAPI(UsdPhysics.CollisionAPI)
                and prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            ]
            print(
                f"[gripper-audit] matched colliders: "
                f"{[str(prim.GetPath()) for prim in collision_prims]}",
                flush=True,
            )
            if len(collision_prims) != 1:
                raise RuntimeError(
                    f"expected one PhysX mesh collider below {link_name}, got {len(collision_prims)}"
                )
            collision_prim = collision_prims[0]
            captured: dict[str, object] = {}

            def on_ready(result: object, convexes: list[object]) -> None:
                captured["result"] = result
                captured["convexes"] = convexes

            cooking.request_convex_collision_representation(
                stage_id=stage_id,
                collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(collision_prim.GetPath())),
                run_asynchronously=False,
                on_result=on_ready,
            )
            print(
                f"[gripper-audit] cooking result={captured.get('result')}, "
                f"hulls={len(captured.get('convexes', []))}",
                flush=True,
            )
            if captured.get("result") != PhysxCollisionRepresentationResult.RESULT_VALID:
                cooked_index.append(
                    {
                        "link_name": link_name,
                        "collision_path": str(collision_prim.GetPath()),
                        "result": str(captured.get("result")),
                        "note": "standalone recooking does not parse the importer Xform wrapper; live PhysX still uses the authored convexHull approximation",
                    }
                )
                continue
            link_to_world = np.asarray(
                UsdGeom.Xformable(link_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                dtype=np.float64,
            ).T
            collision_to_world = np.asarray(
                UsdGeom.Xformable(collision_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                dtype=np.float64,
            ).T
            collision_to_link = np.linalg.inv(link_to_world) @ collision_to_world
            convexes = list(captured["convexes"])
            if len(convexes) != 1:
                raise RuntimeError(f"expected one convex hull for {link_name}, got {len(convexes)}")
            convex = convexes[0]
            vertices = np.asarray(
                [[point.x, point.y, point.z] for point in convex.vertices], dtype=np.float64
            )
            vertices = np.column_stack((vertices, np.ones(len(vertices)))) @ collision_to_link.T
            key = str(CONTACT_LINKS.index(link_name))
            cooked_payload[f"link_{key}_vertices"] = vertices[:, :3].astype(np.float32)
            cooked_payload[f"link_{key}_faces"] = _triangulate_convex(convex)
            cooked_index.append(
                {
                    "link_name": link_name,
                    "collision_path": str(collision_prim.GetPath()),
                    "vertices": len(vertices),
                    "faces": len(cooked_payload[f"link_{key}_faces"]),
                    "approximation": str(
                        UsdPhysics.MeshCollisionAPI(collision_prim).GetApproximationAttr().Get()
                    ),
                    "result": str(captured.get("result")),
                }
            )
        args.cooked_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cooked_output, **cooked_payload)
        result = {
            "root_position_world_m": robot.data.root_pos_w[0].cpu().tolist(),
            "root_quaternion_world_wxyz": robot.data.root_quat_w[0].cpu().tolist(),
            "joint_names": list(robot.joint_names),
            "joint_position_rad": robot.data.joint_pos[0].cpu().tolist(),
            "body_names": list(robot.body_names),
            "selected_bodies": selected,
            "schedule": {
                "closure_fraction": fractions.tolist(),
                "joint_position_rad": schedule_joint_position,
                "bodies": schedule_bodies,
            },
            "physx_cooked_colliders": cooked_index,
            "physx_cooked_output": str(args.cooked_output.resolve()),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if simulation is not None:
            simulation.clear_all_callbacks()
            simulation.clear_instance()
        if app is not None:
            app.close()
        watchdog.stop()


if __name__ == "__main__":
    main()
