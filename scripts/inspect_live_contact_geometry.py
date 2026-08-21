#!/usr/bin/env python3
"""Inspect effective PhysX contact offsets and live collider prims for one SRNO scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _stats(values: np.ndarray) -> dict[str, object]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "shape": list(values.shape),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "unique": np.unique(np.round(flat, 9)).tolist(),
        "values": np.asarray(values, dtype=np.float64).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", default="voda-mineralnaya-psyzh-1-l-27215")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/contact-geometry-diagnostic/live_parameters.json"),
    )
    parser.add_argument("--memory-limit-gib", type=float, default=14.0)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher
    from srno.sim.memory_guard import MemoryWatchdog

    watchdog = MemoryWatchdog(args.memory_limit_gib, 0.25)
    watchdog.start()
    app = None
    simulation = None
    try:
        app = AppLauncher({"headless": True, "device": "cuda:0"}).app

        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

        from srno.sim import SimulatorAssetCatalog
        from srno.sim.config import RelaxationConfig
        from srno.sim.isaac_scene import make_scene_cfg, make_simulation_cfg

        catalog = SimulatorAssetCatalog.load()
        record = catalog.object(args.object)
        simulation = SimulationContext(make_simulation_cfg("cuda:0"))
        scene = InteractiveScene(
            make_scene_cfg(catalog, record, num_envs=1, relaxation=RelaxationConfig())
        )
        simulation.reset()
        for _ in range(4):
            simulation.step(render=False)
            scene.update(simulation.get_physics_dt())

        effective = {}
        for name in ("robot", "object"):
            view = scene[name].root_physx_view
            contact = view.get_contact_offsets().detach().cpu().numpy()
            rest = view.get_rest_offsets().detach().cpu().numpy()
            effective[name] = {
                "body_names": list(scene[name].body_names),
                "contact_offset_m": _stats(contact),
                "rest_offset_m": _stats(rest),
            }

        stage = simulation.stage
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        rows = []
        mesh_rows = []
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith("/World/envs/env_0/"):
                continue
            is_mesh = prim.IsA(UsdGeom.Mesh)
            collision_api = UsdPhysics.CollisionAPI(prim)
            if is_mesh:
                imageable = UsdGeom.Imageable(prim)
                mesh_rows.append(
                    {
                        "path": path,
                        "has_collision_api": bool(collision_api),
                        "visibility": str(imageable.ComputeVisibility()),
                        "purpose": str(imageable.GetPurposeAttr().Get()),
                    }
                )
            if not collision_api:
                continue
            physx_api = PhysxSchema.PhysxCollisionAPI(prim)
            contact_attr = physx_api.GetContactOffsetAttr()
            rest_attr = physx_api.GetRestOffsetAttr()
            mesh_api = UsdPhysics.MeshCollisionAPI(prim)
            world_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            rows.append(
                {
                    "path": path,
                    "type": prim.GetTypeName(),
                    "collision_enabled": collision_api.GetCollisionEnabledAttr().Get(),
                    "contact_offset_authored": contact_attr.HasAuthoredValueOpinion(),
                    "contact_offset_usd_m": contact_attr.Get(),
                    "rest_offset_authored": rest_attr.HasAuthoredValueOpinion(),
                    "rest_offset_usd_m": rest_attr.Get(),
                    "mesh_approximation": (
                        mesh_api.GetApproximationAttr().Get() if mesh_api else None
                    ),
                    "world_bounds_m": [
                        list(world_range.GetMin()),
                        list(world_range.GetMax()),
                    ],
                }
            )

        result = {
            "object_id": args.object,
            "effective_physx_view": effective,
            "collision_prims": rows,
            "mesh_prims": mesh_rows,
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
