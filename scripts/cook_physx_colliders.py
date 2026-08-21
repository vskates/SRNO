#!/usr/bin/env python3
"""Export the exact PhysX-cooked convex decomposition of catalog objects."""

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
        triangles.extend((face[0], face[i], face[i + 1]) for i in range(1, len(face) - 1))
    return np.asarray(triangles, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--memory-limit-gib", type=float, default=12.0)
    parser.add_argument("--memory-check-interval-s", type=float, default=0.25)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher
    from srno.sim.memory_guard import MemoryWatchdog

    watchdog = MemoryWatchdog(args.memory_limit_gib, args.memory_check_interval_s)
    watchdog.start()
    app = None
    try:
        launcher = AppLauncher({"headless": True, "device": "cuda:0"})
        app = launcher.app
        from omni.physx import get_physx_cooking_interface
        from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
        from pxr import PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils

        from srno.sim import SimulatorAssetCatalog

        catalog = SimulatorAssetCatalog.load()
        object_ids = list(catalog.object_ids if args.objects is None else args.objects)
        unknown = set(object_ids) - set(catalog.object_ids)
        if unknown:
            raise ValueError(f"unknown objects: {sorted(unknown)}")
        args.output.mkdir(parents=True, exist_ok=True)
        cooking = get_physx_cooking_interface()
        rows = []
        for index, object_id in enumerate(object_ids, start=1):
            record = catalog.object(object_id)
            stage = Usd.Stage.Open(str(record.usd_path))
            root = stage.GetDefaultPrim()
            collision_prims = [
                prim
                for prim in Usd.PrimRange(root)
                if prim.IsA(UsdGeom.Mesh)
                and prim.HasAPI(UsdPhysics.CollisionAPI)
                and prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            ]
            if len(collision_prims) != 1:
                raise RuntimeError(f"{object_id}: expected one collision mesh, got {len(collision_prims)}")
            prim = collision_prims[0]
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            if approximation != UsdPhysics.Tokens.convexDecomposition:
                raise RuntimeError(f"{object_id}: unexpected approximation {approximation}")

            cache = UsdUtils.StageCache.Get()
            cache_id = cache.Insert(stage)
            stage_id = cache_id.ToLongInt()
            prim_id = PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath()))
            captured: dict[str, object] = {}

            def on_ready(result: object, convexes: list[object]) -> None:
                captured["result"] = result
                captured["convexes"] = convexes

            cooking.request_convex_collision_representation(
                stage_id=stage_id,
                collision_prim_id=prim_id,
                run_asynchronously=False,
                on_result=on_ready,
            )
            if captured.get("result") != PhysxCollisionRepresentationResult.RESULT_VALID:
                raise RuntimeError(f"{object_id}: PhysX cooking failed: {captured.get('result')}")
            convexes = list(captured["convexes"])
            transform = np.asarray(
                UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                dtype=np.float64,
            ).T
            payload: dict[str, np.ndarray] = {
                "hull_count": np.asarray(len(convexes), dtype=np.int32),
                "source_transform": transform,
            }
            total_vertices = 0
            total_triangles = 0
            for hull_index, convex in enumerate(convexes):
                vertices = np.asarray(
                    [[vertex.x, vertex.y, vertex.z] for vertex in convex.vertices],
                    dtype=np.float64,
                )
                homogeneous = np.column_stack((vertices, np.ones(len(vertices))))
                vertices = (homogeneous @ transform.T)[:, :3]
                faces = _triangulate_convex(convex)
                payload[f"hull_{hull_index:04d}_vertices"] = vertices.astype(np.float32)
                payload[f"hull_{hull_index:04d}_faces"] = faces.astype(np.int32)
                total_vertices += len(vertices)
                total_triangles += len(faces)
            destination = args.output / f"{object_id}.npz"
            np.savez_compressed(destination, **payload)
            rows.append(
                {
                    "object_id": object_id,
                    "hulls": len(convexes),
                    "vertices": total_vertices,
                    "triangles": total_triangles,
                    "output": str(destination.resolve()),
                }
            )
            print(
                f"[{index:02d}/{len(object_ids):02d}] {object_id}: "
                f"{len(convexes)} hulls, {total_vertices} vertices, {total_triangles} triangles",
                flush=True,
            )
            cache.Erase(stage)
            del stage
        (args.output / "index.json").write_text(json.dumps(rows, indent=2) + "\n")
    finally:
        if app is not None:
            app.close()
        watchdog.stop()


if __name__ == "__main__":
    main()
