#!/usr/bin/env python3
"""Inspect authored collision schemas in the frozen simulator USD assets."""

from __future__ import annotations

import json

from srno.sim.pxr_bootstrap import ensure_pxr_available


def main() -> None:
    ensure_pxr_available(reexec=True)

    from pxr import Usd, UsdGeom, UsdPhysics

    from srno.sim import SimulatorAssetCatalog

    catalog = SimulatorAssetCatalog.load()
    rows = []
    for record in catalog.objects:
        stage = Usd.Stage.Open(str(record.usd_path))
        root = stage.GetDefaultPrim()
        meshes = []
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            cursor = prim
            inherited_paths = []
            while cursor and cursor.IsValid():
                if cursor.HasAPI(UsdPhysics.CollisionAPI):
                    inherited_paths.append(str(cursor.GetPath()))
                if cursor == root:
                    break
                cursor = cursor.GetParent()
            approximation = None
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            meshes.append(
                {
                    "path": str(prim.GetPath()),
                    "points": len(UsdGeom.Mesh(prim).GetPointsAttr().Get() or []),
                    "faces": len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() or []),
                    "direct_collision": prim.HasAPI(UsdPhysics.CollisionAPI),
                    "collision_ancestors": inherited_paths,
                    "mesh_collision_api": prim.HasAPI(UsdPhysics.MeshCollisionAPI),
                    "approximation": str(approximation) if approximation is not None else None,
                    "applied_schemas": list(prim.GetAppliedSchemas()),
                    "physics_attributes": {
                        attr.GetName(): str(attr.Get())
                        for attr in prim.GetAttributes()
                        if attr.GetName().startswith(("physics:", "physxCollision:"))
                    },
                }
            )
        rows.append(
            {
                "object_id": record.object_id,
                "default_prim": str(root.GetPath()),
                "default_prim_type": root.GetTypeName(),
                "default_prim_schemas": list(root.GetAppliedSchemas()),
                "meshes": meshes,
            }
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
