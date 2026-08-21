#!/usr/bin/env python3
"""Compare runtime gripper contact-link visual and authored collision meshes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from srno.sim.usd_geometry import _triangulate


def _mesh(prim: object) -> trimesh.Trimesh:
    from pxr import Usd, UsdGeom

    geom = UsdGeom.Mesh(prim)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(geom.GetPointsAttr().Get(), dtype=np.float64),
        faces=_triangulate(
            np.asarray(geom.GetFaceVertexCountsAttr().Get()),
            np.asarray(geom.GetFaceVertexIndicesAttr().Get()),
        ),
        process=False,
    )
    transform = np.asarray(
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
        dtype=np.float64,
    ).T
    mesh.apply_transform(transform)
    return mesh


def _distance(source: trimesh.Trimesh, target: trimesh.Trimesh) -> dict[str, float]:
    _, distance, _ = trimesh.proximity.closest_point(target, source.vertices)
    return {
        "mean_m": float(distance.mean()),
        "median_m": float(np.median(distance)),
        "p95_m": float(np.quantile(distance, 0.95)),
        "max_m": float(distance.max()),
    }


def main() -> None:
    # The pip distribution exposes USD bindings only after Kit has initialized.
    from isaaclab.app import AppLauncher

    app = AppLauncher({"headless": True}).app
    from pxr import Usd, UsdGeom

    try:
        source = Path("assets/grippers/astribot/gripper_playground.usd").resolve()
        stage = Usd.Stage.Open(str(source))
        links = (
            "astribot_gripper_right_Link_L11",
            "astribot_gripper_right_Link_R11",
        )
        rows = []
        for link in links:
            meshes = [
                prim
                for prim in stage.Traverse()
                if prim.IsA(UsdGeom.Mesh) and f"/{link}/" in str(prim.GetPath())
            ]
            visual_prims = [prim for prim in meshes if "/visuals/" in str(prim.GetPath())]
            collision_prims = [prim for prim in meshes if "/collisions/" in str(prim.GetPath())]
            visual = trimesh.util.concatenate([_mesh(prim) for prim in visual_prims])
            collision = trimesh.util.concatenate([_mesh(prim) for prim in collision_prims])
            rows.append(
                {
                    "link": link,
                    "visual_paths": [str(prim.GetPath()) for prim in visual_prims],
                    "collision_paths": [str(prim.GetPath()) for prim in collision_prims],
                    "visual_bounds_m": visual.bounds.tolist(),
                    "collision_bounds_m": collision.bounds.tolist(),
                    "visual_vertices_to_collision_surface": _distance(visual, collision),
                    "collision_vertices_to_visual_surface": _distance(collision, visual),
                }
            )
        output = Path("runs/contact-geometry-diagnostic/gripper_visual_collision.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"rows": rows}, indent=2))
    finally:
        app.close()


if __name__ == "__main__":
    main()
