#!/usr/bin/env python3
"""Compare SRNO gripper points with the collision meshes in the runtime USD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from srno.geometry.gripper import GripperAsset
from srno.sim import SimulatorAssetCatalog
from srno.sim.pxr_bootstrap import ensure_pxr_available


def _triangulate(counts: object, indices: object) -> np.ndarray:
    triangles = []
    cursor = 0
    for raw_count in counts:
        count = int(raw_count)
        face = indices[cursor : cursor + count]
        cursor += count
        triangles.extend((face[0], face[index], face[index + 1]) for index in range(1, count - 1))
    return np.asarray(triangles, dtype=np.int64)


def _collision_ancestor(prim: object, link: object) -> bool:
    from pxr import UsdPhysics

    cursor = prim
    while cursor and cursor.IsValid():
        if cursor.HasAPI(UsdPhysics.CollisionAPI):
            return True
        if cursor == link:
            break
        cursor = cursor.GetParent()
    return False


def _stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _pose_matrix(position: list[float], quaternion_wxyz: list[float]) -> np.ndarray:
    w, x, y, z = quaternion_wxyz
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )
    matrix[:3, 3] = position
    return matrix


def _farthest_point_subset(points: np.ndarray, count: int) -> np.ndarray:
    chosen = np.empty(count, dtype=np.int64)
    chosen[0] = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))[0]
    minimum_distance = np.full(len(points), np.inf)
    for index in range(1, count):
        delta = points - points[chosen[index - 1]]
        minimum_distance = np.minimum(
            minimum_distance, np.einsum("ij,ij->i", delta, delta)
        )
        chosen[index] = int(np.argmax(minimum_distance))
    return points[chosen]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-gripper", type=Path, default=Path("data/simulator-v1/gripper.npz"))
    parser.add_argument(
        "--live-frames",
        type=Path,
        default=Path("runs/sdf-collision-diagnostic/live_gripper_frames.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/sdf-collision-diagnostic/gripper_collision_audit.json"),
    )
    parser.add_argument(
        "--runtime-points-output",
        type=Path,
        default=Path("runs/sdf-collision-diagnostic/runtime_gripper_points.npz"),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("runs/sdf-collision-diagnostic/gripper_alignment.png"),
    )
    args = parser.parse_args()
    ensure_pxr_available(reexec=True)

    from pxr import Usd, UsdGeom, UsdPhysics

    catalog = SimulatorAssetCatalog.load()
    asset = GripperAsset.load(args.manifest_gripper)
    stage = Usd.Stage.Open(str(catalog.gripper.runtime_usd))
    root = stage.GetDefaultPrim()
    live = json.loads(args.live_frames.read_text(encoding="utf-8"))
    live_bodies = live["selected_bodies"]
    live_schedule = live["schedule"]
    links_by_name = {
        prim.GetName(): prim
        for prim in Usd.PrimRange(root)
        if prim.GetName() in catalog.gripper.contact_links
    }
    rows = []
    all_schedule_distances = []
    all_schedule_hull_distances = []
    runtime_points_by_link = []
    runtime_hulls: dict[str, np.ndarray] = {}
    for link_index, link_name in enumerate(catalog.gripper.contact_links):
        link = links_by_name.get(link_name)
        if link is None:
            raise RuntimeError(f"runtime contact link is missing: {link_name}")
        link_to_stage = np.asarray(
            UsdGeom.Xformable(link).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
            dtype=np.float64,
        ).T
        stage_to_link = np.linalg.inv(link_to_stage)
        meshes = []
        collider_rows = []
        for prim in Usd.PrimRange(link):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                attributes = {
                    attribute.GetName(): str(attribute.Get())
                    for attribute in prim.GetAttributes()
                    if attribute.GetName().startswith(("physics:", "physxCollision:"))
                }
                approximation = None
                if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                    approximation = str(
                        UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                    )
                collider_rows.append(
                    {
                        "path": str(prim.GetPath()),
                        "type": prim.GetTypeName(),
                        "approximation": approximation,
                        "attributes": attributes,
                    }
                )
            if not prim.IsA(UsdGeom.Mesh) or not _collision_ancestor(prim, link):
                continue
            geometry = UsdGeom.Mesh(prim)
            points = geometry.GetPointsAttr().Get()
            counts = geometry.GetFaceVertexCountsAttr().Get()
            indices = geometry.GetFaceVertexIndicesAttr().Get()
            if points is None or counts is None or indices is None:
                continue
            mesh = trimesh.Trimesh(
                vertices=np.asarray(points, dtype=np.float64),
                faces=_triangulate(counts, indices),
                process=False,
            )
            transform = np.asarray(
                UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                dtype=np.float64,
            ).T
            mesh.apply_transform(stage_to_link @ transform)
            meshes.append(mesh)
        if not meshes:
            raise RuntimeError(f"runtime link has no collision mesh: {link_name}")
        combined_local = trimesh.util.concatenate(meshes)
        convex_local = combined_local.convex_hull
        runtime_hulls[f"link_{link_index}_vertices"] = np.asarray(
            convex_local.vertices, dtype=np.float32
        )
        runtime_hulls[f"link_{link_index}_faces"] = np.asarray(
            convex_local.faces, dtype=np.int32
        )
        candidates, _ = trimesh.sample.sample_surface(
            convex_local, 128 * 8, seed=np.random.default_rng(link_index)
        )
        local_runtime_samples = _farthest_point_subset(np.asarray(candidates), 128)
        schedule_rows = []
        link_distances = []
        link_hull_distances = []
        link_runtime_points = []
        model_index = asset.link_index == link_index
        for fraction_index, fraction in enumerate(live_schedule["closure_fraction"]):
            body_schedule = live_schedule["bodies"][link_name]
            link_to_root = _pose_matrix(
                body_schedule["position_in_root_m"][fraction_index],
                body_schedule["quaternion_in_root_wxyz"][fraction_index],
            )
            runtime_mesh = combined_local.copy()
            runtime_mesh.apply_transform(link_to_root)
            runtime_hull = convex_local.copy()
            runtime_hull.apply_transform(link_to_root)
            homogeneous_samples = np.column_stack(
                (local_runtime_samples, np.ones(len(local_runtime_samples)))
            )
            runtime_sample_points = (homogeneous_samples @ link_to_root.T)[:, :3]
            link_runtime_points.append(runtime_sample_points)
            # GripperAsset knots are closed-to-open, while the simulator
            # schedule is open-to-closed.
            knot_index = len(asset.point_knots) - 1 - fraction_index
            model_points = asset.point_knots[knot_index][model_index].detach().cpu().numpy()
            _, distances, _ = trimesh.proximity.closest_point(runtime_mesh, model_points)
            _, hull_distances, _ = trimesh.proximity.closest_point(
                runtime_hull, model_points
            )
            link_distances.append(distances)
            link_hull_distances.append(hull_distances)
            schedule_rows.append(
                {
                    "closure_fraction": float(fraction),
                    "model_aperture_m": float(asset.aperture_knots[knot_index]),
                    "model_points_to_runtime_collision_abs_m": _stats(distances),
                    "model_points_to_runtime_convex_hull_abs_m": _stats(
                        hull_distances
                    ),
                    "runtime_mesh_bounds_m": runtime_mesh.bounds.tolist(),
                    "model_point_bounds_m": np.asarray(
                        (model_points.min(axis=0), model_points.max(axis=0))
                    ).tolist(),
                }
            )
        distances = np.concatenate(link_distances)
        hull_distances = np.concatenate(link_hull_distances)
        all_schedule_distances.append(distances)
        all_schedule_hull_distances.append(hull_distances)
        runtime_points_by_link.append(np.stack(link_runtime_points))
        open_runtime = combined_local.copy()
        open_body = live_bodies[link_name]
        open_runtime.apply_transform(
            _pose_matrix(
                open_body["position_in_root_m"], open_body["quaternion_in_root_wxyz"]
            )
        )
        open_model_points = asset.point_knots[-1][model_index].detach().cpu().numpy()
        rows.append(
            {
                "link_name": link_name,
                "model_point_count": len(open_model_points),
                "runtime_mesh_vertices": int(len(combined_local.vertices)),
                "runtime_mesh_faces": int(len(combined_local.faces)),
                "runtime_convex_hull_vertices": int(len(convex_local.vertices)),
                "runtime_convex_hull_faces": int(len(convex_local.faces)),
                "runtime_mesh_bounds_m": open_runtime.bounds.tolist(),
                "model_point_bounds_m": np.asarray(
                    (open_model_points.min(axis=0), open_model_points.max(axis=0))
                ).tolist(),
                "all_33_states_model_points_to_runtime_collision_abs_m": _stats(distances),
                "all_33_states_model_points_to_runtime_convex_hull_abs_m": _stats(
                    hull_distances
                ),
                "schedule": schedule_rows,
                "colliders": collider_rows,
            }
        )

    left_position = np.asarray(
        live_schedule["bodies"][catalog.gripper.contact_links[0]]["position_in_root_m"]
    )
    right_position = np.asarray(
        live_schedule["bodies"][catalog.gripper.contact_links[1]]["position_in_root_m"]
    )
    runtime_aperture = np.abs(left_position[:, 0] - right_position[:, 0])
    model_aperture = asset.aperture_knots.detach().cpu().numpy()[::-1]
    aperture_difference = runtime_aperture - model_aperture
    runtime_points = np.concatenate(runtime_points_by_link, axis=1).astype(np.float32)

    result = {
        "runtime_usd": str(catalog.gripper.runtime_usd),
        "runtime_usd_sha256": catalog.gripper.runtime_usd_sha256,
        "model_source_urdf": str(catalog.gripper.source_urdf),
        "model_source_urdf_sha256": asset.source_sha256,
        "aperture_m": asset.aperture_max,
        "comparison_frame": "live Isaac articulation root/base-link frame at zero joint position",
        "live_frame_source": str(args.live_frames.resolve()),
        "links": rows,
        "aggregate": {
            "all_33_states_model_points_to_runtime_collision_abs_m": _stats(
                np.concatenate(all_schedule_distances)
            ),
            "all_33_states_model_points_to_runtime_convex_hull_abs_m": _stats(
                np.concatenate(all_schedule_hull_distances)
            ),
            "runtime_finger_origin_aperture_m": runtime_aperture.tolist(),
            "model_aperture_m": model_aperture.tolist(),
            "runtime_minus_model_aperture_m": _stats(aperture_difference),
            "open_runtime_aperture_m": float(runtime_aperture[0]),
            "open_model_aperture_m": float(model_aperture[0]),
            "closed_runtime_aperture_m": float(runtime_aperture[-1]),
            "closed_model_aperture_m": float(model_aperture[-1]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.runtime_points_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.runtime_points_output,
        points_open_to_closed=runtime_points,
        closure_fraction=np.asarray(live_schedule["closure_fraction"], dtype=np.float32),
        runtime_aperture_m=runtime_aperture.astype(np.float32),
        **runtime_hulls,
    )
    command_index = np.arange(len(runtime_aperture))
    raw_p95 = np.mean(
        [
            [state["model_points_to_runtime_collision_abs_m"]["p95"] for state in row["schedule"]]
            for row in rows
        ],
        axis=0,
    )
    hull_p95 = np.mean(
        [
            [state["model_points_to_runtime_convex_hull_abs_m"]["p95"] for state in row["schedule"]]
            for row in rows
        ],
        axis=0,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(command_index, runtime_aperture * 1000, label="live Isaac finger origins")
    axes[0].plot(command_index, model_aperture * 1000, label="SRNO URDF asset")
    axes[0].fill_between(
        command_index,
        model_aperture * 1000,
        runtime_aperture * 1000,
        color="#C44E52",
        alpha=0.18,
        label="mismatch",
    )
    axes[0].set_xlabel("closure command index")
    axes[0].set_ylabel("finger-origin aperture, mm")
    axes[0].set_title("Runtime vs model aperture geometry")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(command_index, raw_p95 * 1000, label="runtime authored mesh")
    axes[1].plot(command_index, hull_p95 * 1000, label="runtime convexHull proxy")
    axes[1].set_xlabel("closure command index")
    axes[1].set_ylabel("mean link-wise p95 distance, mm")
    axes[1].set_title("SRNO points → runtime finger collision")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("SRNO gripper geometry alignment audit")
    args.plot_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot_output, dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
