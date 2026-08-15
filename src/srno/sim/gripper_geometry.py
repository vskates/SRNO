"""Build the SRNO surface asset from the gripper actually spawned by Isaac.

The validation URDF is useful as source documentation, but it is not the
collision asset used by PhysX.  This module deliberately combines collision
meshes from the runtime USD with body frames measured from the runtime
articulation at the exact 33-command schedule.
"""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path

import numpy as np
import torch

from srno.geometry.gripper import GripperAsset, _farthest_point_subset
from srno.sim.assets import SimulatorAssetCatalog
from srno.sim.config import SimulatorConfig


def _triangulate(face_counts: object, face_indices: object) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    cursor = 0
    for raw_count in face_counts:
        count = int(raw_count)
        face = face_indices[cursor : cursor + count]
        cursor += count
        triangles.extend(
            (int(face[0]), int(face[index]), int(face[index + 1]))
            for index in range(1, count - 1)
        )
    return np.asarray(triangles, dtype=np.int64)


def _has_collision_ancestor(prim: object, link: object) -> bool:
    from pxr import UsdPhysics

    cursor = prim
    while cursor and cursor.IsValid():
        if cursor.HasAPI(UsdPhysics.CollisionAPI):
            return True
        if cursor == link:
            break
        cursor = cursor.GetParent()
    return False


def _runtime_link_samples(
    runtime_usd: Path,
    contact_links: tuple[str, str],
    *,
    samples_per_link: int,
    seed: int,
) -> list[np.ndarray]:
    """Sample the authored convexHull colliders in contact-link coordinates."""

    import trimesh
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(runtime_usd))
    if stage is None:
        raise RuntimeError(f"cannot open runtime gripper USD: {runtime_usd}")
    root = stage.GetDefaultPrim()
    if not root.IsValid():
        raise RuntimeError(f"runtime gripper USD has no default prim: {runtime_usd}")
    links = {
        prim.GetName(): prim
        for prim in Usd.PrimRange(root)
        if prim.GetName() in contact_links
    }

    sampled: list[np.ndarray] = []
    for link_offset, link_name in enumerate(contact_links):
        link = links.get(link_name)
        if link is None:
            raise RuntimeError(f"runtime contact link is missing: {link_name}")
        link_to_stage = np.asarray(
            UsdGeom.Xformable(link).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
            dtype=np.float64,
        ).T
        stage_to_link = np.linalg.inv(link_to_stage)
        meshes = []
        collider_count = 0
        for prim in Usd.PrimRange(link):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                    raise RuntimeError(
                        f"unsupported non-mesh runtime collider below {link_name}: "
                        f"{prim.GetPath()}"
                    )
                approximation = str(
                    UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                )
                if approximation != "convexHull":
                    raise RuntimeError(
                        f"runtime collider {prim.GetPath()} uses {approximation!r}; "
                        "SRNO runtime preprocessing currently requires convexHull"
                    )
                collider_count += 1
            if not prim.IsA(UsdGeom.Mesh) or not _has_collision_ancestor(prim, link):
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
            mesh_to_stage = np.asarray(
                UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                dtype=np.float64,
            ).T
            mesh.apply_transform(stage_to_link @ mesh_to_stage)
            meshes.append(mesh)
        if collider_count == 0 or not meshes:
            raise RuntimeError(f"runtime contact link has no collision mesh: {link_name}")

        # The authored PhysX approximation is one convex hull for this collider.
        # Sampling that hull therefore matches the effective collision surface,
        # including areas spanning concavities in the render/source mesh.
        hull = trimesh.util.concatenate(meshes).convex_hull
        candidates, _ = trimesh.sample.sample_surface(
            hull,
            samples_per_link * 8,
            seed=np.random.default_rng(seed + link_offset),
        )
        sampled.append(
            _farthest_point_subset(np.asarray(candidates), samples_per_link)
        )
    return sampled


def _runtime_schedule_frames(
    app: object,
    config: SimulatorConfig,
    catalog: SimulatorAssetCatalog,
) -> tuple[np.ndarray, np.ndarray]:
    """Return open-to-closed link poses measured from the spawned articulation."""

    import omni.usd
    from isaaclab.assets import Articulation
    from isaaclab.sim import SimulationContext
    from isaaclab.utils.math import subtract_frame_transforms

    from srno.sim.isaac_scene import _gripper_cfg, make_simulation_cfg

    context = omni.usd.get_context()
    context.new_stage()
    for _ in range(5):
        app.update()
    simulation = SimulationContext(make_simulation_cfg(config.device))
    robot = None
    try:
        robot = Articulation(
            _gripper_cfg(catalog, config.relaxation).replace(prim_path="/World/Robot")
        )
        simulation.reset()
        body_indices = []
        for link_name in catalog.gripper.contact_links:
            matches = [
                index for index, name in enumerate(robot.body_names) if name == link_name
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one runtime body named {link_name!r}, got {matches}"
                )
            body_indices.append(matches[0])

        try:
            open_position = torch.tensor(
                [catalog.gripper.open_joint_position_rad[name] for name in robot.joint_names],
                device=robot.device,
                dtype=torch.float32,
            )[None]
            close_position = torch.tensor(
                [catalog.gripper.close_joint_position_rad[name] for name in robot.joint_names],
                device=robot.device,
                dtype=torch.float32,
            )[None]
        except KeyError as error:
            raise RuntimeError(
                f"runtime gripper has unexpected joint {error.args[0]!r}"
            ) from error

        schedule_positions: list[np.ndarray] = []
        schedule_quaternions: list[np.ndarray] = []
        for fraction in np.linspace(0.0, 1.0, 33, dtype=np.float64):
            joint_position = open_position + float(fraction) * (
                close_position - open_position
            )
            robot.write_joint_state_to_sim(
                joint_position, torch.zeros_like(joint_position)
            )
            robot.set_joint_position_target(joint_position)
            robot.write_data_to_sim()
            simulation.step(render=False)
            robot.update(simulation.get_physics_dt())

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
            schedule_positions.append(
                relative_position[0, body_indices].detach().cpu().numpy()
            )
            schedule_quaternions.append(
                relative_quaternion[0, body_indices].detach().cpu().numpy()
            )
        return np.stack(schedule_positions), np.stack(schedule_quaternions)
    finally:
        del robot
        simulation.clear_all_callbacks()
        simulation.clear_instance()
        del simulation
        gc.collect()
        context.close_stage()
        for _ in range(5):
            app.update()


def _quaternion_wxyz_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def preprocess_runtime_gripper(
    app: object,
    config: SimulatorConfig,
    catalog: SimulatorAssetCatalog,
    *,
    samples_per_link: int = 128,
) -> GripperAsset:
    """Create the production gripper asset from runtime collision geometry."""

    if len(catalog.gripper.contact_links) != 2:
        raise ValueError("SRNO v1 requires exactly two runtime contact links")
    contact_links = tuple(catalog.gripper.contact_links)
    local_samples = _runtime_link_samples(
        catalog.gripper.runtime_usd,
        contact_links,
        samples_per_link=samples_per_link,
        seed=config.seed,
    )
    positions, quaternions = _runtime_schedule_frames(app, config, catalog)

    points_open_to_closed: list[np.ndarray] = []
    for state in range(33):
        transformed = []
        for link_index, points in enumerate(local_samples):
            rotation = _quaternion_wxyz_matrix(quaternions[state, link_index])
            transformed.append(points @ rotation.T + positions[state, link_index])
        points_open_to_closed.append(np.concatenate(transformed, axis=0))
    point_array = np.stack(points_open_to_closed)
    aperture = np.abs(positions[:, 0, 0] - positions[:, 1, 0])
    if not np.all(np.diff(aperture) < 0.0):
        raise RuntimeError("runtime finger-origin aperture is not strictly decreasing")

    aperture_knots = aperture[::-1].copy()
    point_knots = point_array[::-1].copy()
    slope = (point_knots[-1] - point_knots[0]) / (
        aperture_knots[-1] - aperture_knots[0]
    )
    intercept = point_knots[0] - aperture_knots[0] * slope
    digest = hashlib.sha256(catalog.gripper.runtime_usd.read_bytes()).hexdigest()
    if digest != catalog.gripper.runtime_usd_sha256:
        raise RuntimeError("runtime gripper USD hash does not match the frozen catalog")
    return GripperAsset(
        torch.from_numpy(intercept.astype(np.float32)),
        torch.from_numpy(slope.astype(np.float32)),
        torch.from_numpy(
            np.repeat(np.arange(2, dtype=np.int64), samples_per_link)
        ),
        float(aperture_knots[0]),
        float(aperture_knots[-1]),
        float(aperture_knots[-1]),
        digest,
        torch.from_numpy(aperture_knots.astype(np.float32)),
        torch.from_numpy(point_knots.astype(np.float32)),
    )
