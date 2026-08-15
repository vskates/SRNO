"""PhysX-cooked collision extraction and dense object-frame SDF generation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import trimesh

from srno.sim.config import SDFGenerationConfig


@dataclass(frozen=True)
class DenseSDF:
    values: np.ndarray
    origin_xyz: np.ndarray
    voxel_size_xyz: np.ndarray
    geometry_sha256: str = ""
    representation: str = "unspecified"

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("SDF values must have shape [depth, height, width]")
        if self.origin_xyz.shape != (3,) or self.voxel_size_xyz.shape != (3,):
            raise ValueError("SDF origin and voxel size must have shape [3]")
        if not np.all(self.voxel_size_xyz > 0.0):
            raise ValueError("SDF voxel size must be positive")


def _triangulate(face_counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    cursor = 0
    for raw_count in face_counts:
        count = int(raw_count)
        polygon = indices[cursor : cursor + count]
        cursor += count
        if count < 3:
            continue
        triangles.extend((int(polygon[0]), int(polygon[i]), int(polygon[i + 1])) for i in range(1, count - 1))
    return np.asarray(triangles, dtype=np.int64)


def _triangulate_convex(convex: object) -> np.ndarray:
    """Triangulate PhysX's polygon representation without changing vertices."""

    triangles: list[tuple[int, int, int]] = []
    indices = list(convex.indices)
    for polygon in convex.polygons:
        face = indices[polygon.index_base : polygon.index_base + polygon.num_vertices]
        triangles.extend(
            (int(face[0]), int(face[index]), int(face[index + 1]))
            for index in range(1, len(face) - 1)
        )
    return np.asarray(triangles, dtype=np.int64)


def _has_collision_api(prim: object, stop_path: str) -> bool:
    from pxr import UsdPhysics

    cursor = prim
    while cursor and cursor.IsValid():
        if cursor.HasAPI(UsdPhysics.CollisionAPI):
            return True
        if str(cursor.GetPath()) == stop_path:
            break
        cursor = cursor.GetParent()
    return False


def load_object_mesh_from_usd(path: str | Path) -> trimesh.Trimesh:
    """Load composed meshes in coordinates of the USD default prim.

    Collision-tagged geometry is preferred when the asset provides it; otherwise
    visible mesh geometry is used. USD matrices are transposed when handed to
    trimesh because Gf uses row-vector convention.
    """

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(Path(path).resolve()))
    if stage is None:
        raise ValueError(f"cannot open USD stage: {path}")
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        roots = list(stage.GetPseudoRoot().GetChildren())
        if len(roots) != 1:
            raise ValueError(f"USD must have a default prim or one root: {path}")
        root = roots[0]
    root_path = str(root.GetPath())
    all_meshes: list[trimesh.Trimesh] = []
    collision_meshes: list[trimesh.Trimesh] = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        geometry = UsdGeom.Mesh(prim)
        points = geometry.GetPointsAttr().Get()
        face_counts = geometry.GetFaceVertexCountsAttr().Get()
        face_indices = geometry.GetFaceVertexIndicesAttr().Get()
        if points is None or face_counts is None or face_indices is None:
            continue
        faces = _triangulate(np.asarray(face_counts), np.asarray(face_indices))
        if len(faces) == 0:
            continue
        mesh = trimesh.Trimesh(
            vertices=np.asarray(points, dtype=np.float64),
            faces=faces,
            process=False,
        )
        world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        # Keep the default prim's authored scale: validation spawns the USD at
        # unit scale, so this scale is part of the physical object geometry.
        matrix = np.asarray(world, dtype=np.float64).T
        mesh.apply_transform(matrix)
        all_meshes.append(mesh)
        if _has_collision_api(prim, root_path):
            collision_meshes.append(mesh)
    selected = collision_meshes or all_meshes
    if not selected:
        raise ValueError(f"no mesh geometry found in {path}")
    combined = trimesh.util.concatenate(selected)
    combined.process(validate=True)
    trimesh.repair.fix_normals(combined, multibody=True)
    return combined


def load_physx_cooked_hulls_from_usd(path: str | Path) -> tuple[trimesh.Trimesh, ...]:
    """Return the convex shapes produced by PhysX for an object's collider.

    An Isaac/Kit application must already be running.  Importantly, this asks
    the PhysX cooking interface for its collision representation; it does not
    use the authored triangle surface as an SDF surrogate.
    """

    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils

    source = Path(path).resolve()
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise ValueError(f"cannot open USD stage: {source}")
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        roots = list(stage.GetPseudoRoot().GetChildren())
        if len(roots) != 1:
            raise ValueError(f"USD must have a default prim or one root: {source}")
        root = roots[0]

    collision_prims = [
        prim
        for prim in Usd.PrimRange(root)
        if prim.IsA(UsdGeom.Mesh)
        and prim.HasAPI(UsdPhysics.CollisionAPI)
        and prim.HasAPI(UsdPhysics.MeshCollisionAPI)
    ]
    if not collision_prims:
        raise ValueError(f"no PhysX mesh collider found in {source}")

    stage_cache = UsdUtils.StageCache.Get()
    cache_id = stage_cache.Insert(stage)
    stage_id = cache_id.ToLongInt()
    cooking = get_physx_cooking_interface()
    hulls: list[trimesh.Trimesh] = []
    try:
        for prim in collision_prims:
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            if approximation != UsdPhysics.Tokens.convexDecomposition:
                raise ValueError(
                    f"{source}:{prim.GetPath()} uses {approximation!r}; "
                    "SRNO object SDF requires PhysX convexDecomposition"
                )
            captured: dict[str, object] = {}

            def on_ready(result: object, convexes: list[object]) -> None:
                captured["result"] = result
                captured["convexes"] = convexes

            cooking.request_convex_collision_representation(
                stage_id=stage_id,
                collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())),
                run_asynchronously=False,
                on_result=on_ready,
            )
            if captured.get("result") != PhysxCollisionRepresentationResult.RESULT_VALID:
                raise RuntimeError(
                    f"PhysX cooking failed for {source}:{prim.GetPath()}: "
                    f"{captured.get('result')}"
                )
            transform = np.asarray(
                UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                dtype=np.float64,
            ).T
            for convex in captured.get("convexes", []):
                vertices = np.asarray(
                    [[vertex.x, vertex.y, vertex.z] for vertex in convex.vertices],
                    dtype=np.float64,
                )
                homogeneous = np.column_stack((vertices, np.ones(len(vertices))))
                vertices = (homogeneous @ transform.T)[:, :3]
                mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=_triangulate_convex(convex),
                    process=False,
                )
                mesh.fix_normals(multibody=False)
                # The cooking API returns convexes by contract.  Rechecking
                # ``is_convex`` in trimesh rejects valid PhysX hulls when
                # float32 cooking leaves nearly-coplanar faces a few ulps apart.
                if not mesh.is_watertight:
                    raise RuntimeError(
                        f"PhysX returned an open cooked hull for "
                        f"{source}:{prim.GetPath()}"
                    )
                hulls.append(mesh)
    finally:
        stage_cache.Erase(stage)
    if not hulls:
        raise RuntimeError(f"PhysX returned no cooked hulls for {source}")
    return tuple(hulls)


SurfaceDistance = Callable[[int | None, np.ndarray], np.ndarray]


def _trimesh_surface_distance(
    hulls: Sequence[trimesh.Trimesh],
) -> SurfaceDistance:
    combined = trimesh.util.concatenate(hulls)
    meshes = (combined, *hulls)

    def query(hull_index: int | None, points: np.ndarray) -> np.ndarray:
        mesh = meshes[0 if hull_index is None else hull_index + 1]
        return np.asarray(trimesh.proximity.closest_point(mesh, points)[1], dtype=np.float64)

    return query


def _convex_inside(mesh: trimesh.Trimesh, points: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    offsets = -np.einsum("ij,ij->i", normals, np.asarray(mesh.triangles_center))
    return np.all(points @ normals.T + offsets[None, :] <= tolerance, axis=1)


def convex_union_signed_distance(
    hulls: Sequence[trimesh.Trimesh],
    points: np.ndarray,
    *,
    surface_distance: SurfaceDistance | None = None,
) -> np.ndarray:
    """Evaluate ``min_i sdf(hull_i)`` with positive values outside.

    Taking the minimum of the individual convex fields is important for
    overlapping decomposed hulls: a hidden component face must not become a
    spurious zero level set inside the collider union.
    """

    meshes = tuple(hulls)
    if not meshes:
        raise ValueError("at least one convex hull is required")
    query_points = np.asarray(points, dtype=np.float64)
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    distance = surface_distance or _trimesh_surface_distance(meshes)
    result = distance(None, query_points)
    for index, mesh in enumerate(meshes):
        inside = _convex_inside(mesh, query_points)
        if np.any(inside):
            result[inside] = np.minimum(
                result[inside],
                -distance(index, query_points[inside]),
            )
    return np.asarray(result, dtype=np.float32)


def _geometry_sha256(hulls: Sequence[trimesh.Trimesh]) -> str:
    digest = hashlib.sha256()
    digest.update(b"srno-physx-cooked-convex-union-v1\0")
    for mesh in hulls:
        vertices = np.asarray(mesh.vertices, dtype="<f4")
        faces = np.asarray(mesh.faces, dtype="<i4")
        digest.update(np.asarray(vertices.shape, dtype="<i8").tobytes())
        digest.update(vertices.tobytes())
        digest.update(np.asarray(faces.shape, dtype="<i8").tobytes())
        digest.update(faces.tobytes())
    return digest.hexdigest()


def generate_dense_sdf(
    geometry: trimesh.Trimesh | Sequence[trimesh.Trimesh],
    cfg: SDFGenerationConfig,
    *,
    surface_distance: SurfaceDistance | None = None,
) -> DenseSDF:
    hulls = (geometry,) if isinstance(geometry, trimesh.Trimesh) else tuple(geometry)
    if not hulls:
        raise ValueError("at least one convex hull is required")
    for hull in hulls:
        hull.fix_normals(multibody=False)
        if not hull.is_watertight:
            raise ValueError("SDF geometry must consist of closed convex hulls")
    combined = trimesh.util.concatenate(hulls)
    bounds = np.asarray(combined.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or np.any(bounds[1] <= bounds[0]):
        raise ValueError("object mesh has invalid bounds")
    lower = bounds[0] - cfg.padding_m
    upper = bounds[1] + cfg.padding_m
    voxel_size = (upper - lower) / float(cfg.resolution - 1)
    axes = [
        np.linspace(lower[axis], upper[axis], cfg.resolution, dtype=np.float64)
        for axis in range(3)
    ]
    values = np.empty((cfg.resolution, cfg.resolution, cfg.resolution), dtype=np.float32)
    flat = values.reshape(-1)
    total = flat.size
    yz_plane = cfg.resolution * cfg.resolution
    for start in range(0, total, cfg.chunk_points):
        stop = min(start + cfg.chunk_points, total)
        linear = np.arange(start, stop, dtype=np.int64)
        z_index = linear // yz_plane
        remainder = linear % yz_plane
        y_index = remainder // cfg.resolution
        x_index = remainder % cfg.resolution
        query = np.column_stack((axes[0][x_index], axes[1][y_index], axes[2][z_index]))
        flat[start:stop] = convex_union_signed_distance(
            hulls,
            query,
            surface_distance=surface_distance,
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("SDF generation produced non-finite values")
    return DenseSDF(
        values,
        lower.astype(np.float32),
        voxel_size.astype(np.float32),
        _geometry_sha256(hulls),
        "physx_cooked_convex_decomposition",
    )


def load_or_generate_sdf(
    usd_path: str | Path,
    cache_path: str | Path,
    cfg: SDFGenerationConfig,
    *,
    distance_backend: str = "trimesh",
    device: str = "cpu",
) -> DenseSDF:
    source = Path(usd_path).resolve()
    cache = Path(cache_path).resolve()
    identity = _cache_identity(source, cfg)
    if cache.is_file():
        try:
            with np.load(cache, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))
                if metadata == identity:
                    return DenseSDF(
                        np.asarray(archive["values"], dtype=np.float32),
                        np.asarray(archive["origin_xyz"], dtype=np.float32),
                        np.asarray(archive["voxel_size_xyz"], dtype=np.float32),
                        str(archive["geometry_sha256"].item()),
                        str(archive["representation"].item()),
                    )
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            pass
    hulls = load_physx_cooked_hulls_from_usd(source)
    if distance_backend == "warp":
        from srno.sim.warp_sdf import WarpConvexSurfaceDistance

        distance_query: SurfaceDistance | None = WarpConvexSurfaceDistance(hulls, device=device)
    elif distance_backend == "trimesh":
        distance_query = None
    else:
        raise ValueError(f"unsupported SDF distance backend: {distance_backend!r}")
    dense = generate_dense_sdf(hulls, cfg, surface_distance=distance_query)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(cache.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            values=dense.values,
            origin_xyz=dense.origin_xyz,
            voxel_size_xyz=dense.voxel_size_xyz,
            geometry_sha256=np.asarray(dense.geometry_sha256),
            representation=np.asarray(dense.representation),
            metadata=np.asarray(json.dumps(identity, sort_keys=True)),
        )
    os.replace(temporary, cache)
    return dense


def sdf_cache_is_current(
    usd_path: str | Path,
    cache_path: str | Path,
    cfg: SDFGenerationConfig,
) -> bool:
    cache = Path(cache_path)
    if not cache.is_file():
        return False
    try:
        with np.load(cache, allow_pickle=False) as archive:
            required = {
                "values",
                "origin_xyz",
                "voxel_size_xyz",
                "geometry_sha256",
                "representation",
                "metadata",
            }
            if not required.issubset(archive.files):
                return False
            metadata = json.loads(str(archive["metadata"].item()))
            return metadata == _cache_identity(Path(usd_path).resolve(), cfg)
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def _cache_identity(source: Path, cfg: SDFGenerationConfig) -> dict[str, object]:
    try:
        isaacsim_version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError:
        isaacsim_version = "unavailable"
    return {
        "format_version": 3,
        "representation": "physx_cooked_convex_decomposition",
        "isaacsim_version": isaacsim_version,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "resolution": cfg.resolution,
        "padding_m": cfg.padding_m,
    }
