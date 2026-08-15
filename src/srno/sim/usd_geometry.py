"""USD mesh extraction and dense object-frame SDF generation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from srno.sim.config import SDFGenerationConfig


@dataclass(frozen=True)
class DenseSDF:
    values: np.ndarray
    origin_xyz: np.ndarray
    voxel_size_xyz: np.ndarray

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


def generate_dense_sdf(mesh: trimesh.Trimesh, cfg: SDFGenerationConfig) -> DenseSDF:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
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
        # trimesh returns positive values inside; SRNO's contract is positive outside.
        flat[start:stop] = -trimesh.proximity.signed_distance(mesh, query).astype(np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("SDF generation produced non-finite values")
    return DenseSDF(values, lower.astype(np.float32), voxel_size.astype(np.float32))


def load_or_generate_sdf(
    usd_path: str | Path,
    cache_path: str | Path,
    cfg: SDFGenerationConfig,
) -> DenseSDF:
    source = Path(usd_path).resolve()
    cache = Path(cache_path).resolve()
    identity = _cache_identity(source, cfg)
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata == identity:
                return DenseSDF(
                    np.asarray(archive["values"], dtype=np.float32),
                    np.asarray(archive["origin_xyz"], dtype=np.float32),
                    np.asarray(archive["voxel_size_xyz"], dtype=np.float32),
                )
    dense = generate_dense_sdf(load_object_mesh_from_usd(source), cfg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(cache.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            values=dense.values,
            origin_xyz=dense.origin_xyz,
            voxel_size_xyz=dense.voxel_size_xyz,
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
            metadata = json.loads(str(archive["metadata"].item()))
            return metadata == _cache_identity(Path(usd_path).resolve(), cfg)
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def _cache_identity(source: Path, cfg: SDFGenerationConfig) -> dict[str, object]:
    return {
        "format_version": 2,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "resolution": cfg.resolution,
        "padding_m": cfg.padding_m,
    }
