"""Fast closest-surface queries for PhysX-cooked SDF generation.

This module is imported only by the Isaac SDF worker.  Warp is bundled with
Isaac Sim and deliberately remains an optional simulator-side dependency.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import trimesh
import warp as wp


@wp.kernel(enable_backward=False)
def _unsigned_mesh_distance(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=wp.float32),
):
    index = wp.tid()
    point = points[index]
    query = wp.mesh_query_point_no_sign(mesh, point, 1.0e6)
    if query.result:
        closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        distances[index] = wp.length(point - closest)
    else:
        distances[index] = 1.0e6


class WarpConvexSurfaceDistance:
    """Callable closest-distance backend for a convex-hull union."""

    def __init__(self, hulls: Sequence[trimesh.Trimesh], *, device: str) -> None:
        self.device = device
        combined = trimesh.util.concatenate(tuple(hulls))
        self._meshes = tuple(self._make_mesh(mesh) for mesh in (combined, *hulls))

    def _make_mesh(self, mesh: trimesh.Trimesh) -> wp.Mesh:
        return wp.Mesh(
            points=wp.array(
                np.asarray(mesh.vertices, dtype=np.float32),
                dtype=wp.vec3,
                device=self.device,
            ),
            indices=wp.array(
                np.asarray(mesh.faces, dtype=np.int32).reshape(-1),
                dtype=wp.int32,
                device=self.device,
            ),
            support_winding_number=False,
        )

    def __call__(self, hull_index: int | None, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.empty(0, dtype=np.float32)
        mesh = self._meshes[0 if hull_index is None else hull_index + 1]
        query = wp.array(
            np.asarray(points, dtype=np.float32),
            dtype=wp.vec3,
            device=self.device,
        )
        result = wp.empty(len(points), dtype=wp.float32, device=self.device)
        wp.launch(
            _unsigned_mesh_distance,
            dim=len(points),
            inputs=[mesh.id, query, result],
            device=self.device,
        )
        return result.numpy()
