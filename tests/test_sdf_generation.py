from __future__ import annotations

import numpy as np
import trimesh

from srno.sim.config import SDFGenerationConfig
from srno.sim.usd_geometry import convex_union_signed_distance, generate_dense_sdf


def test_generated_dense_sdf_is_negative_inside_and_positive_outside() -> None:
    mesh = trimesh.creation.box(extents=(0.1, 0.08, 0.06))
    dense = generate_dense_sdf(
        mesh,
        SDFGenerationConfig(resolution=9, padding_m=0.02, chunk_points=100),
    )
    assert dense.values.shape == (9, 9, 9)
    assert dense.values[4, 4, 4] < 0.0
    assert dense.values[0, 0, 0] > 0.0
    assert np.all(dense.voxel_size_xyz > 0.0)


def test_convex_union_has_no_zero_level_set_on_hidden_hull_faces() -> None:
    left = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    right = trimesh.creation.box(
        extents=(2.0, 2.0, 2.0),
        transform=trimesh.transformations.translation_matrix((1.0, 0.0, 0.0)),
    )
    values = convex_union_signed_distance(
        (left, right),
        np.asarray(
            [
                [0.0, 0.0, 0.0],  # hidden left face of the right hull
                [1.5, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        ),
    )
    np.testing.assert_allclose(values, (-1.0, -0.5, 1.0), atol=1e-7)
