from __future__ import annotations

import numpy as np
import trimesh

from srno.sim.config import SDFGenerationConfig
from srno.sim.usd_geometry import generate_dense_sdf


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
