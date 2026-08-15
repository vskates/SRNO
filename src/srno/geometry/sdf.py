from __future__ import annotations

import torch
from torch import Tensor


def sample_sdf(
    values: Tensor,
    origin: Tensor,
    voxel_size: Tensor,
    coordinates: Tensor,
    *,
    sample_to_object: Tensor | None = None,
    outside_value: float,
) -> Tensor:
    """Differentiably sample unique object SDFs without repeating their grids.

    ``values`` is ``[O, D, H, W]`` with storage order Z/Y/X. ``coordinates`` is
    ``[B, ..., 3]`` in metric XYZ coordinates. ``sample_to_object[B]`` maps each
    leading sample to a unique object grid.
    """

    if values.ndim != 4:
        raise ValueError("values must have shape [objects, depth, height, width]")
    if coordinates.ndim < 2 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape [samples, ..., 3]")
    object_count = values.shape[0]
    sample_count = coordinates.shape[0]
    if origin.shape != (object_count, 3) or voxel_size.shape != (object_count, 3):
        raise ValueError("origin and voxel_size must have shape [objects, 3]")
    if torch.any(voxel_size <= 0):
        raise ValueError("voxel_size must be positive")
    if sample_to_object is None:
        if sample_count != object_count:
            raise ValueError("sample_to_object is required when samples != objects")
        sample_to_object = torch.arange(object_count, device=coordinates.device)
    if sample_to_object.shape != (sample_count,):
        raise ValueError("sample_to_object must have shape [samples]")

    mapping = sample_to_object.to(device=coordinates.device, dtype=torch.long)
    selected_origin = origin.to(coordinates.device).index_select(0, mapping)
    selected_voxel = voxel_size.to(coordinates.device).index_select(0, mapping)
    extra_dims = (1,) * (coordinates.ndim - 2)
    grid_coordinates = (coordinates - selected_origin.view(sample_count, *extra_dims, 3))
    grid_coordinates = grid_coordinates / selected_voxel.view(sample_count, *extra_dims, 3)

    output_shape = coordinates.shape[:-1]
    flat_coordinates = grid_coordinates.reshape(-1, 3)
    points_per_sample = flat_coordinates.shape[0] // sample_count
    flat_objects = mapping[:, None].expand(sample_count, points_per_sample).reshape(-1)

    x, y, z = flat_coordinates.unbind(-1)
    depth, height, width = values.shape[1:]
    in_bounds = (
        (x >= 0)
        & (x <= width - 1)
        & (y >= 0)
        & (y <= height - 1)
        & (z >= 0)
        & (z <= depth - 1)
    )
    x0 = torch.floor(x).to(torch.long).clamp(0, width - 1)
    y0 = torch.floor(y).to(torch.long).clamp(0, height - 1)
    z0 = torch.floor(z).to(torch.long).clamp(0, depth - 1)
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)
    z1 = (z0 + 1).clamp(max=depth - 1)
    wx = x - x0.to(x.dtype)
    wy = y - y0.to(y.dtype)
    wz = z - z0.to(z.dtype)

    values = values.to(coordinates.device)

    def gather(zi: Tensor, yi: Tensor, xi: Tensor) -> Tensor:
        return values[flat_objects, zi, yi, xi].to(coordinates.dtype)

    c000 = gather(z0, y0, x0)
    c001 = gather(z0, y0, x1)
    c010 = gather(z0, y1, x0)
    c011 = gather(z0, y1, x1)
    c100 = gather(z1, y0, x0)
    c101 = gather(z1, y0, x1)
    c110 = gather(z1, y1, x0)
    c111 = gather(z1, y1, x1)

    c00 = c000 * (1 - wx) + c001 * wx
    c01 = c010 * (1 - wx) + c011 * wx
    c10 = c100 * (1 - wx) + c101 * wx
    c11 = c110 * (1 - wx) + c111 * wx
    c0 = c00 * (1 - wy) + c01 * wy
    c1 = c10 * (1 - wy) + c11 * wy
    sampled = c0 * (1 - wz) + c1 * wz
    sampled = torch.where(in_bounds, sampled, sampled.new_full((), outside_value))
    return sampled.reshape(output_shape)

