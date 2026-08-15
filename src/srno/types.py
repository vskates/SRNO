from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PoseState:
    """Object-to-gripper pose and actual gripper aperture.

    All tensors share the same leading dimensions. ``rotation`` ends in (3, 3),
    ``position`` ends in (3,), and ``aperture`` has only the leading dimensions.
    """

    rotation: Tensor
    position: Tensor
    aperture: Tensor

    def __post_init__(self) -> None:
        if self.rotation.shape[-2:] != (3, 3):
            raise ValueError("rotation must end in (3, 3)")
        if self.position.shape[-1:] != (3,):
            raise ValueError("position must end in (3,)")
        if self.rotation.shape[:-2] != self.position.shape[:-1]:
            raise ValueError("rotation and position leading shapes differ")
        if self.rotation.shape[:-2] != self.aperture.shape:
            raise ValueError("rotation and aperture leading shapes differ")

    @property
    def shape(self) -> torch.Size:
        return self.aperture.shape

    @property
    def device(self) -> torch.device:
        return self.rotation.device

    def to(self, *args: object, **kwargs: object) -> "PoseState":
        return PoseState(
            self.rotation.to(*args, **kwargs),
            self.position.to(*args, **kwargs),
            self.aperture.to(*args, **kwargs),
        )

    def detach(self) -> "PoseState":
        return PoseState(
            self.rotation.detach(), self.position.detach(), self.aperture.detach()
        )

    def index_select(self, dim: int, index: Tensor) -> "PoseState":
        return PoseState(
            self.rotation.index_select(dim, index),
            self.position.index_select(dim, index),
            self.aperture.index_select(dim, index),
        )

    @staticmethod
    def stack(states: list["PoseState"], dim: int = 0) -> "PoseState":
        return PoseState(
            torch.stack([s.rotation for s in states], dim=dim),
            torch.stack([s.position for s in states], dim=dim),
            torch.stack([s.aperture for s in states], dim=dim),
        )


@dataclass(frozen=True)
class SDFBatch:
    """Unique object SDF grids plus a state-sample to object mapping."""

    values: Tensor
    origin: Tensor
    voxel_size: Tensor
    sample_to_object: Tensor
    outside_value: float

    def __post_init__(self) -> None:
        if self.values.ndim != 4:
            raise ValueError("values must have shape [objects, depth, height, width]")
        objects = self.values.shape[0]
        if self.origin.shape != (objects, 3):
            raise ValueError("origin must have shape [objects, 3]")
        if self.voxel_size.shape != (objects, 3):
            raise ValueError("voxel_size must have shape [objects, 3]")
        if self.sample_to_object.ndim != 1:
            raise ValueError("sample_to_object must be one-dimensional")
        if self.sample_to_object.numel() and (
            self.sample_to_object.min() < 0
            or self.sample_to_object.max() >= objects
        ):
            raise ValueError("sample_to_object contains an invalid object index")

    def to(self, *args: object, **kwargs: object) -> "SDFBatch":
        converted_values = self.values.to(*args, **kwargs)
        return SDFBatch(
            converted_values,
            self.origin.to(*args, **kwargs),
            self.voxel_size.to(*args, **kwargs),
            self.sample_to_object.to(device=converted_values.device),
            self.outside_value,
        )
