from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

from srno.data.schema import NUM_STATES, NUM_STEPS, SCHEMA_VERSION


DIAGNOSTIC_SPECS: dict[str, tuple[int, ...]] = {
    "contact_count": (),
    "actuator_effort": (),
    "max_penetration": (),
    "linear_velocity": (3,),
    "angular_velocity": (3,),
    "settling_substeps": (),
}


class H5DatasetWriter:
    """Atomic, simulator-agnostic writer for one object-centric HDF5 shard."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        self._file: h5py.File | None = None
        self.object_ids: list[str] = []

    def __enter__(self) -> "H5DatasetWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.temporary_path.exists():
            self.temporary_path.unlink()
        self._file = h5py.File(self.temporary_path, "w", libver="latest")
        self._file.attrs["schema_version"] = SCHEMA_VERSION
        self._file.attrs["units"] = "m"
        self._file.attrs["pose_convention"] = "object_to_gripper"
        self._file.attrs["quaternion_order"] = "xyzw"
        self._file.attrs["sdf_sign"] = "positive_outside"
        self._file.create_group("objects")
        return self

    def add_object(
        self,
        object_id: str,
        *,
        sdf: np.ndarray,
        grid_origin: np.ndarray,
        voxel_size: np.ndarray | float,
        position: np.ndarray,
        quaternion_xyzw: np.ndarray,
        actual_aperture: np.ndarray,
        source_pose_index: np.ndarray | None = None,
        diagnostics: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        if self._file is None:
            raise RuntimeError("writer must be used as a context manager")
        if not object_id or object_id in self.object_ids:
            raise ValueError("object_id must be non-empty and unique within a shard")
        sdf = np.asarray(sdf)
        if sdf.ndim != 3 or min(sdf.shape) < 2:
            raise ValueError("sdf must be a three-dimensional grid with each side >= 2")
        position = np.asarray(position, dtype=np.float32)
        quaternion_xyzw = np.asarray(quaternion_xyzw, dtype=np.float32)
        actual_aperture = np.asarray(actual_aperture, dtype=np.float32)
        if position.ndim != 3 or position.shape[1:] != (NUM_STATES, 3):
            raise ValueError(f"position must have shape [trajectories, {NUM_STATES}, 3]")
        trajectories = position.shape[0]
        if trajectories == 0:
            raise ValueError("an object must contain at least one trajectory")
        if quaternion_xyzw.shape != (trajectories, NUM_STATES, 4):
            raise ValueError("quaternion_xyzw has an invalid shape")
        if actual_aperture.shape != (trajectories, NUM_STATES):
            raise ValueError("actual_aperture has an invalid shape")
        if source_pose_index is not None:
            source_pose_index = np.asarray(source_pose_index, dtype=np.int64)
            if source_pose_index.shape != (trajectories,) or np.any(source_pose_index < 0):
                raise ValueError(
                    "source_pose_index must have shape [trajectories] and be non-negative"
                )
        origin = np.broadcast_to(np.asarray(grid_origin, dtype=np.float32), (3,)).copy()
        spacing = np.broadcast_to(np.asarray(voxel_size, dtype=np.float32), (3,)).copy()
        if np.any(spacing <= 0):
            raise ValueError("voxel_size must be positive")

        group = self._file["objects"].create_group(f"{len(self.object_ids):06d}")
        group.attrs["object_id"] = object_id
        group.attrs["grid_origin"] = origin
        group.attrs["voxel_size"] = spacing
        group.create_dataset(
            "sdf",
            data=sdf.astype(np.float16),
            chunks=sdf.shape,
            compression="lzf",
            shuffle=True,
        )
        chunk_trajectories = min(64, max(1, trajectories))
        group.create_dataset(
            "position",
            data=position,
            chunks=(chunk_trajectories, NUM_STATES, 3),
            compression="lzf",
            shuffle=True,
        )
        group.create_dataset(
            "quaternion_xyzw",
            data=quaternion_xyzw,
            chunks=(chunk_trajectories, NUM_STATES, 4),
            compression="lzf",
            shuffle=True,
        )
        group.create_dataset(
            "actual_aperture",
            data=actual_aperture,
            chunks=(chunk_trajectories, NUM_STATES),
            compression="lzf",
            shuffle=True,
        )
        if source_pose_index is not None:
            group.create_dataset(
                "source_pose_index",
                data=source_pose_index,
                chunks=(chunk_trajectories,),
                compression="lzf",
                shuffle=True,
            )

        if diagnostics:
            debug_group = group.create_group("diagnostics")
            unknown = set(diagnostics) - set(DIAGNOSTIC_SPECS)
            if unknown:
                raise ValueError(f"unknown diagnostic fields: {sorted(unknown)}")
            for name, values in diagnostics.items():
                array = np.asarray(values)
                expected = (trajectories, NUM_STEPS) + DIAGNOSTIC_SPECS[name]
                if array.shape != expected:
                    raise ValueError(f"diagnostic {name!r} must have shape {expected}")
                debug_group.create_dataset(
                    name,
                    data=array,
                    chunks=(chunk_trajectories, NUM_STEPS) + DIAGNOSTIC_SPECS[name],
                    compression="lzf",
                    shuffle=True,
                )
        self.object_ids.append(object_id)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
        if exc_type is None:
            os.replace(self.temporary_path, self.path)
        elif self.temporary_path.exists():
            self.temporary_path.unlink()
