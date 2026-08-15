"""Compact, simulator-independent grasp-pose seed files.

The validation repository stores poses as verbose JSON 4x4 matrices.  SRNO keeps
the same base/grasp poses, but converts them once to a compressed NumPy archive so
the Isaac collector can load a single object in milliseconds.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


POSE_SEED_FORMAT_VERSION = 1


@dataclass(frozen=True)
class PoseSeeds:
    position_m: np.ndarray
    quaternion_wxyz: np.ndarray
    validation_success: np.ndarray
    source_sha256: str = ""
    source_index: np.ndarray | None = None

    def __post_init__(self) -> None:
        count = len(self.position_m)
        if self.position_m.shape != (count, 3):
            raise ValueError("position_m must have shape [poses, 3]")
        if self.quaternion_wxyz.shape != (count, 4):
            raise ValueError("quaternion_wxyz must have shape [poses, 4]")
        if self.validation_success.shape != (count,):
            raise ValueError("validation_success must have shape [poses]")
        source_index = (
            np.arange(count, dtype=np.int64)
            if self.source_index is None
            else np.asarray(self.source_index, dtype=np.int64)
        )
        if source_index.shape != (count,) or np.any(source_index < 0):
            raise ValueError("source_index must have shape [poses] and be non-negative")
        object.__setattr__(self, "source_index", source_index)
        if count == 0:
            raise ValueError("pose seed file must contain at least one pose")
        if not np.all(np.isfinite(self.position_m)) or not np.all(
            np.isfinite(self.quaternion_wxyz)
        ):
            raise ValueError("pose seeds must be finite")
        norms = np.linalg.norm(self.quaternion_wxyz, axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-5, rtol=0.0):
            raise ValueError("pose-seed quaternions must be normalized")

    def select(
        self,
        count: int | None,
        *,
        successful_only: bool,
        seed: int,
    ) -> "PoseSeeds":
        eligible = np.flatnonzero(self.validation_success) if successful_only else np.arange(len(self.position_m))
        if len(eligible) == 0:
            raise ValueError("no pose seeds satisfy successful_only")
        if count is None:
            chosen = eligible
        else:
            if count <= 0:
                raise ValueError("pose count must be positive")
            generator = np.random.default_rng(seed)
            chosen = generator.choice(eligible, size=count, replace=count > len(eligible))
        return PoseSeeds(
            self.position_m[chosen].copy(),
            self.quaternion_wxyz[chosen].copy(),
            self.validation_success[chosen].copy(),
            self.source_sha256,
            self.source_index[chosen].copy(),
        )

    def take(
        self,
        indices: list[int] | tuple[int, ...],
        *,
        require_successful: bool,
    ) -> "PoseSeeds":
        """Select explicit source indices for reproducible simulator inspection."""

        chosen = np.asarray(indices, dtype=np.int64)
        if chosen.ndim != 1 or len(chosen) == 0:
            raise ValueError("pose indices must be a non-empty one-dimensional sequence")
        if len(np.unique(chosen)) != len(chosen):
            raise ValueError("pose indices must not contain duplicates")
        if np.any(chosen < 0) or np.any(chosen >= len(self.position_m)):
            raise IndexError(
                f"pose index outside [0, {len(self.position_m) - 1}]"
            )
        if require_successful and not np.all(self.validation_success[chosen]):
            failed = chosen[~self.validation_success[chosen]].tolist()
            raise ValueError(f"pose indices are not validation-successful: {failed}")
        return PoseSeeds(
            self.position_m[chosen].copy(),
            self.quaternion_wxyz[chosen].copy(),
            self.validation_success[chosen].copy(),
            self.source_sha256,
            self.source_index[chosen].copy(),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        metadata = json.dumps(
            {
                "format_version": POSE_SEED_FORMAT_VERSION,
                "units": "m",
                "quaternion_order": "wxyz",
                "pose": "gripper_base_in_object_frame",
                "source_sha256": self.source_sha256,
            },
            sort_keys=True,
        )
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                position_m=np.asarray(self.position_m, dtype=np.float32),
                quaternion_wxyz=np.asarray(self.quaternion_wxyz, dtype=np.float32),
                validation_success=np.asarray(self.validation_success, dtype=np.bool_),
                metadata=np.asarray(metadata),
            )
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: str | Path) -> "PoseSeeds":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            expected = (POSE_SEED_FORMAT_VERSION, "m", "wxyz", "gripper_base_in_object_frame")
            actual = (
                metadata.get("format_version"),
                metadata.get("units"),
                metadata.get("quaternion_order"),
                metadata.get("pose"),
            )
            if actual != expected:
                raise ValueError(f"unsupported pose-seed conventions: {actual}")
            return cls(
                np.asarray(archive["position_m"], dtype=np.float32),
                np.asarray(archive["quaternion_wxyz"], dtype=np.float32),
                np.asarray(archive["validation_success"], dtype=np.bool_),
                str(metadata.get("source_sha256", "")),
            )


def _matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Vectorized, sign-canonical rotation-matrix conversion."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.ndim != 3 or matrix.shape[1:] != (3, 3):
        raise ValueError("rotation must have shape [poses, 3, 3]")
    result = np.empty((len(matrix), 4), dtype=np.float64)
    for index, value in enumerate(matrix):
        trace = float(np.trace(value))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            quaternion = np.array(
                [0.25 * scale, (value[2, 1] - value[1, 2]) / scale,
                 (value[0, 2] - value[2, 0]) / scale, (value[1, 0] - value[0, 1]) / scale]
            )
        else:
            axis = int(np.argmax(np.diag(value)))
            if axis == 0:
                scale = np.sqrt(max(0.0, 1.0 + value[0, 0] - value[1, 1] - value[2, 2])) * 2.0
                quaternion = np.array(
                    [(value[2, 1] - value[1, 2]) / scale, 0.25 * scale,
                     (value[0, 1] + value[1, 0]) / scale, (value[0, 2] + value[2, 0]) / scale]
                )
            elif axis == 1:
                scale = np.sqrt(max(0.0, 1.0 + value[1, 1] - value[0, 0] - value[2, 2])) * 2.0
                quaternion = np.array(
                    [(value[0, 2] - value[2, 0]) / scale,
                     (value[0, 1] + value[1, 0]) / scale, 0.25 * scale,
                     (value[1, 2] + value[2, 1]) / scale]
                )
            else:
                scale = np.sqrt(max(0.0, 1.0 + value[2, 2] - value[0, 0] - value[1, 1])) * 2.0
                quaternion = np.array(
                    [(value[1, 0] - value[0, 1]) / scale,
                     (value[0, 2] + value[2, 0]) / scale,
                     (value[1, 2] + value[2, 1]) / scale, 0.25 * scale]
                )
        quaternion /= np.linalg.norm(quaternion)
        if quaternion[0] < 0.0:
            quaternion *= -1.0
        result[index] = quaternion
    return result.astype(np.float32)


def import_validation_pose_json(source: str | Path, destination: str | Path) -> PoseSeeds:
    """Convert one ``grasp_dataset/grasp_data/astribot/*.json`` file."""

    source_path = Path(source)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    grasps = payload.get("grasps", {})
    transforms = np.asarray(grasps.get("transforms"), dtype=np.float64)
    successes = np.asarray(grasps.get("object_in_gripper"), dtype=np.bool_)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(f"{source_path}: grasps.transforms must have shape [poses, 4, 4]")
    if successes.shape != (len(transforms),):
        raise ValueError(f"{source_path}: object_in_gripper length does not match transforms")
    if not np.allclose(transforms[:, 3], (0.0, 0.0, 0.0, 1.0), atol=1e-7):
        raise ValueError(f"{source_path}: invalid homogeneous transform row")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    seeds = PoseSeeds(
        transforms[:, :3, 3].astype(np.float32),
        _matrix_to_quaternion_wxyz(transforms[:, :3, :3]),
        successes,
        source_sha256,
    )
    seeds.save(destination)
    return seeds


def import_validation_pose_directory(
    source_directory: str | Path,
    destination_directory: str | Path,
    object_ids: tuple[str, ...] | list[str],
) -> dict[str, int]:
    source_root = Path(source_directory)
    destination_root = Path(destination_directory)
    result: dict[str, int] = {}
    for object_id in object_ids:
        seeds = import_validation_pose_json(
            source_root / f"{object_id}.json", destination_root / f"{object_id}.npz"
        )
        result[object_id] = len(seeds.position_m)
    return result
