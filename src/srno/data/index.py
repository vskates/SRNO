from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ActiveIndex:
    manifest_sha256: str
    shards_sha256: str
    gripper_sha256: str
    delta_gate_m: float
    object_ids: tuple[str, ...]
    offsets: np.ndarray
    trajectory_index: np.ndarray
    step_index: np.ndarray

    def pairs_for(self, object_id: str) -> np.ndarray:
        try:
            object_index = self.object_ids.index(object_id)
        except ValueError as error:
            raise KeyError(object_id) from error
        start, end = self.offsets[object_index : object_index + 2]
        return np.stack(
            (self.trajectory_index[start:end], self.step_index[start:end]), axis=-1
        ).astype(np.int64, copy=False)

    def save(self, path: str | Path) -> None:
        destination = Path(path).resolve()
        temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
        metadata = json.dumps(
            {
                "format_version": 2,
                "manifest_sha256": self.manifest_sha256,
                "shards_sha256": self.shards_sha256,
                "gripper_sha256": self.gripper_sha256,
                "delta_gate_m": self.delta_gate_m,
                "object_ids": self.object_ids,
            }
        )
        np.savez_compressed(
            temporary,
            metadata=np.asarray(metadata),
            offsets=self.offsets.astype(np.int64),
            trajectory_index=self.trajectory_index.astype(np.int32),
            step_index=self.step_index.astype(np.uint8),
        )
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: str | Path) -> "ActiveIndex":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("format_version") != 2:
                raise ValueError(
                    "unsupported active-index version; rebuild it after the PhysX SDF update"
                )
            return cls(
                str(metadata["manifest_sha256"]),
                str(metadata["shards_sha256"]),
                str(metadata["gripper_sha256"]),
                float(metadata["delta_gate_m"]),
                tuple(map(str, metadata["object_ids"])),
                archive["offsets"].copy(),
                archive["trajectory_index"].copy(),
                archive["step_index"].copy(),
            )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_sha256(paths: tuple[Path, ...]) -> str:
    """Order-sensitive digest binding an index to the exact HDF5 contents."""

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()
