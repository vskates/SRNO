from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
NUM_STEPS = 32
NUM_STATES = NUM_STEPS + 1
NUM_SURFACE_SAMPLES = 256
SDF_RESOLUTION = (96, 96, 96)


@dataclass(frozen=True)
class ShardSpec:
    path: str
    object_ids: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ShardSpec":
        return cls(path=str(raw["path"]), object_ids=tuple(map(str, raw["object_ids"])))


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: int
    units: str
    pose_convention: str
    quaternion_order: str
    sdf_sign: str
    length_scale_m: float
    sdf_scale_m: float
    delta_gate_m: float
    commanded_aperture_m: tuple[float, ...]
    gripper_asset: str
    gripper_sha256: str
    shards: tuple[ShardSpec, ...]
    splits: dict[str, tuple[str, ...]]
    root: Path

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        manifest_path = Path(path).resolve()
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = cls(
            schema_version=int(raw["schema_version"]),
            units=str(raw["units"]),
            pose_convention=str(raw["pose_convention"]),
            quaternion_order=str(raw["quaternion_order"]),
            sdf_sign=str(raw["sdf_sign"]),
            length_scale_m=float(raw["length_scale_m"]),
            sdf_scale_m=float(raw["sdf_scale_m"]),
            delta_gate_m=float(raw["delta_gate_m"]),
            commanded_aperture_m=tuple(map(float, raw["commanded_aperture_m"])),
            gripper_asset=str(raw["gripper_asset"]),
            gripper_sha256=str(raw["gripper_sha256"]),
            shards=tuple(ShardSpec.from_dict(item) for item in raw["shards"]),
            splits={key: tuple(map(str, value)) for key, value in raw["splits"].items()},
            root=manifest_path.parent,
        )
        manifest.validate()
        return manifest

    @classmethod
    def create(
        cls,
        *,
        root: str | Path,
        length_scale_m: float,
        sdf_scale_m: float,
        delta_gate_m: float,
        commanded_aperture_m: list[float] | tuple[float, ...],
        gripper_asset: str,
        gripper_sha256: str,
        shards: list[ShardSpec] | tuple[ShardSpec, ...],
        splits: dict[str, list[str] | tuple[str, ...]],
    ) -> "DatasetManifest":
        manifest = cls(
            schema_version=SCHEMA_VERSION,
            units="m",
            pose_convention="object_to_gripper",
            quaternion_order="xyzw",
            sdf_sign="positive_outside",
            length_scale_m=float(length_scale_m),
            sdf_scale_m=float(sdf_scale_m),
            delta_gate_m=float(delta_gate_m),
            commanded_aperture_m=tuple(map(float, commanded_aperture_m)),
            gripper_asset=gripper_asset,
            gripper_sha256=gripper_sha256,
            shards=tuple(shards),
            splits={key: tuple(map(str, value)) for key, value in splits.items()},
            root=Path(root).resolve(),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported dataset schema {self.schema_version}, expected {SCHEMA_VERSION}"
            )
        expected = ("m", "object_to_gripper", "xyzw", "positive_outside")
        actual = (self.units, self.pose_convention, self.quaternion_order, self.sdf_sign)
        if actual != expected:
            raise ValueError(f"dataset conventions must be {expected}, got {actual}")
        if self.length_scale_m <= 0 or self.sdf_scale_m <= 0 or self.delta_gate_m <= 0:
            raise ValueError("length, SDF scale, and gate threshold must be positive")
        if len(self.commanded_aperture_m) != NUM_STATES:
            raise ValueError(f"commanded aperture schedule must have {NUM_STATES} states")
        schedule = np.asarray(self.commanded_aperture_m)
        if not np.all(np.diff(schedule) < 0):
            raise ValueError("commanded aperture schedule must be strictly decreasing")
        if set(self.splits) != {"train", "val", "test"}:
            raise ValueError("splits must contain exactly train, val, and test")
        shard_ids = [object_id for shard in self.shards for object_id in shard.object_ids]
        if len(shard_ids) != len(set(shard_ids)):
            raise ValueError("object IDs must be globally unique across shards")
        split_ids = [object_id for split in self.splits.values() for object_id in split]
        if len(split_ids) != len(set(split_ids)):
            raise ValueError("train/val/test object IDs overlap")
        if set(split_ids) != set(shard_ids):
            raise ValueError("split object IDs must exactly cover all shard objects")

    def shard_path(self, shard: ShardSpec) -> Path:
        return (self.root / shard.path).resolve()

    @property
    def gripper_path(self) -> Path:
        return (self.root / self.gripper_asset).resolve()

    def object_locations(self) -> dict[str, tuple[Path, str]]:
        locations: dict[str, tuple[Path, str]] = {}
        for shard in self.shards:
            path = self.shard_path(shard)
            for group_index, object_id in enumerate(shard.object_ids):
                locations[object_id] = (path, f"objects/{group_index:06d}")
        return locations

    def sha256(self) -> str:
        serializable = self.to_dict()
        payload = json.dumps(serializable, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "units": self.units,
            "pose_convention": self.pose_convention,
            "quaternion_order": self.quaternion_order,
            "sdf_sign": self.sdf_sign,
            "length_scale_m": self.length_scale_m,
            "sdf_scale_m": self.sdf_scale_m,
            "delta_gate_m": self.delta_gate_m,
            "commanded_aperture_m": list(self.commanded_aperture_m),
            "gripper_asset": self.gripper_asset,
            "gripper_sha256": self.gripper_sha256,
            "shards": [
                {"path": shard.path, "object_ids": list(shard.object_ids)}
                for shard in self.shards
            ],
            "splits": {key: list(value) for key, value in self.splits.items()},
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def objectwise_split(
    object_ids: list[str] | tuple[str, ...],
    *,
    seed: int = 0,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, tuple[str, ...]]:
    if len(object_ids) < 3:
        raise ValueError("at least three objects are needed for train/val/test splits")
    if not np.isclose(sum(fractions), 1.0) or any(value <= 0 for value in fractions):
        raise ValueError("split fractions must be positive and sum to one")
    shuffled = np.asarray(sorted(object_ids), dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    count = len(shuffled)
    train_end = max(1, int(np.floor(count * fractions[0])))
    val_end = max(train_end + 1, int(np.floor(count * (fractions[0] + fractions[1]))))
    val_end = min(val_end, count - 1)
    return {
        "train": tuple(map(str, shuffled[:train_end])),
        "val": tuple(map(str, shuffled[train_end:val_end])),
        "test": tuple(map(str, shuffled[val_end:])),
    }

