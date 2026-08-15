"""Small TOML contract for the zero-gravity Isaac collector."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib
from typing import Any, Mapping


SIMULATOR_CONFIG_VERSION = 1


@dataclass(frozen=True)
class SettlingConfig:
    min_steps: int = 20
    max_steps: int = 600
    consecutive_steps: int = 10
    position_delta_m: float = 5e-4
    linear_velocity_m_s: float = 0.01
    angular_velocity_rad_s: float = 0.1
    joint_velocity_rad_s: float = 0.01


@dataclass(frozen=True)
class RelaxationConfig:
    """Dissipation required for a free body to reach equilibrium without gravity."""

    object_linear_damping_s_inv: float = 5.0
    object_angular_damping_s_inv: float = 5.0
    gripper_velocity_limit_rad_s: float = 0.1
    gripper_damping: float = 0.35


@dataclass(frozen=True)
class TrajectoryConfig:
    command_steps: int = 32


@dataclass(frozen=True)
class SDFGenerationConfig:
    resolution: int = 96
    padding_m: float = 0.02
    chunk_points: int = 8_192


@dataclass(frozen=True)
class DatasetExportConfig:
    sdf_scale_m: float = 0.02
    delta_gate_m: float = 0.01
    split_seed: int = 0


@dataclass(frozen=True)
class SimulatorConfig:
    source_path: Path
    catalog: Path
    output_dir: Path
    headless: bool
    device: str
    num_envs: int
    trajectories_per_object: int
    successful_seed_poses_only: bool
    seed: int
    overwrite: bool
    memory_limit_gib: float
    memory_check_interval_s: float
    settling: SettlingConfig
    relaxation: RelaxationConfig
    trajectory: TrajectoryConfig
    sdf: SDFGenerationConfig
    dataset: DatasetExportConfig

    @classmethod
    def load(cls, path: str | Path) -> "SimulatorConfig":
        source_path = Path(path).resolve()
        with source_path.open("rb") as stream:
            raw = tomllib.load(stream)
        allowed = {
            "schema_version", "catalog", "output_dir", "headless", "device", "num_envs",
            "trajectories_per_object", "successful_seed_poses_only", "seed", "overwrite",
            "memory_limit_gib", "memory_check_interval_s",
            "settling", "relaxation", "trajectory", "sdf", "dataset",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown simulator config keys: {sorted(unknown)}")
        if raw.get("schema_version") != SIMULATOR_CONFIG_VERSION:
            raise ValueError(
                f"unsupported simulator config schema {raw.get('schema_version')!r}; "
                f"expected {SIMULATOR_CONFIG_VERSION}"
            )
        root = source_path.parent
        config = cls(
            source_path=source_path,
            catalog=_relative_path(root, raw.get("catalog"), "catalog"),
            output_dir=_relative_path(root, raw.get("output_dir"), "output_dir"),
            headless=_boolean(raw.get("headless"), "headless"),
            device=str(raw.get("device", "cuda:0")),
            num_envs=int(raw.get("num_envs", 256)),
            trajectories_per_object=int(raw.get("trajectories_per_object", 2000)),
            successful_seed_poses_only=_boolean(
                raw.get("successful_seed_poses_only", True), "successful_seed_poses_only"
            ),
            seed=int(raw.get("seed", 0)),
            overwrite=_boolean(raw.get("overwrite", False), "overwrite"),
            memory_limit_gib=float(raw.get("memory_limit_gib", 12.0)),
            memory_check_interval_s=float(raw.get("memory_check_interval_s", 0.25)),
            settling=_section(SettlingConfig, raw.get("settling"), "settling"),
            relaxation=_section(RelaxationConfig, raw.get("relaxation"), "relaxation"),
            trajectory=_section(TrajectoryConfig, raw.get("trajectory"), "trajectory"),
            sdf=_section(SDFGenerationConfig, raw.get("sdf"), "sdf"),
            dataset=_section(DatasetExportConfig, raw.get("dataset"), "dataset"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be 'cpu' or a CUDA device")
        if self.num_envs <= 0 or self.trajectories_per_object <= 0:
            raise ValueError("num_envs and trajectories_per_object must be positive")
        if not math.isfinite(self.memory_limit_gib) or self.memory_limit_gib <= 0.0:
            raise ValueError("memory_limit_gib must be positive and finite")
        if (
            not math.isfinite(self.memory_check_interval_s)
            or self.memory_check_interval_s <= 0.0
        ):
            raise ValueError("memory_check_interval_s must be positive and finite")
        if self.trajectory.command_steps != 32:
            raise ValueError("SRNO v1 collection requires exactly 32 command steps")
        if self.sdf.resolution != 96:
            raise ValueError("SRNO v1 collection requires 96^3 SDF grids")
        if self.settling.min_steps < 0 or self.settling.max_steps < self.settling.min_steps:
            raise ValueError("invalid settling min/max steps")
        if not 1 <= self.settling.consecutive_steps <= self.settling.max_steps:
            raise ValueError("invalid consecutive settling steps")
        positive = (
            self.settling.position_delta_m,
            self.settling.linear_velocity_m_s,
            self.settling.angular_velocity_rad_s,
            self.settling.joint_velocity_rad_s,
            self.relaxation.object_linear_damping_s_inv,
            self.relaxation.object_angular_damping_s_inv,
            self.relaxation.gripper_velocity_limit_rad_s,
            self.relaxation.gripper_damping,
            self.sdf.padding_m,
            self.dataset.sdf_scale_m,
            self.dataset.delta_gate_m,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("all physical thresholds and scales must be positive and finite")
        if self.sdf.chunk_points <= 0:
            raise ValueError("sdf.chunk_points must be positive")

    def sha256(self) -> str:
        payload = asdict(self)
        payload["source_path"] = str(self.source_path)
        payload["catalog"] = str(self.catalog)
        payload["output_dir"] = str(self.output_dir)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _relative_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be true or false")
    return value


def _section(cls: type[Any], raw: Any, label: str) -> Any:
    if raw is None:
        return cls()
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a TOML table")
    allowed = set(cls.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown {label} keys: {sorted(unknown)}")
    return cls(**raw)
