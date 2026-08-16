from __future__ import annotations

import dataclasses
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PathsConfig:
    manifest: Path
    active_index: Path
    output_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 64


@dataclass(frozen=True)
class LossConfig:
    lambda_rotation: float = 1.0
    lambda_joints: float = 1.0
    lambda_feasibility: float = 1.0
    admissible_gap_m: float = 0.0


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    warmup_fraction: float = 0.05


@dataclass(frozen=True)
class LoaderConfig:
    objects_per_batch: int = 4
    local_samples_per_object: int = 256
    rollout_trajectories_per_object: int = 8
    workers: int = 4


@dataclass(frozen=True)
class TrainingConfig:
    local_epochs: int = 100
    rollout_epochs_per_horizon: int = 25
    rollout_horizons: tuple[int, ...] = (4, 8, 16, 32)
    early_stopping_patience: int = 10
    use_bfloat16: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    paths: PathsConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 0
    device: str = "cuda"

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path).resolve()
        with config_path.open("rb") as stream:
            raw = tomllib.load(stream)
        paths = raw.get("paths", {})
        if "manifest" not in paths or "active_index" not in paths or "output_dir" not in paths:
            raise ValueError("config [paths] must define manifest, active_index, and output_dir")

        def resolve(value: str) -> Path:
            candidate = Path(value)
            return candidate.resolve() if candidate.is_absolute() else (config_path.parent / candidate).resolve()

        training_raw = dict(raw.get("training", {}))
        if "rollout_horizons" in training_raw:
            training_raw["rollout_horizons"] = tuple(training_raw["rollout_horizons"])
        config = cls(
            paths=PathsConfig(
                resolve(paths["manifest"]),
                resolve(paths["active_index"]),
                resolve(paths["output_dir"]),
            ),
            model=ModelConfig(**raw.get("model", {})),
            loss=LossConfig(**raw.get("loss", {})),
            optimizer=OptimizerConfig(**raw.get("optimizer", {})),
            loader=LoaderConfig(**raw.get("loader", {})),
            training=TrainingConfig(**training_raw),
            seed=int(raw.get("seed", 0)),
            device=str(raw.get("device", "cuda")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.model.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not math.isfinite(self.loss.admissible_gap_m):
            raise ValueError("admissible_gap_m must be finite")
        if (
            self.loader.objects_per_batch <= 0
            or self.loader.local_samples_per_object <= 0
            or self.loader.rollout_trajectories_per_object <= 0
        ):
            raise ValueError(
                "object, local-transition, and rollout-trajectory batch sizes must be positive"
            )
        if self.loader.workers < 0:
            raise ValueError("workers cannot be negative")
        horizons = self.training.rollout_horizons
        if not horizons or tuple(sorted(set(horizons))) != horizons or horizons[-1] != 32:
            raise ValueError("rollout horizons must be strictly increasing and end at 32")
        if self.training.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if not 0 <= self.optimizer.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        raw["paths"] = {key: str(value) for key, value in raw["paths"].items()}
        raw["training"]["rollout_horizons"] = list(raw["training"]["rollout_horizons"])
        return raw
