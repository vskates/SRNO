from __future__ import annotations

import dataclasses
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class PathsConfig:
    manifest: Path
    active_index: Path
    output_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 64
    operator_layers: int = 1
    contact_features: Literal["gap", "gap_jq", "trial_current_gap"] = "gap"
    contact_head: Literal[
        "direct", "normal_cone", "friction_cone", "implicit_resolvent",
        "dual_potential", "monotone_resolvent",
    ] = "direct"
    friction_coefficient: float = 2.4
    actuation_conditioning: Literal["aperture", "drive_error"] = "aperture"
    pose_predictor: Literal["identity", "continuation"] = "identity"
    pose_corrector: Literal["cone", "predictor_only"] = "cone"
    continuation_factor: float = 0.75
    resolvent_iterations: int = 32
    resolvent_pose_weight: float = 100.0
    resolvent_constraint_gap_m: float = 0.0
    resolvent_pose_query_factor: float = 0.0
    pose_update: Literal["se3_left", "decoupled"] = "se3_left"
    history_conditioning: Literal["none", "pose_delta"] = "none"


@dataclass(frozen=True)
class LossConfig:
    lambda_rotation: float = 1.0
    lambda_joints: float = 1.0
    lambda_feasibility: float = 1.0
    local_lambda_joints: float | None = None
    local_lambda_feasibility: float | None = None
    admissible_gap_m: float = 0.0
    pose_penalty: Literal["squared", "huber"] = "squared"
    pose_huber_delta: float = 0.02


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
class TrustRegionConfig:
    """Function-space trust region used by differentiable rollout training.

    ``rho=0`` selects the scale-aware default ``1 / epsilon_dx**2``.  The
    radius is expressed in the same dimensionless state metric as ``d_X``.
    """

    epsilon_dx: float = 0.002991000423207879
    rho: float = 0.0
    reference_cache: Path | None = None
    constraint_target: Literal["frozen_output", "physical_one_step"] = (
        "frozen_output"
    )


@dataclass(frozen=True)
class ExperimentConfig:
    paths: PathsConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    trust_region: TrustRegionConfig = field(default_factory=TrustRegionConfig)
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
        trust_region_raw = dict(raw.get("trust_region", {}))
        if trust_region_raw.get("reference_cache") is not None:
            trust_region_raw["reference_cache"] = resolve(
                trust_region_raw["reference_cache"]
            )
        model_raw = dict(raw.get("model", {}))
        legacy_conditioning = model_raw.pop("global_conditioning", "aperture")
        if legacy_conditioning != "aperture":
            raise ValueError(
                "drive_error conditioning was removed; only aperture "
                "conditioning is supported"
            )
        config = cls(
            paths=PathsConfig(
                resolve(paths["manifest"]),
                resolve(paths["active_index"]),
                resolve(paths["output_dir"]),
            ),
            model=ModelConfig(**model_raw),
            loss=LossConfig(**raw.get("loss", {})),
            optimizer=OptimizerConfig(**raw.get("optimizer", {})),
            loader=LoaderConfig(**raw.get("loader", {})),
            training=TrainingConfig(**training_raw),
            trust_region=TrustRegionConfig(**trust_region_raw),
            seed=int(raw.get("seed", 0)),
            device=str(raw.get("device", "cuda")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.model.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.model.operator_layers <= 0:
            raise ValueError("operator_layers must be positive")
        if self.model.contact_features not in {
            "gap",
            "gap_jq",
            "trial_current_gap",
        }:
            raise ValueError(
                "contact_features must be 'gap', 'gap_jq', or "
                "'trial_current_gap'"
            )
        if self.model.contact_head not in {
            "direct",
            "normal_cone",
            "friction_cone",
            "implicit_resolvent",
            "dual_potential",
            "monotone_resolvent",
        }:
            raise ValueError(
                "contact_head must be 'direct', 'normal_cone', "
                "'friction_cone', 'implicit_resolvent', 'dual_potential', or "
                "'monotone_resolvent'"
            )
        if (
            not math.isfinite(self.model.friction_coefficient)
            or self.model.friction_coefficient <= 0
        ):
            raise ValueError("friction_coefficient must be finite and positive")
        if self.model.actuation_conditioning not in {"aperture", "drive_error"}:
            raise ValueError(
                "actuation_conditioning must be 'aperture' or 'drive_error'"
            )
        if (
            self.model.actuation_conditioning == "drive_error"
            and self.model.contact_head == "direct"
        ):
            raise ValueError(
                "drive_error actuation conditioning is available only for cone heads"
            )
        if self.model.pose_predictor not in {"identity", "continuation"}:
            raise ValueError("pose_predictor must be 'identity' or 'continuation'")
        if self.model.pose_corrector not in {"cone", "predictor_only"}:
            raise ValueError("pose_corrector must be 'cone' or 'predictor_only'")
        if (
            not math.isfinite(self.model.continuation_factor)
            or self.model.continuation_factor < 0
        ):
            raise ValueError("continuation_factor must be finite and non-negative")
        if (
            self.model.pose_predictor == "continuation"
            and self.model.contact_head == "direct"
        ):
            raise ValueError("continuation pose predictor requires a cone head")
        if (
            self.model.pose_corrector == "predictor_only"
            and self.model.pose_predictor != "continuation"
        ):
            raise ValueError("predictor_only corrector requires continuation predictor")
        if self.model.resolvent_iterations <= 0:
            raise ValueError("resolvent_iterations must be positive")
        if (
            not math.isfinite(self.model.resolvent_pose_weight)
            or self.model.resolvent_pose_weight <= 0
        ):
            raise ValueError("resolvent_pose_weight must be finite and positive")
        if not math.isfinite(self.model.resolvent_constraint_gap_m):
            raise ValueError("resolvent_constraint_gap_m must be finite")
        if (
            not math.isfinite(self.model.resolvent_pose_query_factor)
            or self.model.resolvent_pose_query_factor < 0
        ):
            raise ValueError(
                "resolvent_pose_query_factor must be finite and non-negative"
            )
        if self.model.contact_head == "implicit_resolvent":
            if self.model.contact_features != "gap":
                raise ValueError("implicit_resolvent requires contact_features='gap'")
            if self.model.actuation_conditioning != "drive_error":
                raise ValueError("implicit_resolvent requires drive_error actuation")
            if (
                self.model.pose_predictor != "identity"
                or self.model.pose_corrector != "cone"
            ):
                raise ValueError("implicit_resolvent uses a single coupled state solve")
            if self.model.pose_update != "decoupled":
                raise ValueError("implicit_resolvent requires the product retraction")
            if (
                self.model.resolvent_pose_query_factor > 0
                and self.model.history_conditioning != "pose_delta"
            ):
                raise ValueError(
                    "a nonzero resolvent pose query requires pose_delta history"
                )
        if self.model.contact_head in {"dual_potential", "monotone_resolvent"}:
            if self.model.contact_features != "gap":
                raise ValueError(
                    f"{self.model.contact_head} requires contact_features='gap'"
                )
            if self.model.actuation_conditioning != "drive_error":
                raise ValueError(
                    f"{self.model.contact_head} requires drive_error actuation"
                )
            if (
                self.model.pose_predictor != "identity"
                or self.model.pose_corrector != "cone"
            ):
                raise ValueError(
                    f"{self.model.contact_head} directly returns the coupled increment"
                )
            if self.model.pose_update != "decoupled":
                raise ValueError(
                    f"{self.model.contact_head} requires the product retraction"
                )
            if (
                self.model.resolvent_pose_query_factor > 0
                and self.model.history_conditioning != "pose_delta"
            ):
                raise ValueError(
                    f"a nonzero {self.model.contact_head} pose query requires "
                    "pose_delta history"
                )
        if self.model.pose_update not in {"se3_left", "decoupled"}:
            raise ValueError("pose_update must be 'se3_left' or 'decoupled'")
        if self.model.history_conditioning not in {"none", "pose_delta"}:
            raise ValueError(
                "history_conditioning must be 'none' or 'pose_delta'"
            )
        for name, value in (
            ("lambda_rotation", self.loss.lambda_rotation),
            ("lambda_joints", self.loss.lambda_joints),
            ("lambda_feasibility", self.loss.lambda_feasibility),
            ("local_lambda_joints", self.loss.local_lambda_joints),
            ("local_lambda_feasibility", self.loss.local_lambda_feasibility),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"loss {name} must be finite and non-negative")
        if not math.isfinite(self.loss.admissible_gap_m):
            raise ValueError("admissible_gap_m must be finite")
        if self.loss.pose_penalty not in {"squared", "huber"}:
            raise ValueError("pose_penalty must be 'squared' or 'huber'")
        if (
            not math.isfinite(self.loss.pose_huber_delta)
            or self.loss.pose_huber_delta <= 0
        ):
            raise ValueError("pose_huber_delta must be finite and positive")
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
        if not math.isfinite(self.trust_region.epsilon_dx) or self.trust_region.epsilon_dx <= 0:
            raise ValueError("trust_region.epsilon_dx must be finite and positive")
        if not math.isfinite(self.trust_region.rho) or self.trust_region.rho < 0:
            raise ValueError("trust_region.rho must be finite and non-negative")
        if self.trust_region.constraint_target not in {
            "frozen_output",
            "physical_one_step",
        }:
            raise ValueError(
                "trust_region.constraint_target must be 'frozen_output' or "
                "'physical_one_step'"
            )
        if (
            self.trust_region.constraint_target == "physical_one_step"
            and self.trust_region.reference_cache is not None
        ):
            raise ValueError(
                "trust_region.reference_cache is only valid for frozen_output"
            )

    def to_dict(self) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        raw["paths"] = {key: str(value) for key, value in raw["paths"].items()}
        raw["training"]["rollout_horizons"] = list(raw["training"]["rollout_horizons"])
        if raw["trust_region"]["reference_cache"] is not None:
            raw["trust_region"]["reference_cache"] = str(
                raw["trust_region"]["reference_cache"]
            )
        return raw
