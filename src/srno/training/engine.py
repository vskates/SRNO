from __future__ import annotations

import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Iterable, Literal

import numpy as np
import torch
from torch import Tensor

from srno.data.dataset import (
    H5ObjectDataset,
    LocalTransitionBatch,
    TrajectoryBatch,
    make_dataloader,
)
from srno.data.index import ActiveIndex
from srno.data.index import file_sha256
from srno.data.schema import DatasetManifest
from srno.geometry.gripper import GripperAsset
from srno.losses import combined_loss, state_error
from srno.model import SRNOModel
from srno.training.checkpoint import load_checkpoint, save_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.metrics import MetricAccumulator, accumulate_trajectory_metrics
from srno.training.trust_region import (
    FrozenLocalReference,
    PhysicalBaseline,
    build_frozen_local_reference,
    effective_physical_rho,
    effective_rho,
    functional_drift,
    hpr_inequality_penalty,
    physical_one_step_error,
    reference_contract,
    update_multiplier,
)
from srno.types import PoseState


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _build_model(config: ExperimentConfig, manifest: DatasetManifest, device: torch.device) -> SRNOModel:
    gripper = GripperAsset.load(manifest.gripper_path)
    if gripper.sha256() != manifest.gripper_sha256:
        raise ValueError("gripper hash does not match manifest")
    return SRNOModel(
        gripper,
        sdf_scale=manifest.sdf_scale_m,
        delta_gate=manifest.delta_gate_m,
        contact_offset_sum=manifest.contact_offset_sum_m,
        hidden_dim=config.model.hidden_dim,
        operator_layers=config.model.operator_layers,
        contact_features=config.model.contact_features,
        contact_head=config.model.contact_head,
        friction_coefficient=config.model.friction_coefficient,
        actuation_conditioning=config.model.actuation_conditioning,
        pose_predictor=config.model.pose_predictor,
        pose_corrector=config.model.pose_corrector,
        continuation_factor=config.model.continuation_factor,
        resolvent_iterations=config.model.resolvent_iterations,
        resolvent_pose_weight=config.model.resolvent_pose_weight,
        resolvent_constraint_gap_m=config.model.resolvent_constraint_gap_m,
        resolvent_pose_query_factor=config.model.resolvent_pose_query_factor,
        pose_update=config.model.pose_update,
        history_conditioning=config.model.history_conditioning,
    ).to(device)


def _autocast(config: ExperimentConfig, device: torch.device):
    enabled = config.training.use_bfloat16 and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def _optimizer(model: SRNOModel, config: ExperimentConfig, total_steps: int):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        fused=model.surface_local_points.is_cuda,
    )
    warmup_steps = int(total_steps * config.optimizer.warmup_fraction)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-3)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _state_at(states: PoseState, step: int) -> PoseState:
    return PoseState(
        states.rotation[:, step],
        states.position[:, step],
        states.joint_position[:, step],
    )


def _local_iteration(
    model: SRNOModel,
    batch: LocalTransitionBatch,
    config: ExperimentConfig,
) -> tuple[Tensor, dict[str, float]]:
    prediction = model.forward_step(
        batch.current,
        batch.next_command,
        batch.sdf,
        previous_state=batch.previous,
    )
    assert isinstance(prediction, PoseState)
    # Feasibility is tied to rest_offset=0, not to the wider contact envelope.
    gap = model.query_geometric_gap(prediction, batch.sdf)
    terms = combined_loss(
        prediction,
        batch.target,
        gap,
        length_scale=model.length_scale,
        joint_scale=model.joint_travel_range,
        sdf_scale=model.sdf_scale,
        lambda_rotation=config.loss.lambda_rotation,
        lambda_joints=(
            config.loss.lambda_joints
            if config.loss.local_lambda_joints is None
            else config.loss.local_lambda_joints
        ),
        lambda_feasibility=(
            config.loss.lambda_feasibility
            if config.loss.local_lambda_feasibility is None
            else config.loss.local_lambda_feasibility
        ),
        admissible_gap=config.loss.admissible_gap_m,
        pose_penalty=config.loss.pose_penalty,
        pose_huber_delta=config.loss.pose_huber_delta,
    )
    values = {
        "loss": float(terms.total.detach()),
        "flow": float(terms.flow.detach()),
        "feasibility": float(terms.feasibility.detach()),
        "translation": float(terms.translation.detach()),
        "rotation": float(terms.rotation.detach()),
        "joints": float(terms.joints.detach()),
        "aperture_m": float(
            (
                model.aperture_from_joints(prediction.joint_position)
                - batch.target_aperture
            )
            .abs()
            .mean()
            .detach()
        ),
        "dx": float(
            state_error(
                prediction,
                batch.target,
                length_scale=model.length_scale,
                joint_scale=model.joint_travel_range,
                lambda_rotation=config.loss.lambda_rotation,
                lambda_joints=config.loss.lambda_joints,
            )[0]
            .sqrt()
            .mean()
            .detach()
        ),
    }
    return terms.total, values


def _rollout_iteration(
    model: SRNOModel,
    batch: TrajectoryBatch,
    config: ExperimentConfig,
    horizon: int,
) -> tuple[Tensor, dict[str, float], PoseState, Tensor]:
    initial = _state_at(batch.states, 0)
    prediction = model.rollout(
        initial, batch.command_schedule[1 : horizon + 1], batch.sdf
    )
    total = prediction.position.new_zeros(())
    sums = {
        name: 0.0
        for name in ("flow", "feasibility", "translation", "rotation", "joints")
    }
    aperture_error = 0.0
    gaps: list[Tensor] = []
    for step in range(1, horizon + 1):
        predicted_step = _state_at(prediction, step)
        target_step = _state_at(batch.states, step)
        gap = model.query_geometric_gap(predicted_step, batch.sdf)
        gaps.append(gap)
        terms = combined_loss(
            predicted_step,
            target_step,
            gap,
            length_scale=model.length_scale,
            joint_scale=model.joint_travel_range,
            sdf_scale=model.sdf_scale,
            lambda_rotation=config.loss.lambda_rotation,
            lambda_joints=config.loss.lambda_joints,
            lambda_feasibility=config.loss.lambda_feasibility,
            admissible_gap=config.loss.admissible_gap_m,
            pose_penalty=config.loss.pose_penalty,
            pose_huber_delta=config.loss.pose_huber_delta,
        )
        total = total + terms.total / horizon
        for name in sums:
            sums[name] += float(getattr(terms, name).detach()) / horizon
        aperture_error += float(
            (
                model.aperture_from_joints(predicted_step.joint_position)
                - batch.actual_aperture[:, step]
            )
            .abs()
            .mean()
            .detach()
        ) / horizon
    terminal_dx = (
        state_error(
            _state_at(prediction, horizon),
            _state_at(batch.states, horizon),
            length_scale=model.length_scale,
            joint_scale=model.joint_travel_range,
            lambda_rotation=config.loss.lambda_rotation,
            lambda_joints=config.loss.lambda_joints,
        )[0]
        .sqrt()
        .mean()
    )
    return (
        total,
        {
            "loss": float(total.detach()),
            "terminal_dx": float(terminal_dx.detach()),
            "aperture_m": aperture_error,
            "terminal_aperture_m": float(
                (
                    model.aperture_from_joints(
                        _state_at(prediction, horizon).joint_position
                    )
                    - batch.actual_aperture[:, horizon]
                )
                .abs()
                .mean()
                .detach()
            ),
            **sums,
        },
        prediction,
        torch.stack(gaps, dim=1),
    )


def _mean_dict(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        raise ValueError("loader produced no batches")
    return {key: float(np.mean([value[key] for value in values])) for key in values[0]}


def _set_loader_epoch(loader, epoch: int) -> None:
    set_epoch = getattr(loader.batch_sampler, "set_epoch", None)
    if set_epoch is not None:
        set_epoch(epoch)


def _run_epoch(
    model: SRNOModel,
    loader: Iterable[LocalTransitionBatch | TrajectoryBatch],
    config: ExperimentConfig,
    device: torch.device,
    *,
    mode: Literal["local", "rollout"],
    horizon: int,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    values: list[dict[str, float]] = []
    context = nullcontext() if training else torch.no_grad()
    with context:
        for raw_batch in loader:
            batch = raw_batch.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with _autocast(config, device):
                if mode == "local":
                    assert isinstance(batch, LocalTransitionBatch)
                    loss, metrics = _local_iteration(model, batch, config)
                else:
                    assert isinstance(batch, TrajectoryBatch)
                    loss, metrics, _, _ = _rollout_iteration(
                        model, batch, config, horizon
                    )
            if training:
                if loss.requires_grad:
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optimizer.gradient_clip
                    )
                    optimizer.step()
                else:
                    # Exact free bypass can make a complete batch independent of
                    # the neural cell. Do not manufacture zero gradients here:
                    # AdamW would otherwise still apply decoupled weight decay.
                    gradient_norm = loss.new_zeros(())
                assert scheduler is not None
                scheduler.step()
                metrics["gradient_norm"] = float(gradient_norm.detach())
                metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
            values.append(metrics)
    return _mean_dict(values)


def _run_trust_region_epoch(
    model: SRNOModel,
    rollout_loader: Iterable[TrajectoryBatch],
    local_loader: Iterable[LocalTransitionBatch],
    reference: FrozenLocalReference | None,
    config: ExperimentConfig,
    device: torch.device,
    *,
    horizon: int,
    constraint_baseline: float,
    multiplier: float,
    rho: float,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> dict[str, float]:
    """One differentiable rollout epoch with a minibatch function constraint."""

    model.train(True)
    values: list[dict[str, float]] = []
    local_iterator = iter(local_loader)
    for raw_rollout in rollout_loader:
        try:
            raw_local = next(local_iterator)
        except StopIteration:
            local_iterator = iter(local_loader)
            raw_local = next(local_iterator)
        rollout_batch = raw_rollout.to(device, non_blocking=True)
        local_batch = raw_local.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(config, device):
            rollout_loss, metrics, _, _ = _rollout_iteration(
                model, rollout_batch, config, horizon
            )
            current = model.forward_step(
                local_batch.current, local_batch.next_command, local_batch.sdf
            )
            assert isinstance(current, PoseState)
            if config.trust_region.constraint_target == "frozen_output":
                if reference is None:
                    raise RuntimeError("frozen_output constraint has no reference")
                comparator = reference.lookup(local_batch, device=device)
                constraint_values = functional_drift(
                    current,
                    comparator,
                    length_scale=model.length_scale,
                    joint_scale=model.joint_travel_range,
                )
            else:
                constraint_values = physical_one_step_error(
                    current,
                    local_batch.target,
                    length_scale=model.length_scale,
                    joint_scale=model.joint_travel_range,
                )
            constraint_value, constraint_t, constraint_r, constraint_j = (
                constraint_values
            )
            constraint = constraint_value - constraint_baseline
            penalty = hpr_inequality_penalty(constraint, multiplier, rho)
            loss = rollout_loss + penalty
        if loss.requires_grad:
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.optimizer.gradient_clip
            )
            optimizer.step()
        else:
            gradient_norm = loss.new_zeros(())
        scheduler.step()
        constraint_metrics = {
            "constraint_value_sq": float(constraint_value.detach()),
            "constraint_value_dx": float(constraint_value.detach().sqrt()),
            "constraint_translation_sq": float(constraint_t.detach()),
            "constraint_rotation_sq": float(constraint_r.detach()),
            "constraint_joints_sq": float(constraint_j.detach()),
        }
        if config.trust_region.constraint_target == "frozen_output":
            constraint_metrics.update(
                {
                    "functional_drift_sq": float(constraint_value.detach()),
                    "functional_drift_dx": float(constraint_value.detach().sqrt()),
                    "functional_drift_translation_sq": float(constraint_t.detach()),
                    "functional_drift_rotation_sq": float(constraint_r.detach()),
                    "functional_drift_joints_sq": float(constraint_j.detach()),
                }
            )
        else:
            constraint_metrics.update(
                {
                    "physical_one_step_sq": float(constraint_value.detach()),
                    "physical_one_step_dx": float(constraint_value.detach().sqrt()),
                    "physical_one_step_translation_sq": float(constraint_t.detach()),
                    "physical_one_step_rotation_sq": float(constraint_r.detach()),
                    "physical_one_step_joints_sq": float(constraint_j.detach()),
                }
            )
        metrics.update(
            {
                "loss": float(loss.detach()),
                "rollout_loss": float(rollout_loss.detach()),
                "trust_penalty": float(penalty.detach()),
                "constraint": float(constraint.detach()),
                "constraint_ratio": float(
                    constraint_value.detach() / constraint_baseline
                ),
                "constraint_baseline": float(constraint_baseline),
                "multiplier": float(multiplier),
                "rho": float(rho),
                "gradient_norm": float(gradient_norm.detach()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **constraint_metrics,
            }
        )
        values.append(metrics)
    return _mean_dict(values)


def _evaluate_trust_constraint(
    model: SRNOModel,
    loader: Iterable[LocalTransitionBatch],
    reference: FrozenLocalReference | None,
    config: ExperimentConfig,
    device: torch.device,
    *,
    constraint_baseline: float,
) -> dict[str, float]:
    """Evaluate the exact sample-weighted training constraint over active states."""

    model.eval()
    sums = torch.zeros(4, dtype=torch.float64)
    count = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = raw_batch.to(device, non_blocking=True)
            with _autocast(config, device):
                prediction = model.forward_step(
                    batch.current, batch.next_command, batch.sdf
                )
                assert isinstance(prediction, PoseState)
                if config.trust_region.constraint_target == "frozen_output":
                    if reference is None:
                        raise RuntimeError("frozen_output constraint has no reference")
                    comparator = reference.lookup(batch, device=device)
                    values = functional_drift(
                        prediction,
                        comparator,
                        length_scale=model.length_scale,
                        joint_scale=model.joint_travel_range,
                    )
                else:
                    values = physical_one_step_error(
                        prediction,
                        batch.target,
                        length_scale=model.length_scale,
                        joint_scale=model.joint_travel_range,
                    )
            batch_count = int(batch.next_command.numel())
            sums += torch.tensor(
                [float(value.detach()) for value in values], dtype=torch.float64
            ) * batch_count
            count += batch_count
    if count == 0:
        raise ValueError("cannot evaluate a trust region on an empty active set")
    mean = sums / count
    result = {
        "constraint_value_sq_exact": float(mean[0]),
        "constraint_value_dx_exact": float(mean[0].sqrt()),
        "constraint_translation_sq_exact": float(mean[1]),
        "constraint_rotation_sq_exact": float(mean[2]),
        "constraint_joints_sq_exact": float(mean[3]),
        "constraint_exact": float(mean[0] - constraint_baseline),
        "constraint_ratio_exact": float(mean[0] / constraint_baseline),
        "constraint_baseline": float(constraint_baseline),
        "constraint_samples": float(count),
    }
    if config.trust_region.constraint_target == "frozen_output":
        result.update(
            {
                "functional_drift_sq_exact": float(mean[0]),
                "functional_drift_dx_exact": float(mean[0].sqrt()),
                "functional_drift_translation_sq_exact": float(mean[1]),
                "functional_drift_rotation_sq_exact": float(mean[2]),
                "functional_drift_joints_sq_exact": float(mean[3]),
            }
        )
    else:
        result.update(
            {
                "physical_one_step_sq_exact": float(mean[0]),
                "physical_one_step_dx_exact": float(mean[0].sqrt()),
                "physical_one_step_translation_sq_exact": float(mean[1]),
                "physical_one_step_rotation_sq_exact": float(mean[2]),
                "physical_one_step_joints_sq_exact": float(mean[3]),
            }
        )
    return result


def _open_summary_writer(output_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:  # pragma: no cover - depends on the runtime environment
        raise RuntimeError(
            "TensorBoard logging requires the 'train' extra: pip install -e '.[train]'"
        ) from error
    return SummaryWriter(log_dir=output_dir / "tensorboard", flush_secs=10)


def _write_tensorboard_epoch(
    writer,
    *,
    stage: Literal["local", "rollout", "trust_rollout"],
    horizon: int,
    global_epoch: int,
    horizon_epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    train_seconds: float,
    val_seconds: float,
    best_metric: float,
) -> None:
    prefix = stage if stage == "local" else f"{stage}/horizon_{horizon:02d}"
    for split, metrics in (("train", train_metrics), ("val", val_metrics)):
        for name, value in metrics.items():
            writer.add_scalar(f"{prefix}/{split}/{name}", value, global_epoch)
    writer.add_scalar(f"{prefix}/time/train_epoch_seconds", train_seconds, global_epoch)
    writer.add_scalar(f"{prefix}/time/val_epoch_seconds", val_seconds, global_epoch)
    writer.add_scalar(f"{prefix}/progress/best_metric", best_metric, global_epoch)
    writer.add_scalar(f"{prefix}/progress/horizon_epoch", horizon_epoch, global_epoch)
    writer.add_scalar("progress/horizon", horizon, global_epoch)
    writer.flush()


def train(
    config: ExperimentConfig,
    *,
    stage: Literal["local", "rollout", "trust_rollout"],
    resume: str | Path | None = None,
    advance_horizon: bool = False,
) -> Path:
    _seed_everything(config.seed)
    device = _device(config.device)
    manifest = DatasetManifest.load(config.paths.manifest)
    model = _build_model(config, manifest, device)
    trust_training = stage == "trust_rollout"
    rollout_training = stage in {"rollout", "trust_rollout"}
    active_index = (
        ActiveIndex.load(config.paths.active_index)
        if stage == "local" or trust_training
        else None
    )
    train_dataset = H5ObjectDataset(
        manifest,
        split="train",
        active_index=active_index if stage == "local" else None,
        active_only=stage == "local",
    )
    val_dataset = H5ObjectDataset(
        manifest,
        split="val",
        active_index=active_index if stage == "local" else None,
        active_only=stage == "local",
    )
    trust_dataset = (
        H5ObjectDataset(
            manifest,
            split="train",
            active_index=active_index,
            active_only=True,
        )
        if trust_training
        else None
    )
    writer = None
    try:
        loader_mode = "local" if stage == "local" else "rollout"
        train_loader = make_dataloader(
            train_dataset,
            mode=loader_mode,
            objects_per_batch=config.loader.objects_per_batch,
            samples_per_object=(
                config.loader.local_samples_per_object
                if stage == "local"
                else config.loader.rollout_trajectories_per_object
            ),
            workers=config.loader.workers,
            seed=config.seed,
            shuffle=True,
        )
        val_loader = make_dataloader(
            val_dataset,
            mode=loader_mode,
            objects_per_batch=1,
            samples_per_object=0,
            workers=config.loader.workers,
            seed=config.seed + 1,
            shuffle=False,
        )
        trust_loader = (
            make_dataloader(
                trust_dataset,
                mode="local",
                objects_per_batch=config.loader.objects_per_batch,
                samples_per_object=config.loader.local_samples_per_object,
                workers=config.loader.workers,
                seed=config.seed + 17,
                shuffle=True,
            )
            if trust_dataset is not None
            else None
        )
        horizons = (1,) if stage == "local" else config.training.rollout_horizons
        epochs_per_horizon = (
            config.training.local_epochs
            if stage == "local"
            else config.training.rollout_epochs_per_horizon
        )
        total_steps = max(1, len(train_loader) * epochs_per_horizon * len(horizons))
        optimizer, scheduler = _optimizer(model, config, total_steps)
        start_epoch = 0
        best_metric = float("inf")
        resumed_horizon = 0
        resumed_horizon_epoch = 0
        resumed_stale_epochs = 0
        resumed_best_feasible = False
        resumed_best_constraint_ratio = float("inf")
        checkpoint_stage: str | None = None
        resumed_trust_state: dict[str, object] | None = None
        local_checkpoint_sha256 = ""
        multiplier = 0.0
        advanced_from_horizon = 0
        if resume is not None:
            checkpoint = load_checkpoint(
                resume,
                model=model,
                map_location=device,
            )
            if checkpoint["manifest_sha256"] != manifest.sha256():
                raise ValueError("checkpoint manifest hash mismatch")
            if checkpoint["gripper_sha256"] != manifest.gripper_sha256:
                raise ValueError("checkpoint gripper hash mismatch")
            checkpoint_stage = str(checkpoint["stage"])
            if checkpoint_stage == stage:
                optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
                from srno.training.checkpoint import restore_rng_state

                restore_rng_state(checkpoint["rng"])
                start_epoch = int(checkpoint["epoch"]) + 1
                resumed_horizon = int(checkpoint["horizon"])
                resumed_horizon_epoch = int(checkpoint.get("horizon_epoch", -1)) + 1
                resumed_stale_epochs = int(checkpoint.get("stale_epochs", 0))
                best_metric = float(checkpoint["best_metric"])
                if trust_training:
                    trust_state = checkpoint.get("extra_state", {}).get("trust_region")
                    if not isinstance(trust_state, dict):
                        raise ValueError("trust-rollout checkpoint has no trust-region state")
                    checkpoint_target = str(
                        trust_state.get("constraint_target", "frozen_output")
                    )
                    if checkpoint_target != config.trust_region.constraint_target:
                        raise ValueError("trust-rollout constraint_target mismatch")
                    resumed_trust_state = trust_state
                    multiplier = float(trust_state["multiplier"])
                    local_checkpoint_sha256 = str(
                        trust_state["local_checkpoint_sha256"]
                    )
                    resumed_best_feasible = bool(
                        trust_state.get("best_feasible", False)
                    )
                    resumed_best_constraint_ratio = float(
                        trust_state.get("best_constraint_ratio", float("inf"))
                    )
                if advance_horizon:
                    if stage == "local":
                        raise ValueError("advance_horizon is only valid for rollout stages")
                    advanced_from_horizon = resumed_horizon
                    resumed_horizon = 0
                    resumed_horizon_epoch = 0
                    resumed_stale_epochs = 0
                    resumed_best_feasible = False
                    resumed_best_constraint_ratio = float("inf")
            elif not (
                checkpoint_stage == "local" and rollout_training
            ):
                raise ValueError(
                    f"cannot initialize {stage!r} training from {checkpoint_stage!r} checkpoint"
                )
            elif trust_training:
                local_checkpoint_sha256 = file_sha256(resume)
        elif trust_training:
            raise ValueError("trust_rollout must be initialized from a frozen local checkpoint")

        config.paths.output_dir.mkdir(parents=True, exist_ok=True)
        (config.paths.output_dir / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        reference: FrozenLocalReference | None = None
        constraint_metadata: dict[str, object] | None = None
        constraint_baseline = config.trust_region.epsilon_dx**2
        rho = effective_rho(config.trust_region.epsilon_dx, config.trust_region.rho)
        if trust_training:
            assert active_index is not None
            assert trust_loader is not None
            active_index_sha256 = file_sha256(config.paths.active_index)
            if config.trust_region.constraint_target == "frozen_output":
                reference_path = (
                    config.trust_region.reference_cache
                    or config.paths.output_dir / "frozen-local-reference.pt"
                )
                if reference_path.exists():
                    reference = FrozenLocalReference.load(
                        reference_path,
                        manifest_sha256=manifest.sha256(),
                        gripper_sha256=manifest.gripper_sha256,
                        active_index_sha256=active_index_sha256,
                        local_checkpoint_sha256=local_checkpoint_sha256,
                    )
                else:
                    if checkpoint_stage != "local":
                        raise FileNotFoundError(
                            "frozen-local reference cache is missing; resume from the local "
                            "checkpoint once to build it"
                        )
                    _set_loader_epoch(trust_loader, 0)
                    reference = build_frozen_local_reference(
                        model,
                        trust_loader,
                        device=device,
                        autocast_context=lambda: _autocast(config, device),
                        active_index=active_index,
                        active_index_path=config.paths.active_index,
                        manifest_sha256=manifest.sha256(),
                        gripper_sha256=manifest.gripper_sha256,
                        local_checkpoint_path=resume,
                    )
                    reference.save(reference_path)
                constraint_metadata = reference_contract(reference, reference_path)
                (config.paths.output_dir / "frozen-reference-contract.json").write_text(
                    json.dumps(constraint_metadata, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            else:
                baseline_path = config.paths.output_dir / "physical-baseline-contract.json"
                if checkpoint_stage == "local":
                    _set_loader_epoch(trust_loader, 0)
                    baseline_metrics = _evaluate_trust_constraint(
                        model,
                        trust_loader,
                        None,
                        config,
                        device,
                        constraint_baseline=1.0,
                    )
                    baseline = PhysicalBaseline(
                        manifest_sha256=manifest.sha256(),
                        gripper_sha256=manifest.gripper_sha256,
                        active_index_sha256=active_index_sha256,
                        local_checkpoint_sha256=local_checkpoint_sha256,
                        loss=baseline_metrics["physical_one_step_sq_exact"],
                        translation=baseline_metrics[
                            "physical_one_step_translation_sq_exact"
                        ],
                        rotation=baseline_metrics[
                            "physical_one_step_rotation_sq_exact"
                        ],
                        joints=baseline_metrics["physical_one_step_joints_sq_exact"],
                        transitions=int(baseline_metrics["constraint_samples"]),
                    )
                    baseline.save(baseline_path)
                else:
                    if resumed_trust_state is None:
                        raise ValueError("physical trust resume has no saved constraint state")
                    baseline = PhysicalBaseline.load(
                        baseline_path,
                        manifest_sha256=manifest.sha256(),
                        gripper_sha256=manifest.gripper_sha256,
                        active_index_sha256=active_index_sha256,
                        local_checkpoint_sha256=local_checkpoint_sha256,
                    )
                    expected_baseline_hash = str(
                        resumed_trust_state.get("physical_baseline_sha256", "")
                    )
                    if file_sha256(baseline_path) != expected_baseline_hash:
                        raise ValueError("physical baseline sidecar hash mismatch")
                    if not math.isclose(
                        baseline.loss,
                        float(resumed_trust_state.get("constraint_baseline", math.nan)),
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    ):
                        raise ValueError("physical baseline checkpoint value mismatch")
                constraint_baseline = baseline.loss
                rho = effective_physical_rho(
                    constraint_baseline, config.trust_region.rho
                )
                constraint_metadata = {
                    **baseline.to_dict(),
                    "path": str(baseline_path.resolve()),
                    "sha256": file_sha256(baseline_path),
                }
                if baseline.transitions != sum(
                    len(active_index.pairs_for(object_id))
                    for object_id in manifest.splits["train"]
                ):
                    raise ValueError("physical baseline active transition count mismatch")
        writer = _open_summary_writer(config.paths.output_dir)
        writer.add_text(
            f"{stage}/config",
            "```json\n" + json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n```",
            start_epoch,
        )
        global_epoch = start_epoch
        stale_epochs = 0
        stage_filename = stage.replace("_", "-")
        best_path = config.paths.output_dir / f"best-{stage_filename}.pt"
        best_feasible = resumed_best_feasible
        best_constraint_ratio = resumed_best_constraint_ratio

        def trust_extra_state() -> dict[str, object] | None:
            if not trust_training:
                return None
            assert constraint_metadata is not None
            trust_state: dict[str, object] = {
                "constraint_target": config.trust_region.constraint_target,
                "epsilon_dx": config.trust_region.epsilon_dx,
                "constraint_baseline": constraint_baseline,
                "rho": rho,
                "multiplier": multiplier,
                "local_checkpoint_sha256": local_checkpoint_sha256,
                "best_feasible": best_feasible,
                "best_constraint_ratio": best_constraint_ratio,
            }
            if config.trust_region.constraint_target == "frozen_output":
                trust_state["reference_sha256"] = constraint_metadata["sha256"]
            else:
                trust_state["physical_baseline_sha256"] = constraint_metadata[
                    "sha256"
                ]
                trust_state["physical_baseline"] = {
                    key: constraint_metadata[key]
                    for key in (
                        "loss",
                        "translation",
                        "rotation",
                        "joints",
                        "transitions",
                    )
                }
            return {"trust_region": trust_state}

        selected_horizons = tuple(
            horizon for horizon in horizons if horizon > advanced_from_horizon
        )
        if not selected_horizons:
            raise ValueError("no curriculum horizon remains after the resumed checkpoint")
        for horizon in selected_horizons:
            if resumed_horizon and horizon < resumed_horizon:
                continue
            if horizon != resumed_horizon:
                best_metric = float("inf")
                best_feasible = False
                best_constraint_ratio = float("inf")
            stale_epochs = resumed_stale_epochs if horizon == resumed_horizon else 0
            first_horizon_epoch = resumed_horizon_epoch if horizon == resumed_horizon else 0
            if trust_training and horizon != resumed_horizon:
                # The previous curriculum checkpoint is already a valid point
                # for the new horizon.  Evaluate and preserve it before taking
                # any primal step, so an inexact ALM epoch can never replace a
                # feasible operator with an infeasible one merely because its
                # rollout error is lower.
                assert trust_loader is not None
                _set_loader_epoch(trust_loader, global_epoch)
                initial_constraint = _evaluate_trust_constraint(
                    model,
                    trust_loader,
                    reference,
                    config,
                    device,
                    constraint_baseline=constraint_baseline,
                )
                initial_val = _run_epoch(
                    model,
                    val_loader,
                    config,
                    device,
                    mode="rollout",
                    horizon=horizon,
                    optimizer=None,
                    scheduler=None,
                )
                initial_ratio = initial_constraint["constraint_ratio_exact"]
                feasibility_tolerance = (
                    1e-8
                    if config.trust_region.constraint_target == "physical_one_step"
                    else 0.0
                )
                if initial_constraint["constraint_exact"] > feasibility_tolerance:
                    raise RuntimeError(
                        "the curriculum entered a new horizon from an infeasible checkpoint"
                    )
                best_metric = initial_val["terminal_dx"]
                best_feasible = True
                best_constraint_ratio = initial_ratio
                initial_record = {
                    "stage": stage,
                    "kind": "horizon_initial",
                    "epoch": global_epoch,
                    "horizon": horizon,
                    "train": initial_constraint,
                    "val": initial_val,
                }
                with (config.paths.output_dir / "metrics.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps(initial_record, sort_keys=True) + "\n")
                for destination in (
                    best_path,
                    config.paths.output_dir
                    / f"best-{stage_filename}-h{horizon:02d}.pt",
                ):
                    save_checkpoint(
                        destination,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config.to_dict(),
                        manifest_sha256=manifest.sha256(),
                        gripper_sha256=manifest.gripper_sha256,
                        stage=stage,
                        epoch=global_epoch,
                        horizon=horizon,
                        horizon_epoch=-1,
                        stale_epochs=0,
                        best_metric=best_metric,
                        extra_state=trust_extra_state(),
                    )
                print(json.dumps(initial_record, sort_keys=True), flush=True)
            for horizon_epoch in range(first_horizon_epoch, epochs_per_horizon):
                _set_loader_epoch(train_loader, global_epoch)
                if trust_loader is not None:
                    _set_loader_epoch(trust_loader, global_epoch)
                train_started = perf_counter()
                if trust_training:
                    assert trust_loader is not None
                    multiplier_before = multiplier
                    train_metrics = _run_trust_region_epoch(
                        model,
                        train_loader,
                        trust_loader,
                        reference,
                        config,
                        device,
                        horizon=horizon,
                        constraint_baseline=constraint_baseline,
                        multiplier=multiplier,
                        rho=rho,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )
                    exact_constraint = _evaluate_trust_constraint(
                        model,
                        trust_loader,
                        reference,
                        config,
                        device,
                        constraint_baseline=constraint_baseline,
                    )
                    train_metrics.update(exact_constraint)
                    multiplier = update_multiplier(
                        multiplier, exact_constraint["constraint_exact"], rho
                    )
                    train_metrics["multiplier_before_update"] = multiplier_before
                    train_metrics["multiplier_after_update"] = multiplier
                else:
                    train_metrics = _run_epoch(
                        model,
                        train_loader,
                        config,
                        device,
                        mode=loader_mode,
                        horizon=horizon,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )
                train_seconds = perf_counter() - train_started
                val_started = perf_counter()
                val_metrics = _run_epoch(
                    model,
                    val_loader,
                    config,
                    device,
                    mode=loader_mode,
                    horizon=horizon,
                    optimizer=None,
                    scheduler=None,
                )
                val_seconds = perf_counter() - val_started
                record = {
                    "stage": stage,
                    "epoch": global_epoch,
                    "horizon": horizon,
                    "train": train_metrics,
                    "val": val_metrics,
                    "timing_seconds": {
                        "train": train_seconds,
                        "val": val_seconds,
                    },
                }
                with (config.paths.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                metric = val_metrics["dx" if stage == "local" else "terminal_dx"]
                if trust_training:
                    constraint_ratio = train_metrics["constraint_ratio_exact"]
                    feasibility_tolerance = (
                        1e-8
                        if config.trust_region.constraint_target == "physical_one_step"
                        else 0.0
                    )
                    feasible = (
                        train_metrics["constraint_exact"] <= feasibility_tolerance
                    )
                    improved = (
                        (feasible and not best_feasible)
                        or (
                            feasible == best_feasible
                            and (
                                metric < best_metric
                                if feasible
                                else constraint_ratio < best_constraint_ratio
                            )
                        )
                    )
                else:
                    constraint_ratio = 0.0
                    feasible = True
                    improved = metric < best_metric
                if improved:
                    best_metric = metric
                    best_feasible = feasible
                    best_constraint_ratio = constraint_ratio
                    stale_epochs = 0
                    save_checkpoint(
                        best_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config.to_dict(),
                        manifest_sha256=manifest.sha256(),
                        gripper_sha256=manifest.gripper_sha256,
                        stage=stage,
                        epoch=global_epoch,
                        horizon=horizon,
                        horizon_epoch=horizon_epoch,
                        stale_epochs=stale_epochs,
                        best_metric=best_metric,
                        extra_state=trust_extra_state(),
                    )
                    if rollout_training:
                        # ``best-rollout.pt`` remains the backward-compatible
                        # pointer to the current curriculum best.  Preserve an
                        # immutable best checkpoint per horizon as well so a
                        # clean H4/H8/H16/H32 ablation can be evaluated after
                        # the curriculum has completed.
                        save_checkpoint(
                            config.paths.output_dir
                            / f"best-{stage_filename}-h{horizon:02d}.pt",
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            config=config.to_dict(),
                            manifest_sha256=manifest.sha256(),
                            gripper_sha256=manifest.gripper_sha256,
                            stage=stage,
                            epoch=global_epoch,
                            horizon=horizon,
                            horizon_epoch=horizon_epoch,
                            stale_epochs=stale_epochs,
                            best_metric=best_metric,
                            extra_state=trust_extra_state(),
                        )
                else:
                    stale_epochs += 1
                _write_tensorboard_epoch(
                    writer,
                    stage=stage,
                    horizon=horizon,
                    global_epoch=global_epoch,
                    horizon_epoch=horizon_epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    train_seconds=train_seconds,
                    val_seconds=val_seconds,
                    best_metric=best_metric,
                )
                print(json.dumps(record, sort_keys=True), flush=True)
                save_checkpoint(
                    config.paths.output_dir / f"last-{stage_filename}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config.to_dict(),
                    manifest_sha256=manifest.sha256(),
                    gripper_sha256=manifest.gripper_sha256,
                    stage=stage,
                    epoch=global_epoch,
                    horizon=horizon,
                    horizon_epoch=horizon_epoch,
                    stale_epochs=stale_epochs,
                    best_metric=best_metric,
                    extra_state=trust_extra_state(),
                )
                global_epoch += 1
                if stale_epochs >= config.training.early_stopping_patience:
                    break
            if rollout_training and horizon != selected_horizons[-1]:
                best_checkpoint = load_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    map_location=device,
                )
                if int(best_checkpoint["horizon"]) != horizon:
                    raise RuntimeError(
                        "best rollout checkpoint does not belong to the completed horizon"
                    )
                if trust_training:
                    trust_state = best_checkpoint["extra_state"]["trust_region"]
                    multiplier = float(trust_state["multiplier"])
                    best_feasible = bool(trust_state["best_feasible"])
                    best_constraint_ratio = float(
                        trust_state["best_constraint_ratio"]
                    )
                writer.add_scalar(
                    f"{stage}/horizon_{horizon:02d}/progress/restored_best_epoch",
                    int(best_checkpoint["epoch"]),
                    global_epoch,
                )
                writer.flush()
                print(
                    json.dumps(
                        {
                            "stage": stage,
                            "completed_horizon": horizon,
                            "restored_best_epoch": int(best_checkpoint["epoch"]),
                            "best_terminal_dx": float(best_checkpoint["best_metric"]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        return best_path
    finally:
        if writer is not None:
            writer.close()
        train_dataset.close()
        val_dataset.close()
        if trust_dataset is not None:
            trust_dataset.close()


def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    *,
    split: Literal["val", "test"] = "test",
) -> dict[str, float]:
    device = _device(config.device)
    manifest = DatasetManifest.load(config.paths.manifest)
    model = _build_model(config, manifest, device)
    checkpoint = load_checkpoint(checkpoint_path, model=model, map_location=device)
    if checkpoint["manifest_sha256"] != manifest.sha256():
        raise ValueError("checkpoint manifest hash mismatch")
    dataset = H5ObjectDataset(manifest, split=split)
    try:
        loader = make_dataloader(
            dataset,
            mode="rollout",
            objects_per_batch=1,
            # Test and final validation report every trajectory for each object.
            samples_per_object=0,
            workers=config.loader.workers,
            seed=config.seed + 2,
            shuffle=False,
        )
        model.eval()
        accumulator = MetricAccumulator()
        with torch.no_grad():
            for raw_batch in loader:
                batch = raw_batch.to(device, non_blocking=True)
                with _autocast(config, device):
                    prediction = model.rollout(
                        _state_at(batch.states, 0),
                        batch.command_schedule[1:],
                        batch.sdf,
                    )
                gaps = torch.stack(
                    [
                        model.query_geometric_gap(_state_at(prediction, step), batch.sdf)
                        for step in range(33)
                    ],
                    dim=1,
                )
                accumulate_trajectory_metrics(
                    accumulator,
                    prediction,
                    batch.states,
                    batch.command_schedule,
                    model.aperture_from_joints(prediction.joint_position),
                    batch.actual_aperture,
                    gaps,
                    length_scale=model.length_scale,
                    joint_scale=model.joint_travel_range,
                    lag_threshold=manifest.delta_gate_m,
                )
        metrics = accumulator.compute()
        writer = _open_summary_writer(config.paths.output_dir)
        try:
            checkpoint_name = Path(checkpoint_path).stem
            prefix = f"evaluation/{split}/{checkpoint_name}"
            checkpoint_epoch = int(checkpoint.get("epoch", 0))
            for name, value in metrics.items():
                writer.add_scalar(f"{prefix}/{name}", value, checkpoint_epoch)
            writer.add_text(
                f"{prefix}/checkpoint",
                str(Path(checkpoint_path).resolve()),
                checkpoint_epoch,
            )
            writer.flush()
        finally:
            writer.close()
        return metrics
    finally:
        dataset.close()
