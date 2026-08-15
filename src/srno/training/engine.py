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
from srno.data.schema import DatasetManifest
from srno.geometry.gripper import GripperAsset
from srno.losses import combined_loss
from srno.losses import state_error
from srno.model import SRNOModel
from srno.training.checkpoint import load_checkpoint, save_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.metrics import MetricAccumulator, accumulate_trajectory_metrics
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
        hidden_dim=config.model.hidden_dim,
    ).to(device)


def _autocast(config: ExperimentConfig, device: torch.device):
    enabled = config.training.use_bfloat16 and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def _optimizer(model: SRNOModel, config: ExperimentConfig, total_steps: int):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        fused=model.surface_intercept.is_cuda,
    )
    warmup_steps = int(total_steps * config.optimizer.warmup_fraction)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-3)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _state_at(states: PoseState, step: int) -> PoseState:
    return PoseState(states.rotation[:, step], states.position[:, step], states.aperture[:, step])


def _local_iteration(
    model: SRNOModel,
    batch: LocalTransitionBatch,
    config: ExperimentConfig,
) -> tuple[Tensor, dict[str, float]]:
    prediction = model.forward_step(batch.current, batch.next_command, batch.sdf)
    assert isinstance(prediction, PoseState)
    gap = model.query_gap(prediction, batch.sdf)
    terms = combined_loss(
        prediction,
        batch.target,
        gap,
        length_scale=model.length_scale,
        sdf_scale=model.sdf_scale,
        lambda_rotation=config.loss.lambda_rotation,
        lambda_aperture=config.loss.lambda_aperture,
        lambda_feasibility=config.loss.lambda_feasibility,
    )
    values = {
        "loss": float(terms.total.detach()),
        "flow": float(terms.flow.detach()),
        "feasibility": float(terms.feasibility.detach()),
        "translation": float(terms.translation.detach()),
        "rotation": float(terms.rotation.detach()),
        "aperture": float(terms.aperture.detach()),
        "dx": float(
            state_error(
                prediction,
                batch.target,
                length_scale=model.length_scale,
                lambda_rotation=config.loss.lambda_rotation,
                lambda_aperture=config.loss.lambda_aperture,
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
    prediction = model.rollout(initial, batch.command_schedule[1 : horizon + 1], batch.sdf)
    total = prediction.position.new_zeros(())
    sums = {name: 0.0 for name in ("flow", "feasibility", "translation", "rotation", "aperture")}
    gaps: list[Tensor] = []
    for step in range(1, horizon + 1):
        predicted_step = _state_at(prediction, step)
        target_step = _state_at(batch.states, step)
        gap = model.query_gap(predicted_step, batch.sdf)
        gaps.append(gap)
        terms = combined_loss(
            predicted_step,
            target_step,
            gap,
            length_scale=model.length_scale,
            sdf_scale=model.sdf_scale,
            lambda_rotation=config.loss.lambda_rotation,
            lambda_aperture=config.loss.lambda_aperture,
            lambda_feasibility=config.loss.lambda_feasibility,
        )
        total = total + terms.total / horizon
        for name in sums:
            sums[name] += float(getattr(terms, name).detach()) / horizon
    terminal_dx = (
        state_error(
            _state_at(prediction, horizon),
            _state_at(batch.states, horizon),
            length_scale=model.length_scale,
            lambda_rotation=config.loss.lambda_rotation,
            lambda_aperture=config.loss.lambda_aperture,
        )[0]
        .sqrt()
        .mean()
    )
    return (
        total,
        {"loss": float(total.detach()), "terminal_dx": float(terminal_dx.detach()), **sums},
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
                    loss, metrics, _, _ = _rollout_iteration(model, batch, config, horizon)
            if training:
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optimizer.gradient_clip
                )
                optimizer.step()
                assert scheduler is not None
                scheduler.step()
                metrics["gradient_norm"] = float(gradient_norm.detach())
                metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
            values.append(metrics)
    return _mean_dict(values)


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
    stage: Literal["local", "rollout"],
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
    stage: Literal["local", "rollout"],
    resume: str | Path | None = None,
) -> Path:
    _seed_everything(config.seed)
    device = _device(config.device)
    manifest = DatasetManifest.load(config.paths.manifest)
    model = _build_model(config, manifest, device)
    active_index = ActiveIndex.load(config.paths.active_index) if stage == "local" else None
    train_dataset = H5ObjectDataset(
        manifest,
        split="train",
        active_index=active_index,
        active_only=stage == "local",
    )
    val_dataset = H5ObjectDataset(
        manifest,
        split="val",
        active_index=active_index,
        active_only=stage == "local",
    )
    writer = None
    try:
        train_loader = make_dataloader(
            train_dataset,
            mode=stage,
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
            mode=stage,
            objects_per_batch=1,
            samples_per_object=0,
            workers=config.loader.workers,
            seed=config.seed + 1,
            shuffle=False,
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
            elif not (checkpoint_stage == "local" and stage == "rollout"):
                raise ValueError(
                    f"cannot initialize {stage!r} training from {checkpoint_stage!r} checkpoint"
                )

        config.paths.output_dir.mkdir(parents=True, exist_ok=True)
        (config.paths.output_dir / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        writer = _open_summary_writer(config.paths.output_dir)
        writer.add_text(
            f"{stage}/config",
            "```json\n" + json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n```",
            start_epoch,
        )
        global_epoch = start_epoch
        stale_epochs = 0
        best_path = config.paths.output_dir / f"best-{stage}.pt"
        for horizon in horizons:
            if resumed_horizon and horizon < resumed_horizon:
                continue
            if horizon != resumed_horizon:
                best_metric = float("inf")
            stale_epochs = resumed_stale_epochs if horizon == resumed_horizon else 0
            first_horizon_epoch = resumed_horizon_epoch if horizon == resumed_horizon else 0
            for horizon_epoch in range(first_horizon_epoch, epochs_per_horizon):
                _set_loader_epoch(train_loader, global_epoch)
                train_started = perf_counter()
                train_metrics = _run_epoch(
                    model,
                    train_loader,
                    config,
                    device,
                    mode=stage,
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
                    mode=stage,
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
                improved = metric < best_metric
                if improved:
                    best_metric = metric
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
                    config.paths.output_dir / f"last-{stage}.pt",
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
                )
                global_epoch += 1
                if stale_epochs >= config.training.early_stopping_patience:
                    break
            if stage == "rollout" and horizon != horizons[-1]:
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
                writer.add_scalar(
                    f"rollout/horizon_{horizon:02d}/progress/restored_best_epoch",
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
                        _state_at(batch.states, 0), batch.command_schedule[1:], batch.sdf
                    )
                gaps = torch.stack(
                    [model.query_gap(_state_at(prediction, step), batch.sdf) for step in range(33)],
                    dim=1,
                )
                accumulate_trajectory_metrics(
                    accumulator,
                    prediction,
                    batch.states,
                    batch.command_schedule,
                    gaps,
                    length_scale=model.length_scale,
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
