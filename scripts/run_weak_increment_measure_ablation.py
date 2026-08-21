#!/usr/bin/env python3
"""Learn the SRNO increment measure through multiscale weak defects.

For teacher-forced trajectory increments let e_k be the signed generalized
increment defect in the spatial tangent coordinates used by the production
metric.  Pointwise regression minimizes only ||e_k||^2.  This experiment uses
the dyadic weak norm

    mean_{h in {1,2,4,8,16,32}}
      mean_W ||sum_{k in W} e_k / sqrt(h)||^2,

where W ranges over aligned windows of length h.  The sqrt(h) normalization
keeps zero-mean defect variance at the pointwise scale, while a coherent signed
bias grows like h.  Thus the learned object is the vector-valued increment
measure through its integrals, rather than 32 unrelated endpoint regressions.

Both ablation arms use the same L=1 model, checkpoint initialization, data,
optimizer and sample stream.  The uniform arm uses only h=1.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from srno.data.dataset import (
    H5ObjectDataset,
    ObjectBatchCollator,
    TrajectoryBatch,
    make_dataloader,
)
from srno.data.index import ActiveIndex, file_sha256
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import so3_log_vector
from srno.losses import state_error
from srno.training.checkpoint import load_checkpoint, save_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.training.metrics import MetricAccumulator, accumulate_trajectory_metrics
from srno.types import PoseState, SDFBatch


Objective = Literal["pointwise", "weak"]
OBJECTIVES: tuple[Objective, ...] = ("pointwise", "weak")
DYADIC_SCALES = (1, 2, 4, 8, 16, 32)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _state_at(states: PoseState, step: int) -> PoseState:
    return PoseState(
        states.rotation[:, step],
        states.position[:, step],
        states.joint_position[:, step],
    )


def _autocast(config: ExperimentConfig, device: torch.device):
    enabled = config.training.use_bfloat16 and device.type == "cuda"
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=enabled
    )


def _optimizer(model, config: ExperimentConfig, total_steps: int):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        fused=next(model.parameters()).is_cuda,
    )
    warmup = int(total_steps * config.optimizer.warmup_fraction)

    def multiplier(step: int) -> float:
        if warmup and step < warmup:
            return max((step + 1) / warmup, 1e-3)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return optimizer, scheduler


def _flatten_teacher_batch(batch: TrajectoryBatch) -> tuple[PoseState, PoseState, Tensor, SDFBatch]:
    trajectories = batch.states.position.shape[0]
    steps = batch.states.position.shape[1] - 1
    current = PoseState(
        batch.states.rotation[:, :-1].reshape(-1, 3, 3),
        batch.states.position[:, :-1].reshape(-1, 3),
        batch.states.joint_position[:, :-1].reshape(-1, 6),
    )
    target = PoseState(
        batch.states.rotation[:, 1:].reshape(-1, 3, 3),
        batch.states.position[:, 1:].reshape(-1, 3),
        batch.states.joint_position[:, 1:].reshape(-1, 6),
    )
    commands = batch.command_schedule[1:].view(1, steps).expand(
        trajectories, steps
    ).reshape(-1)
    sdf = SDFBatch(
        batch.sdf.values,
        batch.sdf.origin,
        batch.sdf.voxel_size,
        batch.sdf.sample_to_object.repeat_interleave(steps),
        batch.sdf.outside_value,
    )
    return current, target, commands, sdf


def _signed_defect(model, prediction: PoseState, target: PoseState) -> Tensor:
    translation = (prediction.position - target.position) / model.length_scale
    # Spatial (left-trivialized) error matches the production SE(3) update.
    rotation = so3_log_vector(
        prediction.rotation @ target.rotation.transpose(-1, -2)
    )
    joints = (
        (prediction.joint_position - target.joint_position)
        / model.joint_travel_range
        / math.sqrt(6.0)
    )
    return torch.cat((translation, rotation, joints), dim=-1)


def _multiscale_losses(defect: Tensor) -> dict[int, Tensor]:
    if defect.ndim != 3 or defect.shape[1:] != (32, 12):
        raise ValueError("defect must have shape [trajectories, 32, 12]")
    losses: dict[int, Tensor] = {}
    for scale in DYADIC_SCALES:
        windows = defect.reshape(
            defect.shape[0], 32 // scale, scale, defect.shape[-1]
        )
        integrated = windows.sum(dim=2) / math.sqrt(scale)
        losses[scale] = integrated.square().sum(dim=-1).mean()
    return losses


def _teacher_iteration(model, batch, config, objective: Objective):
    current, target, commands, sdf = _flatten_teacher_batch(batch)
    prediction = model.forward_step(current, commands, sdf)
    assert isinstance(prediction, PoseState)
    trajectories = batch.states.position.shape[0]
    defect = _signed_defect(model, prediction, target).reshape(
        trajectories, 32, 12
    )
    scale_losses = _multiscale_losses(defect)
    scales = (1,) if objective == "pointwise" else DYADIC_SCALES
    flow = torch.stack([scale_losses[scale] for scale in scales]).mean()
    gap = model.query_geometric_gap(prediction, sdf)
    feasibility = torch.relu(
        (config.loss.admissible_gap_m - gap) / model.sdf_scale
    ).square().mean()
    loss = flow + config.loss.lambda_feasibility * feasibility
    metrics = {
        "loss": float(loss.detach()),
        "flow": float(flow.detach()),
        "feasibility": float(feasibility.detach()),
    }
    metrics.update(
        {
            f"defect_h{scale:02d}": float(value.detach())
            for scale, value in scale_losses.items()
        }
    )
    return loss, metrics


def _mean(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in records[0]
    }


def _evaluate_teacher(model, loader, config, objective, device):
    model.eval()
    records: list[dict[str, float]] = []
    with torch.no_grad():
        for raw in loader:
            batch = raw.to(device)
            with _autocast(config, device):
                _, metrics = _teacher_iteration(model, batch, config, objective)
            records.append(metrics)
    return _mean(records)


def _evaluate_rollout(model, loader, manifest, config, device):
    accumulator = MetricAccumulator()
    model.eval()
    with torch.no_grad():
        for raw in loader:
            batch = raw.to(device)
            with _autocast(config, device):
                prediction = model.rollout(
                    _state_at(batch.states, 0), batch.command_schedule[1:], batch.sdf
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
    return accumulator.compute()


def _stochastic_rollout_loader(dataset, config, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.loader.objects_per_batch,
        shuffle=True,
        num_workers=0,
        collate_fn=ObjectBatchCollator(
            dataset.manifest,
            mode="rollout",
            samples_per_object=config.loader.rollout_trajectories_per_object,
            seed=seed,
            resample=True,
        ),
        pin_memory=False,
        generator=generator,
    )


def _train_arm(
    objective,
    config,
    manifest,
    train_loader,
    val_loader,
    device,
    initialize,
    epochs,
    patience,
):
    output = config.paths.output_dir / objective
    output.mkdir(parents=True, exist_ok=True)
    model = _build_model(config, manifest, device)
    load_checkpoint(initialize, model=model, map_location=device)
    optimizer, scheduler = _optimizer(model, config, max(1, len(train_loader) * epochs))
    best = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = output / "best-weak-increment.pt"
    for epoch in range(epochs):
        model.train(True)
        records: list[dict[str, float]] = []
        start = perf_counter()
        for raw in train_loader:
            batch = raw.to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(config, device):
                loss, metrics = _teacher_iteration(
                    model, batch, config, objective
                )
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.optimizer.gradient_clip
            )
            optimizer.step()
            scheduler.step()
            metrics["gradient_norm"] = float(gradient.detach())
            metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
            records.append(metrics)
        validation = _evaluate_teacher(
            model, val_loader, config, objective, device
        )
        selection = validation["flow"]
        record = {
            "epoch": epoch,
            "seconds": perf_counter() - start,
            "train": _mean(records),
            "val": validation,
            "selection": selection,
        }
        history.append(record)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if selection < best:
            best = selection
            stale = 0
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config.to_dict(),
                manifest_sha256=manifest.sha256(),
                gripper_sha256=manifest.gripper_sha256,
                stage=f"increment_measure_{objective}",
                epoch=epoch,
                horizon=32,
                horizon_epoch=epoch,
                stale_epochs=0,
                best_metric=best,
                extra_state={
                    "objective": objective,
                    "scales": [1] if objective == "pointwise" else list(DYADIC_SCALES),
                },
            )
        else:
            stale += 1
        print(
            f"[WEAK] arm={objective} epoch={epoch:03d} "
            f"val_h01={validation['defect_h01']:.6f} "
            f"val_h32={validation['defect_h32']:.6f} "
            f"selection={selection:.6f} best={best:.6f} stale={stale}",
            flush=True,
        )
        if stale >= patience:
            break
    load_checkpoint(checkpoint_path, model=model, map_location=device)
    return model, checkpoint_path, history


def _save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("epochs and patience must be positive")

    base = ExperimentConfig.load(args.config)
    config = replace(
        base,
        device=args.device,
        paths=replace(base.paths, output_dir=args.output.resolve()),
        loader=replace(base.loader, workers=0),
    )
    if config.model.operator_layers != 1 or config.model.contact_head != "direct":
        raise ValueError("controlled weak-defect ablation requires direct L=1")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = config.paths.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest = DatasetManifest.load(config.paths.manifest)
    train_dataset = H5ObjectDataset(manifest, split="train")
    val_dataset = H5ObjectDataset(manifest, split="val")
    test_dataset = H5ObjectDataset(manifest, split="test")
    val_loader = make_dataloader(
        val_dataset,
        mode="rollout",
        objects_per_batch=1,
        samples_per_object=0,
        workers=0,
        seed=config.seed + 1,
        shuffle=False,
    )
    test_loader = make_dataloader(
        test_dataset,
        mode="rollout",
        objects_per_batch=1,
        samples_per_object=0,
        workers=0,
        seed=config.seed + 2,
        shuffle=False,
    )
    try:
        baseline_model = _build_model(config, manifest, device)
        load_checkpoint(args.initialize, model=baseline_model, map_location=device)
        baseline = {
            "teacher_val": _evaluate_teacher(
                baseline_model, val_loader, config, "weak", device
            ),
            "teacher_test": _evaluate_teacher(
                baseline_model, test_loader, config, "weak", device
            ),
            "rollout_val": _evaluate_rollout(
                baseline_model, val_loader, manifest, config, device
            ),
            "rollout_test": _evaluate_rollout(
                baseline_model, test_loader, manifest, config, device
            ),
        }
        del baseline_model
        runs: dict[str, Any] = {}
        for objective in OBJECTIVES:
            _seed_everything(config.seed)
            train_loader = _stochastic_rollout_loader(
                train_dataset, config, config.seed
            )
            model, checkpoint, history = _train_arm(
                objective,
                config,
                manifest,
                train_loader,
                val_loader,
                device,
                args.initialize.resolve(),
                args.epochs,
                args.patience,
            )
            runs[objective] = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "history": history,
                "teacher_val": _evaluate_teacher(
                    model, val_loader, config, "weak", device
                ),
                "teacher_test": _evaluate_teacher(
                    model, test_loader, config, "weak", device
                ),
                "rollout_val": _evaluate_rollout(
                    model, val_loader, manifest, config, device
                ),
                "rollout_test": _evaluate_rollout(
                    model, test_loader, manifest, config, device
                ),
            }
            del model
        result = {
            "definition": {
                "signed_defect": "[dp/L, Log(R_hat R_true^-1), dq/(sqrt(6) travel)]",
                "pointwise": "h={1}",
                "weak": "mean over h={1,2,4,8,16,32} of ||sum_W e/sqrt(h)||^2",
                "architecture_changed": False,
                "physics_data_changed": False,
            },
            "contract": {
                "config": str(args.config.resolve()),
                "manifest": str(config.paths.manifest),
                "manifest_sha256": manifest.sha256(),
                "initial_checkpoint": str(args.initialize.resolve()),
                "initial_checkpoint_sha256": file_sha256(args.initialize),
                "seed": config.seed,
                "device": str(device),
            },
            "baseline": baseline,
            "runs": runs,
            "comparison": {
                objective: {
                    split: {
                        "terminal_dx_change": runs[objective][f"rollout_{split}"]["terminal_dx"]
                        - baseline[f"rollout_{split}"]["terminal_dx"],
                        "h32_defect_change": runs[objective][f"teacher_{split}"]["defect_h32"]
                        - baseline[f"teacher_{split}"]["defect_h32"],
                    }
                    for split in ("val", "test")
                }
                for objective in OBJECTIVES
            },
        }
        _save_json(output / "results.json", result)
        print(json.dumps({
            "baseline_test_H32": baseline["rollout_test"]["terminal_dx"],
            **{
                objective: {
                    "test_H32": runs[objective]["rollout_test"]["terminal_dx"],
                    "test_defect_h32": runs[objective]["teacher_test"]["defect_h32"],
                }
                for objective in OBJECTIVES
            },
        }, indent=2), flush=True)
    finally:
        train_dataset.close()
        val_dataset.close()
        test_dataset.close()


if __name__ == "__main__":
    main()
