#!/usr/bin/env python3
"""Controlled uniform-vs-BV-measure objective ablation for local SRNO.

Rate-independent trajectories are BV curves in the load coordinate.  Uniform
step regression approximates an L2(d lambda) risk and gives vanishing mass to
jumps under load refinement.  The graph-completion measure

    d sigma = d lambda + |D x|

has atoms at jumps.  This script changes only the empirical integration
measure: the direct L=1 cell, data, state metric, physics loss, optimizer, and
initial checkpoint are shared between both arms.
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
    LocalTransitionBatch,
    ObjectBatchCollator,
    TrajectoryBatch,
    make_dataloader,
)
from srno.data.index import ActiveIndex, file_sha256
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import rotation_geodesic_angle
from srno.losses import state_error
from srno.training.checkpoint import load_checkpoint, save_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.training.metrics import MetricAccumulator, accumulate_trajectory_metrics
from srno.types import PoseState


Measure = Literal["uniform", "bv"]
MEASURES: tuple[Measure, ...] = ("uniform", "bv")


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


def _variation_mass(
    model,
    batch: LocalTransitionBatch,
    manifest: DatasetManifest,
) -> tuple[Tensor, Tensor, Tensor]:
    state_increment = state_error(
        batch.target,
        batch.current,
        length_scale=model.length_scale,
        joint_scale=model.joint_travel_range,
    )[0].sqrt()
    schedule = batch.next_command.new_tensor(manifest.commanded_aperture_m)
    load_increment = (
        schedule.index_select(0, batch.step_index + 1)
        - schedule.index_select(0, batch.step_index)
    ).abs() / model.length_scale
    pose_increment = torch.sqrt(
        (
            torch.linalg.vector_norm(
                batch.target.position - batch.current.position, dim=-1
            )
            / model.length_scale
        ).square()
        + rotation_geodesic_angle(
            batch.target.rotation, batch.current.rotation
        ).square()
    )
    return load_increment + state_increment, state_increment, pose_increment


def _sample_terms(model, prediction, target, gap, config):
    _, translation, rotation, joints = state_error(
        prediction,
        target,
        length_scale=model.length_scale,
        joint_scale=model.joint_travel_range,
        lambda_rotation=config.loss.lambda_rotation,
        lambda_joints=config.loss.lambda_joints,
    )
    flow = (
        translation
        + config.loss.lambda_rotation * rotation
        + config.loss.lambda_joints * joints
    )
    feasibility = torch.relu(
        (config.loss.admissible_gap_m - gap) / model.sdf_scale
    ).square().mean(dim=-1)
    total = flow + config.loss.lambda_feasibility * feasibility
    return total, flow, feasibility, translation, rotation, joints


def _weighted_mean(values: Tensor, mass: Tensor) -> Tensor:
    return (values * mass).sum() / mass.sum().clamp_min(1e-12)


def _iteration(model, batch, config, manifest, measure: Measure):
    prediction = model.forward_step(batch.current, batch.next_command, batch.sdf)
    assert isinstance(prediction, PoseState)
    gap = model.query_geometric_gap(prediction, batch.sdf)
    terms = _sample_terms(model, prediction, batch.target, gap, config)
    variation_mass, state_increment, pose_increment = _variation_mass(
        model, batch, manifest
    )
    mass = torch.ones_like(variation_mass) if measure == "uniform" else variation_mass
    loss = _weighted_mean(terms[0], mass)
    dx = _dx(model, prediction, batch.target, config)
    jump = pose_increment > 0.05
    return loss, {
        "loss": float(loss.detach()),
        "uniform_dx": float(dx.mean().detach()),
        "measure_dx": float(_weighted_mean(dx, mass).detach()),
        "jump_dx": float(dx[jump].mean().detach()) if jump.any() else math.nan,
        "jump_fraction": float(jump.float().mean().detach()),
        "mass_mean": float(variation_mass.mean().detach()),
        "mass_max": float(variation_mass.max().detach()),
    }


def _dx(model, prediction, target, config) -> Tensor:
    return state_error(
        prediction,
        target,
        length_scale=model.length_scale,
        joint_scale=model.joint_travel_range,
        lambda_rotation=config.loss.lambda_rotation,
        lambda_joints=config.loss.lambda_joints,
    )[0].sqrt()


def _mean(records: list[dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in records[0]:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        result[key] = float(np.nanmean(values))
    return result


def _evaluate_local(model, loader, config, manifest, device) -> dict[str, float]:
    sums = {
        "uniform_dx": 0.0,
        "bv_dx_numerator": 0.0,
        "bv_mass": 0.0,
        "jump_dx": 0.0,
        "jump_count": 0.0,
        "nonjump_dx": 0.0,
        "nonjump_count": 0.0,
        "count": 0.0,
    }
    model.eval()
    with torch.no_grad():
        for raw in loader:
            batch = raw.to(device)
            with _autocast(config, device):
                prediction = model.forward_step(
                    batch.current, batch.next_command, batch.sdf
                )
                assert isinstance(prediction, PoseState)
            dx = _dx(model, prediction, batch.target, config)
            mass, _, pose_increment = _variation_mass(model, batch, manifest)
            jump = pose_increment > 0.05
            size = dx.numel()
            sums["uniform_dx"] += float(dx.sum().detach())
            sums["bv_dx_numerator"] += float((dx * mass).sum().detach())
            sums["bv_mass"] += float(mass.sum().detach())
            sums["jump_dx"] += float(dx[jump].sum().detach())
            sums["jump_count"] += float(jump.sum().detach())
            sums["nonjump_dx"] += float(dx[~jump].sum().detach())
            sums["nonjump_count"] += float((~jump).sum().detach())
            sums["count"] += size
    return {
        "uniform_dx": sums["uniform_dx"] / sums["count"],
        "bv_dx": sums["bv_dx_numerator"] / sums["bv_mass"],
        "jump_dx": sums["jump_dx"] / max(1.0, sums["jump_count"]),
        "nonjump_dx": sums["nonjump_dx"] / max(1.0, sums["nonjump_count"]),
        "jump_fraction": sums["jump_count"] / sums["count"],
        "count": sums["count"],
    }


def _evaluate_rollout(model, loader, config, manifest, device) -> dict[str, float]:
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


def _stochastic_local_loader(dataset, config, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.loader.objects_per_batch,
        shuffle=True,
        num_workers=config.loader.workers,
        collate_fn=ObjectBatchCollator(
            dataset.manifest,
            mode="local",
            samples_per_object=config.loader.local_samples_per_object,
            seed=seed,
            resample=True,
        ),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.loader.workers > 0,
        prefetch_factor=2 if config.loader.workers > 0 else None,
        generator=generator,
    )


def _train_arm(
    measure: Measure,
    config,
    manifest,
    train_loader,
    val_loader,
    device,
    initialize,
    epochs,
    patience,
):
    arm_output = config.paths.output_dir / measure
    arm_output.mkdir(parents=True, exist_ok=True)
    model = _build_model(config, manifest, device)
    load_checkpoint(initialize, model=model, map_location=device)
    optimizer, scheduler = _optimizer(model, config, max(1, len(train_loader) * epochs))
    best = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    best_path = arm_output / "best-bv-local.pt"
    for epoch in range(epochs):
        model.train(True)
        records: list[dict[str, float]] = []
        start = perf_counter()
        for raw in train_loader:
            batch = raw.to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(config, device):
                loss, metrics = _iteration(
                    model, batch, config, manifest, measure
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
        validation = _evaluate_local(model, val_loader, config, manifest, device)
        selection = validation["uniform_dx" if measure == "uniform" else "bv_dx"]
        record = {
            "epoch": epoch,
            "seconds": perf_counter() - start,
            "train": _mean(records),
            "val": validation,
            "selection": selection,
        }
        history.append(record)
        with (arm_output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if selection < best:
            best = selection
            stale = 0
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config.to_dict(),
                manifest_sha256=manifest.sha256(),
                gripper_sha256=manifest.gripper_sha256,
                stage=f"bv_local_{measure}",
                epoch=epoch,
                horizon=1,
                horizon_epoch=epoch,
                stale_epochs=0,
                best_metric=best,
                extra_state={"measure": measure},
            )
        else:
            stale += 1
        print(
            f"[BV] arm={measure} epoch={epoch:03d} "
            f"val_dx={validation['uniform_dx']:.6f} "
            f"val_bv={validation['bv_dx']:.6f} "
            f"val_jump={validation['jump_dx']:.6f} "
            f"best={best:.6f} stale={stale}",
            flush=True,
        )
        if stale >= patience:
            break
    load_checkpoint(best_path, model=model, map_location=device)
    return model, best_path, history


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
    )
    if config.model.operator_layers != 1 or config.model.contact_head != "direct":
        raise ValueError("controlled BV ablation requires direct L=1")
    if config.loss.pose_penalty != "squared":
        raise ValueError("BV screen currently preserves the production squared loss")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = config.paths.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest = DatasetManifest.load(config.paths.manifest)
    active = ActiveIndex.load(config.paths.active_index)
    train_dataset = H5ObjectDataset(
        manifest, split="train", active_index=active, active_only=True
    )
    val_local_dataset = H5ObjectDataset(
        manifest, split="val", active_index=active, active_only=True
    )
    test_local_dataset = H5ObjectDataset(
        manifest, split="test", active_index=active, active_only=True
    )
    val_rollout_dataset = H5ObjectDataset(manifest, split="val")
    test_rollout_dataset = H5ObjectDataset(manifest, split="test")
    val_local_loader = make_dataloader(
        val_local_dataset,
        mode="local",
        objects_per_batch=1,
        samples_per_object=0,
        workers=config.loader.workers,
        seed=config.seed + 1,
        shuffle=False,
    )
    test_local_loader = make_dataloader(
        test_local_dataset,
        mode="local",
        objects_per_batch=1,
        samples_per_object=0,
        workers=config.loader.workers,
        seed=config.seed + 2,
        shuffle=False,
    )
    val_rollout_loader = make_dataloader(
        val_rollout_dataset,
        mode="rollout",
        objects_per_batch=1,
        samples_per_object=0,
        workers=config.loader.workers,
        seed=config.seed + 1,
        shuffle=False,
    )
    test_rollout_loader = make_dataloader(
        test_rollout_dataset,
        mode="rollout",
        objects_per_batch=1,
        samples_per_object=0,
        workers=config.loader.workers,
        seed=config.seed + 2,
        shuffle=False,
    )
    try:
        runs: dict[str, Any] = {}
        for arm_index, measure in enumerate(MEASURES):
            _seed_everything(config.seed)
            train_loader = _stochastic_local_loader(
                train_dataset, config, config.seed
            )
            model, checkpoint, history = _train_arm(
                measure,
                config,
                manifest,
                train_loader,
                val_local_loader,
                device,
                args.initialize.resolve(),
                args.epochs,
                args.patience,
            )
            runs[measure] = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "history": history,
                "local": {
                    "val": _evaluate_local(
                        model, val_local_loader, config, manifest, device
                    ),
                    "test": _evaluate_local(
                        model, test_local_loader, config, manifest, device
                    ),
                },
                "rollout": {
                    "val": _evaluate_rollout(
                        model, val_rollout_loader, config, manifest, device
                    ),
                    "test": _evaluate_rollout(
                        model, test_rollout_loader, config, manifest, device
                    ),
                },
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        result = {
            "definition": {
                "uniform_measure": "d lambda",
                "bv_measure": "d sigma = d lambda/L + d_X(x_{k+1},x_k)",
                "jump": "pose d_X(x_{k+1},x_k) > 0.05",
                "architecture_changed": False,
                "physics_data_changed": False,
                "stochastic_samples_per_object_per_epoch": config.loader.local_samples_per_object,
            },
            "contract": {
                "config": str(args.config.resolve()),
                "manifest": str(config.paths.manifest),
                "manifest_sha256": manifest.sha256(),
                "active_index": str(config.paths.active_index),
                "active_index_sha256": file_sha256(config.paths.active_index),
                "initial_checkpoint": str(args.initialize.resolve()),
                "initial_checkpoint_sha256": file_sha256(args.initialize),
                "seed": config.seed,
                "device": str(device),
            },
            "runs": runs,
            "comparison_bv_minus_uniform": {
                split: {
                    "local_uniform_dx": runs["bv"]["local"][split]["uniform_dx"]
                    - runs["uniform"]["local"][split]["uniform_dx"],
                    "local_jump_dx": runs["bv"]["local"][split]["jump_dx"]
                    - runs["uniform"]["local"][split]["jump_dx"],
                    "terminal_dx": runs["bv"]["rollout"][split]["terminal_dx"]
                    - runs["uniform"]["rollout"][split]["terminal_dx"],
                }
                for split in ("val", "test")
            },
        }
        _save_json(output / "results.json", result)
        print(json.dumps({
            measure: {
                "test_local": runs[measure]["local"]["test"],
                "test_H32": runs[measure]["rollout"]["test"]["terminal_dx"],
            }
            for measure in MEASURES
        }, indent=2), flush=True)
    finally:
        train_dataset.close()
        val_local_dataset.close()
        test_local_dataset.close()
        val_rollout_dataset.close()
        test_rollout_dataset.close()


if __name__ == "__main__":
    main()
