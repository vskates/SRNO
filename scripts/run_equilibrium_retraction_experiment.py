#!/usr/bin/env python3
"""Test a fixed-load equilibrium retraction formulation for SRNO.

Instead of fitting only transport x_k -> x_{k+1}, fit one shared map P_phi,u
whose target equilibrium is a fixed point and whose repeated application from
the preceding equilibrium stays in the same basin:

    P(x_k; u_{k+1}) -> x_{k+1},
    P(P(x_k; u_{k+1}); u_{k+1}) -> x_{k+1},
    P(x_{k+1}; u_{k+1}) -> x_{k+1}.

At inference each load increment is a warm-started fixed-point solve.  The
network architecture and physics/data contract remain unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import random
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor

from srno.data.dataset import (
    H5ObjectDataset,
    LocalTransitionBatch,
    TrajectoryBatch,
    make_dataloader,
)
from srno.data.index import ActiveIndex, file_sha256
from srno.data.schema import DatasetManifest
from srno.losses import combined_loss, state_error
from srno.training.checkpoint import load_checkpoint, save_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.training.metrics import MetricAccumulator, accumulate_trajectory_metrics
from srno.types import PoseState


REFINEMENTS = (1, 2, 4, 8)


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


def _physical_loss(model, prediction, target, sdf, config):
    gap = model.query_geometric_gap(prediction, sdf)
    return combined_loss(
        prediction,
        target,
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


def _dx(model, prediction, target, config) -> Tensor:
    return state_error(
        prediction,
        target,
        length_scale=model.length_scale,
        joint_scale=model.joint_travel_range,
        lambda_rotation=config.loss.lambda_rotation,
        lambda_joints=config.loss.lambda_joints,
    )[0].sqrt()


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


def _iteration(model, batch: LocalTransitionBatch, config, fixed_weight: float):
    first = model.forward_step(batch.current, batch.next_command, batch.sdf)
    assert isinstance(first, PoseState)
    refined = model.forward_step(first, batch.next_command, batch.sdf)
    assert isinstance(refined, PoseState)
    fixed = model.forward_step(batch.target, batch.next_command, batch.sdf)
    assert isinstance(fixed, PoseState)

    first_terms = _physical_loss(model, first, batch.target, batch.sdf, config)
    refined_terms = _physical_loss(model, refined, batch.target, batch.sdf, config)
    fixed_terms = _physical_loss(model, fixed, batch.target, batch.sdf, config)
    transport = 0.5 * (first_terms.total + refined_terms.total)
    loss = transport + fixed_weight * fixed_terms.total
    return loss, {
        "loss": float(loss.detach()),
        "transport_loss": float(transport.detach()),
        "fixed_loss": float(fixed_terms.total.detach()),
        "first_dx": float(_dx(model, first, batch.target, config).mean().detach()),
        "refined_dx": float(
            _dx(model, refined, batch.target, config).mean().detach()
        ),
        "fixed_dx": float(_dx(model, fixed, batch.target, config).mean().detach()),
    }


def _mean(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in records[0]
    }


def _evaluate_local(model, loader, config, device) -> dict[str, float]:
    totals = torch.zeros(4, dtype=torch.float64)
    count = 0
    fixed_total = 0.0
    model.eval()
    with torch.no_grad():
        for raw in loader:
            batch = raw.to(device)
            with _autocast(config, device):
                first = model.forward_step(batch.current, batch.next_command, batch.sdf)
                assert isinstance(first, PoseState)
                refined = model.forward_step(first, batch.next_command, batch.sdf)
                assert isinstance(refined, PoseState)
                fixed = model.forward_step(batch.target, batch.next_command, batch.sdf)
                assert isinstance(fixed, PoseState)
            values = (
                _dx(model, first, batch.target, config),
                _dx(model, refined, batch.target, config),
                _dx(model, fixed, batch.target, config),
            )
            size = values[0].numel()
            totals[:3] += torch.tensor(
                [float(value.sum().detach()) for value in values], dtype=torch.float64
            )
            fixed_total += float(
                _dx(model, refined, first, config).sum().detach()
            )
            count += size
    totals[3] = fixed_total
    return {
        "first_dx": float(totals[0] / count),
        "refined_dx": float(totals[1] / count),
        "equilibrium_fixed_dx": float(totals[2] / count),
        "first_to_refined_dx": float(totals[3] / count),
        "count": float(count),
    }


def _rollout_refined(model, initial, schedule, sdf, refinements: int) -> PoseState:
    states = [initial]
    current = initial
    for command in schedule:
        for _ in range(refinements):
            current = model.forward_step(current, command, sdf)
            assert isinstance(current, PoseState)
        states.append(current)
    return PoseState.stack(states, dim=1)


def _evaluate_rollout(model, loader, config, manifest, device) -> dict[str, Any]:
    accumulators = {str(value): MetricAccumulator() for value in REFINEMENTS}
    by_object: dict[str, dict[str, dict[str, float]]] = {}
    model.eval()
    with torch.no_grad():
        for raw in loader:
            batch = raw.to(device)
            by_object[batch.object_ids[0]] = {}
            for refinements in REFINEMENTS:
                with _autocast(config, device):
                    prediction = _rollout_refined(
                        model,
                        _state_at(batch.states, 0),
                        batch.command_schedule[1:],
                        batch.sdf,
                        refinements,
                    )
                    gaps = torch.stack(
                        [
                            model.query_geometric_gap(
                                _state_at(prediction, step), batch.sdf
                            )
                            for step in range(33)
                        ],
                        dim=1,
                    )
                accumulate_trajectory_metrics(
                    accumulators[str(refinements)],
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
                terminal = _dx(
                    model,
                    _state_at(prediction, 32),
                    _state_at(batch.states, 32),
                    config,
                )
                by_object[batch.object_ids[0]][str(refinements)] = {
                    "terminal_dx": float(terminal.mean().detach())
                }
    return {
        "refinements": {
            key: accumulator.compute() for key, accumulator in accumulators.items()
        },
        "by_object": by_object,
    }


def _save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--fixed-weight", type=float, default=1.0)
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--evaluate-only", type=Path)
    args = parser.parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.fixed_weight < 0:
        raise ValueError("invalid optimization arguments")

    base = ExperimentConfig.load(args.config)
    config = replace(
        base,
        device=args.device,
        paths=replace(base.paths, output_dir=args.output.resolve()),
    )
    if config.model.operator_layers != 1 or config.model.contact_head != "direct":
        raise ValueError("controlled retraction experiment requires direct L=1")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _seed_everything(config.seed)
    output = config.paths.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest = DatasetManifest.load(config.paths.manifest)
    active = ActiveIndex.load(config.paths.active_index)
    model = _build_model(config, manifest, device)

    train_dataset = H5ObjectDataset(
        manifest, split="train", active_index=active, active_only=True
    )
    val_local_dataset = H5ObjectDataset(
        manifest, split="val", active_index=active, active_only=True
    )
    val_rollout_dataset = H5ObjectDataset(manifest, split="val")
    test_rollout_dataset = H5ObjectDataset(manifest, split="test")
    train_loader = make_dataloader(
        train_dataset,
        mode="local",
        objects_per_batch=config.loader.objects_per_batch,
        samples_per_object=config.loader.local_samples_per_object,
        workers=config.loader.workers,
        seed=config.seed,
        shuffle=True,
    )
    val_local_loader = make_dataloader(
        val_local_dataset,
        mode="local",
        objects_per_batch=1,
        samples_per_object=0,
        workers=config.loader.workers,
        seed=config.seed + 1,
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
        history: list[dict[str, Any]] = []
        checkpoint_path: Path
        if args.evaluate_only is not None:
            checkpoint_path = args.evaluate_only.resolve()
            load_checkpoint(checkpoint_path, model=model, map_location=device)
        else:
            if args.initialize is not None:
                load_checkpoint(args.initialize, model=model, map_location=device)
            optimizer, scheduler = _optimizer(
                model, config, max(1, len(train_loader) * args.epochs)
            )
            best = float("inf")
            stale = 0
            best_path = output / "best-equilibrium-retraction.pt"
            for epoch in range(args.epochs):
                set_epoch = getattr(train_loader.batch_sampler, "set_epoch", None)
                if set_epoch is not None:
                    set_epoch(epoch)
                model.train(True)
                records: list[dict[str, float]] = []
                start = perf_counter()
                for raw in train_loader:
                    batch = raw.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with _autocast(config, device):
                        loss, metrics = _iteration(
                            model, batch, config, args.fixed_weight
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
                validation = _evaluate_local(
                    model, val_local_loader, config, device
                )
                # The refinement formulation is selected on its solved local
                # equilibrium, not on the first iterate.
                selection = validation["refined_dx"]
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
                        best_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config.to_dict(),
                        manifest_sha256=manifest.sha256(),
                        gripper_sha256=manifest.gripper_sha256,
                        stage="equilibrium_retraction",
                        epoch=epoch,
                        horizon=1,
                        horizon_epoch=epoch,
                        stale_epochs=0,
                        best_metric=best,
                        extra_state={"fixed_weight": args.fixed_weight},
                    )
                else:
                    stale += 1
                print(
                    f"[RETRACTION] epoch={epoch:03d} "
                    f"train_first={record['train']['first_dx']:.6f} "
                    f"train_refined={record['train']['refined_dx']:.6f} "
                    f"val_first={validation['first_dx']:.6f} "
                    f"val_refined={validation['refined_dx']:.6f} "
                    f"val_fixed={validation['equilibrium_fixed_dx']:.6f} "
                    f"best={best:.6f} stale={stale}",
                    flush=True,
                )
                if stale >= args.patience:
                    break
            checkpoint_path = best_path
            load_checkpoint(checkpoint_path, model=model, map_location=device)

        result = {
            "definition": {
                "learned_object": "fixed-load equilibrium retraction P_phi,u",
                "transport": "P(x_k; u_{k+1}) -> x_{k+1}",
                "refinement": "P(P(x_k;u);u) -> x_{k+1}",
                "fixed_point": "P(x_{k+1};u) -> x_{k+1}",
                "fixed_weight": args.fixed_weight,
                "refinement_counts": list(REFINEMENTS),
            },
            "contract": {
                "config": str(args.config.resolve()),
                "manifest": str(config.paths.manifest),
                "manifest_sha256": manifest.sha256(),
                "active_index": str(config.paths.active_index),
                "active_index_sha256": file_sha256(config.paths.active_index),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "seed": config.seed,
                "device": str(device),
                "model_parameters": sum(p.numel() for p in model.parameters()),
            },
            "history": history,
            "val_local": _evaluate_local(model, val_local_loader, config, device),
            "val": _evaluate_rollout(
                model, val_rollout_loader, config, manifest, device
            ),
            "test": _evaluate_rollout(
                model, test_rollout_loader, config, manifest, device
            ),
        }
        _save_json(output / "results.json", result)
        print(
            json.dumps(
                {
                    "val_local": result["val_local"],
                    "val_H32": {
                        key: value["terminal_dx"]
                        for key, value in result["val"]["refinements"].items()
                    },
                    "test_H32": {
                        key: value["terminal_dx"]
                        for key, value in result["test"]["refinements"].items()
                    },
                },
                indent=2,
            ),
            flush=True,
        )
    finally:
        train_dataset.close()
        val_local_dataset.close()
        val_rollout_dataset.close()
        test_rollout_dataset.close()


if __name__ == "__main__":
    main()
