#!/usr/bin/env python3
"""Train the SRNO cell as a finite-load evolution family.

The production objective only observes adjacent maps

    x_k -> R(command_{k+1}, x_k).

This experiment keeps the dataset, geometry encoder, state representation, and
L=1 direct SRNO cell fixed, but changes the learned mathematical object.  It
supervises finite-load propagators U(b, a)x and can optionally impose the
evolution-family identity U(c, b)U(b, a)=U(c, a).  Evaluation composes the same
cell over several partitions of the unchanged 32-step loading path.

This is intentionally an experimental script rather than a new production
training stage: a failed mathematical hypothesis should not enlarge the public
model/config surface.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
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

from srno.data.dataset import H5ObjectDataset, TrajectoryBatch, make_dataloader
from srno.data.index import file_sha256
from srno.data.schema import DatasetManifest
from srno.losses import combined_loss, state_error
from srno.training.checkpoint import load_checkpoint, save_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.training.metrics import MetricAccumulator, accumulate_trajectory_metrics
from srno.types import PoseState


Policy = Literal["terminal", "interval", "cocycle"]
PARTITIONS = (1, 2, 4, 8, 16, 32)


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
        fused=device_is_cuda(model),
    )
    warmup_steps = int(total_steps * config.optimizer.warmup_fraction)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-3)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return optimizer, scheduler


def device_is_cuda(model) -> bool:
    return next(model.parameters()).is_cuda


def _physical_loss(
    model,
    prediction: PoseState,
    target: PoseState,
    sdf,
    config: ExperimentConfig,
):
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


def _dx(model, prediction: PoseState, target: PoseState, config: ExperimentConfig) -> Tensor:
    return state_error(
        prediction,
        target,
        length_scale=model.length_scale,
        joint_scale=model.joint_travel_range,
        lambda_rotation=config.loss.lambda_rotation,
        lambda_joints=config.loss.lambda_joints,
    )[0].sqrt()


def _draw_interval(policy: Policy, rng: np.random.Generator) -> tuple[int, int, int]:
    if policy == "terminal":
        return 0, 16, 32
    if policy == "interval":
        start = int(rng.integers(0, 32))
        end = int(rng.integers(start + 1, 33))
        return start, (start + end) // 2, end
    start = int(rng.integers(0, 31))
    middle = int(rng.integers(start + 1, 32))
    end = int(rng.integers(middle + 1, 33))
    return start, middle, end


def _train_iteration(
    model,
    batch: TrajectoryBatch,
    config: ExperimentConfig,
    *,
    policy: Policy,
    rng: np.random.Generator,
    cocycle_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    start, middle, end = _draw_interval(policy, rng)
    current = _state_at(batch.states, start)
    target_end = _state_at(batch.states, end)
    direct = model.forward_step(current, batch.command_schedule[end], batch.sdf)
    assert isinstance(direct, PoseState)
    direct_terms = _physical_loss(model, direct, target_end, batch.sdf, config)

    if policy != "cocycle":
        return direct_terms.total, {
            "loss": float(direct_terms.total.detach()),
            "direct_dx": float(_dx(model, direct, target_end, config).mean().detach()),
            "direct_flow": float(direct_terms.flow.detach()),
            "direct_feasibility": float(direct_terms.feasibility.detach()),
            "start": float(start),
            "middle": float(middle),
            "end": float(end),
        }

    target_middle = _state_at(batch.states, middle)
    predicted_middle = model.forward_step(
        current, batch.command_schedule[middle], batch.sdf
    )
    assert isinstance(predicted_middle, PoseState)
    composed = model.forward_step(
        predicted_middle, batch.command_schedule[end], batch.sdf
    )
    assert isinstance(composed, PoseState)
    middle_terms = _physical_loss(
        model, predicted_middle, target_middle, batch.sdf, config
    )
    composed_terms = _physical_loss(model, composed, target_end, batch.sdf, config)
    cocycle_squared = state_error(
        composed,
        direct,
        length_scale=model.length_scale,
        joint_scale=model.joint_travel_range,
        lambda_rotation=config.loss.lambda_rotation,
        lambda_joints=config.loss.lambda_joints,
    )[0].mean()
    physical = (middle_terms.total + direct_terms.total + composed_terms.total) / 3.0
    loss = physical + cocycle_weight * cocycle_squared
    return loss, {
        "loss": float(loss.detach()),
        "physical_loss": float(physical.detach()),
        "cocycle_sq": float(cocycle_squared.detach()),
        "cocycle_dx": float(cocycle_squared.detach().sqrt()),
        "direct_dx": float(_dx(model, direct, target_end, config).mean().detach()),
        "composed_dx": float(_dx(model, composed, target_end, config).mean().detach()),
        "middle_dx": float(
            _dx(model, predicted_middle, target_middle, config).mean().detach()
        ),
        "start": float(start),
        "middle": float(middle),
        "end": float(end),
    }


def _mean(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in records[0]
    }


def _direct_terminal_validation(
    model,
    loader,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    values: list[dict[str, float]] = []
    with torch.no_grad():
        for raw in loader:
            batch = raw.to(device)
            initial = _state_at(batch.states, 0)
            target = _state_at(batch.states, 32)
            with _autocast(config, device):
                prediction = model.forward_step(
                    initial, batch.command_schedule[32], batch.sdf
                )
                assert isinstance(prediction, PoseState)
                terms = _physical_loss(model, prediction, target, batch.sdf, config)
            values.append(
                {
                    "loss": float(terms.total.detach()),
                    "dx": float(_dx(model, prediction, target, config).mean().detach()),
                    "flow": float(terms.flow.detach()),
                    "feasibility": float(terms.feasibility.detach()),
                }
            )
    return _mean(values)


def _compose_partition(model, batch: TrajectoryBatch, segments: int) -> PoseState:
    if 32 % segments:
        raise ValueError("partition count must divide 32")
    current = _state_at(batch.states, 0)
    stride = 32 // segments
    for end in range(stride, 33, stride):
        current = model.forward_step(current, batch.command_schedule[end], batch.sdf)
        assert isinstance(current, PoseState)
    return current


def _terminal_components(model, prediction: PoseState, target: PoseState) -> dict[str, Tensor]:
    translation_m = torch.linalg.vector_norm(
        prediction.position - target.position, dim=-1
    )
    _, translation_sq, rotation_sq, joint_sq = state_error(
        prediction,
        target,
        length_scale=model.length_scale,
        joint_scale=model.joint_travel_range,
    )
    return {
        "terminal_dx": (translation_sq + rotation_sq + joint_sq).sqrt(),
        "terminal_translation_m": translation_m,
        "terminal_translation_over_length": translation_sq.sqrt(),
        "terminal_rotation_rad": rotation_sq.sqrt(),
        "terminal_joint_rmse_over_travel": joint_sq.sqrt(),
    }


def _evaluate_evolution(
    model,
    loader,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    partition_accumulators = {
        str(segments): MetricAccumulator() for segments in PARTITIONS
    }
    path_accumulator = MetricAccumulator()
    cocycle_accumulator = MetricAccumulator()
    object_partitions: dict[str, dict[str, dict[str, float]]] = {}
    object_direct_paths: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for raw in loader:
            batch = raw.to(device)
            initial = _state_at(batch.states, 0)
            direct_states = [initial]
            with _autocast(config, device):
                for step in range(1, 33):
                    direct = model.forward_step(
                        initial, batch.command_schedule[step], batch.sdf
                    )
                    assert isinstance(direct, PoseState)
                    direct_states.append(direct)
                direct_path = PoseState.stack(direct_states, dim=1)
                gaps = torch.stack(
                    [
                        model.query_geometric_gap(_state_at(direct_path, step), batch.sdf)
                        for step in range(33)
                    ],
                    dim=1,
                )
            accumulate_trajectory_metrics(
                path_accumulator,
                direct_path,
                batch.states,
                batch.command_schedule,
                model.aperture_from_joints(direct_path.joint_position),
                batch.actual_aperture,
                gaps,
                length_scale=model.length_scale,
                joint_scale=model.joint_travel_range,
                lag_threshold=DatasetManifest.load(config.paths.manifest).delta_gate_m,
            )
            object_direct_paths[batch.object_ids[0]] = {
                key: float(value.mean().detach())
                for key, value in _terminal_components(
                    model, _state_at(direct_path, 32), _state_at(batch.states, 32)
                ).items()
            }
            object_partitions[batch.object_ids[0]] = {}
            for segments in PARTITIONS:
                with _autocast(config, device):
                    prediction = _compose_partition(model, batch, segments)
                target = _state_at(batch.states, 32)
                components = _terminal_components(model, prediction, target)
                for name, values in components.items():
                    partition_accumulators[str(segments)].add(name, values)
                object_partitions[batch.object_ids[0]][str(segments)] = {
                    name: float(values.mean().detach())
                    for name, values in components.items()
                }

            # A fixed, interpretable cocycle audit at load levels 0 < 16 < 32.
            direct_32 = _state_at(direct_path, 32)
            direct_16 = _state_at(direct_path, 16)
            with _autocast(config, device):
                composed_16_32 = model.forward_step(
                    direct_16, batch.command_schedule[32], batch.sdf
                )
                assert isinstance(composed_16_32, PoseState)
            cocycle = _dx(model, composed_16_32, direct_32, config)
            physical = _dx(
                model, composed_16_32, _state_at(batch.states, 32), config
            )
            cocycle_accumulator.add("u_32_16_u_16_0_vs_u_32_0_dx", cocycle)
            cocycle_accumulator.add("two_segment_terminal_dx", physical)

    return {
        "direct_path": path_accumulator.compute(),
        "partitions": {
            key: accumulator.compute()
            for key, accumulator in partition_accumulators.items()
        },
        "cocycle": cocycle_accumulator.compute(),
        "by_object_direct": object_direct_paths,
        "by_object_partitions": object_partitions,
    }


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy", choices=("terminal", "interval", "cocycle"), default="terminal"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--cocycle-weight", type=float, default=1.0)
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--evaluate-only", type=Path)
    args = parser.parse_args()
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("epochs and patience must be positive")
    if args.cocycle_weight < 0:
        raise ValueError("cocycle weight must be non-negative")

    base = ExperimentConfig.load(args.config)
    config = replace(
        base,
        device=args.device,
        paths=replace(base.paths, output_dir=args.output.resolve()),
    )
    if config.model.operator_layers != 1 or config.model.contact_head != "direct":
        raise ValueError("controlled evolution-family experiment requires direct L=1")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _seed_everything(config.seed)
    output = config.paths.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest = DatasetManifest.load(config.paths.manifest)
    model = _build_model(config, manifest, device)

    train_dataset = H5ObjectDataset(manifest, split="train")
    val_dataset = H5ObjectDataset(manifest, split="val")
    test_dataset = H5ObjectDataset(manifest, split="test")
    train_loader = make_dataloader(
        train_dataset,
        mode="rollout",
        objects_per_batch=config.loader.objects_per_batch,
        samples_per_object=config.loader.rollout_trajectories_per_object,
        workers=config.loader.workers,
        seed=config.seed,
        shuffle=True,
    )
    val_loader = make_dataloader(
        val_dataset,
        mode="rollout",
        objects_per_batch=1,
        samples_per_object=0,
        workers=config.loader.workers,
        seed=config.seed + 1,
        shuffle=False,
    )
    test_loader = make_dataloader(
        test_dataset,
        mode="rollout",
        objects_per_batch=1,
        samples_per_object=0,
        workers=config.loader.workers,
        seed=config.seed + 2,
        shuffle=False,
    )
    try:
        checkpoint_path: Path
        history: list[dict[str, Any]] = []
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
            best_path = output / "best-evolution-family.pt"
            rng = np.random.default_rng(config.seed + 7919)
            for epoch in range(args.epochs):
                set_epoch = getattr(train_loader.batch_sampler, "set_epoch", None)
                if set_epoch is not None:
                    set_epoch(epoch)
                model.train(True)
                train_values: list[dict[str, float]] = []
                start_time = perf_counter()
                for raw in train_loader:
                    batch = raw.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with _autocast(config, device):
                        loss, values = _train_iteration(
                            model,
                            batch,
                            config,
                            policy=args.policy,
                            rng=rng,
                            cocycle_weight=args.cocycle_weight,
                        )
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optimizer.gradient_clip
                    )
                    optimizer.step()
                    scheduler.step()
                    values["gradient_norm"] = float(gradient_norm.detach())
                    values["learning_rate"] = float(optimizer.param_groups[0]["lr"])
                    train_values.append(values)
                validation = _direct_terminal_validation(
                    model, val_loader, config, device
                )
                train_summary = _mean(train_values)
                record = {
                    "epoch": epoch,
                    "seconds": perf_counter() - start_time,
                    "train": train_summary,
                    "val_direct_terminal": validation,
                }
                history.append(record)
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                improved = validation["dx"] < best
                if improved:
                    best = validation["dx"]
                    stale = 0
                    save_checkpoint(
                        best_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config.to_dict(),
                        manifest_sha256=manifest.sha256(),
                        gripper_sha256=manifest.gripper_sha256,
                        stage="evolution_family",
                        epoch=epoch,
                        horizon=32,
                        horizon_epoch=epoch,
                        stale_epochs=stale,
                        best_metric=best,
                        extra_state={
                            "policy": args.policy,
                            "cocycle_weight": args.cocycle_weight,
                        },
                    )
                else:
                    stale += 1
                print(
                    f"[EVOLUTION] epoch={epoch:03d} policy={args.policy} "
                    f"train_dx={train_summary['direct_dx']:.6f} "
                    f"val_direct_H32={validation['dx']:.6f} "
                    f"best={best:.6f} stale={stale}",
                    flush=True,
                )
                if stale >= args.patience:
                    break
            checkpoint_path = best_path
            load_checkpoint(checkpoint_path, model=model, map_location=device)

        evaluation = {
            "definition": {
                "learned_object": "finite-load propagator U_phi(b,a)",
                "identity": "U_phi(a,a)=I (implemented exactly by zero applications)",
                "cocycle": "U_phi(c,b) o U_phi(b,a) = U_phi(c,a)",
                "policy": args.policy,
                "cocycle_weight": args.cocycle_weight,
                "partitions": list(PARTITIONS),
            },
            "contract": {
                "config": str(args.config.resolve()),
                "manifest": str(config.paths.manifest),
                "manifest_sha256": manifest.sha256(),
                "gripper_sha256": manifest.gripper_sha256,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "seed": config.seed,
                "device": str(device),
                "model_parameters": sum(p.numel() for p in model.parameters()),
            },
            "history": history,
            "val": _evaluate_evolution(model, val_loader, config, device),
            "test": _evaluate_evolution(model, test_loader, config, device),
        }
        _save_json(output / "results.json", evaluation)
        print(json.dumps({
            split: {
                "direct_H32": evaluation[split]["direct_path"]["terminal_dx"],
                "partition_H32": {
                    key: value["terminal_dx"]
                    for key, value in evaluation[split]["partitions"].items()
                },
                "cocycle_dx": evaluation[split]["cocycle"][
                    "u_32_16_u_16_0_vs_u_32_0_dx"
                ],
            }
            for split in ("val", "test")
        }, indent=2), flush=True)
    finally:
        train_dataset.close()
        val_dataset.close()
        test_dataset.close()


if __name__ == "__main__":
    main()
