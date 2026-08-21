#!/usr/bin/env python3
"""Compare recurrent local-resolvent learning with a cumulative path operator.

The path arm learns the fixed-loading sweeping-process solution map
``S_phi(x_0, u_k) = x_k``.  It uses the same SRNO network, SDFs, trajectories,
optimizer steps, and number of supervised states as the local control; only
the graph presented to the learner and the corresponding inference semantics
change.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Iterable, Literal

import numpy as np
import torch
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
from srno.training.checkpoint import load_checkpoint, save_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import (
    _autocast,
    _build_model,
    _local_iteration,
    _optimizer,
    _run_epoch,
    _state_at,
)
from srno.training.metrics import MetricAccumulator, accumulate_trajectory_metrics
from srno.types import PoseState, SDFBatch


Arm = Literal["local_resolvent_control", "cumulative_path_operator"]
ARMS: tuple[Arm, ...] = ("local_resolvent_control", "cumulative_path_operator")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sampled_loader(
    dataset: H5ObjectDataset,
    config: ExperimentConfig,
    *,
    mode: Literal["local", "rollout"],
    samples_per_object: int,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.loader.objects_per_batch,
        shuffle=True,
        num_workers=0,
        collate_fn=ObjectBatchCollator(
            dataset.manifest,
            mode=mode,
            samples_per_object=samples_per_object,
            seed=seed,
            resample=True,
        ),
        generator=torch.Generator().manual_seed(seed),
    )


def _cumulative_batch(
    batch: TrajectoryBatch,
    *,
    step_weight_power: float = 0.0,
    seed: int = 0,
) -> LocalTransitionBatch:
    trajectories, states = batch.states.shape
    if states != 33:
        raise ValueError("cumulative path experiment requires 33-state trajectories")
    steps = states - 1
    initial = PoseState(
        batch.states.rotation[:, :1].expand(-1, steps, -1, -1).reshape(-1, 3, 3),
        batch.states.position[:, :1].expand(-1, steps, -1).reshape(-1, 3),
        batch.states.joint_position[:, :1].expand(-1, steps, -1).reshape(-1, 6),
    )
    if step_weight_power == 0.0:
        selected_step = torch.arange(
            steps, device=batch.command_schedule.device
        ).repeat(trajectories)
    else:
        weights = np.arange(1, steps + 1, dtype=np.float64) ** step_weight_power
        weights /= weights.sum()
        generator = np.random.default_rng(seed)
        selected_step = torch.from_numpy(
            generator.choice(steps, size=trajectories * steps, replace=True, p=weights)
        ).to(device=batch.command_schedule.device, dtype=torch.long)
    trajectory_row = torch.arange(
        trajectories, device=batch.command_schedule.device
    ).repeat_interleave(steps)
    target = PoseState(
        batch.states.rotation[trajectory_row, selected_step + 1],
        batch.states.position[trajectory_row, selected_step + 1],
        batch.states.joint_position[trajectory_row, selected_step + 1],
    )
    commands = batch.command_schedule.index_select(0, selected_step + 1)
    step_index = selected_step
    return LocalTransitionBatch(
        SDFBatch(
            batch.sdf.values,
            batch.sdf.origin,
            batch.sdf.voxel_size,
            batch.sdf.sample_to_object.repeat_interleave(steps),
            batch.sdf.outside_value,
        ),
        initial,
        target,
        commands,
        batch.actual_aperture[trajectory_row, selected_step + 1],
        batch.object_ids,
        batch.trajectory_index.repeat_interleave(steps),
        step_index,
        None,
    )


def _cumulative_prediction(
    model: torch.nn.Module,
    batch: TrajectoryBatch,
) -> PoseState:
    flat = _cumulative_batch(batch)
    prediction = model.forward_step(flat.current, flat.next_command, flat.sdf)
    trajectories = batch.states.shape[0]
    rotation = prediction.rotation.reshape(trajectories, 32, 3, 3)
    position = prediction.position.reshape(trajectories, 32, 3)
    joint = prediction.joint_position.reshape(trajectories, 32, 6)
    return PoseState(
        torch.cat((batch.states.rotation[:, :1], rotation), dim=1),
        torch.cat((batch.states.position[:, :1], position), dim=1),
        torch.cat((batch.states.joint_position[:, :1], joint), dim=1),
    )


def _trajectory_metrics(
    model: torch.nn.Module,
    loader: Iterable[TrajectoryBatch],
    manifest: DatasetManifest,
    config: ExperimentConfig,
    device: torch.device,
    *,
    semantics: Literal["recursive", "cumulative"],
) -> dict[str, float]:
    model.eval()
    accumulator = MetricAccumulator()
    with torch.no_grad():
        for raw in loader:
            batch = raw.to(device)
            with _autocast(config, device):
                if semantics == "recursive":
                    prediction = model.rollout(
                        _state_at(batch.states, 0), batch.command_schedule[1:], batch.sdf
                    )
                else:
                    prediction = _cumulative_prediction(model, batch)
                gaps = torch.stack([
                    model.query_geometric_gap(_state_at(prediction, step), batch.sdf)
                    for step in range(33)
                ], dim=1)
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


def _train_arm(
    arm: Arm,
    config: ExperimentConfig,
    manifest: DatasetManifest,
    train_active: H5ObjectDataset,
    train_trajectories: H5ObjectDataset,
    val_loader: Iterable[TrajectoryBatch],
    initial: Path,
    output: Path,
    device: torch.device,
    *,
    epochs: int,
    patience: int,
    samples_per_object: int,
    trajectories_per_object: int,
    path_step_weight_power: float,
) -> dict[str, Any]:
    _seed_everything(config.seed)
    model = _build_model(config, manifest, device)
    saved = load_checkpoint(initial, model=model, map_location=device)
    if saved["manifest_sha256"] != manifest.sha256():
        raise ValueError("initial checkpoint manifest mismatch")
    batches_per_epoch = math.ceil(len(train_active.object_ids) / config.loader.objects_per_batch)
    optimizer, scheduler = _optimizer(model, config, batches_per_epoch * epochs)
    arm_output = output / arm
    arm_output.mkdir(parents=True, exist_ok=True)
    metrics_path = arm_output / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    checkpoint = arm_output / "best-solution-operator.pt"
    history: list[dict[str, Any]] = []
    best = float("inf")
    stale = 0
    semantics: Literal["recursive", "cumulative"] = (
        "recursive" if arm == "local_resolvent_control" else "cumulative"
    )
    for epoch in range(epochs):
        started = perf_counter()
        if arm == "local_resolvent_control":
            loader = _sampled_loader(
                train_active, config, mode="local",
                samples_per_object=samples_per_object,
                seed=config.seed + epoch,
            )
        else:
            loader = _sampled_loader(
                train_trajectories, config, mode="rollout",
                samples_per_object=trajectories_per_object,
                seed=config.seed + epoch,
            )
        model.train(True)
        accumulator: dict[str, list[float]] = {}
        for batch_index, raw in enumerate(loader):
            local = (
                raw
                if isinstance(raw, LocalTransitionBatch)
                else _cumulative_batch(
                    raw,
                    step_weight_power=path_step_weight_power,
                    seed=config.seed + epoch * 10_000 + batch_index,
                )
            )
            batch = local.to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(config, device):
                loss, values = _local_iteration(model, batch, config)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.optimizer.gradient_clip
            )
            optimizer.step()
            scheduler.step()
            values = dict(values)
            values["gradient_norm"] = float(gradient.detach())
            for name, value in values.items():
                accumulator.setdefault(name, []).append(float(value))
        train_metrics = {
            name: float(np.mean(values)) for name, values in accumulator.items()
        }
        val = _trajectory_metrics(
            model, val_loader, manifest, config, device, semantics=semantics
        )
        selection = val["terminal_dx"]
        record = {
            "epoch": epoch,
            "seconds": perf_counter() - started,
            "train": train_metrics,
            "val": val,
            "selection": selection,
        }
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if selection < best:
            best = selection
            stale = 0
            save_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config.to_dict(),
                manifest_sha256=manifest.sha256(),
                gripper_sha256=manifest.gripper_sha256,
                stage=arm,
                epoch=epoch,
                horizon=32,
                horizon_epoch=epoch,
                stale_epochs=0,
                best_metric=best,
                extra_state={
                    "operator_semantics": semantics,
                    "supervised_states_per_object": samples_per_object,
                    "path_step_weight_power": path_step_weight_power,
                },
            )
        else:
            stale += 1
        print(
            f"[PATH train] arm={arm} epoch={epoch:03d} "
            f"train_dX={train_metrics['dx']:.6f} val_H32={selection:.6f} "
            f"best={best:.6f} stale={stale}",
            flush=True,
        )
        if stale >= patience:
            break
    load_checkpoint(checkpoint, model=model, map_location=device)
    return {"model": model, "checkpoint": checkpoint, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--samples-per-object", type=int, default=256)
    parser.add_argument("--trajectories-per-object", type=int, default=8)
    parser.add_argument("--path-step-weight-power", type=float, default=0.0)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.samples_per_object, args.trajectories_per_object) <= 0:
        parser.error("training counts must be positive")
    if args.samples_per_object != 32 * args.trajectories_per_object:
        parser.error("paired arms require samples-per-object = 32 * trajectories-per-object")
    if args.path_step_weight_power < 0.0:
        parser.error("--path-step-weight-power must be non-negative")

    base = ExperimentConfig.load(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = replace(
        base,
        device=args.device,
        paths=replace(base.paths, output_dir=output),
        loader=replace(base.loader, workers=0),
    )
    device = torch.device(config.device)
    manifest = DatasetManifest.load(config.paths.manifest)
    active = ActiveIndex.load(config.paths.active_index)
    train_active = H5ObjectDataset(
        manifest, split="train", active_index=active, active_only=True
    )
    train_trajectories = H5ObjectDataset(manifest, split="train")
    val_active = H5ObjectDataset(
        manifest, split="val", active_index=active, active_only=True
    )
    val_dataset = H5ObjectDataset(manifest, split="val")
    test_dataset = H5ObjectDataset(manifest, split="test")
    try:
        val_loader = make_dataloader(
            val_dataset, mode="rollout", objects_per_batch=1,
            samples_per_object=0, workers=0, seed=config.seed + 1, shuffle=False,
        )
        test_loader = make_dataloader(
            test_dataset, mode="rollout", objects_per_batch=1,
            samples_per_object=0, workers=0, seed=config.seed + 2, shuffle=False,
        )
        val_local_loader = make_dataloader(
            val_active, mode="local", objects_per_batch=1,
            samples_per_object=0, workers=0, seed=config.seed + 1, shuffle=False,
        )
        baseline_model = _build_model(config, manifest, device)
        load_checkpoint(args.initial, model=baseline_model, map_location=device)
        baseline = {
            semantics: _trajectory_metrics(
                baseline_model, val_loader, manifest, config, device,
                semantics=semantics,
            )
            for semantics in ("recursive", "cumulative")
        }
        baseline["test_recursive"] = _trajectory_metrics(
            baseline_model, test_loader, manifest, config, device, semantics="recursive"
        )
        baseline["test_cumulative"] = _trajectory_metrics(
            baseline_model, test_loader, manifest, config, device, semantics="cumulative"
        )
        del baseline_model

        runs: dict[str, Any] = {}
        for arm in ARMS:
            trained = _train_arm(
                arm, config, manifest, train_active, train_trajectories,
                val_loader, args.initial.resolve(), output, device,
                epochs=args.epochs, patience=args.patience,
                samples_per_object=args.samples_per_object,
                trajectories_per_object=args.trajectories_per_object,
                path_step_weight_power=args.path_step_weight_power,
            )
            model = trained.pop("model")
            checkpoint = trained["checkpoint"]
            trained["checkpoint"] = str(checkpoint)
            trained["checkpoint_sha256"] = file_sha256(checkpoint)
            trained["val_recursive"] = _trajectory_metrics(
                model, val_loader, manifest, config, device, semantics="recursive"
            )
            trained["val_cumulative"] = _trajectory_metrics(
                model, val_loader, manifest, config, device, semantics="cumulative"
            )
            trained["test_recursive"] = _trajectory_metrics(
                model, test_loader, manifest, config, device, semantics="recursive"
            )
            trained["test_cumulative"] = _trajectory_metrics(
                model, test_loader, manifest, config, device, semantics="cumulative"
            )
            trained["local_val"] = _run_epoch(
                model, val_local_loader, config, device,
                mode="local", horizon=1, optimizer=None, scheduler=None,
            )
            runs[arm] = trained
            del model
        control = runs["local_resolvent_control"]["test_recursive"]["terminal_dx"]
        treatment = runs["cumulative_path_operator"]["test_cumulative"]["terminal_dx"]
        result = {
            "definition": {
                "control": "learn R_phi(x_k,u_{k+1}) and compose it recursively",
                "treatment": "learn fixed-loading solution path S_phi(x_0,u_k) directly",
                "same_architecture": True,
                "same_initial_checkpoint": True,
                "same_optimizer_steps": True,
                "same_supervised_states_per_object_epoch": True,
                "canonical_val_test_changed": False,
            },
            "contract": {
                "config": str(args.config.resolve()),
                "manifest_sha256": manifest.sha256(),
                "initial": str(args.initial.resolve()),
                "initial_sha256": file_sha256(args.initial),
                "epochs": args.epochs,
                "patience": args.patience,
                "samples_per_object": args.samples_per_object,
                "trajectories_per_object": args.trajectories_per_object,
                "path_step_weight_power": args.path_step_weight_power,
                "device": str(device),
            },
            "baseline": baseline,
            "runs": runs,
            "paired_test": {
                "recursive_control_terminal_dx": control,
                "cumulative_path_terminal_dx": treatment,
                "relative_change": float((treatment - control) / control),
            },
        }
        _write_json(output / "results.json", result)
        print(json.dumps(result["paired_test"], indent=2, sort_keys=True), flush=True)
    finally:
        train_active.close()
        train_trajectories.close()
        val_active.close()
        val_dataset.close()
        test_dataset.close()


if __name__ == "__main__":
    main()
