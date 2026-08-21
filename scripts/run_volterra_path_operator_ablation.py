#!/usr/bin/env python3
"""Controlled direct-path versus causal Volterra SRNO formulation.

The same L=1 contact cell is evaluated against the initial state at every load
level.  In the direct arm its outputs are endpoint states.  In the Volterra arm
the cell outputs generalized increment atoms; causal cumulative quadrature
constructs the entire BV trajectory.  No recurrent predicted state is fed back
into the contact cell.
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
from torch.utils.data import DataLoader

from srno.data.dataset import H5ObjectDataset, ObjectBatchCollator, TrajectoryBatch, make_dataloader
from srno.data.index import file_sha256
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import so3_exp, so3_log_vector
from srno.losses import state_error
from srno.training.checkpoint import load_checkpoint, save_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.training.metrics import MetricAccumulator, accumulate_trajectory_metrics
from srno.types import PoseState, SDFBatch


Formulation = Literal["direct_path", "volterra"]
FORMULATIONS: tuple[Formulation, ...] = ("direct_path", "volterra")


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


def _autocast(config, device):
    enabled = config.training.use_bfloat16 and device.type == "cuda"
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=enabled
    )


def _optimizer(model, config, total_steps):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        fused=next(model.parameters()).is_cuda,
    )
    warmup = int(total_steps * config.optimizer.warmup_fraction)

    def multiplier(step):
        if warmup and step < warmup:
            return max((step + 1) / warmup, 1e-3)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return optimizer, scheduler


def _reference_candidates(model, batch: TrajectoryBatch):
    trajectories = batch.states.position.shape[0]
    steps = 32
    initial = _state_at(batch.states, 0)
    flat_initial = PoseState(
        initial.rotation[:, None].expand(-1, steps, -1, -1).reshape(-1, 3, 3),
        initial.position[:, None].expand(-1, steps, -1).reshape(-1, 3),
        initial.joint_position[:, None].expand(-1, steps, -1).reshape(-1, 6),
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
    candidates = model.forward_step(flat_initial, commands, sdf)
    assert isinstance(candidates, PoseState)
    return initial, flat_initial, commands, sdf, candidates


def _predict_path(model, batch, formulation: Formulation) -> PoseState:
    initial, flat_initial, commands, _, candidates = _reference_candidates(model, batch)
    trajectories = initial.position.shape[0]
    if formulation == "direct_path":
        tail = PoseState(
            candidates.rotation.reshape(trajectories, 32, 3, 3),
            candidates.position.reshape(trajectories, 32, 3),
            candidates.joint_position.reshape(trajectories, 32, 6),
        )
    else:
        translation_atoms = (
            candidates.position - flat_initial.position
        ).reshape(trajectories, 32, 3)
        rotation_atoms = so3_log_vector(
            candidates.rotation @ flat_initial.rotation.transpose(-1, -2)
        ).reshape(trajectories, 32, 3)
        free_joint = model.free_joint_configuration(commands)
        joint_atoms = (
            candidates.joint_position - free_joint
        ).reshape(trajectories, 32, 6)
        positions = initial.position[:, None] + translation_atoms.cumsum(dim=1)
        joints = free_joint.reshape(trajectories, 32, 6) + joint_atoms.cumsum(dim=1)
        rotations = []
        current_rotation = initial.rotation
        for step in range(32):
            current_rotation = so3_exp(rotation_atoms[:, step]) @ current_rotation
            rotations.append(current_rotation)
        tail = PoseState(torch.stack(rotations, dim=1), positions, joints)
    return PoseState(
        torch.cat((initial.rotation[:, None], tail.rotation), dim=1),
        torch.cat((initial.position[:, None], tail.position), dim=1),
        torch.cat((initial.joint_position[:, None], tail.joint_position), dim=1),
    )


def _trajectory_loss(model, prediction, batch, config):
    trajectories = prediction.position.shape[0]
    flat_prediction = PoseState(
        prediction.rotation[:, 1:].reshape(-1, 3, 3),
        prediction.position[:, 1:].reshape(-1, 3),
        prediction.joint_position[:, 1:].reshape(-1, 6),
    )
    flat_target = PoseState(
        batch.states.rotation[:, 1:].reshape(-1, 3, 3),
        batch.states.position[:, 1:].reshape(-1, 3),
        batch.states.joint_position[:, 1:].reshape(-1, 6),
    )
    sdf = SDFBatch(
        batch.sdf.values,
        batch.sdf.origin,
        batch.sdf.voxel_size,
        batch.sdf.sample_to_object.repeat_interleave(32),
        batch.sdf.outside_value,
    )
    total, translation, rotation, joints = state_error(
        flat_prediction,
        flat_target,
        length_scale=model.length_scale,
        joint_scale=model.joint_travel_range,
        lambda_rotation=config.loss.lambda_rotation,
        lambda_joints=config.loss.lambda_joints,
    )
    gap = model.query_geometric_gap(flat_prediction, sdf)
    feasibility = torch.relu(
        (config.loss.admissible_gap_m - gap) / model.sdf_scale
    ).square().mean()
    flow = total.mean()
    loss = flow + config.loss.lambda_feasibility * feasibility
    terminal = total.reshape(trajectories, 32)[:, -1].sqrt().mean()
    return loss, {
        "loss": float(loss.detach()),
        "flow": float(flow.detach()),
        "feasibility": float(feasibility.detach()),
        "path_dx": float(total.sqrt().mean().detach()),
        "terminal_dx": float(terminal.detach()),
        "translation": float(translation.mean().detach()),
        "rotation": float(rotation.mean().detach()),
        "joints": float(joints.mean().detach()),
    }


def _iteration(model, batch, formulation, config):
    prediction = _predict_path(model, batch, formulation)
    loss, metrics = _trajectory_loss(model, prediction, batch, config)
    return loss, metrics, prediction


def _mean(records):
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in records[0]
    }


def _evaluate(model, loader, formulation, manifest, config, device):
    accumulator = MetricAccumulator()
    loss_records = []
    model.eval()
    with torch.no_grad():
        for raw in loader:
            batch = raw.to(device)
            with _autocast(config, device):
                _, metrics, prediction = _iteration(
                    model, batch, formulation, config
                )
                gaps = torch.stack(
                    [
                        model.query_geometric_gap(_state_at(prediction, step), batch.sdf)
                        for step in range(33)
                    ],
                    dim=1,
                )
            loss_records.append(metrics)
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
    return {"objective": _mean(loss_records), "trajectory": accumulator.compute()}


def _stochastic_loader(dataset, config, seed):
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
        generator=torch.Generator().manual_seed(seed),
    )


def _train_arm(formulation, config, manifest, train_loader, val_loader, device, initialize, epochs, patience):
    output = config.paths.output_dir / formulation
    output.mkdir(parents=True, exist_ok=True)
    model = _build_model(config, manifest, device)
    if initialize is not None:
        load_checkpoint(initialize, model=model, map_location=device)
    optimizer, scheduler = _optimizer(model, config, len(train_loader) * epochs)
    best = float("inf")
    stale = 0
    history = []
    checkpoint = output / "best-volterra-path.pt"
    for epoch in range(epochs):
        model.train(True)
        records = []
        start = perf_counter()
        for raw in train_loader:
            batch = raw.to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(config, device):
                loss, metrics, _ = _iteration(model, batch, formulation, config)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.optimizer.gradient_clip
            )
            optimizer.step()
            scheduler.step()
            metrics["gradient_norm"] = float(gradient.detach())
            metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
            records.append(metrics)
        validation = _evaluate(
            model, val_loader, formulation, manifest, config, device
        )["objective"]
        selection = validation["path_dx"]
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
                checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config.to_dict(),
                manifest_sha256=manifest.sha256(),
                gripper_sha256=manifest.gripper_sha256,
                stage=f"path_{formulation}",
                epoch=epoch,
                horizon=32,
                horizon_epoch=epoch,
                stale_epochs=0,
                best_metric=best,
                extra_state={"formulation": formulation},
            )
        else:
            stale += 1
        print(
            f"[VOLTERRA] arm={formulation} epoch={epoch:03d} "
            f"val_path={validation['path_dx']:.6f} "
            f"val_H32={validation['terminal_dx']:.6f} "
            f"best={best:.6f} stale={stale}",
            flush=True,
        )
        if stale >= patience:
            break
    load_checkpoint(checkpoint, model=model, map_location=device)
    return model, checkpoint, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()
    config = replace(
        ExperimentConfig.load(args.config),
        device=args.device,
        paths=replace(
            ExperimentConfig.load(args.config).paths,
            output_dir=args.output.resolve(),
        ),
        loader=replace(ExperimentConfig.load(args.config).loader, workers=0),
    )
    if config.model.operator_layers != 1 or config.model.contact_head != "direct":
        raise ValueError("controlled path ablation requires direct L=1")
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("epochs and patience must be positive")
    device = torch.device(config.device)
    manifest = DatasetManifest.load(config.paths.manifest)
    output = config.paths.output_dir
    output.mkdir(parents=True, exist_ok=True)
    train_dataset = H5ObjectDataset(manifest, split="train")
    val_dataset = H5ObjectDataset(manifest, split="val")
    test_dataset = H5ObjectDataset(manifest, split="test")
    val_loader = make_dataloader(
        val_dataset, mode="rollout", objects_per_batch=1, samples_per_object=0,
        workers=0, seed=config.seed + 1, shuffle=False,
    )
    test_loader = make_dataloader(
        test_dataset, mode="rollout", objects_per_batch=1, samples_per_object=0,
        workers=0, seed=config.seed + 2, shuffle=False,
    )
    try:
        runs = {}
        for formulation in FORMULATIONS:
            _seed_everything(config.seed)
            train_loader = _stochastic_loader(train_dataset, config, config.seed)
            model, checkpoint, history = _train_arm(
                formulation, config, manifest, train_loader, val_loader, device,
                (
                    None
                    if args.initialize is None
                    else args.initialize.resolve()
                ),
                args.epochs,
                args.patience,
            )
            runs[formulation] = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "history": history,
                "val": _evaluate(model, val_loader, formulation, manifest, config, device),
                "test": _evaluate(model, test_loader, formulation, manifest, config, device),
            }
            del model
        result = {
            "definition": {
                "direct_path": "cell(x0,u_k) interpreted as x_k",
                "volterra": "cell(x0,u_k) interpreted as increment atom; x_k=x0+sum_{l<=k} atom_l",
                "predicted_state_feedback": False,
                "architecture_parameters_changed": False,
                "physics_data_changed": False,
            },
            "contract": {
                "config": str(args.config.resolve()),
                "manifest_sha256": manifest.sha256(),
                "initial_checkpoint": (
                    None if args.initialize is None else str(args.initialize.resolve())
                ),
                "initial_checkpoint_sha256": (
                    None if args.initialize is None else file_sha256(args.initialize)
                ),
                "seed": config.seed,
                "device": str(device),
            },
            "runs": runs,
        }
        (output / "results.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({
            formulation: {
                "test_path_dx": runs[formulation]["test"]["objective"]["path_dx"],
                "test_H32": runs[formulation]["test"]["trajectory"]["terminal_dx"],
            }
            for formulation in FORMULATIONS
        }, indent=2), flush=True)
    finally:
        train_dataset.close()
        val_dataset.close()
        test_dataset.close()


if __name__ == "__main__":
    main()
