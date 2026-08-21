#!/usr/bin/env python3
"""Evaluate Picard iterations of the learned causal Volterra path map."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import torch

from srno.data.dataset import H5ObjectDataset, make_dataloader
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import so3_exp, so3_log_vector
from srno.losses import state_error
from srno.training.checkpoint import load_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.types import PoseState, SDFBatch


ITERATIONS = (0, 1, 2, 4, 8)


def _state_at(states: PoseState, step: int) -> PoseState:
    return PoseState(
        states.rotation[:, step], states.position[:, step], states.joint_position[:, step]
    )


def _free_path(model, batch) -> PoseState:
    initial = _state_at(batch.states, 0)
    trajectories = initial.position.shape[0]
    free = model.free_joint_configuration(batch.command_schedule[1:])
    return PoseState(
        torch.cat(
            (
                initial.rotation[:, None],
                initial.rotation[:, None].expand(-1, 32, -1, -1),
            ),
            dim=1,
        ),
        torch.cat(
            (
                initial.position[:, None],
                initial.position[:, None].expand(-1, 32, -1),
            ),
            dim=1,
        ),
        torch.cat(
            (
                initial.joint_position[:, None],
                free[None].expand(trajectories, -1, -1),
            ),
            dim=1,
        ),
    )


def _picard_step(model, guess: PoseState, commands, sdf: SDFBatch) -> PoseState:
    trajectories = guess.position.shape[0]
    current = PoseState(
        guess.rotation[:, :-1].reshape(-1, 3, 3),
        guess.position[:, :-1].reshape(-1, 3),
        guess.joint_position[:, :-1].reshape(-1, 6),
    )
    flat_commands = commands.view(1, 32).expand(trajectories, -1).reshape(-1)
    flat_sdf = SDFBatch(
        sdf.values,
        sdf.origin,
        sdf.voxel_size,
        sdf.sample_to_object.repeat_interleave(32),
        sdf.outside_value,
    )
    candidate = model.forward_step(current, flat_commands, flat_sdf)
    assert isinstance(candidate, PoseState)
    translation = (candidate.position - current.position).reshape(trajectories, 32, 3)
    rotation = so3_log_vector(
        candidate.rotation @ current.rotation.transpose(-1, -2)
    ).reshape(trajectories, 32, 3)
    joints = (candidate.joint_position - current.joint_position).reshape(
        trajectories, 32, 6
    )
    initial = _state_at(guess, 0)
    positions = initial.position[:, None] + translation.cumsum(dim=1)
    joint_path = initial.joint_position[:, None] + joints.cumsum(dim=1)
    rotations = []
    current_rotation = initial.rotation
    for step in range(32):
        current_rotation = so3_exp(rotation[:, step]) @ current_rotation
        rotations.append(current_rotation)
    return PoseState(
        torch.cat((initial.rotation[:, None], torch.stack(rotations, dim=1)), dim=1),
        torch.cat((initial.position[:, None], positions), dim=1),
        torch.cat((initial.joint_position[:, None], joint_path), dim=1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    base = ExperimentConfig.load(args.config)
    config = replace(base, device=args.device, loader=replace(base.loader, workers=0))
    device = torch.device(config.device)
    manifest = DatasetManifest.load(config.paths.manifest)
    model = _build_model(config, manifest, device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    result = {}
    model.eval()
    with torch.no_grad():
        for split_index, split in enumerate(("val", "test"), start=1):
            dataset = H5ObjectDataset(manifest, split=split)
            loader = make_dataloader(
                dataset,
                mode="rollout",
                objects_per_batch=1,
                samples_per_object=0,
                workers=0,
                seed=config.seed + split_index,
                shuffle=False,
            )
            sums = {iteration: 0.0 for iteration in ITERATIONS}
            count = 0
            try:
                for raw in loader:
                    batch = raw.to(device)
                    guess = _free_path(model, batch)
                    terminal_free = state_error(
                        _state_at(guess, 32),
                        _state_at(batch.states, 32),
                        length_scale=model.length_scale,
                        joint_scale=model.joint_travel_range,
                    )[0].sqrt()
                    sums[0] += float(terminal_free.sum().cpu())
                    for iteration in range(1, max(ITERATIONS) + 1):
                        guess = _picard_step(
                            model, guess, batch.command_schedule[1:], batch.sdf
                        )
                        if iteration in ITERATIONS:
                            terminal = state_error(
                                _state_at(guess, 32),
                                _state_at(batch.states, 32),
                                length_scale=model.length_scale,
                                joint_scale=model.joint_travel_range,
                            )[0].sqrt()
                            sums[iteration] += float(terminal.sum().cpu())
                    count += batch.states.position.shape[0]
            finally:
                dataset.close()
            result[split] = {
                str(iteration): sums[iteration] / count for iteration in ITERATIONS
            }
    result = {
        "definition": "X^(m+1)=V_theta[X^(m)], initialized by the free path",
        "checkpoint": str(args.checkpoint.resolve()),
        "terminal_dx": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
