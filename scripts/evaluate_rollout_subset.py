#!/usr/bin/env python3
"""Evaluate a checkpoint on a deterministic trajectory subset per object."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from srno.data.dataset import H5ObjectDataset, ObjectBatchCollator, ObjectSampleKey
from srno.data.schema import DatasetManifest
from srno.losses import state_error
from srno.training.checkpoint import load_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.types import PoseState


def _state_at(states: PoseState, step: int) -> PoseState:
    return PoseState(
        states.rotation[:, step],
        states.position[:, step],
        states.joint_position[:, step],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--trajectories-per-object", type=int, default=8)
    parser.add_argument("--resolvent-iterations", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.trajectories_per_object <= 0:
        raise ValueError("trajectory subset size must be positive")
    config = replace(ExperimentConfig.load(args.config), device=args.device)
    if args.resolvent_iterations is not None:
        config = replace(
            config,
            model=replace(
                config.model, resolvent_iterations=args.resolvent_iterations
            ),
        )
    device = torch.device(config.device)
    manifest = DatasetManifest.load(config.paths.manifest)
    model = _build_model(config, manifest, device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    dataset = H5ObjectDataset(manifest, split=args.split)
    collator = ObjectBatchCollator(
        manifest, mode="rollout", samples_per_object=0, resample=False
    )
    by_object = {}
    model.eval()
    try:
        with torch.no_grad():
            for object_index, object_id in enumerate(dataset.object_ids):
                count = min(
                    args.trajectories_per_object,
                    dataset.sample_counts("rollout")[object_index],
                )
                record = dataset[
                    ObjectSampleKey(object_index, tuple(range(count)))
                ]
                batch = collator([record]).to(device)
                prediction = model.rollout(
                    _state_at(batch.states, 0), batch.command_schedule[1:], batch.sdf
                )
                terminal = state_error(
                    _state_at(prediction, 32),
                    _state_at(batch.states, 32),
                    length_scale=model.length_scale,
                    joint_scale=model.joint_travel_range,
                )[0].sqrt()
                by_object[object_id] = {
                    "trajectories": count,
                    "terminal_dx": float(terminal.mean().cpu()),
                    "terminal_dx_values": terminal.cpu().tolist(),
                }
                print(
                    f"[SUBSET] {object_id} H32={by_object[object_id]['terminal_dx']:.6f}",
                    flush=True,
                )
    finally:
        dataset.close()
    result = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "trajectories_per_object": args.trajectories_per_object,
        "resolvent_iterations": config.model.resolvent_iterations,
        "equal_object_terminal_dx": sum(
            value["terminal_dx"] for value in by_object.values()
        )
        / len(by_object),
        "by_object": by_object,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "equal_object_terminal_dx": result["equal_object_terminal_dx"]
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
