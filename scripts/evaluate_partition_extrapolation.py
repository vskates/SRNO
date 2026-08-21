#!/usr/bin/env python3
"""Richardson extrapolation across learned-operator load partitions."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from srno.data.dataset import H5ObjectDataset, make_dataloader
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import so3_exp, so3_log_vector
from srno.losses import state_error
from srno.training.checkpoint import load_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.types import PoseState


def _state_at(states: PoseState, step: int) -> PoseState:
    return PoseState(
        states.rotation[:, step], states.position[:, step], states.joint_position[:, step]
    )


def _partition(model, batch, segments):
    current = _state_at(batch.states, 0)
    stride = 32 // segments
    for end in range(stride, 33, stride):
        current = model.forward_step(current, batch.command_schedule[end], batch.sdf)
        assert isinstance(current, PoseState)
    return current


def _extrapolate(coarse: PoseState, fine: PoseState, alpha: float) -> PoseState:
    spatial = so3_log_vector(fine.rotation @ coarse.rotation.transpose(-1, -2))
    return PoseState(
        so3_exp(alpha * spatial) @ fine.rotation,
        fine.position + alpha * (fine.position - coarse.position),
        fine.joint_position + alpha * (fine.joint_position - coarse.joint_position),
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
                dataset, mode="rollout", objects_per_batch=1, samples_per_object=0,
                workers=0, seed=config.seed + split_index, shuffle=False,
            )
            sums = {"p16": 0.0, "p32": 0.0, "richardson_p1": 0.0, "richardson_p2": 0.0}
            count = 0
            try:
                for raw in loader:
                    batch = raw.to(device)
                    coarse = _partition(model, batch, 16)
                    fine = _partition(model, batch, 32)
                    target = _state_at(batch.states, 32)
                    predictions = {
                        "p16": coarse,
                        "p32": fine,
                        "richardson_p1": _extrapolate(coarse, fine, 1.0),
                        "richardson_p2": _extrapolate(coarse, fine, 1.0 / 3.0),
                    }
                    for name, prediction in predictions.items():
                        dx = state_error(
                            prediction,
                            target,
                            length_scale=model.length_scale,
                            joint_scale=model.joint_travel_range,
                        )[0].sqrt()
                        sums[name] += float(dx.sum().cpu())
                    count += target.position.shape[0]
            finally:
                dataset.close()
            result[split] = {name: value / count for name, value in sums.items()}
    payload = {
        "definition": {
            "p1": "x_R=x_h+(x_h-x_2h) in product tangent coordinates",
            "p2": "x_R=x_h+(x_h-x_2h)/3 in product tangent coordinates",
            "coarse_partitions": 16,
            "fine_partitions": 32,
        },
        "terminal_dx": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
