#!/usr/bin/env python3
"""Evaluate geodesic consensus of recurrent and cumulative SRNO paths."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from srno.data.dataset import H5ObjectDataset, TrajectoryBatch, make_dataloader
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import so3_exp, so3_log_vector
from srno.training.checkpoint import load_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _autocast, _build_model, _state_at
from srno.training.metrics import MetricAccumulator, accumulate_trajectory_metrics
from srno.types import PoseState

from run_solution_path_operator_ablation import _cumulative_prediction


def _blend(first: PoseState, second: PoseState, alpha: float) -> PoseState:
    relative = second.rotation.float() @ first.rotation.float().transpose(-1, -2)
    rotation = so3_exp(alpha * so3_log_vector(relative)) @ first.rotation.float()
    return PoseState(
        rotation.to(first.rotation.dtype),
        (1.0 - alpha) * first.position + alpha * second.position,
        (1.0 - alpha) * first.joint_position + alpha * second.joint_position,
    )


def _evaluate(
    recurrent: torch.nn.Module,
    cumulative: torch.nn.Module,
    loader: Any,
    manifest: DatasetManifest,
    config: ExperimentConfig,
    device: torch.device,
    alphas: tuple[float, ...],
) -> dict[str, dict[str, float]]:
    accumulators = {alpha: MetricAccumulator() for alpha in alphas}
    recurrent.eval()
    cumulative.eval()
    with torch.no_grad():
        for raw in loader:
            assert isinstance(raw, TrajectoryBatch)
            batch = raw.to(device)
            with _autocast(config, device):
                recurrent_path = recurrent.rollout(
                    _state_at(batch.states, 0), batch.command_schedule[1:], batch.sdf
                )
                cumulative_path = _cumulative_prediction(cumulative, batch)
                for alpha in alphas:
                    prediction = _blend(recurrent_path, cumulative_path, alpha)
                    gaps = torch.stack([
                        recurrent.query_geometric_gap(
                            _state_at(prediction, step), batch.sdf
                        )
                        for step in range(33)
                    ], dim=1)
                    accumulate_trajectory_metrics(
                        accumulators[alpha],
                        prediction,
                        batch.states,
                        batch.command_schedule,
                        recurrent.aperture_from_joints(prediction.joint_position),
                        batch.actual_aperture,
                        gaps,
                        length_scale=recurrent.length_scale,
                        joint_scale=recurrent.joint_travel_range,
                        lag_threshold=manifest.delta_gate_m,
                    )
    return {f"{alpha:.6f}": accumulators[alpha].compute() for alpha in alphas}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--recurrent", type=Path, required=True)
    parser.add_argument("--cumulative", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grid-size", type=int, default=21)
    args = parser.parse_args()
    if args.grid_size < 2:
        parser.error("--grid-size must be at least two")

    base = ExperimentConfig.load(args.config)
    config = replace(base, device=args.device, loader=replace(base.loader, workers=0))
    device = torch.device(config.device)
    manifest = DatasetManifest.load(config.paths.manifest)
    recurrent = _build_model(config, manifest, device)
    cumulative = _build_model(config, manifest, device)
    for path, model in ((args.recurrent, recurrent), (args.cumulative, cumulative)):
        saved = load_checkpoint(path, model=model, map_location=device)
        if saved["manifest_sha256"] != manifest.sha256():
            raise ValueError("checkpoint and manifest hashes differ")
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
        alphas = tuple(map(float, np.linspace(0.0, 1.0, args.grid_size)))
        val = _evaluate(
            recurrent, cumulative, val_loader, manifest, config, device, alphas
        )
        best_alpha = min(alphas, key=lambda alpha: val[f"{alpha:.6f}"]["terminal_dx"])
        test = _evaluate(
            recurrent, cumulative, test_loader, manifest, config, device,
            (0.0, best_alpha, 1.0),
        )
        baseline = test["0.000000"]["terminal_dx"]
        selected = test[f"{best_alpha:.6f}"]["terminal_dx"]
        result = {
            "definition": {
                "formula": "Exp(alpha Log(R_cumulative R_recurrent^T)) R_recurrent; Euclidean blend for p,q",
                "selection": "single global alpha selected by full validation terminal d_X",
                "test_touched_for_selection": False,
            },
            "contract": {
                "config": str(args.config.resolve()),
                "recurrent": str(args.recurrent.resolve()),
                "cumulative": str(args.cumulative.resolve()),
                "grid_size": args.grid_size,
            },
            "selected_alpha": best_alpha,
            "val": val,
            "test": test,
            "paired_test": {
                "recurrent_terminal_dx": baseline,
                "consensus_terminal_dx": selected,
                "relative_change": float((selected - baseline) / baseline),
            },
        }
        _write_json(args.output, result)
        print(json.dumps({
            "selected_alpha": best_alpha,
            "val_terminal_dx": val[f"{best_alpha:.6f}"]["terminal_dx"],
            **result["paired_test"],
        }, indent=2, sort_keys=True), flush=True)
    finally:
        val_dataset.close()
        test_dataset.close()


if __name__ == "__main__":
    main()
