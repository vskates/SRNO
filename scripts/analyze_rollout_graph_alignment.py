#!/usr/bin/env python3
"""Audit whether rollout error is mainly BV jump-timing misalignment.

The diagnostic compares synchronous trajectory error with band-constrained
dynamic-time-warping distances in the production state metric.  It is a
falsification test for a graph-parametrized/Skorokhod learning formulation, not
an alternative headline metric: terminal states remain paired exactly.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from srno.data.dataset import H5ObjectDataset, make_dataloader
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import rotation_geodesic_angle
from srno.training.checkpoint import load_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.types import PoseState


BANDS = (1, 2, 4, 8)


def _state_at(states: PoseState, step: int) -> PoseState:
    return PoseState(
        states.rotation[:, step],
        states.position[:, step],
        states.joint_position[:, step],
    )


def _pairwise_dx(model, first: PoseState, second: PoseState) -> torch.Tensor:
    translation = torch.linalg.vector_norm(
        first.position[:, :, None] - second.position[:, None, :], dim=-1
    ) / model.length_scale
    rotation = rotation_geodesic_angle(
        first.rotation[:, :, None], second.rotation[:, None, :]
    )
    joints = (
        (
            first.joint_position[:, :, None]
            - second.joint_position[:, None, :]
        )
        / model.joint_travel_range
    ).square().mean(dim=-1).sqrt()
    return torch.sqrt(translation.square() + rotation.square() + joints.square())


def _jump_mask(model, target: PoseState) -> torch.Tensor:
    translation = torch.linalg.vector_norm(
        target.position[:, 1:] - target.position[:, :-1], dim=-1
    ) / model.length_scale
    rotation = rotation_geodesic_angle(
        target.rotation[:, 1:], target.rotation[:, :-1]
    )
    return torch.sqrt(translation.square() + rotation.square()).amax(dim=1) > 0.05


def _dtw(cost: np.ndarray, band: int) -> tuple[float, float, float]:
    size = cost.shape[0]
    total = np.full((size, size), np.inf, dtype=np.float64)
    length = np.zeros((size, size), dtype=np.int32)
    previous = np.full((size, size, 2), -1, dtype=np.int16)
    total[0, 0] = float(cost[0, 0])
    length[0, 0] = 1
    for i in range(size):
        for j in range(max(0, i - band), min(size, i + band + 1)):
            if i == 0 and j == 0:
                continue
            candidates = []
            if i > 0 and np.isfinite(total[i - 1, j]):
                candidates.append((total[i - 1, j], i - 1, j))
            if j > 0 and np.isfinite(total[i, j - 1]):
                candidates.append((total[i, j - 1], i, j - 1))
            if i > 0 and j > 0 and np.isfinite(total[i - 1, j - 1]):
                candidates.append((total[i - 1, j - 1], i - 1, j - 1))
            if not candidates:
                continue
            value, pi, pj = min(candidates, key=lambda item: item[0])
            total[i, j] = value + float(cost[i, j])
            length[i, j] = length[pi, pj] + 1
            previous[i, j] = (pi, pj)
    i = j = size - 1
    offsets: list[int] = []
    while i >= 0 and j >= 0:
        offsets.append(abs(i - j))
        if i == 0 and j == 0:
            break
        i, j = map(int, previous[i, j])
    return (
        float(total[-1, -1] / max(1, length[-1, -1])),
        float(np.mean(offsets)),
        float(np.max(offsets)),
    )


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
    }


def _evaluate(config, checkpoint, split):
    device = torch.device(config.device)
    manifest = DatasetManifest.load(config.paths.manifest)
    model = _build_model(config, manifest, device)
    load_checkpoint(checkpoint, model=model, map_location=device)
    dataset = H5ObjectDataset(manifest, split=split)
    loader = make_dataloader(
        dataset,
        mode="rollout",
        objects_per_batch=1,
        samples_per_object=0,
        workers=0,
        seed=config.seed + (1 if split == "val" else 2),
        shuffle=False,
    )
    synchronous: list[np.ndarray] = []
    terminal: list[np.ndarray] = []
    jumps: list[np.ndarray] = []
    aligned = {band: [] for band in BANDS}
    offsets = {band: [] for band in BANDS}
    model.eval()
    try:
        with torch.no_grad():
            for raw in loader:
                batch = raw.to(device)
                prediction = model.rollout(
                    _state_at(batch.states, 0), batch.command_schedule[1:], batch.sdf
                )
                distances = _pairwise_dx(model, prediction, batch.states).cpu().numpy()
                synchronous.append(np.diagonal(distances, axis1=1, axis2=2).mean(axis=1))
                terminal.append(distances[:, -1, -1])
                jumps.append(_jump_mask(model, batch.states).cpu().numpy())
                for trajectory_cost in distances:
                    for band in BANDS:
                        distance, mean_offset, _ = _dtw(trajectory_cost, band)
                        aligned[band].append(distance)
                        offsets[band].append(mean_offset)
    finally:
        dataset.close()
    sync = np.concatenate(synchronous)
    end = np.concatenate(terminal)
    jump = np.concatenate(jumps).astype(bool)
    result = {
        "trajectories": int(len(sync)),
        "jump_trajectory_fraction": float(jump.mean()),
        "synchronous_path_dx": _summary(sync),
        "terminal_dx": _summary(end),
        "bands": {},
    }
    for band in BANDS:
        distance = np.asarray(aligned[band])
        offset = np.asarray(offsets[band])
        result["bands"][str(band)] = {
            "aligned_path_dx": _summary(distance),
            "mean_step_offset": _summary(offset),
            "relative_change_vs_synchronous_percent": float(
                100.0 * (distance.mean() / sync.mean() - 1.0)
            ),
            "jump_trajectories": {
                "synchronous_path_dx": _summary(sync[jump]),
                "aligned_path_dx": _summary(distance[jump]),
                "relative_change_percent": float(
                    100.0 * (distance[jump].mean() / sync[jump].mean() - 1.0)
                ),
            },
            "nonjump_trajectories": {
                "synchronous_path_dx": _summary(sync[~jump]),
                "aligned_path_dx": _summary(distance[~jump]),
                "relative_change_percent": float(
                    100.0 * (distance[~jump].mean() / sync[~jump].mean() - 1.0)
                ),
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = replace(ExperimentConfig.load(args.config), device=args.device)
    result = {
        "definition": {
            "synchronous": "mean_k d_X(x_hat_k,x_k)",
            "aligned": "band-constrained DTW mean path cost in d_X",
            "terminal_pairing_changed": False,
            "bands_steps": list(BANDS),
        },
        "val": _evaluate(config, args.checkpoint.resolve(), "val"),
        "test": _evaluate(config, args.checkpoint.resolve(), "test"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
