#!/usr/bin/env python3
"""Evaluate an untrained architecture on a deterministic local subset.

This is a representation/initialization diagnostic.  It never selects a
checkpoint and therefore must be run on validation rather than test data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from srno.data.dataset import H5ObjectDataset, LocalTransitionBatch, make_dataloader
from srno.data.index import ActiveIndex
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import rotation_geodesic_angle
from srno.losses import state_error
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.types import PoseState, SDFBatch


def _select_state(state: PoseState, index: torch.Tensor) -> PoseState:
    return PoseState(
        state.rotation.index_select(0, index),
        state.position.index_select(0, index),
        state.joint_position.index_select(0, index),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, required=True, help="Architecture/config to initialize."
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--samples-per-object", type=int, default=100)
    parser.add_argument(
        "--pose-query-factor",
        type=float,
        help="Optional validation-only override for resolvent pose query.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples_per_object <= 0:
        raise ValueError("samples-per-object must be positive")

    config = replace(ExperimentConfig.load(args.config), device="cpu")
    if args.pose_query_factor is not None:
        config = replace(
            config,
            model=replace(
                config.model,
                resolvent_pose_query_factor=args.pose_query_factor,
            ),
        )
        config.validate()
    torch.manual_seed(config.seed)
    manifest = DatasetManifest.load(config.paths.manifest)
    active_index = ActiveIndex.load(config.paths.active_index)
    model = _build_model(config, manifest, torch.device("cpu")).eval()
    dataset = H5ObjectDataset(
        manifest,
        split=args.split,
        active_index=active_index,
        active_only=True,
    )
    loader = make_dataloader(
        dataset,
        mode="local",
        objects_per_batch=1,
        samples_per_object=0,
        workers=0,
        seed=config.seed,
        shuffle=False,
    )

    totals = torch.zeros(8, dtype=torch.float64)
    count = 0
    start = perf_counter()
    try:
        with torch.no_grad():
            for batch in loader:
                assert isinstance(batch, LocalTransitionBatch)
                selected = torch.from_numpy(
                    np.linspace(
                        0,
                        batch.next_command.numel() - 1,
                        min(args.samples_per_object, batch.next_command.numel()),
                        dtype=np.int64,
                    )
                )
                current = _select_state(batch.current, selected)
                target = _select_state(batch.target, selected)
                previous = (
                    None
                    if batch.previous is None
                    else _select_state(batch.previous, selected)
                )
                sdf = SDFBatch(
                    batch.sdf.values,
                    batch.sdf.origin,
                    batch.sdf.voxel_size,
                    torch.zeros(len(selected), dtype=torch.long),
                    batch.sdf.outside_value,
                )
                prediction, aux = model.forward_step(
                    current,
                    batch.next_command.index_select(0, selected),
                    sdf,
                    previous_state=previous,
                    return_aux=True,
                )
                assert isinstance(prediction, PoseState)
                total, translation_sq, rotation_sq, joint_sq = state_error(
                    prediction,
                    target,
                    length_scale=model.length_scale,
                    joint_scale=model.joint_travel_range,
                )
                translation_m = torch.linalg.vector_norm(
                    prediction.position - target.position, dim=-1
                )
                rotation_rad = rotation_geodesic_angle(
                    prediction.rotation, target.rotation
                )
                minimum_gap = model.query_geometric_gap(prediction, sdf).amin(dim=-1)
                values = torch.stack(
                    (
                        total.sqrt(),
                        translation_m,
                        rotation_rad,
                        joint_sq.sqrt(),
                        translation_sq.sqrt(),
                        rotation_sq.sqrt(),
                        aux.active.float(),
                        (minimum_gap >= config.loss.admissible_gap_m).float(),
                    ),
                    dim=-1,
                )
                totals += values.double().sum(dim=0)
                count += len(selected)
    finally:
        dataset.close()
    means = totals / count
    result = {
        "config": str(args.config.resolve()),
        "split": args.split,
        "samples": count,
        "samples_per_object": args.samples_per_object,
        "pose_query_factor": config.model.resolvent_pose_query_factor,
        "elapsed_seconds": perf_counter() - start,
        "aggregate_dx": float(means[0]),
        "translation_m": float(means[1]),
        "rotation_rad": float(means[2]),
        "joint_rmse_over_travel": float(means[3]),
        "translation_over_length": float(means[4]),
        "rotation_component": float(means[5]),
        "active_fraction": float(means[6]),
        "nonlinear_feasible_fraction": float(means[7]),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
