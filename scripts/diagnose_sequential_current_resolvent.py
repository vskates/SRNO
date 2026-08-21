#!/usr/bin/env python3
"""Validation screen for sequentially relinearized current-state resolvents."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from diagnose_current_state_resolvent import (
    _errors,
    _repeat_sdf,
    _select_state,
    _solve,
    _state_from_increment,
)
from srno.data.dataset import H5ObjectDataset, LocalTransitionBatch, make_dataloader
from srno.data.index import ActiveIndex
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import so3_log_vector
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.types import PoseState


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("dx", "translation_m", "rotation_rad", "joint_rmse_over_travel")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/srno-r-material-v2.toml")
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--samples-per-object", type=int, default=100)
    parser.add_argument("--iterations", type=int, nargs="+", default=(1, 2, 3, 4))
    parser.add_argument("--pose-weight", type=float, default=100.0)
    parser.add_argument("--pose-query-factor", type=float, default=0.5)
    parser.add_argument("--constraint-gap-m", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/sequential-current-resolvent-val.json"),
    )
    args = parser.parse_args()
    if any(value <= 0 for value in args.iterations):
        raise ValueError("iterations must be positive")

    config = ExperimentConfig.load(args.config)
    config = replace(
        config,
        device="cpu",
        model=replace(config.model, pose_update="decoupled"),
    )
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
    rows = {value: [] for value in args.iterations}
    jump_rows = {value: [] for value in args.iterations}
    statuses = {value: {} for value in args.iterations}
    try:
        with torch.no_grad():
            for batch in loader:
                assert isinstance(batch, LocalTransitionBatch)
                selected = np.linspace(
                    0,
                    len(batch.next_command) - 1,
                    min(args.samples_per_object, len(batch.next_command)),
                    dtype=np.int64,
                )
                for raw_index in selected:
                    index = int(raw_index)
                    current = _select_state(batch.current, index)
                    target = _select_state(batch.target, index)
                    assert batch.previous is not None
                    previous = _select_state(batch.previous, index)
                    sdf = _repeat_sdf(batch.sdf, 1)
                    free_joint = model.free_joint_configuration(
                        batch.next_command[index : index + 1]
                    )
                    query = np.zeros(12, dtype=np.float64)
                    query[:3] = args.pose_query_factor * (
                        (current.position - previous.position) / model.length_scale
                    )[0].numpy()
                    query[3:6] = args.pose_query_factor * so3_log_vector(
                        current.rotation @ previous.rotation.transpose(-1, -2)
                    )[0].numpy()
                    query[6:] = (
                        (free_joint - current.joint_position)
                        / model.joint_travel_range
                    )[0].numpy()
                    joint_min = model.free_joint_knots.amin(dim=0)
                    joint_max = model.free_joint_knots.amax(dim=0)
                    joint_lower = (
                        (joint_min - current.joint_position[0])
                        / model.joint_travel_range
                    ).numpy().astype(np.float64)
                    joint_upper = (
                        (joint_max - current.joint_position[0])
                        / model.joint_travel_range
                    ).numpy().astype(np.float64)
                    pose_motion = float(
                        torch.sqrt(
                            ((target.position - current.position) / model.length_scale)
                            .square()
                            .sum()
                            + so3_log_vector(
                                target.rotation
                                @ current.rotation.transpose(-1, -2)
                            )
                            .square()
                            .sum()
                        )
                    )

                    z = np.zeros(12, dtype=np.float64)
                    for iteration in range(1, max(args.iterations) + 1):
                        estimate = _state_from_increment(model, current, z)
                        contact_gap, _, jacobian = (
                            model._contact_gap_and_full_jacobian(estimate, sdf)
                        )
                        geometric_gap = contact_gap[0] + model.contact_offset_sum
                        full_jacobian = jacobian[0]
                        support = (contact_gap[0] <= model.delta_gate) & (
                            torch.linalg.vector_norm(full_jacobian, dim=-1) > 1e-8
                        )
                        if not bool(support.any()):
                            nearest = torch.argmin(contact_gap[0])
                            if torch.linalg.vector_norm(full_jacobian[nearest]) > 1e-8:
                                support[nearest] = True
                        selected_jacobian = (
                            full_jacobian[support].numpy().astype(np.float64)
                        )
                        # h(x(z)) + J(x(z)) (y-z) >= h_constraint.
                        rhs = (
                            (args.constraint_gap_m - geometric_gap[support])
                            / model.sdf_scale
                            + selected_jacobian @ z
                        )
                        solved, status = _solve(
                            selected_jacobian,
                            rhs,
                            query,
                            joint_lower,
                            joint_upper,
                            pose_weight=args.pose_weight,
                        )
                        counter = statuses[iteration] if iteration in statuses else None
                        if counter is not None:
                            counter[status] = counter.get(status, 0) + 1
                        if solved is None:
                            break
                        z = solved
                        if iteration in rows:
                            prediction = _state_from_increment(model, current, z)
                            metric = _errors(model, prediction, target)
                            rows[iteration].append(metric)
                            if pose_motion > 0.05:
                                jump_rows[iteration].append(metric)
    finally:
        dataset.close()

    output = {
        "format_version": 1,
        "split": args.split,
        "samples_per_object": args.samples_per_object,
        "pose_weight": args.pose_weight,
        "pose_query_factor": args.pose_query_factor,
        "constraint_gap_m": args.constraint_gap_m,
        "arms": {
            str(iteration): {
                "all": _mean(rows[iteration]),
                "jump": _mean(jump_rows[iteration]) if jump_rows[iteration] else None,
                "all_count": len(rows[iteration]),
                "jump_count": len(jump_rows[iteration]),
                "solver_statuses": statuses[iteration],
            }
            for iteration in args.iterations
        },
    }
    rendered = json.dumps(output, indent=2, sort_keys=True)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
