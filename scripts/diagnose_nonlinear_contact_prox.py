#!/usr/bin/env python3
"""Screen a parameter-free nonlinear contact-prox backbone on active transitions.

Unlike ``diagnose_linearized_contact_projection.py``, this diagnostic recomputes
FK, SDF gaps, and their gradients at every optimization step.  It minimizes a
free-state metric plus a squared normal-cone violation penalty.  The target is
used only for reporting errors, never by the solve.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from srno.data.dataset import H5ObjectDataset, LocalTransitionBatch, make_dataloader
from srno.data.index import ActiveIndex
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import rotation_geodesic_angle, so3_exp
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.types import PoseState, SDFBatch


def _select_state(state: PoseState, indices: torch.Tensor) -> PoseState:
    return PoseState(
        state.rotation.index_select(0, indices),
        state.position.index_select(0, indices),
        state.joint_position.index_select(0, indices),
    )


def _single_object_sdf(sdf: SDFBatch, count: int) -> SDFBatch:
    if sdf.values.shape[0] != 1:
        raise ValueError("diagnostic expects one object per loader batch")
    return SDFBatch(
        sdf.values,
        sdf.origin,
        sdf.voxel_size,
        torch.zeros(count, dtype=torch.long),
        sdf.outside_value,
    )


def _state_from_z(model: torch.nn.Module, trial: PoseState, z: torch.Tensor) -> PoseState:
    return PoseState(
        so3_exp(z[:, 3:6]) @ trial.rotation,
        trial.position + z[:, :3] * model.length_scale,
        trial.joint_position + z[:, 6:] * model.joint_travel_range,
    )


def _component_arrays(
    model: torch.nn.Module, prediction: PoseState, target: PoseState
) -> dict[str, np.ndarray]:
    translation_m = torch.linalg.vector_norm(
        prediction.position - target.position, dim=-1
    )
    translation = translation_m / model.length_scale
    rotation = rotation_geodesic_angle(prediction.rotation, target.rotation)
    joints = torch.sqrt(
        (((prediction.joint_position - target.joint_position) / model.joint_travel_range) ** 2)
        .mean(dim=-1)
    )
    dx = torch.sqrt(translation.square() + rotation.square() + joints.square())
    return {
        "dx": dx.detach().numpy(),
        "translation_m": translation_m.detach().numpy(),
        "rotation_rad": rotation.detach().numpy(),
        "joint_rmse_over_travel": joints.detach().numpy(),
    }


def _solve(
    model: torch.nn.Module,
    trial: PoseState,
    sdf: SDFBatch,
    *,
    admissible_gap_m: float,
    penalty: float,
    pose_weight: float,
    steps: int,
    learning_rate: float,
) -> tuple[PoseState, np.ndarray]:
    z = torch.zeros(trial.position.shape[0], 12, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=learning_rate)
    joint_min = model.free_joint_knots.amin(dim=0)
    joint_max = model.free_joint_knots.amax(dim=0)
    lower_joint_z = (joint_min - trial.joint_position) / model.joint_travel_range
    upper_joint_z = (joint_max - trial.joint_position) / model.joint_travel_range
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        state = _state_from_z(model, trial, z)
        normalized_violation = torch.relu(
            (admissible_gap_m - model.query_geometric_gap(state, sdf))
            / model.sdf_scale
        )
        metric = 0.5 * (
            pose_weight * z[:, :6].square().sum(dim=-1)
            + z[:, 6:].square().mean(dim=-1)
        )
        objective = metric + penalty * normalized_violation.square().mean(dim=-1)
        objective.sum().backward()
        optimizer.step()
        with torch.no_grad():
            z[:, :6].clamp_(-0.5, 0.5)
            z[:, 6:] = torch.maximum(
                torch.minimum(z[:, 6:], upper_joint_z), lower_joint_z
            )
    with torch.no_grad():
        state = _state_from_z(model, trial, z)
        min_gap = model.query_geometric_gap(state, sdf).amin(dim=-1).numpy()
    return state, min_gap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/srno-r-material-v2.toml"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--samples-per-object", type=int, default=50)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--penalties", type=float, nargs="+", default=(10.0, 100.0, 1000.0))
    parser.add_argument("--pose-weights", type=float, nargs="+", default=(1.0, 10.0, 100.0))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/nonlinear-contact-prox-val.json")
    )
    args = parser.parse_args()

    config = replace(ExperimentConfig.load(args.config), device="cpu")
    manifest = DatasetManifest.load(config.paths.manifest)
    active_index = ActiveIndex.load(config.paths.active_index)
    model = _build_model(config, manifest, torch.device("cpu"))
    model.requires_grad_(False).eval()
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
    candidates = [
        (float(penalty), float(pose_weight))
        for penalty in args.penalties
        for pose_weight in args.pose_weights
    ]
    values: dict[str, dict[str, list[np.ndarray]]] = {}
    gaps: dict[str, list[np.ndarray]] = {}
    baseline: dict[str, list[np.ndarray]] = {}
    try:
        for batch in loader:
            assert isinstance(batch, LocalTransitionBatch)
            count = batch.current.position.shape[0]
            indices = torch.from_numpy(
                np.linspace(
                    0,
                    count - 1,
                    min(args.samples_per_object, count),
                    dtype=np.int64,
                )
            )
            current = _select_state(batch.current, indices)
            target = _select_state(batch.target, indices)
            command = batch.next_command.index_select(0, indices)
            trial = PoseState(
                current.rotation,
                current.position,
                model.free_joint_configuration(command),
            )
            sdf = _single_object_sdf(batch.sdf, len(indices))
            for name, array in _component_arrays(model, trial, target).items():
                baseline.setdefault(name, []).append(array)
            for penalty, pose_weight in candidates:
                key = f"penalty={penalty:g},pose_weight={pose_weight:g}"
                prediction, min_gap = _solve(
                    model,
                    trial,
                    sdf,
                    admissible_gap_m=config.loss.admissible_gap_m,
                    penalty=penalty,
                    pose_weight=pose_weight,
                    steps=args.steps,
                    learning_rate=args.learning_rate,
                )
                values.setdefault(key, {})
                for name, array in _component_arrays(model, prediction, target).items():
                    values[key].setdefault(name, []).append(array)
                gaps.setdefault(key, []).append(min_gap)
    finally:
        dataset.close()

    baseline_mean = {
        name: float(np.concatenate(chunks).mean()) for name, chunks in baseline.items()
    }
    results: dict[str, object] = {}
    for key, metrics in values.items():
        means = {name: float(np.concatenate(chunks).mean()) for name, chunks in metrics.items()}
        all_gaps = np.concatenate(gaps[key])
        results[key] = {
            "metrics": means,
            "relative_change_percent": {
                name: (means[name] / baseline_mean[name] - 1.0) * 100.0
                for name in means
            },
            "nonlinear_feasible_fraction": float(
                np.mean(all_gaps >= config.loss.admissible_gap_m)
            ),
            "mean_min_gap_m": float(all_gaps.mean()),
        }
    output = {
        "config": str(args.config),
        "split": args.split,
        "samples_per_object": args.samples_per_object,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "admissible_gap_m": config.loss.admissible_gap_m,
        "baseline": baseline_mean,
        "candidates": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
