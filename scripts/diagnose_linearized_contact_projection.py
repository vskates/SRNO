#!/usr/bin/env python3
"""Measure how much a geometry-only normal-cone projection explains SRNO targets.

The diagnostic uses the product coordinates employed by the state metric,

    z = [delta_position / L, spatial_rotation, delta_joint / joint_travel],

linearizes every geometric SDF gap at the free-command trial state, and solves

    min 0.5 * (|z_p|^2 + |z_R|^2 + mean(z_r^2))
    s.t. h_trial + J z >= h_admissible.

It intentionally contains no learned parameter.  Consequently, improvement
over the free-state baseline is evidence for a useful normal-cone backbone;
remaining error measures the constitutive information (metric, friction,
contact mode, and history) that such a backbone still lacks.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import osqp
import scipy.sparse as sparse
import torch

from srno.data.dataset import H5ObjectDataset, LocalTransitionBatch, make_dataloader
from srno.data.index import ActiveIndex
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import rotation_geodesic_angle, so3_exp, so3_log_vector
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.types import PoseState, SDFBatch


def _select_state(state: PoseState, index: int) -> PoseState:
    selection = torch.tensor([index], dtype=torch.long)
    return PoseState(
        state.rotation.index_select(0, selection),
        state.position.index_select(0, selection),
        state.joint_position.index_select(0, selection),
    )


def _repeat_sdf(sdf: SDFBatch, count: int) -> SDFBatch:
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
    count = z.shape[0]
    rotation = so3_exp(z[:, 3:6]) @ trial.rotation.expand(count, 3, 3)
    position = trial.position.expand(count, 3) + z[:, :3] * model.length_scale
    joints = (
        trial.joint_position.expand(count, 6)
        + z[:, 6:] * model.joint_travel_range
    )
    return PoseState(rotation, position, joints)


def _target_coordinates(model: torch.nn.Module, trial: PoseState, target: PoseState) -> np.ndarray:
    rotation = target.rotation @ trial.rotation.transpose(-1, -2)
    return torch.cat(
        (
            (target.position - trial.position) / model.length_scale,
            so3_log_vector(rotation),
            (target.joint_position - trial.joint_position) / model.joint_travel_range,
        ),
        dim=-1,
    )[0].numpy()


def _linearization(
    model: torch.nn.Module,
    trial: PoseState,
    sdf: SDFBatch,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    zero = torch.zeros(1, 12)
    h0 = model.query_geometric_gap(trial, sdf)[0]
    perturbations = []
    for coordinate in range(12):
        plus = zero.clone()
        minus = zero.clone()
        plus[0, coordinate] = epsilon
        minus[0, coordinate] = -epsilon
        perturbations.extend((plus, minus))
    z = torch.cat(perturbations, dim=0)
    states = _state_from_z(model, trial, z)
    gaps = model.query_geometric_gap(states, _repeat_sdf(sdf, len(z)))
    columns = [
        (gaps[2 * coordinate] - gaps[2 * coordinate + 1]) / (2.0 * epsilon)
        for coordinate in range(12)
    ]
    jacobian = torch.stack(columns, dim=-1)
    # Normalize constraint rows by L so all OSQP quantities are O(1).
    return (h0 / model.length_scale).numpy(), (jacobian / model.length_scale).numpy()


def _solve_projection(
    h0: np.ndarray,
    jacobian: np.ndarray,
    admissible_gap_over_length: float,
) -> tuple[np.ndarray | None, str]:
    hessian = np.ones(12, dtype=np.float64)
    hessian[6:] = 1.0 / 6.0
    solver = osqp.OSQP()
    solver.setup(
        P=sparse.diags(hessian, format="csc"),
        q=np.zeros(12),
        A=sparse.csc_matrix(jacobian),
        l=np.full(len(h0), admissible_gap_over_length) - h0,
        u=np.full(len(h0), np.inf),
        eps_abs=1e-7,
        eps_rel=1e-6,
        max_iter=20_000,
        polish=True,
        verbose=False,
    )
    result = solver.solve()
    status = str(result.info.status)
    if result.x is None or not status.startswith("solved"):
        return None, status
    return np.asarray(result.x), status


def _errors(model: torch.nn.Module, state: PoseState, target: PoseState) -> dict[str, float]:
    translation_m = torch.linalg.vector_norm(state.position - target.position, dim=-1)[0]
    translation = translation_m / model.length_scale
    rotation = rotation_geodesic_angle(state.rotation, target.rotation)[0]
    joints = torch.sqrt(
        (((state.joint_position - target.joint_position) / model.joint_travel_range) ** 2)
        .mean(dim=-1)
    )[0]
    dx = torch.sqrt(translation.square() + rotation.square() + joints.square())
    return {
        "dx": float(dx),
        "translation_m": float(translation_m),
        "rotation_rad": float(rotation),
        "joint_rmse_over_travel": float(joints),
    }


def _summary(rows: list[dict[str, float]], prefix: str) -> dict[str, float]:
    names = ("dx", "translation_m", "rotation_rad", "joint_rmse_over_travel")
    return {name: float(np.mean([row[f"{prefix}_{name}"] for row in rows])) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/srno-r-material-v2.toml"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--samples-per-object", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=2e-4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/linearized-contact-projection-test.json"),
    )
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    config = replace(config, device="cpu")
    manifest = DatasetManifest.load(config.paths.manifest)
    index = ActiveIndex.load(config.paths.active_index)
    model = _build_model(config, manifest, torch.device("cpu"))
    model.eval()
    dataset = H5ObjectDataset(
        manifest,
        split=args.split,
        active_index=index,
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
    rows: list[dict[str, float]] = []
    statuses: dict[str, int] = {}
    try:
        with torch.no_grad():
            for batch in loader:
                assert isinstance(batch, LocalTransitionBatch)
                count = batch.current.position.shape[0]
                selected = np.linspace(
                    0,
                    count - 1,
                    min(args.samples_per_object, count),
                    dtype=np.int64,
                )
                for index_in_batch in selected:
                    i = int(index_in_batch)
                    current = _select_state(batch.current, i)
                    target = _select_state(batch.target, i)
                    command = batch.next_command[i : i + 1]
                    trial = PoseState(
                        current.rotation,
                        current.position,
                        model.free_joint_configuration(command),
                    )
                    sdf = _repeat_sdf(batch.sdf, 1)
                    h0, jacobian = _linearization(model, trial, sdf, args.epsilon)
                    projection, status = _solve_projection(
                        h0,
                        jacobian,
                        config.loss.admissible_gap_m / model.length_scale,
                    )
                    statuses[status] = statuses.get(status, 0) + 1
                    if projection is None:
                        continue
                    projected = _state_from_z(
                        model, trial, torch.from_numpy(projection.astype(np.float32))[None]
                    )
                    target_z = _target_coordinates(model, trial, target)
                    nonlinear_min_gap = float(model.query_geometric_gap(projected, sdf).amin())
                    free_error = _errors(model, trial, target)
                    projected_error = _errors(model, projected, target)
                    weighted_inner = float(
                        np.dot(projection[:6], target_z[:6])
                        + np.dot(projection[6:], target_z[6:]) / 6.0
                    )
                    projection_norm = float(
                        np.sqrt(np.dot(projection[:6], projection[:6]) + np.dot(projection[6:], projection[6:]) / 6.0)
                    )
                    target_norm = float(
                        np.sqrt(np.dot(target_z[:6], target_z[:6]) + np.dot(target_z[6:], target_z[6:]) / 6.0)
                    )
                    row: dict[str, float] = {
                        "trial_min_gap_m": float(np.min(h0) * model.length_scale),
                        "projected_min_gap_m": nonlinear_min_gap,
                        "target_min_gap_m": float(model.query_geometric_gap(target, sdf).amin()),
                        "projection_target_cosine": weighted_inner / max(projection_norm * target_norm, 1e-12),
                    }
                    row.update({f"free_{name}": value for name, value in free_error.items()})
                    row.update({f"projected_{name}": value for name, value in projected_error.items()})
                    rows.append(row)
    finally:
        dataset.close()

    free = _summary(rows, "free")
    projected = _summary(rows, "projected")
    relative_change = {
        name: (projected[name] / free[name] - 1.0) * 100.0 for name in free
    }
    result = {
        "config": str(args.config),
        "split": args.split,
        "requested_samples_per_object": args.samples_per_object,
        "solved_samples": len(rows),
        "solver_statuses": statuses,
        "coordinates": "[delta_position/L, spatial_rotation, delta_joint/joint_travel]",
        "admissible_gap_m": config.loss.admissible_gap_m,
        "free": free,
        "linearized_projection": projected,
        "relative_change_percent": relative_change,
        "mean_projection_target_cosine": float(
            np.mean([row["projection_target_cosine"] for row in rows])
        ),
        "nonlinear_feasible_fraction": float(
            np.mean(
                [
                    row["projected_min_gap_m"] >= config.loss.admissible_gap_m
                    for row in rows
                ]
            )
        ),
        "mean_min_gap_m": {
            name: float(np.mean([row[f"{name}_min_gap_m"] for row in rows]))
            for name in ("trial", "projected", "target")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
