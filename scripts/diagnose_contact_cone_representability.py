#!/usr/bin/env python3
"""Oracle test: is the observed correction in the positive contact-normal cone?

For each transition this fits nonnegative contact multipliers at either the
free trial or target geometry.  It is an oracle because target geometry is
used in one arm and the multipliers are fitted directly to the target state.
The result is therefore an architecture upper bound, not a predictor metric.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
import torch

from diagnose_linearized_contact_projection import (
    _errors,
    _linearization,
    _repeat_sdf,
    _select_state,
    _state_from_z,
    _target_coordinates,
)
from srno.data.dataset import H5ObjectDataset, LocalTransitionBatch, make_dataloader
from srno.data.index import ActiveIndex
from srno.data.schema import DatasetManifest
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.types import PoseState


def _fit_cone(
    jacobian: np.ndarray,
    target_z: np.ndarray,
    pose_weight: float,
) -> tuple[np.ndarray, float]:
    hessian = np.full(12, 1.0 / 6.0, dtype=np.float64)
    hessian[:6] = pose_weight
    # Stationarity of 0.5 z^T H z - lambda^T h is H z = J^T lambda.
    # Fit in the same H metric used for state error.
    design = jacobian.T / np.sqrt(hessian)[:, None]
    rhs = target_z * np.sqrt(hessian)
    multipliers, residual = nnls(design, rhs, maxiter=5000)
    prediction = (jacobian.T @ multipliers) / hessian
    return prediction, float(residual)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/srno-r-material-v2.toml"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--samples-per-object", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=2e-4)
    parser.add_argument("--pose-weights", type=float, nargs="+", default=(1.0, 10.0, 100.0))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/contact-cone-representability-test.json")
    )
    args = parser.parse_args()

    config = replace(ExperimentConfig.load(args.config), device="cpu")
    manifest = DatasetManifest.load(config.paths.manifest)
    active_index = ActiveIndex.load(config.paths.active_index)
    model = _build_model(config, manifest, torch.device("cpu"))
    model.eval()
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
    keys = [
        f"{geometry},pose_weight={pose_weight:g}"
        for geometry in ("trial", "target")
        for pose_weight in args.pose_weights
    ]
    rows: dict[str, list[dict[str, float]]] = {key: [] for key in keys}
    free_rows: list[dict[str, float]] = []
    contact_counts: dict[str, list[int]] = {"trial": [], "target": []}
    try:
        with torch.no_grad():
            for batch in loader:
                assert isinstance(batch, LocalTransitionBatch)
                count = batch.current.position.shape[0]
                selected = np.linspace(
                    0, count - 1, min(args.samples_per_object, count), dtype=np.int64
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
                    target_z = _target_coordinates(model, trial, target)
                    free_rows.append(_errors(model, trial, target))
                    for geometry_name, state in (("trial", trial), ("target", target)):
                        gap, jacobian = _linearization(model, state, sdf, args.epsilon)
                        # The learned model's exact gate is the broadest physically
                        # defensible contact support.  Excluding farther points also
                        # prevents an oracle from using non-contact normals.
                        contact = (
                            gap * model.length_scale - model.contact_offset_sum
                            <= model.delta_gate
                        )
                        contact_counts[geometry_name].append(int(contact.sum()))
                        active_jacobian = jacobian[contact]
                        if not len(active_jacobian):
                            active_jacobian = jacobian[np.argmin(gap) : np.argmin(gap) + 1]
                        for pose_weight in args.pose_weights:
                            prediction_z, residual = _fit_cone(
                                active_jacobian, target_z, pose_weight
                            )
                            prediction = _state_from_z(
                                model,
                                trial,
                                torch.from_numpy(prediction_z.astype(np.float32))[None],
                            )
                            values = _errors(model, prediction, target)
                            values["nnls_residual"] = residual
                            rows[f"{geometry_name},pose_weight={pose_weight:g}"].append(values)
    finally:
        dataset.close()

    names = ("dx", "translation_m", "rotation_rad", "joint_rmse_over_travel")
    free = {name: float(np.mean([row[name] for row in free_rows])) for name in names}
    arms: dict[str, object] = {}
    for key, arm_rows in rows.items():
        metrics = {name: float(np.mean([row[name] for row in arm_rows])) for name in names}
        arms[key] = {
            "metrics": metrics,
            "relative_change_percent": {
                name: (metrics[name] / free[name] - 1.0) * 100.0 for name in names
            },
            "mean_nnls_residual": float(
                np.mean([row["nnls_residual"] for row in arm_rows])
            ),
        }
    output = {
        "config": str(args.config),
        "split": args.split,
        "samples_per_object": args.samples_per_object,
        "note": "Oracle representability bound; not a predictive evaluation.",
        "free": free,
        "mean_contact_points": {
            key: float(np.mean(value)) for key, value in contact_counts.items()
        },
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
