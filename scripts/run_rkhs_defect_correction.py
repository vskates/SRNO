#!/usr/bin/env python3
"""Learn a global RKHS defect correction to a frozen recurrent solver.

The frozen production rollout is treated as a coarse numerical solution.  A
vector-valued operator learns its complete path defect in the tangent product
coordinates of each coarse state.  This preserves the coarse contact branch
while removing systematic long-horizon bias without recurrently feeding the
correction back into the local cell.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_loading_profile_kernel_operator import _load_split, _metrics, _profile_embedding
from evaluate_rollout_path_projection import _rollout_split
from run_rkhs_solution_operator import (
    _global_shape_distances,
    _objectwise_metrics,
    _solve,
    _squared_distance,
)
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import so3_exp, so3_log_vector
from srno.training.checkpoint import load_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.types import PoseState


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _target_state(data: dict[str, np.ndarray]) -> PoseState:
    return PoseState(
        torch.from_numpy(data["rotation"]),
        torch.from_numpy(data["position"]),
        torch.from_numpy(data["joint"]),
    )


def _defect_coordinates(
    target: PoseState,
    coarse: PoseState,
    *,
    length_scale: float,
    joint_scale: np.ndarray,
) -> np.ndarray:
    position = (target.position - coarse.position) / length_scale
    rotation_delta = target.rotation @ coarse.rotation.transpose(-1, -2)
    rotation = so3_log_vector(rotation_delta)
    joints = (
        target.joint_position - coarse.joint_position
    ) / torch.from_numpy(joint_scale)
    path = torch.cat((position, rotation, joints), dim=-1)
    return path[:, 1:].reshape(len(path), -1).numpy().astype(np.float32)


def _coarse_coordinates(
    coarse: PoseState,
    *,
    length_scale: float,
    joint_scale: np.ndarray,
) -> np.ndarray:
    position = (coarse.position - coarse.position[:, :1]) / length_scale
    rotation_delta = coarse.rotation @ coarse.rotation[:, :1].transpose(-1, -2)
    rotation = so3_log_vector(rotation_delta)
    joints = (
        coarse.joint_position - coarse.joint_position[:, :1]
    ) / torch.from_numpy(joint_scale)
    path = torch.cat((position, rotation, joints), dim=-1)
    return path[:, 1:].reshape(len(path), -1).numpy().astype(np.float32)


def _apply_defect(coarse: PoseState, coordinates: np.ndarray, *, length_scale: float, joint_scale: np.ndarray) -> PoseState:
    correction = coordinates.reshape(len(coordinates), 32, 12)
    position = coarse.position[:, 1:].numpy() + correction[..., :3] * length_scale
    rotation_delta = so3_exp(torch.from_numpy(correction[..., 3:6]))
    rotation = rotation_delta @ coarse.rotation[:, 1:]
    joints = coarse.joint_position[:, 1:].numpy() + correction[..., 6:] * joint_scale[None, None]
    return PoseState(
        torch.cat((coarse.rotation[:, :1], rotation), dim=1),
        torch.from_numpy(
            np.concatenate((coarse.position[:, :1].numpy(), position), axis=1)
        ),
        torch.from_numpy(
            np.concatenate((coarse.joint_position[:, :1].numpy(), joints), axis=1).astype(np.float32)
        ),
    )


def _flat_profile(data: dict[str, np.ndarray], metric: str) -> np.ndarray:
    value = _profile_embedding(data["descriptor"], metric)
    return np.ascontiguousarray(value.reshape(len(value), -1), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile-metric", default="hinge_100")
    parser.add_argument("--coarse-weights", default="0,0.1,0.25,0.5,1,2")
    parser.add_argument("--shape-weights", default="0,0.1,0.25")
    parser.add_argument("--bandwidth-factors", default="0.25,0.5,1")
    parser.add_argument("--regularizations", default="0.001,0.01,0.1,1")
    args = parser.parse_args()
    coarse_weights = tuple(float(value) for value in args.coarse_weights.split(","))
    shape_weights = tuple(float(value) for value in args.shape_weights.split(","))
    bandwidth_factors = tuple(float(value) for value in args.bandwidth_factors.split(","))
    regularizations = tuple(float(value) for value in args.regularizations.split(","))
    device = torch.device(args.device)
    config = replace(ExperimentConfig.load(args.config), device=str(device))
    manifest = DatasetManifest.load(config.paths.manifest)
    model = _build_model(config, manifest, device)
    saved = load_checkpoint(args.checkpoint, model=model, map_location=device)
    if saved["manifest_sha256"] != manifest.sha256():
        raise ValueError("checkpoint manifest mismatch")
    model.eval()
    data = {
        split: _load_split(manifest, model, split, device)
        for split in ("train", "val", "test")
    }
    coarse = {
        split: _rollout_split(model, config, manifest, data[split], split, device)
        for split in ("train", "val", "test")
    }
    model = model.to("cpu")
    joint_scale = model.joint_travel_range.detach().numpy().astype(np.float32)
    targets = {split: _target_state(data[split]) for split in ("train", "val", "test")}
    train_defect = _defect_coordinates(
        targets["train"], coarse["train"], length_scale=manifest.length_scale_m, joint_scale=joint_scale
    )
    defect_mean = train_defect.mean(axis=0, keepdims=True)
    centered_defect = torch.from_numpy(train_defect - defect_mean).to(device)
    profile = {
        split: _flat_profile(data[split], args.profile_metric)
        for split in ("train", "val", "test")
    }
    coarse_coordinate = {
        split: _coarse_coordinates(
            coarse[split], length_scale=manifest.length_scale_m, joint_scale=joint_scale
        )
        for split in ("train", "val", "test")
    }
    profile_train_distance = _squared_distance(profile["train"], profile["train"], device)
    profile_val_distance = _squared_distance(profile["val"], profile["train"], device)
    profile_off_diagonal = profile_train_distance[
        ~torch.eye(len(profile_train_distance), dtype=torch.bool, device=device)
    ]
    profile_median = float(torch.sqrt(profile_off_diagonal.median()).cpu())
    coarse_train_distance = _squared_distance(
        coarse_coordinate["train"], coarse_coordinate["train"], device
    )
    coarse_val_distance = _squared_distance(
        coarse_coordinate["val"], coarse_coordinate["train"], device
    )
    coarse_nonzero = coarse_train_distance[coarse_train_distance > 0.0]
    coarse_median = float(torch.sqrt(coarse_nonzero.median()).cpu())
    shape_distance = _global_shape_distances(manifest, data, device)
    validation = []
    for coarse_weight in coarse_weights:
        coarse_scale = (coarse_weight * profile_median / coarse_median) ** 2
        for shape_weight in shape_weights:
            shape_scale = (shape_weight * profile_median) ** 2
            train_distance = (
                profile_train_distance
                + coarse_scale * coarse_train_distance
                + shape_scale * shape_distance["train"]
            )
            val_distance = (
                profile_val_distance
                + coarse_scale * coarse_val_distance
                + shape_scale * shape_distance["val"]
            )
            for factor in bandwidth_factors:
                bandwidth = factor * profile_median
                train_kernel = torch.exp(-train_distance / (2.0 * bandwidth**2))
                val_kernel = torch.exp(-val_distance / (2.0 * bandwidth**2))
                for regularization in regularizations:
                    coefficients = _solve(train_kernel, centered_defect, regularization)
                    correction = (val_kernel @ coefficients).cpu().numpy() + defect_mean
                    prediction = _apply_defect(
                        coarse["val"], correction.astype(np.float32),
                        length_scale=manifest.length_scale_m, joint_scale=joint_scale,
                    )
                    values = _metrics(
                        prediction, data["val"], length_scale=manifest.length_scale_m, joint_scale=joint_scale
                    )
                    validation.append({
                        "coarse_weight": coarse_weight,
                        "shape_weight": shape_weight,
                        "bandwidth_factor": factor,
                        "bandwidth": bandwidth,
                        "regularization": regularization,
                        "metrics": values,
                    })
            best = min(validation, key=lambda value: value["metrics"]["terminal_dx"])
            print(
                f"[DEFECT] coarse={coarse_weight:g} shape={shape_weight:g} "
                f"best_val={best['metrics']['terminal_dx']:.8f}", flush=True
            )
    selected = min(validation, key=lambda value: value["metrics"]["terminal_dx"])
    coarse_test_distance = _squared_distance(
        coarse_coordinate["test"], coarse_coordinate["train"], device
    )
    profile_test_distance = _squared_distance(profile["test"], profile["train"], device)
    coarse_scale = (selected["coarse_weight"] * profile_median / coarse_median) ** 2
    shape_scale = (selected["shape_weight"] * profile_median) ** 2
    train_distance = (
        profile_train_distance + coarse_scale * coarse_train_distance + shape_scale * shape_distance["train"]
    )
    test_distance = (
        profile_test_distance + coarse_scale * coarse_test_distance + shape_scale * shape_distance["test"]
    )
    bandwidth = selected["bandwidth"]
    train_kernel = torch.exp(-train_distance / (2.0 * bandwidth**2))
    test_kernel = torch.exp(-test_distance / (2.0 * bandwidth**2))
    coefficients = _solve(train_kernel, centered_defect, selected["regularization"])
    test_correction = (test_kernel @ coefficients).cpu().numpy() + defect_mean
    test_prediction = _apply_defect(
        coarse["test"], test_correction.astype(np.float32),
        length_scale=manifest.length_scale_m, joint_scale=joint_scale,
    )
    production = {
        split: _metrics(
            coarse[split], data[split], length_scale=manifest.length_scale_m, joint_scale=joint_scale
        )
        for split in ("val", "test")
    }
    test = _metrics(
        test_prediction, data["test"], length_scale=manifest.length_scale_m, joint_scale=joint_scale
    )
    result = {
        "definition": {
            "coarse_operator": "frozen production recurrent SRNO",
            "learned_object": "complete product-manifold defect from coarse rollout to physical solution path",
            "correction_feedback": False,
            "test_used_for_selection": False,
        },
        "contract": {
            "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "manifest_sha256": manifest.sha256(),
            "profile_metric": args.profile_metric,
            "validation_grid_size": len(validation),
        },
        "production": production,
        "selected": selected,
        "test": test,
        "test_objectwise": _objectwise_metrics(test_prediction, data["test"], manifest, joint_scale),
        "validation_grid": validation,
    }
    _write_json(args.output, result)
    print(json.dumps({"production": production, "selected": selected, "test": test}, indent=2), flush=True)


if __name__ == "__main__":
    main()
