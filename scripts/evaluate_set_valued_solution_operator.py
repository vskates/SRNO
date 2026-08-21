#!/usr/bin/env python3
"""Evaluate a geometry-conditioned set-valued solution-path operator.

The predictor returns a finite set of complete paths; target states are used
only to evaluate point-to-set distance, exactly as a graph/set-valued operator
metric requires.  Validation fixes the profile metric and the smallest set
budget reaching the predeclared 50% production-risk target.  Matched random
sets prevent a vacuous best-of-many interpretation.
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

from evaluate_branch_defect_selector import _candidate_paths
from evaluate_loading_profile_kernel_operator import (
    DEFAULT_PROFILE_METRICS,
    _load_split,
    _metrics,
    _nearest,
    _predict,
    _profile_embedding,
)
from evaluate_rollout_path_projection import _rollout_split
from run_branch_energy_operator import _candidate_terminal_error
from run_rkhs_solution_operator import _objectwise_metrics
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import rotation_geodesic_angle
from srno.training.checkpoint import load_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.types import PoseState


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _error_tensor(
    candidates: PoseState,
    target: dict[str, np.ndarray],
    *,
    length_scale: float,
    joint_scale: np.ndarray,
) -> dict[str, torch.Tensor]:
    target_position = torch.from_numpy(target["position"])[:, None]
    target_rotation = torch.from_numpy(target["rotation"])[:, None]
    target_joint = torch.from_numpy(target["joint"])[:, None]
    translation = torch.linalg.vector_norm(
        candidates.position - target_position, dim=-1
    ) / length_scale
    rotation = rotation_geodesic_angle(
        candidates.rotation,
        target_rotation.expand_as(candidates.rotation),
    )
    joints = torch.sqrt(
        (((candidates.joint_position - target_joint) / torch.from_numpy(joint_scale)).square()).mean(dim=-1)
    )
    dx = torch.sqrt(translation.square() + rotation.square() + joints.square())
    return {
        "dx": dx,
        "translation_over_length": translation,
        "rotation_rad": rotation,
        "joints": joints,
    }


def _gather_path(candidates: PoseState, selected: torch.Tensor) -> PoseState:
    rows = torch.arange(candidates.position.shape[0])
    return PoseState(
        candidates.rotation[rows, selected],
        candidates.position[rows, selected],
        candidates.joint_position[rows, selected],
    )


def _slice_candidates(candidates: PoseState, cardinality: int) -> PoseState:
    return PoseState(
        candidates.rotation[:, :cardinality],
        candidates.position[:, :cardinality],
        candidates.joint_position[:, :cardinality],
    )


def _terminal_error(
    prediction: PoseState,
    target: dict[str, np.ndarray],
    *,
    length_scale: float,
    joint_scale: np.ndarray,
) -> np.ndarray:
    candidate = PoseState(
        prediction.rotation[:, None],
        prediction.position[:, None],
        prediction.joint_position[:, None],
    )
    return _error_tensor(
        candidate, target, length_scale=length_scale, joint_scale=joint_scale
    )["dx"][:, 0, -1].numpy()


def _coverage(error: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(error.mean()),
        "median": float(np.median(error)),
        "q90": float(np.quantile(error, 0.9)),
        "coverage_at_0p05": float((error <= 0.05).mean()),
        "coverage_at_0p10": float((error <= 0.10).mean()),
        "coverage_at_0p20": float((error <= 0.20).mean()),
    }


def _paired_bootstrap(
    set_error: np.ndarray,
    production_error: np.ndarray,
    object_ids: np.ndarray,
    *,
    samples: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    objects = np.asarray(list(dict.fromkeys(object_ids.tolist())))
    improvements = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        selected_objects = rng.choice(objects, size=len(objects), replace=True)
        selected_rows = []
        for object_id in selected_objects:
            rows = np.flatnonzero(object_ids == object_id)
            selected_rows.append(rng.choice(rows, size=len(rows), replace=True))
        rows = np.concatenate(selected_rows)
        improvements[draw] = 1.0 - set_error[rows].mean() / production_error[rows].mean()
    return {
        "mean_improvement_fraction": float(improvements.mean()),
        "ci95_low": float(np.quantile(improvements, 0.025)),
        "ci95_high": float(np.quantile(improvements, 0.975)),
        "probability_improvement_gt_50_percent": float((improvements > 0.5).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-cardinality", type=int, default=64)
    parser.add_argument("--cardinalities", default="1,2,4,8,16,32,64")
    parser.add_argument("--random-repeats", type=int, default=32)
    args = parser.parse_args()
    cardinalities = tuple(sorted({int(value) for value in args.cardinalities.split(",")}))
    if max(cardinalities) > args.maximum_cardinality:
        parser.error("cardinalities cannot exceed maximum cardinality")
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
    production = {
        split: _rollout_split(model, config, manifest, data[split], split, device)
        for split in ("val", "test")
    }
    joint_scale = model.joint_travel_range.detach().cpu().numpy().astype(np.float32)
    production_metrics = {
        split: _metrics(
            production[split], data[split], length_scale=manifest.length_scale_m, joint_scale=joint_scale
        )
        for split in ("val", "test")
    }
    validation = []
    val_nearest = {}
    for profile_metric in DEFAULT_PROFILE_METRICS:
        neighbours, distances = _nearest(
            _profile_embedding(data["train"]["descriptor"], profile_metric),
            _profile_embedding(data["val"]["descriptor"], profile_metric),
            maximum_k=args.maximum_cardinality,
            device=device,
            chunk_size=32,
        )
        val_nearest[profile_metric] = (neighbours, distances)
        error = _candidate_terminal_error(
            data["train"], data["val"], neighbours,
            length_scale=manifest.length_scale_m, joint_scale=joint_scale,
        )
        for cardinality in cardinalities:
            validation.append({
                "profile_metric": profile_metric,
                "cardinality": cardinality,
                "terminal_point_to_set_dx": float(error[:, :cardinality].min(axis=1).mean()),
            })
    target = 0.5 * production_metrics["val"]["terminal_dx"]
    feasible = [value for value in validation if value["terminal_point_to_set_dx"] <= target]
    if feasible:
        selected = min(
            feasible,
            key=lambda value: (value["cardinality"], value["terminal_point_to_set_dx"]),
        )
    else:
        selected = min(validation, key=lambda value: value["terminal_point_to_set_dx"])
    profile_metric = selected["profile_metric"]
    cardinality = selected["cardinality"]
    test_neighbours, test_distances = _nearest(
        _profile_embedding(data["train"]["descriptor"], profile_metric),
        _profile_embedding(data["test"]["descriptor"], profile_metric),
        maximum_k=args.maximum_cardinality,
        device=device,
        chunk_size=32,
    )
    all_candidates = _candidate_paths(
        data["train"], data["test"], test_neighbours[:, : args.maximum_cardinality]
    )
    all_errors = _error_tensor(
        all_candidates, data["test"], length_scale=manifest.length_scale_m, joint_scale=joint_scale
    )
    candidates = _slice_candidates(all_candidates, cardinality)
    errors = _error_tensor(
        candidates, data["test"], length_scale=manifest.length_scale_m, joint_scale=joint_scale
    )
    terminal_selected = errors["dx"][:, :, -1].argmin(dim=1)
    terminal_path = _gather_path(candidates, terminal_selected)
    terminal_set_error = errors["dx"][:, :, -1].amin(dim=1).numpy()
    path_selected = errors["dx"][:, :, 1:].mean(dim=-1).argmin(dim=1)
    coherent_path = _gather_path(candidates, path_selected)
    stepwise_set_error = errors["dx"][:, :, 1:].amin(dim=1).mean(dim=0).numpy()
    maximum_terminal_selected = all_errors["dx"][:, :, -1].argmin(dim=1)
    maximum_terminal_path = _gather_path(all_candidates, maximum_terminal_selected)
    maximum_terminal_error = all_errors["dx"][:, :, -1].amin(dim=1).numpy()
    maximum_path_selected = all_errors["dx"][:, :, 1:].mean(dim=-1).argmin(dim=1)
    maximum_coherent_path = _gather_path(all_candidates, maximum_path_selected)

    rng = np.random.default_rng(0)
    random_curves = {str(value): [] for value in cardinalities}
    for repeat in range(args.random_repeats):
        random_neighbours = rng.integers(
            0, len(data["train"]["position"]),
            size=(len(data["test"]["position"]), args.maximum_cardinality),
        )
        random_error = _candidate_terminal_error(
            data["train"], data["test"], random_neighbours,
            length_scale=manifest.length_scale_m, joint_scale=joint_scale,
        )
        for value in cardinalities:
            random_curves[str(value)].append(
                float(random_error[:, :value].min(axis=1).mean())
            )
    selected_random = np.asarray(random_curves[str(cardinality)])
    production_error = _terminal_error(
        production["test"], data["test"], length_scale=manifest.length_scale_m, joint_scale=joint_scale
    )
    result = {
        "definition": {
            "operator": "finite set of complete physical train paths nearest in a loading-profile metric",
            "evaluation": "point-to-set distance; target is never used to construct the predicted set",
            "selection": "validation chooses the smallest cardinality reaching half the production terminal risk",
            "test_used_for_selection": False,
        },
        "contract": {
            "manifest_sha256": manifest.sha256(),
            "checkpoint": str(args.checkpoint.resolve()),
            "maximum_cardinality": args.maximum_cardinality,
            "cardinality_grid": cardinalities,
            "random_control_repeats": args.random_repeats,
        },
        "production": production_metrics,
        "production_objectwise": _objectwise_metrics(
            production["test"], data["test"], manifest, joint_scale
        ),
        "validation_target_half_production": target,
        "selected": selected,
        "test_terminal_point_to_set": {
            "metrics_of_selected_member": _metrics(
                terminal_path, data["test"], length_scale=manifest.length_scale_m, joint_scale=joint_scale
            ),
            "coverage": _coverage(terminal_set_error),
            "objectwise": _objectwise_metrics(
                terminal_path, data["test"], manifest, joint_scale
            ),
            "paired_hierarchical_bootstrap_vs_production": _paired_bootstrap(
                terminal_set_error, production_error, data["test"]["object_id"]
            ),
        },
        "test_coherent_path_point_to_set": {
            "metrics_of_selected_member": _metrics(
                coherent_path, data["test"], length_scale=manifest.length_scale_m, joint_scale=joint_scale
            )
        },
        "test_stepwise_point_to_set_dx": stepwise_set_error.tolist(),
        "maximum_cardinality_diagnostic": {
            "cardinality": args.maximum_cardinality,
            "terminal_metrics_of_selected_member": _metrics(
                maximum_terminal_path, data["test"], length_scale=manifest.length_scale_m, joint_scale=joint_scale
            ),
            "terminal_coverage": _coverage(maximum_terminal_error),
            "terminal_bootstrap_vs_production": _paired_bootstrap(
                maximum_terminal_error, production_error, data["test"]["object_id"]
            ),
            "coherent_path_metrics_of_selected_member": _metrics(
                maximum_coherent_path, data["test"], length_scale=manifest.length_scale_m, joint_scale=joint_scale
            ),
        },
        "matched_random_set_control": {
            "mean_terminal_dx": float(selected_random.mean()),
            "std_over_repeats": float(selected_random.std(ddof=1)),
            "nearest_set_improvement_fraction": float(
                1.0 - terminal_set_error.mean() / selected_random.mean()
            ),
            "cardinality_curve_mean": {
                key: float(np.mean(value)) for key, value in random_curves.items()
            },
        },
        "validation_nearest_set_cardinality_curve": {
            str(value): next(
                row["terminal_point_to_set_dx"]
                for row in validation
                if row["profile_metric"] == profile_metric and row["cardinality"] == value
            )
            for value in cardinalities
        },
        "test_nearest_set_cardinality_curve": {
            str(value): float(
                all_errors["dx"][:, :value, -1].amin(dim=1).mean()
            )
            for value in cardinalities
        },
        "validation_grid": validation,
    }
    _write_json(args.output, result)
    print(json.dumps({
        "selected": selected,
        "production_test": production_metrics["test"],
        "terminal_set": result["test_terminal_point_to_set"],
        "coherent_path": result["test_coherent_path_point_to_set"],
        "random_control": result["matched_random_set_control"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
