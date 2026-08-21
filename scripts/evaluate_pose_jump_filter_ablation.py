#!/usr/bin/env python3
"""Compare full-trained and jump-filtered local models on matched subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from run_jq_local_ablation import SPLITS, _evaluate_split
from srno.data.index import ActiveIndex, file_sha256
from srno.data.schema import DatasetManifest
from srno.training.config import ExperimentConfig


METRICS = (
    "dx",
    "pose",
    "translation_over_length",
    "translation_m",
    "rotation_rad",
    "joint_rmse_over_travel",
)


def _enrich(arrays: dict[str, np.ndarray]) -> None:
    arrays["pose"] = np.sqrt(
        arrays["translation_over_length"] ** 2 + arrays["rotation_rad"] ** 2
    )


def _summary(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    labels = [str(value) for value in arrays["object_labels"].tolist()]
    by_object: dict[str, dict[str, float]] = {}
    for index, label in enumerate(labels):
        selected = mask & (arrays["object_index"] == index)
        if not np.any(selected):
            continue
        by_object[label] = {
            metric: float(np.mean(arrays[metric][selected])) for metric in METRICS
        }
    if not by_object:
        raise ValueError("evaluation subset is empty")
    return {
        "sample_count": int(mask.sum()),
        "sample_weighted": {
            metric: float(np.mean(arrays[metric][mask])) for metric in METRICS
        },
        "equal_object": {
            metric: float(
                np.mean([values[metric] for values in by_object.values()])
            )
            for metric in METRICS
        },
        "by_object": by_object,
    }


def _smooth_mask(
    arrays: dict[str, np.ndarray], filtered: ActiveIndex
) -> np.ndarray:
    labels = [str(value) for value in arrays["object_labels"].tolist()]
    retained = {
        object_id: {tuple(map(int, pair)) for pair in filtered.pairs_for(object_id)}
        for object_id in labels
    }
    return np.asarray(
        [
            (int(trajectory), int(step)) in retained[labels[int(object_index)]]
            for object_index, trajectory, step in zip(
                arrays["object_index"],
                arrays["trajectory_index"],
                arrays["step_index"],
                strict=True,
            )
        ],
        dtype=np.bool_,
    )


def _best_training_record(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    local = [record for record in records if record["stage"] == "local"]
    best = min(local, key=lambda record: float(record["val"]["dx"]))
    return {
        "best_epoch": int(best["epoch"]),
        "best_validation_dx_during_training": float(best["val"]["dx"]),
        "epochs_completed": len(local),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-config", type=Path, required=True)
    parser.add_argument("--filtered-config", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--filtered-checkpoint", type=Path, required=True)
    parser.add_argument("--history-config", type=Path)
    parser.add_argument("--history-checkpoint", type=Path)
    parser.add_argument("--filter-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    full_config = ExperimentConfig.load(args.full_config)
    filtered_config = ExperimentConfig.load(args.filtered_config)
    if full_config.model != filtered_config.model:
        raise ValueError("model contracts differ")
    if full_config.loss != filtered_config.loss:
        raise ValueError("loss contracts differ")
    if (args.history_config is None) != (args.history_checkpoint is None):
        raise ValueError("history config and checkpoint must be supplied together")
    manifest = DatasetManifest.load(full_config.paths.manifest)
    filtered_index = ActiveIndex.load(filtered_config.paths.active_index)

    model_specs = {
        "full_trained": (full_config, args.baseline_checkpoint),
        "smooth_trained": (full_config, args.filtered_checkpoint),
    }
    if args.history_config is not None and args.history_checkpoint is not None:
        model_specs["history_reference"] = (
            ExperimentConfig.load(args.history_config),
            args.history_checkpoint,
        )
    evaluations: dict[str, dict[str, Any]] = {}
    for model_name, (evaluation_config, checkpoint) in model_specs.items():
        evaluations[model_name] = {}
        for split in SPLITS:
            _, arrays = _evaluate_split(evaluation_config, checkpoint, split)
            _enrich(arrays)
            smooth = _smooth_mask(arrays, filtered_index)
            evaluations[model_name][split] = {
                "full": _summary(arrays, np.ones(len(smooth), dtype=np.bool_)),
                "smooth": _summary(arrays, smooth),
                "jump": _summary(arrays, ~smooth),
            }
            print(
                f"[JUMP-FILTER] {model_name} {split}: "
                f"full={evaluations[model_name][split]['full']['equal_object']['dx']:.8f} "
                f"smooth={evaluations[model_name][split]['smooth']['equal_object']['dx']:.8f} "
                f"jump={evaluations[model_name][split]['jump']['equal_object']['dx']:.8f}",
                flush=True,
            )

    comparison: dict[str, Any] = {}
    for split in SPLITS:
        comparison[split] = {}
        for subset in ("full", "smooth", "jump"):
            baseline = evaluations["full_trained"][split][subset]["equal_object"]
            candidate = evaluations["smooth_trained"][split][subset]["equal_object"]
            comparison[split][subset] = {
                metric: {
                    "delta": float(candidate[metric] - baseline[metric]),
                    "change_percent": float(
                        100.0 * (candidate[metric] / baseline[metric] - 1.0)
                    ),
                }
                for metric in METRICS
            }

    payload = {
        "format_version": 1,
        "rollout_performed": False,
        "manifest": str(full_config.paths.manifest),
        "manifest_sha256": manifest.sha256(),
        "full_active_index": str(full_config.paths.active_index),
        "full_active_index_sha256": file_sha256(full_config.paths.active_index),
        "filtered_active_index": str(filtered_config.paths.active_index),
        "filtered_active_index_sha256": file_sha256(
            filtered_config.paths.active_index
        ),
        "filter_contract": json.loads(
            args.filter_contract.read_text(encoding="utf-8")
        ),
        "models": {
            name: {
                "checkpoint": str(path.resolve()),
                "checkpoint_sha256": file_sha256(path),
                "training": _best_training_record(
                    path.parent / "metrics.jsonl"
                ),
            }
            for name, (config, path) in model_specs.items()
        },
        "evaluation": evaluations,
        "comparison_smooth_trained_vs_full_trained": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
