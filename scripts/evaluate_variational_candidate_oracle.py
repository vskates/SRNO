#!/usr/bin/env python3
"""Measure whether a branch-selecting variational model has useful headroom.

This is an oracle diagnostic, not a model: for every transition it selects the
candidate with the smallest aggregate state error.  A small oracle gain rejects
a mixture/energy-selector hypothesis before training; a large gain only shows
representational headroom and does not establish that the branch is observable.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from run_jq_local_ablation import _evaluate_split, _transition_keys
from srno.training.config import ExperimentConfig


ARMS = {
    "direct": (
        Path("configs/srno-r-material-v2.toml"),
        Path("runs/ablation-operator-depth-local/L1/best-local.pt"),
    ),
    "split_continuation": (
        Path("configs/ablation-normal-cone-split-resolvent-local.toml"),
        Path("runs/ablation-normal-cone-local/best-local.pt"),
    ),
    "inertial_implicit": (
        Path("configs/ablation-inertial-implicit-resolvent-local.toml"),
        Path("runs/ablation-inertial-implicit-resolvent-local/best-local.pt"),
    ),
}
METRICS = (
    "dx",
    "translation_over_length",
    "translation_m",
    "rotation_rad",
    "joint_rmse_over_travel",
)


def _means(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    labels = arrays["object_labels"].tolist()
    return {
        metric: float(
            np.mean(
                [
                    arrays[metric][arrays["object_index"] == object_index].mean()
                    for object_index in range(len(labels))
                ]
            )
        )
        for metric in METRICS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/variational-candidate-oracle-val.json"),
    )
    args = parser.parse_args()

    evaluations: dict[str, dict[str, np.ndarray]] = {}
    summaries: dict[str, dict[str, float]] = {}
    reference_keys = None
    for name, (config_path, checkpoint) in ARMS.items():
        config = replace(ExperimentConfig.load(config_path), device="cpu")
        _, arrays = _evaluate_split(config, checkpoint, args.split)
        keys = _transition_keys(arrays)
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise RuntimeError(f"transition ordering differs for {name}")
        evaluations[name] = arrays
        summaries[name] = _means(arrays)
        print(f"[CANDIDATE] {name}: dX={summaries[name]['dx']:.8f}", flush=True)

    names = tuple(evaluations)
    stacked_dx = np.stack([evaluations[name]["dx"] for name in names], axis=0)
    choice = stacked_dx.argmin(axis=0)
    oracle = {
        key: value.copy()
        for key, value in evaluations[names[0]].items()
        if key not in METRICS
    }
    for metric in METRICS:
        values = np.stack([evaluations[name][metric] for name in names], axis=0)
        oracle[metric] = np.take_along_axis(values, choice[None], axis=0)[0]
    oracle_means = _means(oracle)
    best_name = min(names, key=lambda name: summaries[name]["dx"])
    best = summaries[best_name]
    output = {
        "split": args.split,
        "note": "target-error oracle; representational upper bound, not predictor",
        "arms": summaries,
        "selection_fraction": {
            name: float(np.mean(choice == index)) for index, name in enumerate(names)
        },
        "oracle": oracle_means,
        "best_single_arm": best_name,
        "oracle_change_from_best_percent": {
            metric: float(100.0 * (oracle_means[metric] / best[metric] - 1.0))
            for metric in METRICS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
