#!/usr/bin/env python3
"""Consolidate the true current-state contact-resolvent local ablation."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from run_jq_local_ablation import _evaluate_split
from srno.data.index import file_sha256
from srno.training.config import ExperimentConfig


NEW_ARMS = {
    "implicit_resolvent": (
        Path("configs/ablation-implicit-resolvent-local.toml"),
        Path("runs/ablation-implicit-resolvent-local/best-local.pt"),
    ),
    "inertial_implicit_resolvent": (
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


def _training_summary(checkpoint: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (checkpoint.parent / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    best_loss = min(rows, key=lambda row: float(row["val"]["loss"]))
    best_dx = min(rows, key=lambda row: float(row["val"]["dx"]))
    return {
        "epochs_completed": len(rows),
        "stopped_as_short_screen": True,
        "best_loss_epoch": int(best_loss["epoch"]),
        "best_validation_loss": float(best_loss["val"]["loss"]),
        "best_dx_epoch": int(best_dx["epoch"]),
        "best_validation_dx": float(best_dx["val"]["dx"]),
    }


def _changes(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, Any]:
    return {
        metric: {
            "delta": float(candidate[metric] - reference[metric]),
            "change_percent": float(100.0 * (candidate[metric] / reference[metric] - 1.0)),
        }
        for metric in METRICS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/ablation-implicit-resolvent-v1/results.json"),
    )
    args = parser.parse_args()

    previous = json.loads(
        Path("runs/ablation-contact-resolvent-v1/results.json").read_text(
            encoding="utf-8"
        )
    )
    evaluations = {
        name: previous["arms"][name]
        for name in ("direct_L1", "split_resolvent")
    }
    for name, (config_path, checkpoint) in NEW_ARMS.items():
        config = dataclasses.replace(
            ExperimentConfig.load(config_path), device="cpu"
        )
        evaluations[name] = {
            "config": str(config_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "training": _training_summary(checkpoint),
            "evaluation": {},
        }
        for split in ("val", "test"):
            summary, _ = _evaluate_split(config, checkpoint, split)
            evaluations[name]["evaluation"][split] = summary
            print(
                f"[IMPLICIT-RESOLVENT] {name} {split}: "
                f"dX={summary['equal_object']['dx']:.8f}",
                flush=True,
            )

    comparisons: dict[str, Any] = {}
    for name in NEW_ARMS:
        comparisons[name] = {}
        for split in ("val", "test"):
            candidate = evaluations[name]["evaluation"][split]["equal_object"]
            comparisons[name][split] = {
                reference: _changes(
                    candidate,
                    evaluations[reference]["evaluation"][split]["equal_object"],
                )
                for reference in ("direct_L1", "split_resolvent")
            }
    comparisons["inertial_vs_stateless"] = {
        split: _changes(
            evaluations["inertial_implicit_resolvent"]["evaluation"][split][
                "equal_object"
            ],
            evaluations["implicit_resolvent"]["evaluation"][split]["equal_object"],
        )
        for split in ("val", "test")
    }
    output = {
        "format_version": 1,
        "scope": "local active transitions only; no rollout performed",
        "selection": {
            "linearization_state": "current observed state",
            "geometric_constraint_gap_m": 0.0,
            "pose_weight": 100.0,
            "solver_iterations": 128,
            "pose_query_factor": 0.5,
            "pose_query_selected_on": "validation exact-QP grid {0, 0.5, 0.75, 1}",
            "test_was_not_used_for_selection": True,
        },
        "arms": evaluations,
        "comparisons": comparisons,
        "diagnostics": {
            "normal_exact_qp": "reports/current-state-resolvent-val-gap0-decoupled.json",
            "friction_exact_qp": "reports/current-state-frictional-resolvent-val.json",
            "inertial_exact_qp": "reports/current-state-inertial-resolvent-val.json",
            "stateless_initialization": "reports/implicit-resolvent-initialization-val-iter128.json",
            "inertial_initialization": "reports/inertial-implicit-resolvent-initialization-val.json",
            "target_compatibility_val": "reports/resolvent-target-compatibility-val.json",
            "target_compatibility_test": "reports/resolvent-target-compatibility-test.json",
            "sequential_resolvent": "reports/sequential-current-resolvent-val.json",
            "relative_error": (
                "runs/ablation-inertial-implicit-resolvent-local/relative-error/"
                "local_pose_relative_error.json"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            comparisons["inertial_implicit_resolvent"]["test"]["direct_L1"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
