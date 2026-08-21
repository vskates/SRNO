#!/usr/bin/env python3
"""Consolidate local screens of general monotone-resolvent hypotheses."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from run_jq_local_ablation import _evaluate_split
from srno.data.index import file_sha256
from srno.training.config import ExperimentConfig


ARMS = {
    "direct_L1": (
        Path("configs/srno-r-material-v2.toml"),
        Path("runs/ablation-operator-depth-local/L1/best-local.pt"),
    ),
    "implicit_contact_QP_history_query": (
        Path("configs/ablation-inertial-implicit-resolvent-local.toml"),
        Path("runs/ablation-inertial-implicit-resolvent-local/best-local.pt"),
    ),
    "convex_dual_zero_pose_query": (
        Path("configs/ablation-dual-potential-mixed-local.toml"),
        Path("runs/ablation-dual-potential-mixed-local/best-local.pt"),
    ),
    "noncyclic_monotone_zero_pose_query": (
        Path("configs/ablation-monotone-resolvent-local.toml"),
        Path("runs/ablation-monotone-resolvent-local/best-local.pt"),
    ),
    "convex_dual_history_query": (
        Path("configs/ablation-dual-potential-history-query-local.toml"),
        Path("runs/ablation-dual-potential-history-query-local/best-local.pt"),
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
        "best_loss_epoch": int(best_loss["epoch"]),
        "best_validation_loss": float(best_loss["val"]["loss"]),
        "best_dx_epoch": int(best_dx["epoch"]),
        "best_validation_dx_during_training": float(best_dx["val"]["dx"]),
    }


def _changes(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, Any]:
    return {
        metric: {
            "delta": float(candidate[metric] - reference[metric]),
            "change_percent": float(
                100.0 * (candidate[metric] / reference[metric] - 1.0)
            ),
        }
        for metric in METRICS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/ablation-general-resolvent-v1/results.json"),
    )
    args = parser.parse_args()

    evaluations: dict[str, Any] = {}
    for name, (config_path, checkpoint) in ARMS.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        config = replace(ExperimentConfig.load(config_path), device="cpu")
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
                f"[GENERAL-RESOLVENT] {name} {split}: "
                f"dX={summary['equal_object']['dx']:.8f}",
                flush=True,
            )

    baseline = evaluations["direct_L1"]
    comparisons = {
        name: {
            split: _changes(
                arm["evaluation"][split]["equal_object"],
                baseline["evaluation"][split]["equal_object"],
            )
            for split in ("val", "test")
        }
        for name, arm in evaluations.items()
        if name != "direct_L1"
    }
    output = {
        "format_version": 1,
        "scope": "local active transitions only; no rollout performed",
        "test_not_used_for_model_selection": True,
        "arms": evaluations,
        "comparisons_vs_direct_L1": comparisons,
        "interpretation_contract": {
            "convex_dual": (
                "gradient of a conditional convex 1-smooth potential; exact "
                "firmly-nonexpansive resolvent for fixed context"
            ),
            "noncyclic_monotone": (
                "resolvent of S+W with S positive semidefinite and W skew; "
                "monotone but not restricted to a scalar potential"
            ),
            "scientific_status": (
                "architecture screens, not a claim that Coulomb friction is monotone"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
