#!/usr/bin/env python3
"""Consolidate the local learned-contact-cone architecture ablation."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from run_jq_local_ablation import _evaluate_split
from srno.data.index import file_sha256
from srno.training.config import ExperimentConfig


ARMS = {
    "direct_L1": (
        "configs/srno-r-material-v2.toml",
        "runs/ablation-operator-depth-local/L1/best-local.pt",
    ),
    "normal_cone": (
        "configs/ablation-normal-cone-local.toml",
        "runs/ablation-normal-cone-local/best-local.pt",
    ),
    "normal_cone_history": (
        "configs/ablation-normal-cone-history-local.toml",
        "runs/ablation-normal-cone-history-local/best-local.pt",
    ),
    "normal_cone_balanced": (
        "configs/ablation-normal-cone-balanced-local.toml",
        "runs/ablation-normal-cone-balanced-local/best-local.pt",
    ),
    "normal_cone_full_state_balanced": (
        "configs/ablation-normal-cone-state-balanced-local.toml",
        "runs/ablation-normal-cone-state-balanced-local/best-local.pt",
    ),
    "friction_cone_full_state_balanced": (
        "configs/ablation-friction-cone-state-balanced-local.toml",
        "runs/ablation-friction-cone-state-balanced-local/best-local.pt",
    ),
    "continuation_huber_balanced": (
        "configs/ablation-normal-cone-continuation-huber-local.toml",
        "runs/ablation-normal-cone-continuation-huber-local/best-local.pt",
    ),
    # Parameter-compatible view of the trained normal-cone checkpoint: q uses
    # the validation-selected causal continuation branch and r uses the cone.
    "split_resolvent": (
        "configs/ablation-normal-cone-split-resolvent-local.toml",
        "runs/ablation-normal-cone-local/best-local.pt",
    ),
}


def _training_summary(checkpoint: Path) -> dict[str, Any] | None:
    path = checkpoint.parent / "metrics.jsonl"
    if not path.exists():
        return None
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    local = [row for row in rows if row.get("stage") == "local"]
    if not local:
        return None
    best_loss = min(local, key=lambda row: float(row["val"]["loss"]))
    best_dx = min(local, key=lambda row: float(row["val"]["dx"]))
    return {
        "epochs_completed": len(local),
        "best_loss_epoch": int(best_loss["epoch"]),
        "best_validation_loss": float(best_loss["val"]["loss"]),
        "best_dx_epoch": int(best_dx["epoch"]),
        "best_validation_dx": float(best_dx["val"]["dx"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/ablation-contact-resolvent-v1/results.json"),
    )
    args = parser.parse_args()
    evaluations: dict[str, Any] = {}
    for name, (config_name, checkpoint_name) in ARMS.items():
        config_path = Path(config_name)
        checkpoint = Path(checkpoint_name)
        config = ExperimentConfig.load(config_path)
        # Consolidation is deliberately CPU-only so that replaying finished
        # checkpoints does not depend on the workstation CUDA runtime.
        config = dataclasses.replace(config, device="cpu")
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
                f"[CONTACT-RESOLVENT] {name} {split}: "
                f"dX={summary['equal_object']['dx']:.8f}",
                flush=True,
            )

    baseline = evaluations["direct_L1"]
    comparison: dict[str, Any] = {}
    metrics = (
        "dx",
        "translation_over_length",
        "translation_m",
        "rotation_rad",
        "joint_rmse_over_travel",
    )
    for name, arm in evaluations.items():
        if name == "direct_L1":
            continue
        comparison[name] = {}
        for split in ("val", "test"):
            reference = baseline["evaluation"][split]["equal_object"]
            candidate = arm["evaluation"][split]["equal_object"]
            comparison[name][split] = {
                metric: {
                    "delta": float(candidate[metric] - reference[metric]),
                    "change_percent": float(
                        100.0 * (candidate[metric] / reference[metric] - 1.0)
                    ),
                }
                for metric in metrics
            }
    output = {
        "format_version": 1,
        "scope": "local active transitions only; no rollout performed",
        "selection": {
            "continuation_factor": 0.75,
            "selected_on": "validation mean d_X grid {0, .25, .5, .75, 1, 1.25}",
            "test_was_not_used_for_selection": True,
        },
        "arms": evaluations,
        "comparison_vs_direct_L1": comparison,
        "diagnostics": {
            "linearized_projection": "reports/linearized-contact-projection-test.json",
            "nonlinear_prox_val": "reports/nonlinear-contact-prox-val.json",
            "nonlinear_prox_test": "reports/nonlinear-contact-prox-test.json",
            "contact_cone_oracle": "reports/contact-cone-representability-test.json",
            "gradient_scales": "runs/ablation-normal-cone-local/gradient-scales.json",
            "continuation_screen": "runs/ablation-normal-cone-local/continuation-hybrid.json",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison["split_resolvent"]["test"], indent=2))


if __name__ == "__main__":
    main()
