#!/usr/bin/env python3
"""Clean single-seed local ablation over residual operator depth L=1..4."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from run_jq_local_ablation import SPLITS, _evaluate_split
from srno.data.index import ActiveIndex
from srno.data.schema import DatasetManifest
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model, train


DEPTHS = (1, 2, 3, 4)
SEED = 0
COLORS = ("#4472C4", "#ED7D31", "#70AD47", "#A64D79")


def _run_config(
    base: ExperimentConfig, output: Path, depth: int
) -> ExperimentConfig:
    return replace(
        base,
        seed=SEED,
        paths=replace(base.paths, output_dir=output / f"L{depth}"),
        model=replace(base.model, operator_layers=depth),
    )


def _train_or_resume(config: ExperimentConfig) -> Path:
    output = config.paths.output_dir
    best = output / "best-local.pt"
    if best.is_file():
        print(f"[DEPTH] reuse completed {best}", flush=True)
        return best
    last = output / "last-local.pt"
    if last.is_file():
        print(f"[DEPTH] resume {last}", flush=True)
        return train(config, stage="local", resume=last)
    if output.exists() and (output / "metrics.jsonl").exists():
        raise RuntimeError(f"run has metrics but no resumable checkpoint: {output}")
    print(f"[DEPTH] train L={config.model.operator_layers} seed={SEED}", flush=True)
    return train(config, stage="local")


def _metrics(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _best_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    local = [record for record in records if record["stage"] == "local"]
    if not local:
        raise ValueError("local run contains no epoch metrics")
    return min(local, key=lambda record: float(record["val"]["dx"]))


def _parameter_count(config: ExperimentConfig) -> int:
    manifest = DatasetManifest.load(config.paths.manifest)
    model = _build_model(config, manifest, torch.device("cpu"))
    return sum(parameter.numel() for parameter in model.parameters())


def _active_counts(config: ExperimentConfig) -> dict[str, int]:
    manifest = DatasetManifest.load(config.paths.manifest)
    index = ActiveIndex.load(config.paths.active_index)
    return {
        split: int(
            sum(len(index.pairs_for(object_id)) for object_id in manifest.splits[split])
        )
        for split in SPLITS
    }


def _plot(
    output: Path,
    results: dict[str, Any],
    raw: dict[int, dict[str, dict[str, np.ndarray]]],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    ax = axes[0, 0]
    for depth, color in zip(DEPTHS, COLORS, strict=True):
        records = results["runs"][str(depth)]["training_curve"]
        ax.plot(
            [record["epoch"] for record in records],
            [record["val_dx"] for record in records],
            color=color,
            linewidth=1.8,
            label=f"L={depth}",
        )
    ax.set_title(r"Validation one-step $d_X$")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$d_X$")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    for depth, color in zip(DEPTHS, COLORS, strict=True):
        records = results["runs"][str(depth)]["training_curve"]
        ax.plot(
            [record["epoch"] for record in records],
            [record["val_flow"] for record in records],
            color=color,
            linewidth=1.8,
            label=f"state L={depth}",
        )
        ax.plot(
            [record["epoch"] for record in records],
            [record["val_feasibility"] for record in records],
            color=color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
        )
    ax.set_title("Validation loss terms (solid state, dashed feasibility)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss term")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)

    ax = axes[0, 2]
    x = np.arange(len(SPLITS))
    width = 0.19
    for offset, depth, color in zip(
        np.linspace(-1.5, 1.5, len(DEPTHS)) * width,
        DEPTHS,
        COLORS,
        strict=True,
    ):
        values = [
            results["runs"][str(depth)]["evaluation"][split]["equal_object"]["dx"]
            for split in SPLITS
        ]
        ax.bar(x + offset, values, width, color=color, label=f"L={depth}")
    ax.set_xticks(x, SPLITS)
    ax.set_ylabel(r"local $d_X$")
    ax.set_title("Full active-set error")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    names = (
        "translation_over_length",
        "rotation_rad",
        "joint_rmse_over_travel",
    )
    labels = ("T/L", "R [rad]", "J/travel")
    x = np.arange(len(names))
    for offset, depth, color in zip(
        np.linspace(-1.5, 1.5, len(DEPTHS)) * width,
        DEPTHS,
        COLORS,
        strict=True,
    ):
        values = [
            results["runs"][str(depth)]["evaluation"]["test"]["equal_object"][name]
            for name in names
        ]
        ax.bar(x + offset, values, width, color=color, label=f"L={depth}")
    ax.set_xticks(x, labels)
    ax.set_ylabel("mean local component")
    ax.set_title("Unseen test decomposition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    for depth, color in zip(DEPTHS, COLORS, strict=True):
        ordered = np.sort(raw[depth]["test"]["dx"])
        cdf = np.arange(1, len(ordered) + 1) / len(ordered)
        ax.plot(ordered, cdf, color=color, linewidth=2, label=f"L={depth}")
    ax.set_xlabel(r"test local $d_X$")
    ax.set_ylabel("empirical CDF")
    ax.set_title("All unseen active transitions")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 2]
    parameters = [results["runs"][str(depth)]["parameter_count"] for depth in DEPTHS]
    test_dx = [
        results["runs"][str(depth)]["evaluation"]["test"]["equal_object"]["dx"]
        for depth in DEPTHS
    ]
    ax.plot(parameters, test_dx, "o-", linewidth=2, color="#5B9BD5")
    for depth, parameters_i, dx_i in zip(DEPTHS, parameters, test_dx, strict=True):
        ax.annotate(f"L={depth}", (parameters_i, dx_i), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel(r"test local $d_X$")
    ax.set_title("Accuracy / capacity")
    ax.grid(alpha=0.25)

    fig.suptitle("SRNO residual integral-operator depth ablation (seed 0, local only)")
    fig.savefig(output / "operator_depth_local_ablation.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = ExperimentConfig.load(args.config)
    if base.seed != SEED:
        raise ValueError("clean depth ablation is frozen to seed 0")
    if base.model.contact_features != "gap":
        raise ValueError("clean depth ablation requires gap contact features")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    configs = {depth: _run_config(base, output, depth) for depth in DEPTHS}
    checkpoints = {
        depth: _train_or_resume(configs[depth]) for depth in DEPTHS
    }
    runs: dict[str, Any] = {}
    raw: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    packed: dict[str, np.ndarray] = {}
    for depth in DEPTHS:
        records = _metrics(configs[depth].paths.output_dir / "metrics.jsonl")
        best = _best_record(records)
        raw[depth] = {}
        evaluation: dict[str, Any] = {}
        for split in SPLITS:
            summary, arrays = _evaluate_split(
                configs[depth], checkpoints[depth], split
            )
            evaluation[split] = summary
            raw[depth][split] = arrays
            for name, value in arrays.items():
                packed[f"L{depth}_{split}_{name}"] = value
            print(
                f"[DEPTH] L={depth} {split} "
                f"dx={summary['equal_object']['dx']:.8f}",
                flush=True,
            )
        runs[str(depth)] = {
            "parameter_count": _parameter_count(configs[depth]),
            "best_epoch": int(best["epoch"]),
            "best_validation_dx_during_training": float(best["val"]["dx"]),
            "training_curve": [
                {
                    "epoch": int(record["epoch"]),
                    "train_dx": float(record["train"]["dx"]),
                    "val_dx": float(record["val"]["dx"]),
                    "train_flow": float(record["train"]["flow"]),
                    "val_flow": float(record["val"]["flow"]),
                    "train_feasibility": float(record["train"]["feasibility"]),
                    "val_feasibility": float(record["val"]["feasibility"]),
                }
                for record in records
                if record["stage"] == "local"
            ],
            "evaluation": evaluation,
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    baseline = runs["1"]["evaluation"]
    comparison = {
        split: {
            str(depth): {
                "delta_dx_vs_L1": float(
                    runs[str(depth)]["evaluation"][split]["equal_object"]["dx"]
                    - baseline[split]["equal_object"]["dx"]
                ),
                "relative_dx_change_percent": float(
                    100.0
                    * (
                        runs[str(depth)]["evaluation"][split]["equal_object"]["dx"]
                        / baseline[split]["equal_object"]["dx"]
                        - 1.0
                    )
                ),
            }
            for depth in DEPTHS
        }
        for split in SPLITS
    }
    result = {
        "definition": {
            "first_layer": "z1=silu(W0*e + mean_j(kappa1(rho_i,rho_j)*W1*e_j) + b1)",
            "residual_layers": "z[l+1]=z[l]+silu(W0[l]*z[l]+mean_j(kappa[l](rho_i,rho_j)*W1[l]*z[l]_j)+b[l])",
            "pooling": "mean_i z[L]",
            "depths": list(DEPTHS),
            "seed": SEED,
            "rollout_performed": False,
        },
        "configuration": {
            "base_config": str(args.config.resolve()),
            "manifest": str(base.paths.manifest),
            "active_index": str(base.paths.active_index),
            "active_transition_counts": _active_counts(base),
            "object_counts": {
                split: len(DatasetManifest.load(base.paths.manifest).splits[split])
                for split in SPLITS
            },
        },
        "runs": runs,
        "comparison_vs_L1": comparison,
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(output / "samples.npz", **packed)
    _plot(output, result, raw)
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
