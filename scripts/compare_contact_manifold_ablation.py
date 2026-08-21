#!/usr/bin/env python3
"""Compare the four controlled contact-manifold ablation arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from srno.data.dataset import H5ObjectDataset, make_dataloader
from srno.data.schema import DatasetManifest
from srno.losses import state_error
from srno.training.checkpoint import load_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _autocast, _build_model, _state_at
from srno.training.metrics import MetricAccumulator, accumulate_trajectory_metrics


HORIZONS = (4, 8, 16, 32)
SEGMENTS = ((0, 4), (4, 8), (8, 16), (16, 32))
RUNS = {
    "baseline": "ablation-contact-baseline.toml",
    "gate-only": "ablation-contact-gate.toml",
    "feasibility-only": "ablation-contact-feasibility.toml",
    "combined": "ablation-contact-combined.toml",
}
COLORS = {
    "baseline": "#4472C4",
    "gate-only": "#70AD47",
    "feasibility-only": "#E67E22",
    "combined": "#A64D79",
}


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _training_summary(path: Path) -> tuple[float, dict[int, float]]:
    rows = _records(path)
    local = min(row["val"]["dx"] for row in rows if row.get("stage") == "local")
    curriculum = {
        horizon: min(
            row["val"]["terminal_dx"]
            for row in rows
            if row.get("stage") == "rollout"
            and row.get("horizon") == horizon
            and "epoch" in row
        )
        for horizon in HORIZONS
    }
    return local, curriculum


def _rollout_metrics(
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    *,
    common_lag_threshold: float,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    config = ExperimentConfig.load(config_path)
    device = torch.device(config.device)
    manifest = DatasetManifest.load(config.paths.manifest)
    model = _build_model(config, manifest, device)
    checkpoint = load_checkpoint(checkpoint_path, model=model, map_location=device)
    if checkpoint["manifest_sha256"] != manifest.sha256():
        raise ValueError(f"manifest hash mismatch for {checkpoint_path}")
    model.eval()
    dataset = H5ObjectDataset(manifest, split=split)
    loader = make_dataloader(
        dataset,
        mode="rollout",
        objects_per_batch=1,
        samples_per_object=0,
        workers=config.loader.workers,
        seed=config.seed + 2,
        shuffle=False,
    )
    errors: list[np.ndarray] = []
    keys: list[str] = []
    accumulator = MetricAccumulator()
    try:
        with torch.no_grad():
            for raw_batch in loader:
                batch = raw_batch.to(device, non_blocking=True)
                with _autocast(config, device):
                    prediction = model.rollout(
                        _state_at(batch.states, 0), batch.command_schedule[1:], batch.sdf
                    )
                squared = state_error(
                    prediction,
                    batch.states,
                    length_scale=model.length_scale,
                    lambda_rotation=config.loss.lambda_rotation,
                    lambda_aperture=config.loss.lambda_aperture,
                )[0]
                errors.append(squared.sqrt().float().cpu().numpy())
                gaps = torch.stack(
                    [model.query_gap(_state_at(prediction, step), batch.sdf) for step in range(33)],
                    dim=1,
                )
                accumulate_trajectory_metrics(
                    accumulator,
                    prediction,
                    batch.states,
                    batch.command_schedule,
                    gaps,
                    length_scale=model.length_scale,
                    lag_threshold=common_lag_threshold,
                )
                if len(batch.object_ids) != 1:
                    raise RuntimeError("comparison expects one object per batch")
                keys.extend(
                    f"{batch.object_ids[0]}:{int(index)}"
                    for index in batch.trajectory_index.cpu()
                )
    finally:
        dataset.close()
    return np.concatenate(errors, axis=0), keys, accumulator.compute()


def _mean_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    ci = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    return mean, ci


def _slope_summary(values: np.ndarray) -> dict[str, Any]:
    steps = np.arange(values.shape[1], dtype=np.float64)
    centered = steps - steps.mean()
    global_slopes = values @ centered / np.dot(centered, centered)
    return {
        "global_0_32": {
            "mean": float(global_slopes.mean()),
            "ci95": float(1.96 * global_slopes.std(ddof=1) / np.sqrt(len(global_slopes))),
        },
        "segments": {
            f"{start}-{stop}": {
                "mean": float(slopes.mean()),
                "ci95": float(1.96 * slopes.std(ddof=1) / np.sqrt(len(slopes))),
            }
            for start, stop in SEGMENTS
            for slopes in [(values[:, stop] - values[:, start]) / (stop - start)]
        },
    }


def _curve_summary(values: np.ndarray) -> dict[str, Any]:
    mean, ci = _mean_ci(values)
    return {
        "count": int(values.shape[0]),
        "milestones": {
            str(step): {"mean": float(mean[step]), "ci95": float(ci[step])}
            for step in HORIZONS
        },
        "slope_dx_per_step": _slope_summary(values),
    }


def _paired_summary(baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for step in HORIZONS:
        difference = candidate[:, step] - baseline[:, step]
        result[str(step)] = {
            "candidate_minus_baseline_mean": float(difference.mean()),
            "ci95": float(1.96 * difference.std(ddof=1) / np.sqrt(len(difference))),
            "candidate_better_fraction": float(np.mean(difference < 0)),
            "relative_mean_change_percent": float(
                100.0 * (candidate[:, step].mean() / baseline[:, step].mean() - 1.0)
            ),
        }
    return result


def _plots(
    output: Path,
    curves: dict[str, dict[str, np.ndarray]],
    curriculum: dict[str, dict[int, float]],
    evaluation: dict[str, dict[str, dict[str, float]]],
) -> None:
    steps = np.arange(33)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ax = axes[0, 0]
    for name, values in curriculum.items():
        ax.plot(HORIZONS, [values[h] for h in HORIZONS], "o-", lw=2, label=name,
                color=COLORS[name])
    ax.set(title="Best validation terminal $d_X$ during curriculum",
           xlabel="training horizon", ylabel="terminal $d_X$")
    ax.set_xticks(HORIZONS)
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    for name, values in curves["test"].items():
        mean, ci = _mean_ci(values)
        ax.plot(steps, mean, lw=2, label=name, color=COLORS[name])
        ax.fill_between(steps, mean - ci, mean + ci, color=COLORS[name], alpha=0.12)
    ax.set(title="Unseen test: mean $d_X(k)$ (trajectory 95% CI)",
           xlabel="rollout step $k$", ylabel="$d_X(k)$")
    ax.set_xticks([0, *HORIZONS])
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    for name, values in curves["test"].items():
        increments = np.diff(values, axis=1)
        mean, ci = _mean_ci(increments)
        ax.plot(np.arange(1, 33), mean, lw=1.7, label=name, color=COLORS[name])
        ax.fill_between(np.arange(1, 33), mean - ci, mean + ci,
                        color=COLORS[name], alpha=0.12)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set(title="Unseen test: discrete slope $d_X(k)-d_X(k-1)$",
           xlabel="rollout step $k$", ylabel="$\Delta d_X$")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    x = np.arange(len(SEGMENTS))
    width = 0.19
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(RUNS))
    for offset, (name, values) in zip(offsets, curves["test"].items(), strict=True):
        means, cis = [], []
        for start, stop in SEGMENTS:
            slopes = (values[:, stop] - values[:, start]) / (stop - start)
            means.append(slopes.mean())
            cis.append(1.96 * slopes.std(ddof=1) / np.sqrt(len(slopes)))
        ax.bar(x + offset, means, width, yerr=cis, capsize=2, label=name,
               color=COLORS[name])
    ax.set(title="Unseen test: mean $d_X$ slope by interval",
           xlabel="rollout interval", ylabel="$\Delta d_X/\Delta k$")
    ax.set_xticks(x, [f"{start}–{stop}" for start, stop in SEGMENTS])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.suptitle("SRNO contact-manifold controlled ablation", fontsize=15)
    fig.savefig(output / "ablation_dashboard.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, split in zip(axes, ("val", "test"), strict=True):
        for name, values in curves[split].items():
            mean, ci = _mean_ci(values)
            ax.plot(steps, mean, lw=2, label=name, color=COLORS[name])
            ax.fill_between(steps, mean - ci, mean + ci, color=COLORS[name], alpha=0.12)
        ax.set(title=f"{split}: mean $d_X(k)$", xlabel="rollout step $k$", ylabel="$d_X(k)$")
        ax.set_xticks([0, *HORIZONS])
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(output / "dx_k_val_test.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for name in RUNS:
        metrics = evaluation["test"][name]
        ax.scatter(metrics["penetration_m"] * 1e3, metrics["terminal_dx"], s=75,
                   color=COLORS[name], label=name)
        ax.annotate(name, (metrics["penetration_m"] * 1e3, metrics["terminal_dx"]),
                    xytext=(5, 4), textcoords="offset points")
    ax.set(title="Unseen test: state-error / feasibility trade-off",
           xlabel="mean predicted penetration (mm)", ylabel="terminal $d_X$")
    ax.grid(alpha=0.25)
    fig.savefig(output / "state_penetration_tradeoff.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/contact-manifold-ablation/comparison"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configs = {name: args.config_dir / filename for name, filename in RUNS.items()}
    loaded = {name: ExperimentConfig.load(path) for name, path in configs.items()}
    baseline_manifest = DatasetManifest.load(loaded["baseline"].paths.manifest)
    common_lag_threshold = baseline_manifest.delta_gate_m

    curves: dict[str, dict[str, np.ndarray]] = {"val": {}, "test": {}}
    evaluation: dict[str, dict[str, dict[str, float]]] = {"val": {}, "test": {}}
    reference_keys: dict[str, list[str]] = {}
    for split in curves:
        for name, config_path in configs.items():
            checkpoint = loaded[name].paths.output_dir / "best-rollout.pt"
            values, keys, metrics = _rollout_metrics(
                config_path, checkpoint, split,
                common_lag_threshold=common_lag_threshold,
            )
            if split in reference_keys and keys != reference_keys[split]:
                raise RuntimeError(f"trajectory ordering differs for {split}")
            reference_keys.setdefault(split, keys)
            curves[split][name] = values
            evaluation[split][name] = metrics

    local, curriculum = {}, {}
    for name, config in loaded.items():
        local[name], curriculum[name] = _training_summary(config.paths.output_dir / "metrics.jsonl")
    summary: dict[str, Any] = {
        "definitions": {
            "state_metric": "d_X=sqrt(||p_hat-p||^2/L^2+d_SO3(R_hat,R)^2+|a_hat-a|^2/L^2)",
            "correction_label": "y_corr=1[c_k>tau_num], c_k=d_X(x*_{k+1},F_free(x*_k))",
            "gate": "active=1[min_i h_i<=delta_gate]",
            "feasibility": "L_K=mean(((h_admissible-h)_+/sdf_scale)^2)",
        },
        "constants": {
            "baseline_delta_gate_m": baseline_manifest.delta_gate_m,
            "correction_delta_gate_m": DatasetManifest.load(
                loaded["gate-only"].paths.manifest
            ).delta_gate_m,
            "calibrated_admissible_gap_m": loaded["feasibility-only"].loss.admissible_gap_m,
            "common_evaluation_lag_threshold_m": common_lag_threshold,
        },
        "local_best_validation_dx": local,
        "curriculum_best_validation_terminal_dx": {
            name: {str(h): value for h, value in values.items()}
            for name, values in curriculum.items()
        },
        "evaluation": evaluation,
        "curves": {
            split: {name: _curve_summary(values) for name, values in split_curves.items()}
            for split, split_curves in curves.items()
        },
        "paired_vs_baseline": {
            split: {
                name: _paired_summary(split_curves["baseline"], values)
                for name, values in split_curves.items() if name != "baseline"
            }
            for split, split_curves in curves.items()
        },
    }
    (args.output / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output / "raw_dx_curves.npz",
        **{f"{name.replace('-', '_')}_{split}": values
           for split, split_curves in curves.items() for name, values in split_curves.items()},
    )
    _plots(args.output, curves, curriculum, evaluation)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
