#!/usr/bin/env python3
"""Paired rollout diagnostics for the lambda_K=0 ablation."""

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


HORIZONS = (4, 8, 16, 32)
SEGMENTS = ((0, 4), (4, 8), (8, 16), (16, 32))
COLORS = {"lambda_K=1": "#4472C4", "lambda_K=0": "#E67E22"}


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _best_validation_by_horizon(path: Path) -> dict[int, float]:
    records = _read_metrics(path)
    return {
        horizon: min(
            record["val"]["terminal_dx"]
            for record in records
            if record["stage"] == "rollout" and record["horizon"] == horizon
        )
        for horizon in HORIZONS
    }


def _best_local(path: Path) -> float:
    return min(
        record["val"]["dx"]
        for record in _read_metrics(path)
        if record["stage"] == "local"
    )


def _rollout_errors(
    config_path: Path,
    checkpoint_path: Path,
    split: str,
) -> tuple[np.ndarray, list[str]]:
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
                if len(batch.object_ids) != 1:
                    raise RuntimeError("comparison expects one object per batch")
                keys.extend(
                    f"{batch.object_ids[0]}:{int(index)}"
                    for index in batch.trajectory_index.cpu()
                )
    finally:
        dataset.close()
    return np.concatenate(errors, axis=0), keys


def _mean_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    ci = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    return mean, ci


def _slope_summary(values: np.ndarray) -> dict[str, Any]:
    steps = np.arange(values.shape[1], dtype=np.float64)
    centered = steps - steps.mean()
    per_trajectory_global = values @ centered / np.dot(centered, centered)
    result: dict[str, Any] = {
        "global_0_32": float(per_trajectory_global.mean()),
        "segments": {},
    }
    for start, stop in SEGMENTS:
        slopes = (values[:, stop] - values[:, start]) / (stop - start)
        result["segments"][f"{start}-{stop}"] = {
            "mean": float(slopes.mean()),
            "ci95": float(1.96 * slopes.std(ddof=1) / np.sqrt(len(slopes))),
        }
    return result


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


def _paired_summary(baseline: np.ndarray, ablation: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for step in HORIZONS:
        difference = ablation[:, step] - baseline[:, step]
        result[str(step)] = {
            "ablation_minus_baseline_mean": float(difference.mean()),
            "ci95": float(1.96 * difference.std(ddof=1) / np.sqrt(len(difference))),
            "ablation_better_fraction": float(np.mean(difference < 0)),
            "relative_mean_change_percent": float(
                100 * (ablation[:, step].mean() / baseline[:, step].mean() - 1)
            ),
        }
    return result


def _plots(
    output: Path,
    curves: dict[str, dict[str, np.ndarray]],
    curriculum: dict[str, dict[int, float]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    ax = axes[0, 0]
    for name, values in curriculum.items():
        y = np.asarray([values[horizon] for horizon in HORIZONS])
        ax.plot(HORIZONS, y, "o-", linewidth=2, label=name, color=COLORS[name])
    ax.set_title("Best validation terminal $d_X$ during curriculum")
    ax.set_xlabel("training horizon")
    ax.set_ylabel("terminal $d_X$")
    ax.set_xticks(HORIZONS)
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    steps = np.arange(33)
    for name, values in curves["test"].items():
        mean, ci = _mean_ci(values)
        ax.plot(steps, mean, linewidth=2, label=name, color=COLORS[name])
        ax.fill_between(steps, mean - ci, mean + ci, color=COLORS[name], alpha=0.15)
        ax.scatter(HORIZONS, mean[list(HORIZONS)], color=COLORS[name], s=28)
    for horizon in HORIZONS:
        ax.axvline(horizon, color="#777777", linewidth=0.7, alpha=0.25)
    ax.set_title("Unseen test: mean $d_X(k)$ (95% CI)")
    ax.set_xlabel("rollout step $k$")
    ax.set_ylabel("$d_X(k)$")
    ax.set_xticks([0, *HORIZONS])
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    for name, values in curves["test"].items():
        increments = np.diff(values, axis=1)
        mean, ci = _mean_ci(increments)
        increment_steps = np.arange(1, 33)
        ax.plot(increment_steps, mean, linewidth=1.8, label=name, color=COLORS[name])
        ax.fill_between(
            increment_steps, mean - ci, mean + ci, color=COLORS[name], alpha=0.15
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Unseen test: discrete slope $d_X(k)-d_X(k-1)$")
    ax.set_xlabel("rollout step $k$")
    ax.set_ylabel("$\Delta d_X$ per step")
    ax.set_xticks(HORIZONS)
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    x = np.arange(len(SEGMENTS))
    width = 0.36
    for offset, (name, values) in zip((-width / 2, width / 2), curves["test"].items()):
        means = []
        cis = []
        for start, stop in SEGMENTS:
            slopes = (values[:, stop] - values[:, start]) / (stop - start)
            means.append(slopes.mean())
            cis.append(1.96 * slopes.std(ddof=1) / np.sqrt(len(slopes)))
        ax.bar(x + offset, means, width, yerr=cis, capsize=3, label=name, color=COLORS[name])
    ax.set_title("Unseen test: average $d_X$ slope by interval")
    ax.set_xlabel("rollout interval")
    ax.set_ylabel("$\Delta d_X / \Delta k$")
    ax.set_xticks(x, [f"{start}–{stop}" for start, stop in SEGMENTS])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.suptitle("SRNO feasibility-loss ablation: $\lambda_K=1$ vs $\lambda_K=0$", fontsize=15)
    fig.savefig(output / "ablation_dashboard.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, split in zip(axes, ("val", "test")):
        for name, values in curves[split].items():
            mean, ci = _mean_ci(values)
            ax.plot(steps, mean, linewidth=2, label=name, color=COLORS[name])
            ax.fill_between(steps, mean - ci, mean + ci, color=COLORS[name], alpha=0.15)
        ax.set_title(f"{split}: mean $d_X(k)$ (95% CI)")
        ax.set_xlabel("rollout step $k$")
        ax.set_ylabel("$d_X(k)$")
        ax.set_xticks([0, *HORIZONS])
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(output / "dx_k_val_test.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument(
        "--ablation-config", type=Path, default=Path("configs/ablation-lambda-k0.toml")
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=Path("runs/srno-contact-v1-full-coverage/best-rollout.pt"),
    )
    parser.add_argument(
        "--ablation-checkpoint",
        type=Path,
        default=Path("runs/srno-contact-v1-lambda-k0/best-rollout.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/srno-contact-v1-lambda-k0/comparison"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    run_specs = {
        "lambda_K=1": (args.baseline_config, args.baseline_checkpoint),
        "lambda_K=0": (args.ablation_config, args.ablation_checkpoint),
    }
    curves: dict[str, dict[str, np.ndarray]] = {"val": {}, "test": {}}
    reference_keys: dict[str, list[str]] = {}
    for split in curves:
        for name, (config, checkpoint) in run_specs.items():
            values, keys = _rollout_errors(config, checkpoint, split)
            if split in reference_keys and keys != reference_keys[split]:
                raise RuntimeError(f"trajectory ordering differs for {split}")
            reference_keys.setdefault(split, keys)
            curves[split][name] = values

    curriculum = {
        name: _best_validation_by_horizon(
            ExperimentConfig.load(config).paths.output_dir / "metrics.jsonl"
        )
        for name, (config, _) in run_specs.items()
    }
    summary: dict[str, Any] = {
        "definition": (
            "d_X(k)=sqrt(||p_hat-p||^2/length_scale^2 + "
            "geodesic_SO3(R_hat,R)^2 + |a_hat-a|^2/length_scale^2)"
        ),
        "curriculum_best_validation_terminal_dx": {
            name: {str(key): value for key, value in values.items()}
            for name, values in curriculum.items()
        },
        "local_best_validation_dx": {
            name: _best_local(ExperimentConfig.load(config).paths.output_dir / "metrics.jsonl")
            for name, (config, _) in run_specs.items()
        },
        "curves": {
            split: {name: _curve_summary(values) for name, values in split_curves.items()}
            for split, split_curves in curves.items()
        },
        "paired": {
            split: _paired_summary(values["lambda_K=1"], values["lambda_K=0"])
            for split, values in curves.items()
        },
    }
    (args.output / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(
        args.output / "raw_dx_curves.npz",
        baseline_val=curves["val"]["lambda_K=1"],
        ablation_val=curves["val"]["lambda_K=0"],
        baseline_test=curves["test"]["lambda_K=1"],
        ablation_test=curves["test"]["lambda_K=0"],
    )
    _plots(args.output, curves, curriculum)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
