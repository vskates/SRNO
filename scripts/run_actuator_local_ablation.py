#!/usr/bin/env python3
"""Run the paired three-seed aperture-vs-drive-error local ablation."""

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

from run_jq_local_ablation import (
    _evaluate_split,
    _transition_keys,
)
from srno.data.schema import DatasetManifest
from srno.training.checkpoint import load_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model, train


ARMS = ("aperture", "drive_error")
SPLITS = ("train", "val", "test")
BOOTSTRAP_SEED = 20260817


def _config(
    base: ExperimentConfig,
    output: Path,
    seed: int,
    conditioning: str,
    contact_features: str,
) -> ExperimentConfig:
    return replace(
        base,
        seed=seed,
        paths=replace(
            base.paths, output_dir=output / conditioning / f"seed-{seed}"
        ),
        model=replace(
            base.model,
            contact_features=contact_features,  # type: ignore[arg-type]
            global_conditioning=conditioning,  # type: ignore[arg-type]
        ),
    )


def _verify_baseline(path: Path, config: ExperimentConfig) -> None:
    manifest = DatasetManifest.load(config.paths.manifest)
    model = _build_model(config, manifest, torch.device("cpu"))
    checkpoint = load_checkpoint(path, model=model, map_location="cpu")
    if checkpoint["stage"] != "local":
        raise ValueError(f"actuator baseline must be a local checkpoint: {path}")
    if checkpoint["manifest_sha256"] != manifest.sha256():
        raise ValueError(f"actuator baseline manifest mismatch: {path}")
    if checkpoint["gripper_sha256"] != manifest.gripper_sha256:
        raise ValueError(f"actuator baseline gripper mismatch: {path}")
    saved = checkpoint["config"]
    expected = config.to_dict()
    saved_model = dict(saved["model"])
    saved_model.setdefault("global_conditioning", "aperture")
    if saved_model != expected["model"]:
        raise ValueError(f"actuator baseline model config mismatch: {path}")
    for key in ("loss", "optimizer", "loader", "training", "seed"):
        if saved[key] != expected[key]:
            raise ValueError(f"actuator baseline {key} mismatch: {path}")


def _train_candidate(config: ExperimentConfig) -> Path:
    output = config.paths.output_dir
    best = output / "best-local.pt"
    if best.is_file():
        print(f"[ACTUATOR] reuse completed {best}", flush=True)
        return best
    last = output / "last-local.pt"
    if last.is_file():
        print(f"[ACTUATOR] resume {last}", flush=True)
        return train(config, stage="local", resume=last)
    if output.exists() and (output / "metrics.jsonl").exists():
        raise RuntimeError(f"local metrics exist without resumable checkpoint: {output}")
    print(f"[ACTUATOR] train drive_error seed={config.seed}", flush=True)
    return train(config, stage="local")


def _bootstrap(
    evaluations: dict[str, dict[int, dict[str, dict[str, np.ndarray]]]],
    seeds: tuple[int, ...],
    replicates: int = 10_000,
) -> dict[str, float | int]:
    differences: dict[int, dict[str, np.ndarray]] = {}
    labels: list[str] | None = None
    for seed in seeds:
        baseline = evaluations["aperture"][seed]["test"]
        candidate = evaluations["drive_error"][seed]["test"]
        if _transition_keys(baseline) != _transition_keys(candidate):
            raise RuntimeError(f"actuator transition order differs for seed {seed}")
        object_labels = [str(value) for value in baseline["object_labels"].tolist()]
        if labels is None:
            labels = object_labels
        elif labels != object_labels:
            raise RuntimeError("test object order differs between actuator seeds")
        differences[seed] = {}
        for object_index, object_id in enumerate(object_labels):
            mask = baseline["object_index"] == object_index
            differences[seed][object_id] = (
                candidate["dx"][mask] - baseline["dx"][mask]
            )
    assert labels is not None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_seeds = rng.choice(seeds, len(seeds), replace=True)
        seed_means: list[float] = []
        for seed in sampled_seeds:
            sampled_objects = rng.choice(labels, len(labels), replace=True)
            object_means: list[float] = []
            for object_id in sampled_objects:
                values = differences[int(seed)][str(object_id)]
                object_means.append(
                    float(rng.choice(values, len(values), replace=True).mean())
                )
            seed_means.append(float(np.mean(object_means)))
        samples[replicate] = float(np.mean(seed_means))
    return {
        "replicates": replicates,
        "seed": BOOTSTRAP_SEED,
        "mean_candidate_minus_baseline": float(samples.mean()),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
    }


def _summarize(
    summaries: dict[str, dict[int, dict[str, Any]]],
    evaluations: dict[str, dict[int, dict[str, dict[str, np.ndarray]]]],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    paired: dict[str, Any] = {}
    for split in ("val", "test"):
        paired[split] = {}
        for seed in seeds:
            baseline = summaries["aperture"][seed][split]["equal_object"]
            candidate = summaries["drive_error"][seed][split]["equal_object"]
            paired[split][str(seed)] = {
                name: float(candidate[name] - baseline[name]) for name in baseline
            }
    bootstrap = _bootstrap(evaluations, seeds)
    val_all_better = all(paired["val"][str(seed)]["dx"] < 0.0 for seed in seeds)
    arms: dict[str, Any] = {}
    for arm in ARMS:
        arms[arm] = {}
        for split in SPLITS:
            arms[arm][split] = {}
            names = summaries[arm][seeds[0]][split]["equal_object"]
            for name in names:
                values = np.asarray(
                    [summaries[arm][seed][split]["equal_object"][name] for seed in seeds]
                )
                arms[arm][split][name] = {
                    "mean": float(values.mean()),
                    "std_across_seeds": float(values.std(ddof=1)),
                    "per_seed": {
                        str(seed): float(value)
                        for seed, value in zip(seeds, values, strict=True)
                    },
                }
    return {
        "criterion": {
            "validation_drive_error_better_all_three_seeds": val_all_better,
            "test_bootstrap_ci95_upper_below_zero": float(bootstrap["ci95_upper"]) < 0.0,
            "confirmed_local_gain": val_all_better
            and float(bootstrap["ci95_upper"]) < 0.0,
        },
        "test_hierarchical_bootstrap": bootstrap,
        "paired_candidate_minus_baseline": paired,
        "arms": arms,
    }


def _plot(
    output: Path,
    comparison: dict[str, Any],
    summaries: dict[str, dict[int, dict[str, Any]]],
    evaluations: dict[str, dict[int, dict[str, dict[str, np.ndarray]]]],
    seeds: tuple[int, ...],
) -> None:
    colors = {"aperture": "#4472C4", "drive_error": "#E67E22"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax = axes[0, 0]
    for split_index, split in enumerate(("val", "test")):
        for seed in seeds:
            values = [summaries[arm][seed][split]["equal_object"]["dx"] for arm in ARMS]
            x = np.asarray([0, 1]) + split_index * 3
            ax.plot(x, values, "o-", alpha=0.75, label=f"{split} seed {seed}")
    ax.set_xticks([0, 1, 3, 4], ["val A(r)", "val u", "test A(r)", "test u"])
    ax.set_ylabel(r"local $d_X$")
    ax.set_title("Paired training seeds")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    names = ("translation_over_length", "rotation_rad", "joint_rmse_over_travel")
    labels = ("T/L", "R [rad]", "J/travel")
    x = np.arange(3)
    width = 0.36
    for arm_index, arm in enumerate(ARMS):
        values = comparison["arms"][arm]["test"]
        means = [values[name]["mean"] for name in names]
        stds = [values[name]["std_across_seeds"] for name in names]
        ax.bar(x + (arm_index - 0.5) * width, means, width, yerr=stds,
               capsize=3, color=colors[arm], label=arm)
    ax.set_xticks(x, labels)
    ax.set_title("Unseen local components")
    ax.legend()

    ax = axes[1, 0]
    labels_test = evaluations["aperture"][seeds[0]]["test"]["object_labels"].tolist()
    deltas: list[list[float]] = []
    for object_index, _ in enumerate(labels_test):
        values = []
        for seed in seeds:
            baseline = evaluations["aperture"][seed]["test"]
            candidate = evaluations["drive_error"][seed]["test"]
            mask = baseline["object_index"] == object_index
            values.append(float((candidate["dx"][mask] - baseline["dx"][mask]).mean()))
        deltas.append(values)
    means = np.asarray(deltas).mean(axis=1)
    stds = np.asarray(deltas).std(axis=1, ddof=1)
    positions = np.arange(len(labels_test))
    ax.bar(positions, means, yerr=stds, capsize=3, color="#70AD47")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(positions, [str(value)[:24] for value in labels_test], rotation=18)
    ax.set_ylabel(r"$d_X^u-d_X^{A(r)}$")
    ax.set_title("Per-object test difference")

    ax = axes[1, 1]
    for arm in ARMS:
        pooled = np.concatenate([evaluations[arm][seed]["test"]["dx"] for seed in seeds])
        ordered = np.sort(pooled)
        ax.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered),
                color=colors[arm], label=arm)
    ax.set_xlabel(r"local test $d_X$")
    ax.set_ylabel("empirical CDF")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.suptitle("SRNO actuator conditioning: scalar aperture vs six-joint drive error")
    fig.savefig(output / "actuator_local_ablation.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-features", choices=("gap", "gap_jq"), default="gap")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    if seeds != (0, 1, 2):
        raise ValueError("the frozen actuator ablation requires seeds 0 1 2")
    base = ExperimentConfig.load(args.config)
    output = args.output.resolve()
    configs = {
        arm: {
            seed: _config(base, output, seed, arm, args.contact_features)
            for seed in seeds
        }
        for arm in ARMS
    }
    checkpoints: dict[str, dict[int, Path]] = {arm: {} for arm in ARMS}
    for seed in seeds:
        baseline = (
            args.baseline_root.resolve()
            / "baseline"
            / f"seed-{seed}"
            / "best-local.pt"
        )
        _verify_baseline(baseline, configs["aperture"][seed])
        checkpoints["aperture"][seed] = baseline
        checkpoints["drive_error"][seed] = _train_candidate(
            configs["drive_error"][seed]
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summaries: dict[str, dict[int, dict[str, Any]]] = {
        arm: {seed: {} for seed in seeds} for arm in ARMS
    }
    evaluations: dict[str, dict[int, dict[str, dict[str, np.ndarray]]]] = {
        arm: {seed: {} for seed in seeds} for arm in ARMS
    }
    for arm in ARMS:
        for seed in seeds:
            packed: dict[str, np.ndarray] = {}
            for split in SPLITS:
                summary, arrays = _evaluate_split(
                    configs[arm][seed], checkpoints[arm][seed], split
                )
                summaries[arm][seed][split] = summary
                evaluations[arm][seed][split] = arrays
                for name, value in arrays.items():
                    packed[f"{split}_{name}"] = value
                print(
                    f"[ACTUATOR] evaluate arm={arm} seed={seed} split={split} "
                    f"dx={summary['equal_object']['dx']:.8f}",
                    flush=True,
                )
            run_output = configs[arm][seed].paths.output_dir
            run_output.mkdir(parents=True, exist_ok=True)
            (run_output / "local_evaluation.json").write_text(
                json.dumps(summaries[arm][seed], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            np.savez_compressed(run_output / "local_evaluation.npz", **packed)

    manifest = DatasetManifest.load(base.paths.manifest)
    aperture_model = _build_model(
        configs["aperture"][0], manifest, torch.device("cpu")
    )
    drive_model = _build_model(
        configs["drive_error"][0], manifest, torch.device("cpu")
    )
    result = {
        "definition": {
            "baseline": "head conditioning [A(r_k)/L, command/L]",
            "candidate": "u_k=(R_free(command)-r_k)/joint_travel",
            "dx": "sqrt((||dp||/L)^2 + theta(R,R*)^2 + mean(((r-r*)/travel)^2))",
            "contact_features": args.contact_features,
            "rollout_performed": False,
        },
        "configuration": {
            "manifest": str(base.paths.manifest),
            "manifest_sha256": manifest.sha256(),
            "gripper_sha256": manifest.gripper_sha256,
            "seeds": list(seeds),
            "parameter_count": {
                "aperture": sum(p.numel() for p in aperture_model.parameters()),
                "drive_error": sum(p.numel() for p in drive_model.parameters()),
            },
        },
        "runs": summaries,
    }
    result["comparison"] = _summarize(summaries, evaluations, seeds)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    comparison_arrays: dict[str, np.ndarray] = {
        "seeds": np.asarray(seeds, dtype=np.int64),
    }
    for arm in ARMS:
        for split in SPLITS:
            comparison_arrays[f"{arm}_{split}_equal_object_dx"] = np.asarray(
                [
                    summaries[arm][seed][split]["equal_object"]["dx"]
                    for seed in seeds
                ],
                dtype=np.float64,
            )
    test_labels = evaluations["aperture"][seeds[0]]["test"][
        "object_labels"
    ]
    comparison_arrays["test_object_labels"] = test_labels
    per_object_delta = np.empty((len(seeds), len(test_labels)), dtype=np.float64)
    for seed_index, seed in enumerate(seeds):
        baseline = evaluations["aperture"][seed]["test"]
        candidate = evaluations["drive_error"][seed]["test"]
        for object_index in range(len(test_labels)):
            mask = baseline["object_index"] == object_index
            per_object_delta[seed_index, object_index] = float(
                (candidate["dx"][mask] - baseline["dx"][mask]).mean()
            )
    comparison_arrays["test_per_object_candidate_minus_baseline"] = (
        per_object_delta
    )
    np.savez_compressed(output / "comparison.npz", **comparison_arrays)
    _plot(output, result["comparison"], summaries, evaluations, seeds)
    print(json.dumps(result["comparison"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
