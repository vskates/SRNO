#!/usr/bin/env python3
"""Run and evaluate the clean three-seed local SRNO gap-vs-gap+Jq ablation."""

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

from srno.data.dataset import H5ObjectDataset, LocalTransitionBatch, make_dataloader
from srno.data.index import ActiveIndex
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import rotation_geodesic_angle
from srno.training.checkpoint import load_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _autocast, _build_model, train
from srno.types import PoseState


ARMS = {"baseline": "gap", "gap_jq": "gap_jq"}
SPLITS = ("train", "val", "test")
SIMULATOR_FLOOR = 0.002991
BOOTSTRAP_SEED = 20260817


def _components(model: Any, prediction: PoseState, target: PoseState) -> dict[str, torch.Tensor]:
    translation_m = torch.linalg.vector_norm(prediction.position - target.position, dim=-1)
    translation = translation_m / model.length_scale
    rotation = rotation_geodesic_angle(prediction.rotation, target.rotation)
    joints = torch.sqrt(
        (
            (prediction.joint_position - target.joint_position)
            / model.joint_travel_range
        )
        .square()
        .mean(dim=-1)
    )
    dx = torch.sqrt(translation.square() + rotation.square() + joints.square())
    return {
        "dx": dx,
        "translation_over_length": translation,
        "translation_m": translation_m,
        "rotation_rad": rotation,
        "joint_rmse_over_travel": joints,
    }


def _mean_metrics(arrays: dict[str, np.ndarray], mask: np.ndarray | None = None) -> dict[str, float]:
    selected = slice(None) if mask is None else mask
    return {
        name: float(np.mean(arrays[name][selected]))
        for name in (
            "dx",
            "translation_over_length",
            "translation_m",
            "rotation_rad",
            "joint_rmse_over_travel",
        )
    }


def _evaluate_split(
    config: ExperimentConfig,
    checkpoint: Path,
    split: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    device = torch.device(config.device)
    manifest = DatasetManifest.load(config.paths.manifest)
    active_index = ActiveIndex.load(config.paths.active_index)
    model = _build_model(config, manifest, device)
    saved = load_checkpoint(checkpoint, model=model, map_location=device)
    if saved["manifest_sha256"] != manifest.sha256():
        raise ValueError("checkpoint manifest hash mismatch")
    model.eval()
    dataset = H5ObjectDataset(
        manifest,
        split=split,  # type: ignore[arg-type]
        active_index=active_index,
        active_only=True,
    )
    loader = make_dataloader(
        dataset,
        mode="local",
        objects_per_batch=1,
        samples_per_object=0,
        workers=config.loader.workers,
        seed=config.seed + 100,
        shuffle=False,
    )
    rows: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "dx",
            "translation_over_length",
            "translation_m",
            "rotation_rad",
            "joint_rmse_over_travel",
            "object_index",
            "trajectory_index",
            "step_index",
        )
    }
    try:
        with torch.no_grad():
            for raw_batch in loader:
                assert isinstance(raw_batch, LocalTransitionBatch)
                if len(raw_batch.object_ids) != 1:
                    raise RuntimeError("local ablation evaluation expects one object per batch")
                batch = raw_batch.to(device, non_blocking=True)
                with _autocast(config, device):
                    prediction = model.forward_step(
                        batch.current,
                        batch.next_command,
                        batch.sdf,
                        previous_state=batch.previous,
                    )
                assert isinstance(prediction, PoseState)
                values = _components(model, prediction, batch.target)
                for name, value in values.items():
                    rows[name].append(value.float().cpu().numpy())
                object_index = dataset.object_ids.index(raw_batch.object_ids[0])
                rows["object_index"].append(
                    np.full(len(batch.trajectory_index), object_index, dtype=np.int32)
                )
                rows["trajectory_index"].append(batch.trajectory_index.cpu().numpy())
                rows["step_index"].append(batch.step_index.cpu().numpy())
    finally:
        dataset.close()

    arrays = {name: np.concatenate(values) for name, values in rows.items()}
    arrays["object_labels"] = np.asarray(dataset.object_ids)
    by_object = {
        object_id: _mean_metrics(arrays, arrays["object_index"] == index)
        for index, object_id in enumerate(dataset.object_ids)
    }
    equal_object = {
        name: float(np.mean([metrics[name] for metrics in by_object.values()]))
        for name in next(iter(by_object.values()))
    }
    return (
        {
            "sample_count": int(len(arrays["dx"])),
            "sample_weighted": _mean_metrics(arrays),
            "equal_object": equal_object,
            "by_object": by_object,
        },
        arrays,
    )


def _run_config(base: ExperimentConfig, output: Path, arm: str, seed: int) -> ExperimentConfig:
    return replace(
        base,
        seed=seed,
        paths=replace(base.paths, output_dir=output / arm / f"seed-{seed}"),
        model=replace(base.model, contact_features=ARMS[arm]),
    )


def _train_or_resume(config: ExperimentConfig) -> Path:
    output = config.paths.output_dir
    best = output / "best-local.pt"
    if best.is_file():
        print(f"[JQ] reuse completed {best}", flush=True)
        return best
    last = output / "last-local.pt"
    if last.is_file():
        print(f"[JQ] resume {last}", flush=True)
        return train(config, stage="local", resume=last)
    if output.exists() and any(output.iterdir()):
        # A failure before the first completed epoch leaves only config.json
        # and a TensorBoard event.  Retrying is safe because no optimizer or
        # reported metric exists yet; anything later must have a checkpoint.
        if (output / "metrics.jsonl").exists():
            raise RuntimeError(
                f"non-empty run has metrics but no resumable checkpoint: {output}"
            )
        print(f"[JQ] retry pre-epoch run {output}", flush=True)
    print(
        f"[JQ] train arm={config.model.contact_features} seed={config.seed}",
        flush=True,
    )
    return train(config, stage="local")


def _transition_keys(arrays: dict[str, np.ndarray]) -> list[tuple[str, int, int]]:
    labels = arrays["object_labels"].tolist()
    return [
        (str(labels[int(object_index)]), int(trajectory), int(step))
        for object_index, trajectory, step in zip(
            arrays["object_index"],
            arrays["trajectory_index"],
            arrays["step_index"],
            strict=True,
        )
    ]


def _hierarchical_bootstrap(
    evaluations: dict[str, dict[int, dict[str, dict[str, np.ndarray]]]],
    seeds: tuple[int, ...],
    *,
    replicates: int = 10_000,
) -> dict[str, float | int]:
    differences: dict[int, dict[str, np.ndarray]] = {}
    labels: list[str] | None = None
    for seed in seeds:
        baseline = evaluations["baseline"][seed]["test"]
        candidate = evaluations["gap_jq"][seed]["test"]
        if _transition_keys(baseline) != _transition_keys(candidate):
            raise RuntimeError(f"baseline/Jq transition order differs for seed {seed}")
        object_labels = [str(value) for value in baseline["object_labels"].tolist()]
        if labels is None:
            labels = object_labels
        elif labels != object_labels:
            raise RuntimeError("test object order differs between seeds")
        differences[seed] = {
            object_id: (
                candidate["dx"][candidate["object_index"] == object_index]
                - baseline["dx"][baseline["object_index"] == object_index]
            )
            for object_index, object_id in enumerate(object_labels)
        }
    assert labels is not None
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        seed_sample = generator.choice(seeds, size=len(seeds), replace=True)
        seed_means = []
        for seed in seed_sample:
            object_sample = generator.choice(labels, size=len(labels), replace=True)
            object_means = []
            for object_id in object_sample:
                values = differences[int(seed)][str(object_id)]
                sampled = generator.choice(values, size=len(values), replace=True)
                object_means.append(float(np.mean(sampled)))
            seed_means.append(float(np.mean(object_means)))
        bootstrap[replicate] = float(np.mean(seed_means))
    return {
        "replicates": replicates,
        "seed": BOOTSTRAP_SEED,
        "mean_candidate_minus_baseline": float(np.mean(bootstrap)),
        "ci95_lower": float(np.quantile(bootstrap, 0.025)),
        "ci95_upper": float(np.quantile(bootstrap, 0.975)),
    }


def _summarize(
    summaries: dict[str, dict[int, dict[str, Any]]],
    evaluations: dict[str, dict[int, dict[str, dict[str, np.ndarray]]]],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    bootstrap = _hierarchical_bootstrap(evaluations, seeds)
    paired: dict[str, Any] = {}
    for split in ("val", "test"):
        paired[split] = {}
        for seed in seeds:
            baseline = summaries["baseline"][seed][split]["equal_object"]
            candidate = summaries["gap_jq"][seed][split]["equal_object"]
            paired[split][str(seed)] = {
                name: candidate[name] - baseline[name]
                for name in baseline
            }
    val_all_better = all(
        paired["val"][str(seed)]["dx"] < 0.0 for seed in seeds
    )
    confirmed = val_all_better and float(bootstrap["ci95_upper"]) < 0.0
    arm_summary: dict[str, Any] = {}
    for arm in ARMS:
        arm_summary[arm] = {}
        for split in SPLITS:
            arm_summary[arm][split] = {}
            metric_names = summaries[arm][seeds[0]][split]["equal_object"]
            for name in metric_names:
                values = np.asarray(
                    [summaries[arm][seed][split]["equal_object"][name] for seed in seeds]
                )
                arm_summary[arm][split][name] = {
                    "mean": float(values.mean()),
                    "std_across_seeds": float(values.std(ddof=1)),
                    "per_seed": {str(seed): float(value) for seed, value in zip(seeds, values, strict=True)},
                }
        arm_summary[arm]["gamma_test_over_simulator_floor"] = (
            arm_summary[arm]["test"]["dx"]["mean"] / SIMULATOR_FLOOR
        )
    return {
        "criterion": {
            "validation_jq_better_all_three_seeds": val_all_better,
            "test_hierarchical_bootstrap_ci95_upper_below_zero": float(bootstrap["ci95_upper"]) < 0.0,
            "confirmed_local_gain": confirmed,
        },
        "test_hierarchical_bootstrap": bootstrap,
        "paired_candidate_minus_baseline": paired,
        "arms": arm_summary,
    }


def _plot(
    output: Path,
    summary: dict[str, Any],
    summaries: dict[str, dict[int, dict[str, Any]]],
    evaluations: dict[str, dict[int, dict[str, dict[str, np.ndarray]]]],
    seeds: tuple[int, ...],
) -> None:
    colors = {"baseline": "#4472C4", "gap_jq": "#E67E22"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    ax = axes[0, 0]
    for split_index, split in enumerate(("val", "test")):
        for seed in seeds:
            values = [
                summaries[arm][seed][split]["equal_object"]["dx"]
                for arm in ARMS
            ]
            x = np.asarray([0, 1]) + split_index * 3
            ax.plot(x, values, "o-", alpha=0.75, label=f"{split} seed {seed}")
    ax.set_xticks([0, 1, 3, 4], ["val base", "val +Jq", "test base", "test +Jq"])
    ax.set_ylabel(r"local $d_X$")
    ax.set_title("Paired training seeds")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    metric_names = ("translation_over_length", "rotation_rad", "joint_rmse_over_travel")
    labels = ("T/L", "R [rad]", "J/travel")
    x = np.arange(len(labels))
    width = 0.36
    for arm_index, arm in enumerate(ARMS):
        means = [summary["arms"][arm]["test"][name]["mean"] for name in metric_names]
        stds = [summary["arms"][arm]["test"][name]["std_across_seeds"] for name in metric_names]
        ax.bar(x + (arm_index - 0.5) * width, means, width, yerr=stds, capsize=3,
               label=arm, color=colors[arm])
    ax.set_xticks(x, labels)
    ax.set_ylabel("mean local component")
    ax.set_title("Unseen test components: mean ± seed std")
    ax.legend()

    ax = axes[1, 0]
    test_labels = evaluations["baseline"][seeds[0]]["test"]["object_labels"].tolist()
    object_delta = []
    for object_index, object_id in enumerate(test_labels):
        deltas = []
        for seed in seeds:
            base = evaluations["baseline"][seed]["test"]
            jq = evaluations["gap_jq"][seed]["test"]
            mask = base["object_index"] == object_index
            deltas.append(float(np.mean(jq["dx"][mask] - base["dx"][mask])))
        object_delta.append(deltas)
    positions = np.arange(len(test_labels))
    means = np.asarray(object_delta).mean(axis=1)
    stds = np.asarray(object_delta).std(axis=1, ddof=1)
    ax.bar(positions, means, yerr=stds, capsize=3, color="#70AD47")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(positions, [str(label)[:24] for label in test_labels], rotation=18)
    ax.set_ylabel(r"$d_X^{J_q}-d_X^{base}$")
    ax.set_title("Per-object test difference: mean ± seed std")

    ax = axes[1, 1]
    for arm in ARMS:
        pooled = np.concatenate(
            [evaluations[arm][seed]["test"]["dx"] for seed in seeds]
        )
        ordered = np.sort(pooled)
        cdf = np.arange(1, len(ordered) + 1) / len(ordered)
        ax.plot(ordered, cdf, linewidth=2, label=arm, color=colors[arm])
    ax.set_xlabel(r"test local $d_X$")
    ax.set_ylabel("empirical CDF")
    ax.set_title("All unseen active transitions")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.suptitle("SRNO local representation ablation: gap vs gap + Jq", fontsize=15)
    fig.savefig(output / "jq_local_ablation.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    if seeds != (0, 1, 2):
        raise ValueError("the frozen clean ablation requires exactly --seeds 0 1 2")

    base = ExperimentConfig.load(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints: dict[str, dict[int, Path]] = {arm: {} for arm in ARMS}
    run_configs: dict[str, dict[int, ExperimentConfig]] = {arm: {} for arm in ARMS}
    for seed in seeds:
        for arm in ARMS:
            config = _run_config(base, output, arm, seed)
            run_configs[arm][seed] = config
            checkpoints[arm][seed] = _train_or_resume(config)
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
            run_output = run_configs[arm][seed].paths.output_dir
            packed: dict[str, np.ndarray] = {}
            for split in SPLITS:
                split_summary, arrays = _evaluate_split(
                    run_configs[arm][seed], checkpoints[arm][seed], split
                )
                summaries[arm][seed][split] = split_summary
                evaluations[arm][seed][split] = arrays
                for name, value in arrays.items():
                    packed[f"{split}_{name}"] = value
                print(
                    f"[JQ] evaluate arm={arm} seed={seed} split={split} "
                    f"dx={split_summary['equal_object']['dx']:.8f}",
                    flush=True,
                )
            (run_output / "local_evaluation.json").write_text(
                json.dumps(summaries[arm][seed], indent=2, sort_keys=True) + "\n"
            )
            np.savez_compressed(run_output / "local_evaluation.npz", **packed)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result = {
        "definition": {
            "arms": ARMS,
            "dx": "sqrt((||dp||/L)^2 + theta(R,R*)^2 + mean(((r-r*)/travel)^2))",
            "jq": "[-n_G, -(y_G/L cross n_G)]",
            "simulator_floor": SIMULATOR_FLOOR,
            "rollout_performed": False,
        },
        "configuration": {
            "base_config": str(args.config.resolve()),
            "manifest": str(base.paths.manifest),
            "active_index": str(base.paths.active_index),
            "seeds": list(seeds),
            "parameter_count": {"baseline": 31_436, "gap_jq": 32_204},
        },
        "runs": summaries,
        "comparison": _summarize(summaries, evaluations, seeds),
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _plot(output, result["comparison"], summaries, evaluations, seeds)
    print(json.dumps(result["comparison"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
