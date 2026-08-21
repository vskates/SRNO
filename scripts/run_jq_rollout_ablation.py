#!/usr/bin/env python3
"""Run the clean three-seed SRNO gap-vs-gap+Jq rollout ablation."""

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

from srno.data.dataset import H5ObjectDataset, TrajectoryBatch, make_dataloader
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import rotation_geodesic_angle
from srno.training.checkpoint import load_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _autocast, _build_model, train
from srno.types import PoseState


ARMS = {"baseline": "gap", "gap_jq": "gap_jq"}
HORIZONS = (4, 8, 16, 32)
SPLITS = ("val", "test")
BOOTSTRAP_SEED = 20260817


def _state_at(state: PoseState, step: int) -> PoseState:
    return PoseState(
        state.rotation[:, step],
        state.position[:, step],
        state.joint_position[:, step],
    )


def _run_config(
    base: ExperimentConfig, output: Path, arm: str, seed: int
) -> ExperimentConfig:
    return replace(
        base,
        seed=seed,
        paths=replace(base.paths, output_dir=output / arm / f"seed-{seed}"),
        model=replace(base.model, contact_features=ARMS[arm]),
    )


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _verify_local_checkpoint(
    path: Path,
    config: ExperimentConfig,
    manifest: DatasetManifest,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing paired local checkpoint: {path}")
    checkpoint = _checkpoint_metadata(path)
    if checkpoint.get("stage") != "local":
        raise ValueError(f"rollout initialization must be local: {path}")
    if checkpoint.get("manifest_sha256") != manifest.sha256():
        raise ValueError(f"local checkpoint manifest mismatch: {path}")
    if checkpoint.get("gripper_sha256") != manifest.gripper_sha256:
        raise ValueError(f"local checkpoint gripper mismatch: {path}")
    saved = checkpoint.get("config", {})
    expected = config.to_dict()
    for key in ("model", "loss", "optimizer", "loader", "training", "seed"):
        saved_value = saved.get(key)
        if key == "model" and isinstance(saved_value, dict):
            saved_value = dict(saved_value)
            saved_value.setdefault("global_conditioning", "aperture")
        if saved_value != expected.get(key):
            raise ValueError(
                f"local checkpoint {key} differs from frozen rollout config: {path}"
            )


def _train_or_resume(config: ExperimentConfig, local_checkpoint: Path) -> None:
    output = config.paths.output_dir
    final = output / "best-rollout-h32.pt"
    if final.is_file():
        checkpoint = _checkpoint_metadata(final)
        if checkpoint.get("stage") == "rollout" and checkpoint.get("horizon") == 32:
            print(f"[JQ-ROLLOUT] reuse completed {final}", flush=True)
            return
        raise RuntimeError(f"invalid completed rollout checkpoint: {final}")
    last = output / "last-rollout.pt"
    if last.is_file():
        print(f"[JQ-ROLLOUT] resume {last}", flush=True)
        train(config, stage="rollout", resume=last)
        return
    if output.exists() and (output / "metrics.jsonl").exists():
        raise RuntimeError(f"rollout metrics exist without a resumable checkpoint: {output}")
    print(
        f"[JQ-ROLLOUT] initialize arm={config.model.contact_features} "
        f"seed={config.seed} from {local_checkpoint}",
        flush=True,
    )
    train(config, stage="rollout", resume=local_checkpoint)


def _metric_arrays(
    model: Any, prediction: PoseState, target: PoseState
) -> dict[str, torch.Tensor]:
    translation_m = torch.linalg.vector_norm(
        prediction.position - target.position, dim=-1
    )
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


def _evaluate(
    config: ExperimentConfig,
    checkpoint_path: Path,
    split: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = DatasetManifest.load(config.paths.manifest)
    device = torch.device(config.device)
    model = _build_model(config, manifest, device)
    checkpoint = load_checkpoint(checkpoint_path, model=model, map_location=device)
    horizon = int(checkpoint["horizon"])
    if horizon not in HORIZONS:
        raise ValueError(f"unexpected rollout horizon {horizon}: {checkpoint_path}")
    if checkpoint["manifest_sha256"] != manifest.sha256():
        raise ValueError("rollout checkpoint manifest mismatch")
    dataset = H5ObjectDataset(manifest, split=split)  # type: ignore[arg-type]
    loader = make_dataloader(
        dataset,
        mode="rollout",
        objects_per_batch=1,
        samples_per_object=0,
        workers=config.loader.workers,
        seed=config.seed + 200 + horizon,
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
        )
    }
    model.eval()
    try:
        with torch.no_grad():
            for raw_batch in loader:
                assert isinstance(raw_batch, TrajectoryBatch)
                if len(raw_batch.object_ids) != 1:
                    raise RuntimeError("paired rollout evaluation requires one object per batch")
                batch = raw_batch.to(device, non_blocking=True)
                with _autocast(config, device):
                    prediction = model.rollout(
                        _state_at(batch.states, 0),
                        batch.command_schedule[1 : horizon + 1],
                        batch.sdf,
                    )
                target = PoseState(
                    batch.states.rotation[:, : horizon + 1],
                    batch.states.position[:, : horizon + 1],
                    batch.states.joint_position[:, : horizon + 1],
                )
                metrics = _metric_arrays(model, prediction, target)
                for name, values in metrics.items():
                    rows[name].append(values.float().cpu().numpy())
                object_index = dataset.object_ids.index(raw_batch.object_ids[0])
                rows["object_index"].append(
                    np.full(len(batch.trajectory_index), object_index, dtype=np.int32)
                )
                rows["trajectory_index"].append(
                    batch.trajectory_index.cpu().numpy().astype(np.int32, copy=False)
                )
    finally:
        dataset.close()
    arrays = {name: np.concatenate(values) for name, values in rows.items()}
    arrays["object_labels"] = np.asarray(dataset.object_ids)
    terminal_names = (
        "dx",
        "translation_over_length",
        "translation_m",
        "rotation_rad",
        "joint_rmse_over_travel",
    )
    by_object: dict[str, dict[str, float]] = {}
    for object_index, object_id in enumerate(dataset.object_ids):
        mask = arrays["object_index"] == object_index
        by_object[object_id] = {
            f"terminal_{name}": float(arrays[name][mask, -1].mean())
            for name in terminal_names
        }
    equal_object = {
        name: float(np.mean([values[name] for values in by_object.values()]))
        for name in next(iter(by_object.values()))
    }
    step_mean = {
        name: arrays[name].mean(axis=0).astype(float).tolist()
        for name in terminal_names
    }
    return (
        {
            "horizon": horizon,
            "trajectory_count": int(len(arrays["object_index"])),
            "equal_object": equal_object,
            "by_object": by_object,
            "step_mean": step_mean,
        },
        arrays,
    )


def _keys(arrays: dict[str, np.ndarray]) -> list[tuple[str, int]]:
    labels = arrays["object_labels"].tolist()
    return [
        (str(labels[int(object_index)]), int(trajectory))
        for object_index, trajectory in zip(
            arrays["object_index"], arrays["trajectory_index"], strict=True
        )
    ]


def _bootstrap(
    evaluations: dict[str, dict[int, dict[int, dict[str, dict[str, np.ndarray]]]]],
    seeds: tuple[int, ...],
    replicates: int = 10_000,
) -> dict[str, float | int]:
    differences: dict[int, dict[str, np.ndarray]] = {}
    labels: list[str] | None = None
    for seed in seeds:
        baseline = evaluations["baseline"][seed][32]["test"]
        candidate = evaluations["gap_jq"][seed][32]["test"]
        if _keys(baseline) != _keys(candidate):
            raise RuntimeError(f"paired H32 trajectory order differs for seed {seed}")
        object_labels = [str(value) for value in baseline["object_labels"].tolist()]
        if labels is None:
            labels = object_labels
        elif labels != object_labels:
            raise RuntimeError("test object order differs between seeds")
        differences[seed] = {}
        for object_index, object_id in enumerate(object_labels):
            mask = baseline["object_index"] == object_index
            differences[seed][object_id] = (
                candidate["dx"][mask, -1] - baseline["dx"][mask, -1]
            )
    assert labels is not None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_means: list[float] = []
        for seed in sampled_seeds:
            sampled_objects = rng.choice(labels, size=len(labels), replace=True)
            object_means: list[float] = []
            for object_id in sampled_objects:
                values = differences[int(seed)][str(object_id)]
                object_means.append(
                    float(rng.choice(values, size=len(values), replace=True).mean())
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
    summaries: dict[str, dict[int, dict[int, dict[str, dict[str, Any]]]]],
    evaluations: dict[str, dict[int, dict[int, dict[str, dict[str, np.ndarray]]]]],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    paired: dict[str, dict[str, dict[str, float]]] = {split: {} for split in SPLITS}
    for split in SPLITS:
        for seed in seeds:
            baseline = summaries["baseline"][seed][32][split]["equal_object"]
            candidate = summaries["gap_jq"][seed][32][split]["equal_object"]
            paired[split][str(seed)] = {
                name: float(candidate[name] - baseline[name]) for name in baseline
            }
    bootstrap = _bootstrap(evaluations, seeds)
    val_all_better = all(
        paired["val"][str(seed)]["terminal_dx"] < 0.0 for seed in seeds
    )
    confirmed = val_all_better and float(bootstrap["ci95_upper"]) < 0.0
    arms: dict[str, Any] = {}
    for arm in ARMS:
        arms[arm] = {}
        for horizon in HORIZONS:
            arms[arm][str(horizon)] = {}
            for split in SPLITS:
                names = summaries[arm][seeds[0]][horizon][split]["equal_object"]
                arms[arm][str(horizon)][split] = {}
                for name in names:
                    values = np.asarray(
                        [
                            summaries[arm][seed][horizon][split]["equal_object"][name]
                            for seed in seeds
                        ],
                        dtype=np.float64,
                    )
                    arms[arm][str(horizon)][split][name] = {
                        "mean": float(values.mean()),
                        "std_across_seeds": float(values.std(ddof=1)),
                        "per_seed": {
                            str(seed): float(value)
                            for seed, value in zip(seeds, values, strict=True)
                        },
                    }
    return {
        "criterion": {
            "validation_h32_jq_better_all_three_seeds": val_all_better,
            "test_h32_bootstrap_ci95_upper_below_zero": float(bootstrap["ci95_upper"]) < 0.0,
            "confirmed_rollout_gain": confirmed,
        },
        "test_h32_hierarchical_bootstrap": bootstrap,
        "paired_h32_candidate_minus_baseline": paired,
        "arms": arms,
    }


def _plot(
    output: Path,
    result: dict[str, Any],
    summaries: dict[str, dict[int, dict[int, dict[str, dict[str, Any]]]]],
    evaluations: dict[str, dict[int, dict[int, dict[str, dict[str, np.ndarray]]]]],
    seeds: tuple[int, ...],
) -> None:
    colors = {"baseline": "#4472C4", "gap_jq": "#E67E22"}
    comparison = result["comparison"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    ax = axes[0, 0]
    for arm in ARMS:
        means = [
            comparison["arms"][arm][str(h)]["test"]["terminal_dx"]["mean"]
            for h in HORIZONS
        ]
        stds = [
            comparison["arms"][arm][str(h)]["test"]["terminal_dx"]["std_across_seeds"]
            for h in HORIZONS
        ]
        ax.errorbar(HORIZONS, means, yerr=stds, marker="o", capsize=3,
                    label=arm, color=colors[arm])
    ax.set_xlabel("rollout horizon")
    ax.set_ylabel(r"terminal $d_X$")
    ax.set_title("Unseen test terminal error")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    for arm in ARMS:
        curves = np.asarray(
            [
                summaries[arm][seed][32]["test"]["step_mean"]["dx"]
                for seed in seeds
            ]
        )
        mean = curves.mean(axis=0)
        std = curves.std(axis=0, ddof=1)
        x = np.arange(len(mean))
        ax.plot(x, mean, label=arm, color=colors[arm])
        ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=colors[arm])
    ax.set_xlabel("closure step k")
    ax.set_ylabel(r"$d_X(k)$")
    ax.set_title("H32 pushforward error: mean ± seed std")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    names = ("terminal_translation_over_length", "terminal_rotation_rad", "terminal_joint_rmse_over_travel")
    labels = ("T/L", "R [rad]", "J/travel")
    x = np.arange(3)
    width = 0.36
    for arm_index, arm in enumerate(ARMS):
        values = comparison["arms"][arm]["32"]["test"]
        means = [values[name]["mean"] for name in names]
        stds = [values[name]["std_across_seeds"] for name in names]
        ax.bar(x + (arm_index - 0.5) * width, means, width, yerr=stds,
               capsize=3, label=arm, color=colors[arm])
    ax.set_xticks(x, labels)
    ax.set_title("H32 terminal components")
    ax.legend()

    ax = axes[1, 1]
    labels_test = evaluations["baseline"][seeds[0]][32]["test"]["object_labels"].tolist()
    values_by_object: list[list[float]] = []
    for object_index, _ in enumerate(labels_test):
        deltas: list[float] = []
        for seed in seeds:
            baseline = evaluations["baseline"][seed][32]["test"]
            candidate = evaluations["gap_jq"][seed][32]["test"]
            mask = baseline["object_index"] == object_index
            deltas.append(float((candidate["dx"][mask, -1] - baseline["dx"][mask, -1]).mean()))
        values_by_object.append(deltas)
    means = np.asarray(values_by_object).mean(axis=1)
    stds = np.asarray(values_by_object).std(axis=1, ddof=1)
    positions = np.arange(len(labels_test))
    ax.bar(positions, means, yerr=stds, capsize=3, color="#70AD47")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(positions, [str(value)[:24] for value in labels_test], rotation=18)
    ax.set_ylabel(r"$d_X^{J_q}-d_X^{gap}$")
    ax.set_title("Per-object H32 test difference")

    fig.suptitle("SRNO rollout representation ablation: gap vs gap + Jq", fontsize=15)
    fig.savefig(output / "jq_rollout_ablation.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    if seeds != (0, 1, 2):
        raise ValueError("the frozen rollout ablation requires seeds 0 1 2")
    base = ExperimentConfig.load(args.config)
    if base.training.rollout_horizons != HORIZONS:
        raise ValueError("the frozen rollout curriculum must be H4/H8/H16/H32")
    manifest = DatasetManifest.load(base.paths.manifest)
    output = args.output.resolve()

    configs: dict[str, dict[int, ExperimentConfig]] = {arm: {} for arm in ARMS}
    for seed in seeds:
        for arm in ARMS:
            config = _run_config(base, output, arm, seed)
            local = args.local_root.resolve() / arm / f"seed-{seed}" / "best-local.pt"
            _verify_local_checkpoint(local, config, manifest)
            configs[arm][seed] = config
            _train_or_resume(config, local)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summaries: dict[str, dict[int, dict[int, dict[str, dict[str, Any]]]]] = {
        arm: {seed: {horizon: {} for horizon in HORIZONS} for seed in seeds}
        for arm in ARMS
    }
    evaluations: dict[str, dict[int, dict[int, dict[str, dict[str, np.ndarray]]]]] = {
        arm: {seed: {horizon: {} for horizon in HORIZONS} for seed in seeds}
        for arm in ARMS
    }
    for arm in ARMS:
        for seed in seeds:
            packed: dict[str, np.ndarray] = {}
            for horizon in HORIZONS:
                checkpoint = (
                    configs[arm][seed].paths.output_dir
                    / f"best-rollout-h{horizon:02d}.pt"
                )
                for split in SPLITS:
                    summary, arrays = _evaluate(configs[arm][seed], checkpoint, split)
                    summaries[arm][seed][horizon][split] = summary
                    evaluations[arm][seed][horizon][split] = arrays
                    for name, value in arrays.items():
                        packed[f"h{horizon}_{split}_{name}"] = value
                    print(
                        f"[JQ-ROLLOUT] evaluate arm={arm} seed={seed} H{horizon} "
                        f"split={split} dx={summary['equal_object']['terminal_dx']:.8f}",
                        flush=True,
                    )
            run_output = configs[arm][seed].paths.output_dir
            (run_output / "rollout_evaluation.json").write_text(
                json.dumps(summaries[arm][seed], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            np.savez_compressed(run_output / "rollout_evaluation.npz", **packed)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result = {
        "definition": {
            "arms": ARMS,
            "dx": "sqrt((||dp||/L)^2 + theta(R,R*)^2 + mean(((r-r*)/travel)^2))",
            "jq": "[-n_G, -(y_G/L cross n_G)]",
            "curriculum": list(HORIZONS),
        },
        "configuration": {
            "base_config": str(args.config.resolve()),
            "local_root": str(args.local_root.resolve()),
            "manifest": str(base.paths.manifest),
            "manifest_sha256": manifest.sha256(),
            "gripper_sha256": manifest.gripper_sha256,
            "seeds": list(seeds),
            "parameter_count": {"baseline": 31_436, "gap_jq": 32_204},
        },
        "runs": summaries,
    }
    result["comparison"] = _summarize(summaries, evaluations, seeds)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot(output, result, summaries, evaluations, seeds)
    print(json.dumps(result["comparison"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
