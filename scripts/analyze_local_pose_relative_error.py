#!/usr/bin/env python3
"""Measure local object-pose error relative to the true one-step motion."""

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

from srno.data.dataset import H5ObjectDataset, LocalTransitionBatch, make_dataloader
from srno.data.index import ActiveIndex
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import rotation_geodesic_angle
from srno.training.checkpoint import load_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _autocast, _build_model
from srno.types import PoseState


SPLITS = ("train", "val", "test")
TRANSLATION_MIN_M = 1e-4
ROTATION_MIN_RAD = 1e-4
POSE_MIN = 1e-3
JOINT_MOTION_MIN = 1e-3


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray, minimum: float) -> np.ndarray:
    ratio = np.full_like(numerator, np.nan, dtype=np.float64)
    valid = denominator >= minimum
    ratio[valid] = numerator[valid] / denominator[valid]
    return ratio


def _evaluate(
    config: ExperimentConfig,
    checkpoint: Path,
    split: str,
) -> dict[str, np.ndarray]:
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
    names = (
        "translation_error_m",
        "translation_motion_m",
        "rotation_error_rad",
        "rotation_motion_rad",
        "pose_error",
        "pose_motion",
        "joint_error",
        "joint_step_motion",
        "joint_contact_residual",
        "object_index",
        "trajectory_index",
        "step_index",
    )
    rows: dict[str, list[np.ndarray]] = {name: [] for name in names}
    loader = make_dataloader(
        dataset,
        mode="local",
        objects_per_batch=1,
        samples_per_object=0,
        workers=config.loader.workers,
        seed=config.seed + 710,
        shuffle=False,
    )
    try:
        with torch.no_grad():
            for raw_batch in loader:
                assert isinstance(raw_batch, LocalTransitionBatch)
                if len(raw_batch.object_ids) != 1:
                    raise RuntimeError("relative-pose evaluation expects one object per batch")
                batch = raw_batch.to(device, non_blocking=True)
                with _autocast(config, device):
                    prediction = model.forward_step(
                        batch.current,
                        batch.next_command,
                        batch.sdf,
                        previous_state=batch.previous,
                    )
                assert isinstance(prediction, PoseState)
                translation_error = torch.linalg.vector_norm(
                    prediction.position - batch.target.position, dim=-1
                )
                translation_motion = torch.linalg.vector_norm(
                    batch.target.position - batch.current.position, dim=-1
                )
                rotation_error = rotation_geodesic_angle(
                    prediction.rotation, batch.target.rotation
                )
                rotation_motion = rotation_geodesic_angle(
                    batch.current.rotation, batch.target.rotation
                )
                pose_error = torch.sqrt(
                    (translation_error / model.length_scale).square()
                    + rotation_error.square()
                )
                pose_motion = torch.sqrt(
                    (translation_motion / model.length_scale).square()
                    + rotation_motion.square()
                )
                joint_error = torch.sqrt(
                    (
                        (prediction.joint_position - batch.target.joint_position)
                        / model.joint_travel_range
                    )
                    .square()
                    .mean(dim=-1)
                )
                joint_step_motion = torch.sqrt(
                    (
                        (batch.target.joint_position - batch.current.joint_position)
                        / model.joint_travel_range
                    )
                    .square()
                    .mean(dim=-1)
                )
                free_joint = model.free_joint_configuration(batch.next_command)
                joint_contact_residual = torch.sqrt(
                    (
                        (batch.target.joint_position - free_joint)
                        / model.joint_travel_range
                    )
                    .square()
                    .mean(dim=-1)
                )
                tensors = {
                    "translation_error_m": translation_error,
                    "translation_motion_m": translation_motion,
                    "rotation_error_rad": rotation_error,
                    "rotation_motion_rad": rotation_motion,
                    "pose_error": pose_error,
                    "pose_motion": pose_motion,
                    "joint_error": joint_error,
                    "joint_step_motion": joint_step_motion,
                    "joint_contact_residual": joint_contact_residual,
                }
                for name, value in tensors.items():
                    rows[name].append(value.float().cpu().numpy())
                object_index = dataset.object_ids.index(raw_batch.object_ids[0])
                count = len(batch.trajectory_index)
                rows["object_index"].append(
                    np.full(count, object_index, dtype=np.int32)
                )
                rows["trajectory_index"].append(
                    batch.trajectory_index.cpu().numpy()
                )
                rows["step_index"].append(batch.step_index.cpu().numpy())
    finally:
        dataset.close()

    arrays = {name: np.concatenate(values) for name, values in rows.items()}
    arrays["translation_ratio"] = _safe_ratio(
        arrays["translation_error_m"],
        arrays["translation_motion_m"],
        TRANSLATION_MIN_M,
    )
    arrays["rotation_ratio"] = _safe_ratio(
        arrays["rotation_error_rad"], arrays["rotation_motion_rad"], ROTATION_MIN_RAD
    )
    arrays["pose_ratio"] = _safe_ratio(
        arrays["pose_error"], arrays["pose_motion"], POSE_MIN
    )
    arrays["joint_step_ratio"] = _safe_ratio(
        arrays["joint_error"], arrays["joint_step_motion"], JOINT_MOTION_MIN
    )
    arrays["joint_contact_ratio"] = _safe_ratio(
        arrays["joint_error"], arrays["joint_contact_residual"], JOINT_MOTION_MIN
    )
    arrays["object_labels"] = np.asarray(dataset.object_ids)
    return arrays


def _ratio_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("relative-error selection is empty")
    return {
        "valid_count": int(len(finite)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p10": float(np.quantile(finite, 0.10)),
        "p25": float(np.quantile(finite, 0.25)),
        "p75": float(np.quantile(finite, 0.75)),
        "p90": float(np.quantile(finite, 0.90)),
        "fraction_below_0_5": float(np.mean(finite < 0.5)),
        "fraction_below_1": float(np.mean(finite < 1.0)),
        "fraction_above_2": float(np.mean(finite > 2.0)),
    }


def _motion_summary(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    total = len(arrays["pose_motion"])
    result: dict[str, Any] = {"sample_count": total}
    specs = {
        "translation": ("translation_ratio", "translation_motion_m", TRANSLATION_MIN_M),
        "rotation": ("rotation_ratio", "rotation_motion_rad", ROTATION_MIN_RAD),
        "pose": ("pose_ratio", "pose_motion", POSE_MIN),
    }
    for name, (ratio_name, motion_name, minimum) in specs.items():
        valid = np.isfinite(arrays[ratio_name])
        result[name] = _ratio_summary(arrays[ratio_name])
        result[name]["minimum_gt_motion"] = minimum
        result[name]["excluded_near_zero_count"] = int(total - valid.sum())
        result[name]["excluded_near_zero_fraction"] = float(1.0 - valid.mean())
        result[name]["gt_motion_median"] = float(np.median(arrays[motion_name]))
    for name, ratio_name, motion_name in (
        ("joint_step", "joint_step_ratio", "joint_step_motion"),
        ("joint_contact", "joint_contact_ratio", "joint_contact_residual"),
    ):
        valid = np.isfinite(arrays[ratio_name])
        result[name] = _ratio_summary(arrays[ratio_name])
        result[name]["minimum_gt_motion"] = JOINT_MOTION_MIN
        result[name]["excluded_near_zero_count"] = int(total - valid.sum())
        result[name]["excluded_near_zero_fraction"] = float(1.0 - valid.mean())
        result[name]["gt_motion_median"] = float(np.median(arrays[motion_name]))
    by_object: dict[str, Any] = {}
    for index, label in enumerate(arrays["object_labels"].tolist()):
        mask = arrays["object_index"] == index
        by_object[str(label)] = {
            name: _ratio_summary(arrays[f"{name}_ratio"][mask])
            for name in ("translation", "rotation", "pose")
            if np.isfinite(arrays[f"{name}_ratio"][mask]).any()
        }
        for name, ratio_name in (
            ("joint_step", "joint_step_ratio"),
            ("joint_contact", "joint_contact_ratio"),
        ):
            if np.isfinite(arrays[ratio_name][mask]).any():
                by_object[str(label)][name] = _ratio_summary(
                    arrays[ratio_name][mask]
                )
    result["by_object"] = by_object
    return result


def _ecdf(ax: plt.Axes, values: np.ndarray, label: str, color: str) -> None:
    values = np.sort(values[np.isfinite(values)])
    y = np.arange(1, len(values) + 1) / len(values)
    ax.plot(values, y, linewidth=2, label=label, color=color)


def _binned_curve(
    motion: np.ndarray, ratio: np.ndarray, bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(ratio) & (motion > 0)
    motion = motion[valid]
    ratio = ratio[valid]
    bucket = np.digitize(motion, bins) - 1
    x: list[float] = []
    median: list[float] = []
    low: list[float] = []
    high: list[float] = []
    for index in range(len(bins) - 1):
        selected = ratio[bucket == index]
        if len(selected) < 10:
            continue
        x.append(float(np.sqrt(bins[index] * bins[index + 1])))
        median.append(float(np.median(selected)))
        low.append(float(np.quantile(selected, 0.10)))
        high.append(float(np.quantile(selected, 0.90)))
    return np.asarray(x), np.asarray(median), np.asarray(low), np.asarray(high)


def _plot(output: Path, raw: dict[str, dict[str, np.ndarray]], summary: dict[str, Any]) -> None:
    colors = {"train": "#4472C4", "val": "#ED7D31", "test": "#70AD47"}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    ax = axes[0, 0]
    for split in SPLITS:
        _ecdf(ax, 100.0 * raw[split]["pose_ratio"], split, colors[split])
    ax.axvline(100.0, color="black", linestyle="--", label="free predictor")
    ax.set_xscale("log")
    ax.set_xlabel("combined pose error / GT motion [%]")
    ax.set_ylabel("ECDF")
    ax.set_title("Relative combined pose error")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    for name, color in (
        ("translation", "#5B9BD5"),
        ("rotation", "#A64D79"),
        ("pose", "#70AD47"),
    ):
        _ecdf(ax, 100.0 * raw["test"][f"{name}_ratio"], name, color)
    ax.axvline(100.0, color="black", linestyle="--", label="free predictor")
    ax.set_xscale("log")
    ax.set_xlabel("prediction error / GT motion [%]")
    ax.set_ylabel("ECDF")
    ax.set_title("Unseen test: T / R / combined")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 2]
    components = ("translation", "rotation", "pose")
    x = np.arange(len(components))
    width = 0.24
    for offset, split in zip((-width, 0.0, width), SPLITS, strict=True):
        values = [
            100.0 * summary["splits"][split][name]["fraction_below_1"]
            for name in components
        ]
        ax.bar(x + offset, values, width, color=colors[split], label=split)
    ax.set_xticks(x, ("translation", "rotation", "combined"))
    ax.set_ylabel("transitions improved over free predictor [%]")
    ax.set_ylim(0, 100)
    ax.set_title("Fraction with relative error < 100%")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    test = raw["test"]
    bins = np.geomspace(POSE_MIN, max(POSE_MIN * 1.01, np.nanmax(test["pose_motion"])), 16)
    bx, median, low, high = _binned_curve(test["pose_motion"], test["pose_ratio"], bins)
    ax.plot(bx, 100.0 * median, "o-", color="#70AD47", label="median")
    ax.fill_between(bx, 100.0 * low, 100.0 * high, color="#70AD47", alpha=0.2, label="p10–p90")
    ax.axhline(100.0, color="black", linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"GT combined pose motion $m_q$")
    ax.set_ylabel("relative combined pose error [%]")
    ax.set_title("Error versus true motion magnitude")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    finite = test["pose_ratio"][np.isfinite(test["pose_ratio"])]
    log_bins = np.geomspace(max(1e-2, np.quantile(finite, 0.001)), np.quantile(finite, 0.999), 60)
    ax.hist(100.0 * finite, bins=100.0 * log_bins, color="#70AD47", alpha=0.8)
    ax.axvline(100.0, color="black", linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("combined pose error / GT motion [%]")
    ax.set_ylabel("transition count")
    ax.set_title("Unseen test distribution")
    ax.grid(alpha=0.25)

    ax = axes[1, 2]
    labels = list(summary["splits"]["test"]["by_object"])
    medians = [
        100.0 * summary["splits"]["test"]["by_object"][label]["pose"]["median"]
        for label in labels
    ]
    p90 = [
        100.0 * summary["splits"]["test"]["by_object"][label]["pose"]["p90"]
        for label in labels
    ]
    positions = np.arange(len(labels))
    ax.bar(positions, medians, color="#5B9BD5", label="median")
    ax.scatter(positions, p90, color="#C00000", marker="_", s=200, label="p90")
    ax.axhline(100.0, color="black", linestyle="--")
    ax.set_xticks(positions, [label[:24] for label in labels], rotation=15)
    ax.set_ylabel("combined relative error [%]")
    ax.set_yscale("log")
    ax.set_title("Unseen test objects")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.suptitle("SRNO L=1 local object-pose error relative to true one-step motion")
    fig.savefig(output / "local_pose_relative_error.png", dpi=180)
    plt.close(fig)


def _plot_joints(
    output: Path,
    raw: dict[str, dict[str, np.ndarray]],
    summary: dict[str, Any],
) -> None:
    colors = {"train": "#4472C4", "val": "#ED7D31", "test": "#70AD47"}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    ax = axes[0, 0]
    for split in SPLITS:
        _ecdf(ax, 100.0 * raw[split]["joint_contact_ratio"], split, colors[split])
    ax.axvline(100.0, color="black", linestyle="--", label="free joints")
    ax.set_xscale("log")
    ax.set_xlabel("joint error / GT contact residual [%]")
    ax.set_ylabel("ECDF")
    ax.set_title("Joint correction relative to free predictor")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    _ecdf(ax, 100.0 * raw["test"]["joint_step_ratio"], "full step", "#A64D79")
    _ecdf(ax, 100.0 * raw["test"]["joint_contact_ratio"], "contact residual", "#70AD47")
    ax.axvline(100.0, color="black", linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("normalized joint error ratio [%]")
    ax.set_ylabel("ECDF")
    ax.set_title("Unseen test: two denominators")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 2]
    x = np.arange(len(SPLITS))
    width = 0.36
    for offset, name, label, color in (
        (-width / 2, "joint_step", "vs full step", "#A64D79"),
        (width / 2, "joint_contact", "vs contact residual", "#70AD47"),
    ):
        values = [
            100.0 * summary["splits"][split][name]["fraction_below_1"]
            for split in SPLITS
        ]
        ax.bar(x + offset, values, width, color=color, label=label)
    ax.set_xticks(x, SPLITS)
    ax.set_ylim(0, 100)
    ax.set_ylabel("transitions with relative error < 100%")
    ax.set_title("Fraction improved over each baseline")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    test = raw["test"]
    maximum = max(JOINT_MOTION_MIN * 1.01, np.nanmax(test["joint_contact_residual"]))
    bins = np.geomspace(JOINT_MOTION_MIN, maximum, 16)
    bx, median, low, high = _binned_curve(
        test["joint_contact_residual"], test["joint_contact_ratio"], bins
    )
    ax.plot(bx, 100.0 * median, "o-", color="#70AD47", label="median")
    ax.fill_between(
        bx, 100.0 * low, 100.0 * high, color="#70AD47", alpha=0.2, label="p10–p90"
    )
    ax.axhline(100.0, color="black", linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"GT normalized contact residual $m_J^{contact}$")
    ax.set_ylabel("joint relative error [%]")
    ax.set_title("Error versus contact-residual magnitude")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    finite = test["joint_contact_ratio"][np.isfinite(test["joint_contact_ratio"])]
    log_bins = np.geomspace(
        max(1e-3, np.quantile(finite, 0.001)), np.quantile(finite, 0.999), 60
    )
    ax.hist(100.0 * finite, bins=100.0 * log_bins, color="#70AD47", alpha=0.8)
    ax.axvline(100.0, color="black", linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("joint error / GT contact residual [%]")
    ax.set_ylabel("transition count")
    ax.set_title("Unseen test distribution")
    ax.grid(alpha=0.25)

    ax = axes[1, 2]
    labels = list(summary["splits"]["test"]["by_object"])
    medians = [
        100.0
        * summary["splits"]["test"]["by_object"][label]["joint_contact"]["median"]
        for label in labels
    ]
    p90 = [
        100.0
        * summary["splits"]["test"]["by_object"][label]["joint_contact"]["p90"]
        for label in labels
    ]
    positions = np.arange(len(labels))
    ax.bar(positions, medians, color="#5B9BD5", label="median")
    ax.scatter(positions, p90, color="#C00000", marker="_", s=200, label="p90")
    ax.axhline(100.0, color="black", linestyle="--")
    ax.set_xticks(positions, [label[:24] for label in labels], rotation=15)
    ax.set_yscale("log")
    ax.set_ylabel("joint contact-relative error [%]")
    ax.set_title("Unseen test objects")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.suptitle("SRNO L=1 local joint error relative to true one-step motion")
    fig.savefig(output / "local_joint_relative_error.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    if config.model.operator_layers != 1:
        raise ValueError("this diagnostic is frozen to operator depth L=1")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = {
        split: _evaluate(config, args.checkpoint.resolve(), split) for split in SPLITS
    }
    summary = {
        "definition": {
            "translation_ratio": "||p_hat-p_next|| / ||p_next-p_current||",
            "rotation_ratio": "theta(R_hat,R_next) / theta(R_current,R_next)",
            "pose_ratio": "sqrt((e_p/L)^2+e_R^2) / sqrt((m_p/L)^2+m_R^2)",
            "joint_step_ratio": "||r_hat-r_next||_s / ||r_next-r_current||_s",
            "joint_contact_ratio": "||r_hat-r_next||_s / ||r_next-r_free(command_next)||_s",
            "free_predictor_ratio": 1.0,
            "near_zero_thresholds": {
                "translation_m": TRANSLATION_MIN_M,
                "rotation_rad": ROTATION_MIN_RAD,
                "pose_dimensionless": POSE_MIN,
                "joint_dimensionless": JOINT_MOTION_MIN,
            },
        },
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "splits": {split: _motion_summary(raw[split]) for split in SPLITS},
    }
    (output / "local_pose_relative_error.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    packed = {
        f"{split}_{name}": values
        for split, arrays in raw.items()
        for name, values in arrays.items()
    }
    np.savez_compressed(output / "local_pose_relative_error.npz", **packed)
    _plot(output, raw, summary)
    _plot_joints(output, raw, summary)
    print(json.dumps(summary["splits"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
