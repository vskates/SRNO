#!/usr/bin/env python3
"""Measure empirical contact-map amplification on a paired PhysX tube."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from srno.data.schema import DatasetManifest
from srno.geometry.gripper import GripperAsset
from srno.geometry.se3 import quaternion_xyzw_to_matrix, rotation_geodesic_angle
from srno.types import PoseState


QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
BANDS = ((1, 8), (9, 16), (17, 24), (25, 31))


def _state(values: Any, prefix: str) -> PoseState:
    return PoseState(
        quaternion_xyzw_to_matrix(torch.from_numpy(
            np.asarray(values[f"{prefix}quaternion_xyzw"], dtype=np.float32)
        )),
        torch.from_numpy(np.asarray(values[f"{prefix}position"], dtype=np.float32)),
        torch.from_numpy(np.asarray(values[f"{prefix}joint"], dtype=np.float32)),
    )


def _distance(
    first: PoseState,
    second: PoseState,
    *,
    length_scale: float,
    joint_scale: torch.Tensor,
) -> np.ndarray:
    translation = torch.linalg.vector_norm(first.position - second.position, dim=-1) / length_scale
    rotation = rotation_geodesic_angle(first.rotation, second.rotation)
    joints = torch.sqrt(
        (((first.joint_position - second.joint_position) / joint_scale).square()).mean(dim=-1)
    )
    return torch.sqrt(
        translation.square() + rotation.square() + joints.square()
    ).numpy()


def _summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    result: dict[str, float | int] = {
        "count": int(len(finite)),
        "mean": float(finite.mean()),
    }
    result.update({
        f"q{int(round(100 * quantile)):02d}": float(value)
        for quantile, value in zip(QUANTILES, np.quantile(finite, QUANTILES), strict=True)
    })
    return result


def _hierarchical_bootstrap(
    values: np.ndarray,
    object_ids: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    objects = np.unique(object_ids)
    groups = {name: np.flatnonzero(object_ids == name) for name in objects}
    generator = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled_objects = generator.choice(objects, size=len(objects), replace=True)
        sampled_rows = np.concatenate([
            generator.choice(groups[name], size=len(groups[name]), replace=True)
            for name in sampled_objects
        ])
        draws[repetition] = statistic(values[sampled_rows])
    return {
        "estimate": float(statistic(values)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "repetitions": repetitions,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _plot(
    path: Path,
    input_separation: np.ndarray,
    successor_separation: np.ndarray,
    amplification: np.ndarray,
    steps: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    scatter = axes[0].scatter(
        input_separation,
        successor_separation,
        c=steps,
        cmap="viridis",
        s=22,
        alpha=0.8,
        linewidths=0.0,
    )
    lower = max(1e-5, min(float(input_separation.min()), float(successor_separation.min())))
    upper = max(float(input_separation.max()), float(successor_separation.max()))
    axes[0].plot((lower, upper), (lower, upper), color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"input separation $d_X(\hat x_k,x_k^*)$")
    axes[0].set_ylabel(r"successor separation $d_X(F(\hat x_k),F(x_k^*))$")
    axes[0].set_title("Paired PhysX map")
    figure.colorbar(scatter, ax=axes[0], label="closure step")

    band_values = [
        amplification[(steps >= first) & (steps <= last)]
        for first, last in BANDS
    ]
    axes[1].boxplot(band_values, tick_labels=[f"{a}-{b}" for a, b in BANDS], showfliers=True)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("horizon band")
    axes[1].set_ylabel("amplification ratio")
    axes[1].set_title("Contact-map amplification")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-labels", type=Path, required=True)
    parser.add_argument("--nominal-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive")

    manifest = DatasetManifest.load(args.manifest)
    gripper = GripperAsset.load(manifest.gripper_path)
    model = np.load(args.model_labels, allow_pickle=False)
    nominal = np.load(args.nominal_labels, allow_pickle=False)
    for labels in (model, nominal):
        if str(labels["manifest_sha256"]) != manifest.sha256():
            raise ValueError("label and manifest hashes differ")
        if not np.asarray(labels["settled"], dtype=bool).all():
            raise ValueError("stability analysis requires settled paired labels")
    for name in ("object_id", "trajectory", "current_step", "source_pose_index"):
        if not np.array_equal(model[name], nominal[name]):
            raise ValueError(f"paired label key differs: {name}")
    if str(nominal["collection_state_source"]) != "nominal":
        raise ValueError("nominal labels were not collected from nominal states")

    scale = gripper.joint_travel_range.float().clamp_min(1e-8)
    distance = lambda first, second: _distance(  # noqa: E731
        first, second,
        length_scale=manifest.length_scale_m,
        joint_scale=scale,
    )
    input_separation = distance(_state(model, "model_"), _state(model, "nominal_current_"))
    successor_separation = distance(_state(model, "successor_"), _state(nominal, "successor_"))
    reset_floor = distance(_state(nominal, "successor_"), _state(nominal, "nominal_successor_"))
    model_to_recorded = distance(_state(model, "successor_"), _state(model, "nominal_successor_"))
    amplification = successor_separation / np.maximum(input_separation, 1e-6)
    floor_separation = successor_separation / np.maximum(reset_floor, 1e-4)
    object_ids = np.asarray(model["object_id"]).astype(str)
    steps = np.asarray(model["current_step"])

    metrics = {
        "input_separation_dx": input_separation,
        "successor_separation_dx": successor_separation,
        "nominal_reset_floor_dx": reset_floor,
        "model_successor_to_recorded_dx": model_to_recorded,
        "amplification_ratio": amplification,
        "successor_over_reset_floor": floor_separation,
    }
    result: dict[str, Any] = {
        "definition": {
            "input_separation_dx": "d(x_hat_k, x^*_k)",
            "successor_separation_dx": "d(F_PhysX(x_hat_k), F_PhysX(x^*_k))",
            "nominal_reset_floor_dx": "d(F_PhysX(x^*_k), x^*_{k+1})",
            "amplification_ratio": "successor separation / input separation",
            "pairing": "same object, source pose, trajectory, command, and fresh-scene protocol",
        },
        "contract": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest.sha256(),
            "model_labels": str(args.model_labels.resolve()),
            "nominal_labels": str(args.nominal_labels.resolve()),
            "samples": int(len(steps)),
            "objects": int(len(np.unique(object_ids))),
        },
        "all": {name: _summary(values) for name, values in metrics.items()},
        "fractions": {
            "amplification_gt_1": float(np.mean(amplification > 1.0)),
            "amplification_gt_2": float(np.mean(amplification > 2.0)),
            "amplification_gt_5": float(np.mean(amplification > 5.0)),
            "amplification_gt_10": float(np.mean(amplification > 10.0)),
        },
        "bands": {},
        "bootstrap": {},
    }
    for first, last in BANDS:
        mask = (steps >= first) & (steps <= last)
        band_ids = object_ids[mask]
        band_amplification = amplification[mask]
        result["bands"][f"{first:02d}-{last:02d}"] = {
            name: _summary(values[mask]) for name, values in metrics.items()
        }
        result["bands"][f"{first:02d}-{last:02d}"]["fractions"] = {
            "amplification_gt_1": float(np.mean(band_amplification > 1.0)),
            "amplification_gt_2": float(np.mean(band_amplification > 2.0)),
        }
        result["bands"][f"{first:02d}-{last:02d}"]["bootstrap"] = {
            "median_amplification": _hierarchical_bootstrap(
                band_amplification, band_ids, lambda value: float(np.median(value)),
                repetitions=args.bootstrap, seed=args.seed + first,
            ),
            "fraction_expanding": _hierarchical_bootstrap(
                band_amplification, band_ids, lambda value: float(np.mean(value > 1.0)),
                repetitions=args.bootstrap, seed=args.seed + last,
            ),
        }
    result["bootstrap"] = {
        "median_amplification": _hierarchical_bootstrap(
            amplification, object_ids, lambda value: float(np.median(value)),
            repetitions=args.bootstrap, seed=args.seed + 101,
        ),
        "fraction_expanding": _hierarchical_bootstrap(
            amplification, object_ids, lambda value: float(np.mean(value > 1.0)),
            repetitions=args.bootstrap, seed=args.seed + 102,
        ),
        "mean_successor_separation_dx": _hierarchical_bootstrap(
            successor_separation, object_ids, lambda value: float(np.mean(value)),
            repetitions=args.bootstrap, seed=args.seed + 103,
        ),
        "mean_reset_floor_dx": _hierarchical_bootstrap(
            reset_floor, object_ids, lambda value: float(np.mean(value)),
            repetitions=args.bootstrap, seed=args.seed + 104,
        ),
    }
    _write_json(args.output, result)
    _plot(
        args.output.with_suffix(".png"),
        input_separation,
        successor_separation,
        amplification,
        steps,
    )
    print(json.dumps({
        "samples": result["contract"]["samples"],
        "objects": result["contract"]["objects"],
        "input_separation_dx": result["all"]["input_separation_dx"],
        "successor_separation_dx": result["all"]["successor_separation_dx"],
        "nominal_reset_floor_dx": result["all"]["nominal_reset_floor_dx"],
        "amplification_ratio": result["all"]["amplification_ratio"],
        "fractions": result["fractions"],
        "late_band": result["bands"]["25-31"],
        "bootstrap": result["bootstrap"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
