#!/usr/bin/env python3
"""Test whether the recorded SRNO-r snapshot determines its local successor.

The diagnostic is deliberately non-parametric and does not train a model.  For
every active transition it searches only other trajectories of the same object
at the same command step.  It compares nearest neighbours under the production
operator input, the full recorded snapshot, current-contact geometry, previous
increments, and the diagnostics that are actually present in material-v2.

This cannot prove non-observability from merely close (rather than identical)
states: a discontinuous but deterministic contact map has the same signature.
Exact/replicated-state controls and an outgoing-contact-count oracle are
reported separately so that those two cases are not conflated.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from srno.data.index import ActiveIndex, file_sha256
from srno.data.schema import DatasetManifest
from srno.geometry.gripper import GripperAsset
from srno.geometry.se3 import quaternion_xyzw_to_matrix
from srno.model import SRNOModel
from srno.types import PoseState, SDFBatch


SPLITS = ("train", "val", "test")
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
METHODS = (
    "operator",
    "operator_current_gap",
    "snapshot",
    "history_pose_k05",
    "history_pose_k10",
    "history_pose_k20",
    "history_full_k10",
    "history_pose_radius125_k20",
    "history_full_radius125_k20",
    "incoming_diagnostics_k10",
    "incoming_diagnostics_radius125_k20",
    "incoming_contact_count_k10",
    "outgoing_contact_count_oracle_k10",
)


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    result: dict[str, float | int | None] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
    }
    if not finite.size:
        return result
    result.update(
        {
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            **{
                f"q{int(round(100 * quantile)):02d}": float(
                    np.quantile(finite, quantile)
                )
                for quantile in QUANTILES
            },
        }
    )
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    return value / np.linalg.norm(value, axis=-1, keepdims=True).clip(min=1e-15)


def _quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=np.float64).copy()
    result[..., :3] *= -1.0
    return result


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )


def _relative_quaternion(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    return _normalize_quaternion(
        _quaternion_multiply(
            _normalize_quaternion(target),
            _quaternion_conjugate(_normalize_quaternion(source)),
        )
    )


def _pairwise_rotation_angle(quaternion: np.ndarray) -> np.ndarray:
    value = _normalize_quaternion(quaternion)
    cosine_half = np.clip(np.abs(value @ value.T), 0.0, 1.0)
    return 2.0 * np.arccos(cosine_half)


def _paired_rotation_angle(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_value = _normalize_quaternion(left)
    right_value = _normalize_quaternion(right)
    cosine_half = np.clip(
        np.abs(np.sum(left_value * right_value, axis=-1)), 0.0, 1.0
    )
    return 2.0 * np.arccos(cosine_half)


def _pairwise_rms(features: np.ndarray) -> np.ndarray:
    value = np.asarray(features, dtype=np.float64)
    squared = np.sum(value * value, axis=-1)
    distance_squared = (
        squared[:, None] + squared[None, :] - 2.0 * value @ value.T
    ) / value.shape[-1]
    return np.sqrt(np.maximum(distance_squared, 0.0))


def _pairwise_norm(features: np.ndarray) -> np.ndarray:
    value = np.asarray(features, dtype=np.float64)
    squared = np.sum(value * value, axis=-1)
    distance_squared = squared[:, None] + squared[None, :] - 2.0 * value @ value.T
    return np.sqrt(np.maximum(distance_squared, 0.0))


def _nearest(distance: np.ndarray) -> np.ndarray:
    value = np.asarray(distance, dtype=np.float64).copy()
    np.fill_diagonal(value, np.inf)
    return np.argmin(value, axis=1)


def _top_k(primary: np.ndarray, count: int) -> np.ndarray:
    value = np.asarray(primary, dtype=np.float64).copy()
    np.fill_diagonal(value, np.inf)
    count = min(count, value.shape[1] - 1)
    return np.argsort(value, axis=1, kind="stable")[:, :count]


def _rerank(primary: np.ndarray, secondary: np.ndarray, count: int) -> np.ndarray:
    candidates = _top_k(primary, count)
    rows = np.arange(primary.shape[0])[:, None]
    secondary_values = secondary[rows, candidates]
    primary_values = primary[rows, candidates]
    # A tiny primary-distance term provides deterministic tie breaking without
    # materially changing the secondary ordering.
    scale = np.nanmax(primary_values, axis=1, keepdims=True).clip(min=1e-12)
    score = secondary_values + 1e-9 * primary_values / scale
    return candidates[np.arange(len(candidates)), np.argmin(score, axis=1)]


def _constrained_rerank(
    primary: np.ndarray,
    secondary: np.ndarray,
    count: int,
    *,
    maximum_ratio: float,
) -> np.ndarray:
    """Rerank without allowing a substantially less similar snapshot."""

    candidates = _top_k(primary, count)
    rows = np.arange(primary.shape[0])[:, None]
    primary_values = primary[rows, candidates]
    secondary_values = secondary[rows, candidates]
    nearest = primary_values[:, :1]
    eligible = primary_values <= nearest * maximum_ratio + 1e-12
    score = np.where(eligible, secondary_values, np.inf)
    return candidates[np.arange(len(candidates)), np.argmin(score, axis=1)]


def _same_label_rerank(
    primary: np.ndarray, labels: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    candidates = _top_k(primary, count)
    rows = np.arange(primary.shape[0])[:, None]
    match = labels[candidates] == labels[:, None]
    first = np.argmax(match, axis=1)
    covered = np.any(match, axis=1)
    selected = candidates[np.arange(len(candidates)), first]
    fallback = candidates[:, 0]
    return np.where(covered, selected, fallback), covered


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left_value = np.asarray(left, dtype=np.float64)
    right_value = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left_value) & np.isfinite(right_value)
    if finite.sum() < 3:
        return None
    left_order = np.argsort(np.argsort(left_value[finite], kind="stable"), kind="stable")
    right_order = np.argsort(
        np.argsort(right_value[finite], kind="stable"), kind="stable"
    )
    correlation = np.corrcoef(left_order, right_order)[0, 1]
    return float(correlation) if np.isfinite(correlation) else None


def _method_errors(
    neighbour: np.ndarray,
    target_translation: np.ndarray,
    target_rotation: np.ndarray,
    target_joint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    translation = np.linalg.norm(
        target_translation - target_translation[neighbour], axis=-1
    )
    rotation = _paired_rotation_angle(
        target_rotation, target_rotation[neighbour]
    )
    pose = np.sqrt(translation**2 + rotation**2)
    joint = np.sqrt(
        np.mean((target_joint - target_joint[neighbour]) ** 2, axis=-1)
    )
    total = np.sqrt(pose**2 + joint**2)
    return total, pose, translation, rotation, joint


def _pair_components(
    neighbour: np.ndarray,
    position: np.ndarray,
    quaternion: np.ndarray,
    joint: np.ndarray,
    *,
    length_scale: float,
    joint_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    translation_m = np.linalg.norm(position - position[neighbour], axis=-1)
    rotation = _paired_rotation_angle(quaternion, quaternion[neighbour])
    joint_rms = np.sqrt(
        np.mean(((joint - joint[neighbour]) / joint_scale) ** 2, axis=-1)
    )
    return translation_m, rotation, joint_rms


def _equal_object_mean(
    values: np.ndarray, object_index: np.ndarray
) -> tuple[float, dict[str, float]]:
    labels = np.unique(object_index)
    per_object = {
        str(int(label)): float(np.mean(values[object_index == label]))
        for label in labels
    }
    return float(np.mean(list(per_object.values()))), per_object


def _paired_object_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    object_index: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | list[float]]:
    labels = np.unique(object_index)
    base = np.asarray([baseline[object_index == label].mean() for label in labels])
    cand = np.asarray([candidate[object_index == label].mean() for label in labels])
    observed = float(cand.mean() - base.mean())
    observed_percent = float(100.0 * (cand.mean() / base.mean() - 1.0))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        chosen = rng.integers(0, len(labels), size=len(labels))
        draws[index] = 100.0 * (cand[chosen].mean() / base[chosen].mean() - 1.0)
    return {
        "equal_object_delta": observed,
        "equal_object_change_percent": observed_percent,
        "object_bootstrap_change_percent_ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
    }


def _binary_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(target, dtype=np.bool_)
    predicted = np.asarray(prediction, dtype=np.bool_)
    true_positive = int(np.sum(truth & predicted))
    true_negative = int(np.sum(~truth & ~predicted))
    false_positive = int(np.sum(~truth & predicted))
    false_negative = int(np.sum(truth & ~predicted))
    recall = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    return {
        "samples": int(len(truth)),
        "positives": int(truth.sum()),
        "prevalence": float(truth.mean()),
        "predicted_positive_fraction": float(predicted.mean()),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "precision": float(precision),
        "f1": float(
            2.0 * precision * recall / max(precision + recall, 1e-15)
        ),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


@torch.inference_mode()
def _contact_gaps(
    model: SRNOModel,
    sdf: SDFBatch,
    position: np.ndarray,
    quaternion: np.ndarray,
    joint: np.ndarray,
    next_command: float,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    position_tensor = torch.from_numpy(position).to(device=device, dtype=torch.float32)
    quaternion_tensor = torch.from_numpy(quaternion).to(
        device=device, dtype=torch.float32
    )
    rotation = quaternion_xyzw_to_matrix(quaternion_tensor)
    joint_tensor = torch.from_numpy(joint).to(device=device, dtype=torch.float32)
    current = PoseState(rotation, position_tensor, joint_tensor)
    current_gap = model.query_gap(current, sdf)
    command = torch.tensor(next_command, device=device, dtype=torch.float32)
    free_joint = model.free_joint_configuration(command).unsqueeze(0).expand(
        len(position), -1
    )
    trial = PoseState(rotation, position_tensor, free_joint)
    trial_gap = model.query_gap(trial, sdf)
    return current_gap.cpu().numpy(), trial_gap.cpu().numpy()


def _make_plot(output: Path, arrays: dict[str, np.ndarray], summary: dict[str, Any]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.7), constrained_layout=True)
    split_names = [str(value) for value in arrays["split_label"]]
    split_index = arrays["split_index"]
    has_history = arrays["has_history"].astype(bool)
    selected_methods = (
        "operator",
        "operator_current_gap",
        "snapshot",
        "history_pose_k10",
        "incoming_contact_count_k10",
        "outgoing_contact_count_oracle_k10",
    )
    width = 0.12
    x = np.arange(len(SPLITS), dtype=np.float64)
    for method_index, method in enumerate(selected_methods):
        means = []
        for split in SPLITS:
            mask = has_history & (split_index == split_names.index(split))
            mean, _ = _equal_object_mean(
                arrays[f"{method}_pose"][mask], arrays["object_index"][mask]
            )
            means.append(mean)
        axes[0].bar(
            x + (method_index - 2.5) * width,
            means,
            width=width,
            label=method.replace("_k10", ""),
        )
    axes[0].set_xticks(x, SPLITS)
    axes[0].set_ylabel("1-NN target pose divergence")
    axes[0].set_title("Same object and command step")
    axes[0].legend(fontsize=7)

    test_mask = has_history & (split_index == split_names.index("test"))
    order = np.argsort(arrays["snapshot_distance"][test_mask])
    if len(order) > 6000:
        order = order[np.linspace(0, len(order) - 1, 6000).astype(int)]
    test_x = arrays["snapshot_distance"][test_mask][order]
    test_y = arrays["snapshot_pose"][test_mask][order]
    color = arrays["snapshot_neighbour_history_pose_distance"][test_mask][order]
    scatter = axes[1].scatter(test_x, test_y, c=color, s=7, alpha=0.55, cmap="viridis")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("nearest recorded-state distance")
    axes[1].set_ylabel("successor pose-increment divergence")
    axes[1].set_title("Test nearest pairs")
    figure.colorbar(scatter, ax=axes[1], label="previous pose-increment distance")

    for method in ("snapshot", "history_pose_k10", "outgoing_contact_count_oracle_k10"):
        values = np.sort(arrays[f"{method}_pose"][test_mask])
        axes[2].plot(values, np.linspace(0.0, 1.0, len(values)), label=method)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("1-NN target pose divergence")
    axes[2].set_ylabel("empirical CDF")
    axes[2].set_title("Test, steps with history")
    axes[2].legend(fontsize=8)
    figure.savefig(output / "conditional_observability.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--active-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")

    manifest = DatasetManifest.load(args.manifest)
    active = ActiveIndex.load(args.active_index)
    if active.manifest_sha256 != manifest.sha256():
        raise ValueError("active index does not match manifest")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    gripper = GripperAsset.load(manifest.gripper_path)
    if gripper.sha256() != manifest.gripper_sha256:
        raise ValueError("manifest gripper hash mismatch")
    torch.manual_seed(args.seed)
    model = SRNOModel(
        gripper,
        sdf_scale=manifest.sdf_scale_m,
        delta_gate=manifest.delta_gate_m,
        contact_offset_sum=manifest.contact_offset_sum_m,
    ).to(device)
    model.eval()
    schedule = np.asarray(manifest.commanded_aperture_m, dtype=np.float64)
    joint_scale = gripper.joint_travel_range.cpu().numpy().astype(np.float64)
    object_to_split = {
        object_id: split
        for split in SPLITS
        for object_id in manifest.splits[split]
    }
    split_labels = np.asarray(SPLITS)
    object_labels = np.asarray(active.object_ids)
    sample_parts: dict[str, list[np.ndarray]] = {}

    def append(name: str, value: np.ndarray) -> None:
        sample_parts.setdefault(name, []).append(np.asarray(value))

    for object_index, object_id in enumerate(active.object_ids):
        shard_path, group_name = manifest.object_locations()[object_id]
        with h5py.File(shard_path, "r", swmr=True) as handle:
            group = handle[group_name]
            position_all = np.asarray(group["position"], dtype=np.float64)
            quaternion_all = _normalize_quaternion(
                np.asarray(group["quaternion_xyzw"], dtype=np.float64)
            )
            joint_all = np.asarray(group["joint_position"], dtype=np.float64)
            aperture_all = np.asarray(group["actual_aperture"], dtype=np.float64)
            source_pose = np.asarray(group["source_pose_index"], dtype=np.int64)
            diagnostics = {
                name: np.asarray(group[f"diagnostics/{name}"])
                for name in (
                    "contact_count",
                    "actuator_effort",
                    "linear_velocity",
                    "angular_velocity",
                    "settling_substeps",
                )
            }
            sdf = SDFBatch(
                torch.from_numpy(np.asarray(group["sdf"], dtype=np.float32))
                .unsqueeze(0)
                .to(device),
                torch.from_numpy(
                    np.asarray(group.attrs["grid_origin"], dtype=np.float32)
                )
                .unsqueeze(0)
                .to(device),
                torch.from_numpy(
                    np.asarray(group.attrs["voxel_size"], dtype=np.float32)
                )
                .unsqueeze(0)
                .to(device),
                torch.zeros(1, dtype=torch.long, device=device),
                manifest.sdf_scale_m,
            )
            pairs = active.pairs_for(object_id)
            for step in np.unique(pairs[:, 1]):
                trajectory = pairs[pairs[:, 1] == step, 0].astype(np.int64)
                step = int(step)
                if len(trajectory) < 2:
                    continue
                position = position_all[trajectory, step]
                quaternion = quaternion_all[trajectory, step]
                joint = joint_all[trajectory, step]
                aperture = aperture_all[trajectory, step]
                target_position = position_all[trajectory, step + 1]
                target_quaternion = quaternion_all[trajectory, step + 1]
                target_joint_raw = joint_all[trajectory, step + 1]
                current_gap, trial_gap = _contact_gaps(
                    model,
                    SDFBatch(
                        sdf.values,
                        sdf.origin,
                        sdf.voxel_size,
                        torch.zeros(len(trajectory), dtype=torch.long, device=device),
                        sdf.outside_value,
                    ),
                    position,
                    quaternion,
                    joint,
                    float(schedule[step + 1]),
                    device=device,
                )

                current_translation_distance = _pairwise_norm(
                    position / manifest.length_scale_m
                )
                current_rotation_distance = _pairwise_rotation_angle(quaternion)
                current_joint_distance = _pairwise_rms(joint / joint_scale)
                snapshot_distance = np.sqrt(
                    current_translation_distance**2
                    + current_rotation_distance**2
                    + current_joint_distance**2
                )
                trial_gap_distance = _pairwise_rms(
                    trial_gap / manifest.sdf_scale_m
                )
                aperture_distance = np.abs(
                    aperture[:, None] - aperture[None, :]
                ) / manifest.length_scale_m
                operator_distance = np.sqrt(
                    trial_gap_distance**2 + aperture_distance**2
                )
                current_gap_distance = _pairwise_rms(
                    current_gap / manifest.sdf_scale_m
                )
                operator_current_gap_distance = np.sqrt(
                    operator_distance**2 + current_gap_distance**2
                )

                target_translation = (
                    target_position - position
                ) / manifest.length_scale_m
                target_rotation = _relative_quaternion(
                    target_quaternion, quaternion
                )
                free_joint = (
                    model.free_joint_configuration(
                        torch.tensor(
                            float(schedule[step + 1]),
                            dtype=torch.float32,
                            device=device,
                        )
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                target_joint = (target_joint_raw - free_joint) / joint_scale
                target_pose_motion = np.sqrt(
                    np.sum(target_translation**2, axis=-1)
                    + _paired_rotation_angle(
                        target_rotation,
                        np.broadcast_to(
                            np.asarray([0.0, 0.0, 0.0, 1.0]),
                            target_rotation.shape,
                        ),
                    )
                    ** 2
                )

                has_history = step > 0
                if has_history:
                    previous_position = position_all[trajectory, step - 1]
                    previous_quaternion = quaternion_all[trajectory, step - 1]
                    previous_joint = joint_all[trajectory, step - 1]
                    history_translation = (
                        position - previous_position
                    ) / manifest.length_scale_m
                    history_rotation = _relative_quaternion(
                        quaternion, previous_quaternion
                    )
                    history_joint = (joint - previous_joint) / joint_scale
                    history_translation_distance = _pairwise_norm(history_translation)
                    history_rotation_distance = _pairwise_rotation_angle(
                        history_rotation
                    )
                    history_joint_distance = _pairwise_rms(history_joint)
                    history_pose_distance = np.sqrt(
                        history_translation_distance**2
                        + history_rotation_distance**2
                    )
                    history_full_distance = np.sqrt(
                        history_pose_distance**2 + history_joint_distance**2
                    )
                    persistence_translation = np.linalg.norm(
                        target_translation - history_translation, axis=-1
                    )
                    persistence_rotation = _paired_rotation_angle(
                        target_rotation, history_rotation
                    )
                    persistence_pose = np.sqrt(
                        persistence_translation**2 + persistence_rotation**2
                    )
                    history_pose_motion = np.sqrt(
                        np.sum(history_translation**2, axis=-1)
                        + _paired_rotation_angle(
                            history_rotation,
                            np.broadcast_to(
                                np.asarray([0.0, 0.0, 0.0, 1.0]),
                                history_rotation.shape,
                            ),
                        )
                        ** 2
                    )
                    incoming_contact = diagnostics["contact_count"][
                        trajectory, step - 1
                    ].astype(np.float64)
                    incoming_raw = np.column_stack(
                        (
                            incoming_contact,
                            diagnostics["actuator_effort"][trajectory, step - 1],
                            diagnostics["linear_velocity"][trajectory, step - 1],
                            diagnostics["angular_velocity"][trajectory, step - 1],
                            diagnostics["settling_substeps"][trajectory, step - 1],
                        )
                    ).astype(np.float64)
                    scale = incoming_raw.std(axis=0)
                    usable = scale > 1e-12
                    if np.any(usable):
                        normalized_diagnostics = (
                            incoming_raw[:, usable]
                            - incoming_raw[:, usable].mean(axis=0)
                        ) / scale[usable]
                        incoming_diagnostic_distance = _pairwise_rms(
                            normalized_diagnostics
                        )
                    else:
                        incoming_diagnostic_distance = np.zeros_like(
                            snapshot_distance
                        )
                else:
                    history_translation = np.zeros_like(position)
                    history_rotation = np.broadcast_to(
                        np.asarray([0.0, 0.0, 0.0, 1.0]), quaternion.shape
                    ).copy()
                    history_joint = np.zeros_like(joint)
                    history_pose_distance = np.zeros_like(snapshot_distance)
                    history_full_distance = np.zeros_like(snapshot_distance)
                    persistence_pose = target_pose_motion.copy()
                    history_pose_motion = np.zeros(len(trajectory), dtype=np.float64)
                    incoming_contact = np.full(len(trajectory), np.nan)
                    incoming_diagnostic_distance = np.zeros_like(snapshot_distance)
                outgoing_contact = diagnostics["contact_count"][trajectory, step].astype(
                    np.float64
                )

                neighbours: dict[str, np.ndarray] = {
                    "operator": _nearest(operator_distance),
                    "operator_current_gap": _nearest(
                        operator_current_gap_distance
                    ),
                    "snapshot": _nearest(snapshot_distance),
                }
                if has_history:
                    for count in (5, 10, 20):
                        neighbours[f"history_pose_k{count:02d}"] = _rerank(
                            snapshot_distance, history_pose_distance, count
                        )
                    neighbours["history_full_k10"] = _rerank(
                        snapshot_distance, history_full_distance, 10
                    )
                    neighbours["history_pose_radius125_k20"] = _constrained_rerank(
                        snapshot_distance,
                        history_pose_distance,
                        20,
                        maximum_ratio=1.25,
                    )
                    neighbours["history_full_radius125_k20"] = _constrained_rerank(
                        snapshot_distance,
                        history_full_distance,
                        20,
                        maximum_ratio=1.25,
                    )
                    neighbours["incoming_diagnostics_k10"] = _rerank(
                        snapshot_distance, incoming_diagnostic_distance, 10
                    )
                    neighbours[
                        "incoming_diagnostics_radius125_k20"
                    ] = _constrained_rerank(
                        snapshot_distance,
                        incoming_diagnostic_distance,
                        20,
                        maximum_ratio=1.25,
                    )
                    (
                        neighbours["incoming_contact_count_k10"],
                        incoming_coverage,
                    ) = _same_label_rerank(snapshot_distance, incoming_contact, 10)
                else:
                    for name in (
                        "history_pose_k05",
                        "history_pose_k10",
                        "history_pose_k20",
                        "history_full_k10",
                        "history_pose_radius125_k20",
                        "history_full_radius125_k20",
                        "incoming_diagnostics_k10",
                        "incoming_diagnostics_radius125_k20",
                        "incoming_contact_count_k10",
                    ):
                        neighbours[name] = neighbours["snapshot"]
                    incoming_coverage = np.zeros(len(trajectory), dtype=np.bool_)
                (
                    neighbours["outgoing_contact_count_oracle_k10"],
                    outgoing_coverage,
                ) = _same_label_rerank(snapshot_distance, outgoing_contact, 10)

                append("split_index", np.full(len(trajectory), SPLITS.index(object_to_split[object_id]), dtype=np.int8))
                append("object_index", np.full(len(trajectory), object_index, dtype=np.int16))
                append("trajectory", trajectory.astype(np.int16))
                append("source_pose_index", source_pose[trajectory])
                append("step", np.full(len(trajectory), step, dtype=np.int8))
                append("has_history", np.full(len(trajectory), has_history, dtype=np.bool_))
                append("target_pose_motion", target_pose_motion.astype(np.float32))
                append("persistence_pose", persistence_pose.astype(np.float32))
                append("history_pose_motion", history_pose_motion.astype(np.float32))
                append("incoming_contact_count", incoming_contact.astype(np.float32))
                append("outgoing_contact_count", outgoing_contact.astype(np.float32))
                append("incoming_contact_rerank_covered", incoming_coverage)
                append("outgoing_contact_oracle_covered", outgoing_coverage)

                snapshot_neighbour = neighbours["snapshot"]
                append(
                    "snapshot_distance",
                    snapshot_distance[np.arange(len(trajectory)), snapshot_neighbour].astype(np.float32),
                )
                append(
                    "operator_distance",
                    operator_distance[
                        np.arange(len(trajectory)), neighbours["operator"]
                    ].astype(np.float32),
                )
                append(
                    "operator_current_gap_distance",
                    operator_current_gap_distance[
                        np.arange(len(trajectory)),
                        neighbours["operator_current_gap"],
                    ].astype(np.float32),
                )
                snapshot_translation_m, snapshot_rotation, snapshot_joint = _pair_components(
                    snapshot_neighbour,
                    position,
                    quaternion,
                    joint,
                    length_scale=manifest.length_scale_m,
                    joint_scale=joint_scale,
                )
                append("snapshot_translation_m", snapshot_translation_m.astype(np.float32))
                append("snapshot_rotation_rad", snapshot_rotation.astype(np.float32))
                append("snapshot_joint_rms", snapshot_joint.astype(np.float32))
                append(
                    "snapshot_neighbour_history_pose_distance",
                    history_pose_distance[
                        np.arange(len(trajectory)), snapshot_neighbour
                    ].astype(np.float32),
                )
                append(
                    "snapshot_neighbour_incoming_diagnostic_distance",
                    incoming_diagnostic_distance[
                        np.arange(len(trajectory)), snapshot_neighbour
                    ].astype(np.float32),
                )
                append(
                    "snapshot_neighbour_same_source_pose",
                    (source_pose[trajectory] == source_pose[trajectory][snapshot_neighbour]),
                )
                append(
                    "snapshot_neighbour_same_incoming_contact_count",
                    (incoming_contact == incoming_contact[snapshot_neighbour]) & has_history,
                )
                append(
                    "snapshot_neighbour_same_outgoing_contact_count",
                    outgoing_contact == outgoing_contact[snapshot_neighbour],
                )

                for method, neighbour in neighbours.items():
                    append(f"{method}_neighbour_trajectory", trajectory[neighbour].astype(np.int16))
                    append(
                        f"{method}_selected_snapshot_distance",
                        snapshot_distance[np.arange(len(trajectory)), neighbour].astype(np.float32),
                    )
                    total, pose, translation, rotation, joint_error = _method_errors(
                        neighbour,
                        target_translation,
                        target_rotation,
                        target_joint,
                    )
                    append(f"{method}_dx", total.astype(np.float32))
                    append(f"{method}_pose", pose.astype(np.float32))
                    append(f"{method}_translation", translation.astype(np.float32))
                    append(f"{method}_rotation", rotation.astype(np.float32))
                    append(f"{method}_joint", joint_error.astype(np.float32))
                    append(
                        f"{method}_neighbour_is_jump",
                        (target_pose_motion[neighbour] > 0.05),
                    )
        print(
            f"[OBSERVABILITY] {object_id}: {len(pairs)} active transitions",
            flush=True,
        )

    arrays = {name: np.concatenate(parts) for name, parts in sample_parts.items()}
    arrays["object_label"] = object_labels
    arrays["split_label"] = split_labels
    summary: dict[str, Any] = {
        "format_version": 1,
        "dataset": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest.sha256(),
            "active_index": str(args.active_index.resolve()),
            "active_index_sha256": file_sha256(args.active_index),
            "active_transitions": int(len(arrays["step"])),
            "objects": int(len(object_labels)),
        },
        "definitions": {
            "search_scope": "other trajectories of the same object and same next-command step",
            "operator": "RMS trial-gap difference / sdf_scale plus scalar current-aperture difference / length_scale",
            "operator_current_gap": "operator metric plus RMS current-gap difference / sdf_scale",
            "snapshot": "sqrt((||dp||/L)^2 + d_SO3^2 + RMS(dr/joint_travel)^2)",
            "target_pose": "difference between normalized spatial translation increments and SO(3) relative-rotation increments",
            "history_rerank": "among K nearest recorded snapshots, choose the closest previous pose/full-state increment",
            "incoming_diagnostics": "previous transition contact_count, approximate max applied actuator effort, residual linear/angular velocity, and settling_substeps; z-scored within object/step",
            "outgoing_contact_oracle": "among 10 nearest snapshots, choose the closest one with the same successor contact_count; this is target-side information and is not a deployable input",
            "jump": "ground-truth target pose motion > 0.05",
        },
        "limitations": {
            "close_pairs": "large successor spread for merely close inputs is compatible with deterministic discontinuity and does not by itself prove non-observability",
            "contact_count": "only a scalar count is stored; contact identities, normals, tangential slip state, impulses, and reaction forces are absent",
            "actuator_effort": "stored actuator_effort is approximate maximum applied drive torque, not a measured joint/contact reaction",
            "diagnostic_causality": "incoming diagnostics come from transition k-1; outgoing contact_count is used only as an oracle label",
            "nonparametric": "1-NN error measures local target spread and representation quality, not the attainable error of a trained architecture",
        },
        "splits": {},
    }

    for split_index, split in enumerate(SPLITS):
        split_mask = arrays["split_index"] == split_index
        history_mask = split_mask & arrays["has_history"].astype(bool)
        split_summary: dict[str, Any] = {
            "samples": int(split_mask.sum()),
            "samples_with_history": int(history_mask.sum()),
            "methods": {},
            "paired_vs_snapshot": {},
        }
        for method in METHODS:
            method_summary: dict[str, Any] = {}
            smooth_mask = split_mask & (arrays["target_pose_motion"] <= 0.05)
            jump_mask = split_mask & (arrays["target_pose_motion"] > 0.05)
            for subset_name, mask in (
                ("all", split_mask),
                ("with_history", history_mask),
                ("smooth_target", smooth_mask),
                ("jump_target", jump_mask),
            ):
                if not np.any(mask):
                    method_summary[subset_name] = {"samples": 0}
                    continue
                equal_pose, pose_by_object = _equal_object_mean(
                    arrays[f"{method}_pose"][mask], arrays["object_index"][mask]
                )
                equal_joint, joint_by_object = _equal_object_mean(
                    arrays[f"{method}_joint"][mask], arrays["object_index"][mask]
                )
                method_summary[subset_name] = {
                    "samples": int(mask.sum()),
                    "pose": _stats(arrays[f"{method}_pose"][mask]),
                    "joint": _stats(arrays[f"{method}_joint"][mask]),
                    "dx": _stats(arrays[f"{method}_dx"][mask]),
                    "equal_object_pose_mean": equal_pose,
                    "equal_object_joint_mean": equal_joint,
                    "pose_mean_by_object_index": pose_by_object,
                    "joint_mean_by_object_index": joint_by_object,
                    "selected_snapshot_distance": _stats(
                        arrays[f"{method}_selected_snapshot_distance"][mask]
                    ),
                }
            split_summary["methods"][method] = method_summary
            if method != "snapshot":
                split_summary["paired_vs_snapshot"][method] = {
                    "pose": _paired_object_bootstrap(
                        arrays["snapshot_pose"][history_mask],
                        arrays[f"{method}_pose"][history_mask],
                        arrays["object_index"][history_mask],
                        samples=args.bootstrap_samples,
                        seed=args.seed + 101 * split_index,
                    ),
                    "joint": _paired_object_bootstrap(
                        arrays["snapshot_joint"][history_mask],
                        arrays[f"{method}_joint"][history_mask],
                        arrays["object_index"][history_mask],
                        samples=args.bootstrap_samples,
                        seed=args.seed + 101 * split_index + 1,
                    ),
                }

        close_bands: dict[str, Any] = {}
        snapshot_distance = arrays["snapshot_distance"][split_mask]
        for quantile in (0.001, 0.01, 0.05):
            threshold = float(np.quantile(snapshot_distance, quantile))
            close = split_mask & (arrays["snapshot_distance"] <= threshold)
            close_history = close & arrays["has_history"].astype(bool)
            close_bands[f"q{100 * quantile:g}"] = {
                "threshold": threshold,
                "samples": int(close.sum()),
                "snapshot_translation_m": _stats(arrays["snapshot_translation_m"][close]),
                "snapshot_rotation_rad": _stats(arrays["snapshot_rotation_rad"][close]),
                "snapshot_joint_rms": _stats(arrays["snapshot_joint_rms"][close]),
                "target_pose_divergence": _stats(arrays["snapshot_pose"][close]),
                "fraction_target_pose_divergence_gt_0_05": float(
                    np.mean(arrays["snapshot_pose"][close] > 0.05)
                ),
                "fraction_same_source_pose": float(
                    np.mean(arrays["snapshot_neighbour_same_source_pose"][close])
                ),
                "fraction_same_incoming_contact_count": (
                    float(
                        np.mean(
                            arrays[
                                "snapshot_neighbour_same_incoming_contact_count"
                            ][close_history]
                        )
                    )
                    if close_history.any()
                    else None
                ),
                "fraction_same_outgoing_contact_count": float(
                    np.mean(
                        arrays["snapshot_neighbour_same_outgoing_contact_count"][close]
                    )
                ),
                "spearman_target_pose_vs_previous_pose_distance": _rank_correlation(
                    arrays["snapshot_pose"][close_history],
                    arrays["snapshot_neighbour_history_pose_distance"][close_history],
                ),
                "spearman_target_pose_vs_incoming_diagnostic_distance": _rank_correlation(
                    arrays["snapshot_pose"][close_history],
                    arrays[
                        "snapshot_neighbour_incoming_diagnostic_distance"
                    ][close_history],
                ),
            }
        split_summary["close_snapshot_bands"] = close_bands
        representation_bands: dict[str, Any] = {}
        for representation in ("operator", "operator_current_gap"):
            input_distance = arrays[f"{representation}_distance"][split_mask]
            bands: dict[str, Any] = {}
            for quantile in (0.001, 0.01, 0.05):
                threshold = float(np.quantile(input_distance, quantile))
                close = split_mask & (
                    arrays[f"{representation}_distance"] <= threshold
                )
                bands[f"q{100 * quantile:g}"] = {
                    "threshold": threshold,
                    "samples": int(close.sum()),
                    "selected_snapshot_distance": _stats(
                        arrays[f"{representation}_selected_snapshot_distance"][close]
                    ),
                    "target_pose_divergence": _stats(
                        arrays[f"{representation}_pose"][close]
                    ),
                    "target_joint_divergence": _stats(
                        arrays[f"{representation}_joint"][close]
                    ),
                    "fraction_target_pose_divergence_gt_0_05": float(
                        np.mean(arrays[f"{representation}_pose"][close] > 0.05)
                    ),
                }
            machine_near_representation = split_mask & (
                arrays[f"{representation}_distance"] <= 1e-6
            )
            bands["machine_near"] = {
                "definition": "normalized input distance <= 1e-6",
                "samples": int(machine_near_representation.sum()),
                "selected_snapshot_distance": _stats(
                    arrays[f"{representation}_selected_snapshot_distance"][
                        machine_near_representation
                    ]
                ),
                "target_pose_divergence": _stats(
                    arrays[f"{representation}_pose"][machine_near_representation]
                ),
                "target_joint_divergence": _stats(
                    arrays[f"{representation}_joint"][machine_near_representation]
                ),
            }
            representation_bands[representation] = bands
        split_summary["representation_neighbourhoods"] = representation_bands
        split_summary["nearest_snapshot_lipschitz_ratio"] = _stats(
            arrays["snapshot_pose"][split_mask]
            / np.maximum(arrays["snapshot_distance"][split_mask], 1e-12)
        )
        machine_near = split_mask & (
            (arrays["snapshot_translation_m"] <= 1e-7)
            & (arrays["snapshot_rotation_rad"] <= 1e-6)
            & (arrays["snapshot_joint_rms"] <= 1e-6)
        )
        split_summary["machine_near_duplicates"] = {
            "definition": "translation <= 0.1 um, rotation <= 1e-6 rad, normalized joint RMS <= 1e-6",
            "samples": int(machine_near.sum()),
            "target_pose_divergence": _stats(arrays["snapshot_pose"][machine_near]),
            "target_joint_divergence": _stats(arrays["snapshot_joint"][machine_near]),
        }
        split_summary["reference_predictors"] = {
            "identity_pose": _stats(arrays["target_pose_motion"][history_mask]),
            "previous_increment_pose": _stats(arrays["persistence_pose"][history_mask]),
        }
        jump_truth = arrays["target_pose_motion"][history_mask] > 0.05
        split_summary["jump_classification"] = {
            "previous_increment": _binary_metrics(
                jump_truth,
                arrays["history_pose_motion"][history_mask] > 0.05,
            ),
            **{
                method: _binary_metrics(
                    jump_truth,
                    arrays[f"{method}_neighbour_is_jump"][history_mask],
                )
                for method in METHODS
            },
        }
        split_summary["incoming_contact_rerank_coverage"] = float(
            np.mean(arrays["incoming_contact_rerank_covered"][history_mask])
        )
        split_summary["outgoing_contact_oracle_coverage"] = float(
            np.mean(arrays["outgoing_contact_oracle_covered"][history_mask])
        )
        summary["splits"][split] = split_summary

    close_all_threshold = float(np.quantile(arrays["snapshot_distance"], 0.01))
    close_all = arrays["snapshot_distance"] <= close_all_threshold
    candidate = np.flatnonzero(close_all)
    order = candidate[np.argsort(arrays["snapshot_pose"][candidate])[::-1]]
    seen: set[tuple[int, int, int, int]] = set()
    examples: list[dict[str, Any]] = []
    for index in order:
        object_index = int(arrays["object_index"][index])
        step = int(arrays["step"][index])
        trajectory = int(arrays["trajectory"][index])
        neighbour = int(arrays["snapshot_neighbour_trajectory"][index])
        key = (object_index, step, min(trajectory, neighbour), max(trajectory, neighbour))
        if key in seen:
            continue
        seen.add(key)
        examples.append(
            {
                "split": str(split_labels[int(arrays["split_index"][index])]),
                "object_id": str(object_labels[object_index]),
                "step": step,
                "trajectory": trajectory,
                "nearest_trajectory": neighbour,
                "source_pose_index": int(arrays["source_pose_index"][index]),
                "snapshot_distance": float(arrays["snapshot_distance"][index]),
                "snapshot_translation_m": float(
                    arrays["snapshot_translation_m"][index]
                ),
                "snapshot_rotation_rad": float(
                    arrays["snapshot_rotation_rad"][index]
                ),
                "snapshot_joint_rms": float(arrays["snapshot_joint_rms"][index]),
                "target_pose_divergence": float(arrays["snapshot_pose"][index]),
                "previous_pose_increment_distance": float(
                    arrays["snapshot_neighbour_history_pose_distance"][index]
                ),
                "incoming_contact_count": float(
                    arrays["incoming_contact_count"][index]
                ),
                "outgoing_contact_count": float(
                    arrays["outgoing_contact_count"][index]
                ),
                "same_incoming_contact_count": bool(
                    arrays["snapshot_neighbour_same_incoming_contact_count"][index]
                ),
                "same_outgoing_contact_count": bool(
                    arrays["snapshot_neighbour_same_outgoing_contact_count"][index]
                ),
            }
        )
        if len(examples) == 20:
            break
    summary["most_divergent_close_pairs"] = examples

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(args.output / "samples.npz", **arrays)
    _make_plot(args.output, arrays, summary)
    compact = {
        split: {
            "snapshot_pose": summary["splits"][split]["methods"]["snapshot"][
                "with_history"
            ]["equal_object_pose_mean"],
            "history_pose_k10": summary["splits"][split]["methods"][
                "history_pose_k10"
            ]["with_history"]["equal_object_pose_mean"],
            "operator_current_gap": summary["splits"][split]["methods"][
                "operator_current_gap"
            ]["with_history"]["equal_object_pose_mean"],
            "outgoing_contact_oracle": summary["splits"][split]["methods"][
                "outgoing_contact_count_oracle_k10"
            ]["with_history"]["equal_object_pose_mean"],
            "machine_near_duplicates": summary["splits"][split][
                "machine_near_duplicates"
            ]["samples"],
        }
        for split in SPLITS
    }
    print(json.dumps(compact, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
