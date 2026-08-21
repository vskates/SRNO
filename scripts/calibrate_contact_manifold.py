#!/usr/bin/env python3
"""Read-only calibration of correction gate and admissible settled gap."""

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

from srno.data.dataset import H5ObjectDataset, ObjectRecord
from srno.data.schema import DatasetManifest
from srno.geometry.gripper import GripperAsset
from srno.geometry.sdf import sample_sdf
from srno.geometry.se3 import quaternion_xyzw_to_matrix, rotation_geodesic_angle


SPLITS = ("train", "val", "test")
QUANTILES = (0.0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0)


def _stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        **{
            f"q{100 * quantile:g}": float(np.quantile(values, quantile))
            for quantile in QUANTILES
        },
    }


def _minimum_gaps(
    record: ObjectRecord,
    gripper: GripperAsset,
    schedule: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    position = record.position.to(device=device, dtype=torch.float32)
    rotation = quaternion_xyzw_to_matrix(
        record.quaternion_xyzw.to(device=device, dtype=torch.float32)
    )
    aperture = record.aperture.to(device=device, dtype=torch.float32)
    sdf = record.sdf[None].to(device=device)
    origin = record.origin[None].to(device=device)
    voxel = record.voxel_size[None].to(device=device)
    trajectories = position.shape[0]
    mapping = torch.zeros(trajectories, dtype=torch.long, device=device)
    trial = torch.empty((trajectories, 32), device=device)
    target = torch.empty_like(trial)
    with torch.no_grad():
        for step in range(32):
            trial_points = gripper.points(schedule[step + 1]).expand(
                trajectories, -1, -1
            )
            target_points = gripper.points(aperture[:, step + 1])
            for output, points, pose_step in (
                (trial, trial_points, step),
                (target, target_points, step + 1),
            ):
                relative = points - position[:, pose_step, None, :]
                points_object = torch.einsum(
                    "bij,bmj->bmi",
                    rotation[:, pose_step].transpose(-1, -2),
                    relative,
                )
                gap = sample_sdf(
                    sdf,
                    origin,
                    voxel,
                    points_object,
                    sample_to_object=mapping,
                    outside_value=0.02,
                )
                output[:, step] = gap.amin(dim=-1)
    return trial.cpu().numpy(), target.cpu().numpy()


def _correction_distance(
    record: ObjectRecord,
    schedule: np.ndarray,
    length_scale: float,
) -> np.ndarray:
    position = record.position.float()
    rotation = quaternion_xyzw_to_matrix(record.quaternion_xyzw.float())
    translation = torch.linalg.vector_norm(
        position[:, 1:] - position[:, :-1], dim=-1
    ) / length_scale
    angle = rotation_geodesic_angle(rotation[:, 1:], rotation[:, :-1])
    command = torch.from_numpy(schedule[1:]).to(record.aperture)
    aperture = torch.abs(record.aperture[:, 1:] - command[None]) / length_scale
    return torch.sqrt(translation.square() + angle.square() + aperture.square()).numpy()


def _classification(
    gap: np.ndarray,
    label: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predicted = gap <= threshold
    true_positive = np.count_nonzero(predicted & label)
    false_positive = np.count_nonzero(predicted & ~label)
    return {
        "threshold_m": float(threshold),
        "positive_transitions": int(np.count_nonzero(label)),
        "active_transitions": int(np.count_nonzero(predicted)),
        "active_fraction": float(np.mean(predicted)),
        "recall": float(true_positive / max(1, np.count_nonzero(label))),
        "precision": float(true_positive / max(1, np.count_nonzero(predicted))),
        "false_positive_rate": float(false_positive / max(1, np.count_nonzero(~label))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-recall", type=float, default=0.995)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/contact-manifold-calibration")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    manifest = DatasetManifest.load(args.manifest)
    gripper = GripperAsset.load(manifest.gripper_path).to(device)
    schedule_np = np.asarray(manifest.commanded_aperture_m, dtype=np.float32)
    schedule = torch.from_numpy(schedule_np).to(device)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    object_rows: list[dict[str, Any]] = []
    maximum_voxel = 0.0
    for split in SPLITS:
        trial_parts = []
        target_parts = []
        correction_parts = []
        contact_parts = []
        dataset = H5ObjectDataset(manifest, split=split)
        try:
            for record in dataset:
                trial, target = _minimum_gaps(record, gripper, schedule, device)
                correction = _correction_distance(
                    record, schedule_np, manifest.length_scale_m
                )
                trial_parts.append(trial.reshape(-1))
                target_parts.append(target.reshape(-1))
                correction_parts.append(correction.reshape(-1))
                contact_parts.append(
                    (record.diagnostics["contact_count"].numpy() > 0).reshape(-1)
                )
                maximum_voxel = max(maximum_voxel, float(record.voxel_size.max()))
                object_rows.append(
                    {
                        "split": split,
                        "object_id": record.object_id,
                        "target_min_gap_m": _stats(target.reshape(-1)),
                        "target_infeasible_fraction": float(np.mean(target < 0.0)),
                        "correction_dx": _stats(correction.reshape(-1)),
                    }
                )
        finally:
            dataset.close()
        arrays[split] = {
            "trial_gap_m": np.concatenate(trial_parts),
            "target_gap_m": np.concatenate(target_parts),
            "correction_dx": np.concatenate(correction_parts),
            "contact": np.concatenate(contact_parts),
        }

    train_val = {
        name: np.concatenate((arrays["train"][name], arrays["val"][name]))
        for name in arrays["train"]
    }
    no_contact_correction = train_val["correction_dx"][~train_val["contact"]]
    tau_candidates = {
        "noncontact_q95": float(np.quantile(no_contact_correction, 0.95)),
        "noncontact_q99": float(np.quantile(no_contact_correction, 0.99)),
        "noncontact_q99.5": float(np.quantile(no_contact_correction, 0.995)),
    }
    gate_candidates: dict[str, Any] = {}
    for name, tau in tau_candidates.items():
        label = train_val["correction_dx"] > tau
        positive_gaps = train_val["trial_gap_m"][label]
        threshold = float(
            np.quantile(positive_gaps, args.target_recall, method="higher")
        )
        gate_candidates[name] = {
            "tau_num_dx": tau,
            "raw_threshold_m": threshold,
            "two_voxel_floor_m": 2.01 * maximum_voxel,
            "splits": {
                split: _classification(
                    arrays[split]["trial_gap_m"],
                    arrays[split]["correction_dx"] > tau,
                    threshold,
                )
                for split in SPLITS
            },
        }

    # Rigid-body restOffset defaults to zero when neither asset family authors
    # an override.  Quantiles therefore measure geometry/interpolation/solver
    # residual directly as penetration below the zero-distance surface.
    residual = np.maximum(-train_val["target_gap_m"], 0.0)
    epsilon = float(np.quantile(residual, 0.995, method="higher"))
    admissible_gap = -epsilon
    result = {
        "definitions": {
            "correction_dx": (
                "sqrt(||p[k+1]-p[k]||^2/L^2 + "
                "d_SO3(R[k+1],R[k])^2 + |a[k+1]-command[k+1]|^2/L^2)"
            ),
            "gate_label": "correction_dx > tau_num_dx",
            "rest_distance_m": 0.0,
            "epsilon_geom_solver_m": epsilon,
            "admissible_gap_m": admissible_gap,
        },
        "maximum_voxel_size_m": maximum_voxel,
        "noncontact_correction_dx": _stats(no_contact_correction),
        "all_correction_dx": _stats(train_val["correction_dx"]),
        "settled_target_min_gap_m": {
            split: _stats(arrays[split]["target_gap_m"]) for split in SPLITS
        },
        "settled_penetration_residual_m_train_val": _stats(residual),
        "gate_candidates": gate_candidates,
        "objects": object_rows,
    }
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output / "arrays.npz",
        **{
            f"{split}_{name}": values
            for split, split_arrays in arrays.items()
            for name, values in split_arrays.items()
        },
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    axes[0].hist(
        np.clip(no_contact_correction, 0, np.quantile(no_contact_correction, 0.999)),
        bins=100,
        color="#4472C4",
    )
    for name, value in tau_candidates.items():
        axes[0].axvline(value, label=name)
    axes[0].set(title="Numerical/free correction floor", xlabel="$c_k=d_X(x^*,F_{free})$")
    axes[0].legend(fontsize=8)
    axes[1].hist(
        np.clip(train_val["trial_gap_m"] * 1000, -20, 20),
        bins=120,
        color="#55A868",
    )
    axes[1].set(title="Trial minimum gap", xlabel="$m_k^{trial}$, mm")
    axes[2].hist(
        np.clip(residual * 1000, 0, np.quantile(residual, 0.999) * 1000),
        bins=100,
        color="#C44E52",
    )
    axes[2].axvline(epsilon * 1000, color="black", label="Q99.5")
    axes[2].set(title="Settled GT residual", xlabel="$[-h^*(x)]_+$, mm")
    axes[2].legend()
    fig.savefig(args.output / "calibration.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
