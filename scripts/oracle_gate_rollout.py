#!/usr/bin/env python3
"""Inference-only H32 oracle-gate diagnostic for a frozen SRNO checkpoint.

The predicted state remains autoregressive.  At transition k, the only oracle
quantity is the recorded simulator contact flag: free transitions use the
exact bypass and contact transitions are forced through the existing cell.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from srno.data.dataset import H5ObjectDataset, ObjectBatchCollator, TrajectoryBatch
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import rotation_geodesic_angle
from srno.model import SRNOModel
from srno.training.checkpoint import load_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _autocast, _build_model, _state_at
from srno.types import PoseState, SDFBatch


def _subset_sdf(sdf: SDFBatch, count: int) -> SDFBatch:
    # The diagnostic collates one object at a time, so every selected sample
    # maps to the one unique resident grid without duplicating it.
    return SDFBatch(
        sdf.values,
        sdf.origin,
        sdf.voxel_size,
        torch.zeros(count, dtype=torch.long, device=sdf.values.device),
        sdf.outside_value,
    )


def _oracle_step(
    model: SRNOModel,
    state: PoseState,
    command: Tensor,
    sdf: SDFBatch,
    contact: Tensor,
) -> PoseState:
    """Apply only the oracle bypass/cell decision, keeping the frozen cell exact."""

    batch = state.shape[0]
    if contact.shape != (batch,):
        raise ValueError("oracle contact mask must have shape [batch]")
    trial_joint = model.free_joint_configuration(command).unsqueeze(0).expand(batch, 6)
    next_rotation = state.rotation.clone()
    next_position = state.position.clone()
    next_joint = trial_joint.clone()
    indices = torch.nonzero(contact, as_tuple=False).flatten()
    if indices.numel():
        active_state = PoseState(
            state.rotation.index_select(0, indices),
            state.position.index_select(0, indices),
            state.joint_position.index_select(0, indices),
        )
        # +inf forces the existing geometric path through the existing cell;
        # no weight, feature, geometry or cell implementation is changed.
        previous_gate = model.delta_gate
        try:
            model.delta_gate = math.inf
            active_result = model.forward_step(
                active_state,
                command,
                _subset_sdf(sdf, len(indices)),
            )
        finally:
            model.delta_gate = previous_gate
        assert isinstance(active_result, PoseState)
        next_rotation[indices] = active_result.rotation
        next_position[indices] = active_result.position
        next_joint[indices] = active_result.joint_position
    return PoseState(next_rotation, next_position, next_joint)


def _rollouts(
    model: SRNOModel,
    batch: TrajectoryBatch,
    oracle_contact: Tensor,
) -> tuple[PoseState, PoseState, Tensor]:
    baseline_states = [_state_at(batch.states, 0)]
    oracle_states = [_state_at(batch.states, 0)]
    baseline_active: list[Tensor] = []
    baseline = baseline_states[0]
    oracle = oracle_states[0]
    for step, command in enumerate(batch.command_schedule[1:]):
        baseline_result = model.forward_step(baseline, command, batch.sdf, return_aux=True)
        assert isinstance(baseline_result, tuple)
        baseline, aux = baseline_result
        oracle = _oracle_step(model, oracle, command, batch.sdf, oracle_contact[:, step])
        baseline_states.append(baseline)
        oracle_states.append(oracle)
        baseline_active.append(aux.active)
    return (
        PoseState.stack(baseline_states, dim=1),
        PoseState.stack(oracle_states, dim=1),
        torch.stack(baseline_active, dim=1),
    )


def _components(model: SRNOModel, prediction: PoseState, target: PoseState) -> dict[str, Tensor]:
    translation_m = torch.linalg.vector_norm(prediction.position - target.position, dim=-1)
    translation = translation_m / model.length_scale
    rotation = rotation_geodesic_angle(prediction.rotation, target.rotation)
    joint = torch.sqrt(
        (
            (prediction.joint_position - target.joint_position)
            / model.joint_travel_range
        )
        .square()
        .mean(dim=-1)
    )
    dx = torch.sqrt(translation.square() + rotation.square() + joint.square())
    return {
        "dx": dx,
        "translation_m": translation_m,
        "rotation_rad": rotation,
        "joint_rmse_over_travel": joint,
    }


def _stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _evaluate_split(
    model: SRNOModel,
    manifest: DatasetManifest,
    config: ExperimentConfig,
    split: str,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    dataset = H5ObjectDataset(manifest, split=split)  # type: ignore[arg-type]
    collator = ObjectBatchCollator(manifest, mode="rollout", samples_per_object=0, resample=False)
    arrays: dict[str, list[np.ndarray]] = {}
    by_object: dict[str, Any] = {}
    tp = fp = tn = fn = 0
    try:
        for index, object_id in enumerate(dataset.object_ids):
            record = dataset[index]
            if "contact_count" not in record.diagnostics:
                raise ValueError(f"{object_id}: diagnostics/contact_count is required")
            raw_batch = collator([record])
            assert isinstance(raw_batch, TrajectoryBatch)
            batch = raw_batch.to(device)
            oracle_contact = record.diagnostics["contact_count"].to(device) > 0
            expected_shape = (batch.states.shape[0], 32)
            if oracle_contact.shape != expected_shape:
                raise ValueError(
                    f"{object_id}: contact_count shape {tuple(oracle_contact.shape)} != {expected_shape}"
                )
            with torch.no_grad(), _autocast(config, device):
                baseline, oracle, baseline_active = _rollouts(model, batch, oracle_contact)
            baseline_values = _components(model, baseline, batch.states)
            oracle_values = _components(model, oracle, batch.states)
            for prefix, values in (("baseline", baseline_values), ("oracle", oracle_values)):
                for name, value in values.items():
                    arrays.setdefault(f"{prefix}_{name}", []).append(value.float().cpu().numpy())
            arrays.setdefault("object_index", []).append(
                np.full(batch.states.shape[0], index, dtype=np.int32)
            )
            active = baseline_active.bool()
            tp += int((active & oracle_contact).sum())
            fp += int((active & ~oracle_contact).sum())
            tn += int((~active & ~oracle_contact).sum())
            fn += int((~active & oracle_contact).sum())
            baseline_terminal = baseline_values["dx"][:, -1].float().cpu().numpy()
            oracle_terminal = oracle_values["dx"][:, -1].float().cpu().numpy()
            by_object[object_id] = {
                "trajectories": int(len(baseline_terminal)),
                "baseline_terminal_dx": _stats(baseline_terminal),
                "oracle_terminal_dx": _stats(oracle_terminal),
                "mean_delta_terminal_dx": float(np.mean(oracle_terminal - baseline_terminal)),
            }
            print(
                f"[ORACLE] {split}/{object_id}: baseline={baseline_terminal.mean():.6f}, "
                f"oracle={oracle_terminal.mean():.6f}",
                flush=True,
            )
    finally:
        dataset.close()

    merged = {name: np.concatenate(values, axis=0) for name, values in arrays.items()}
    baseline_terminal = merged["baseline_dx"][:, -1]
    oracle_terminal = merged["oracle_dx"][:, -1]
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    result = {
        "trajectories": int(len(baseline_terminal)),
        "transitions": int((tp + fp + tn + fn)),
        "baseline_terminal_dx": _stats(baseline_terminal),
        "oracle_terminal_dx": _stats(oracle_terminal),
        "absolute_change": float(np.mean(oracle_terminal) - np.mean(baseline_terminal)),
        "relative_change_percent": float(
            100.0 * (np.mean(oracle_terminal) - np.mean(baseline_terminal))
            / np.mean(baseline_terminal)
        ),
        "baseline": {
            name: {
                "all_states": _stats(merged[f"baseline_{name}"][:, 1:]),
                "terminal": _stats(merged[f"baseline_{name}"][:, -1]),
            }
            for name in ("dx", "translation_m", "rotation_rad", "joint_rmse_over_travel")
        },
        "oracle": {
            name: {
                "all_states": _stats(merged[f"oracle_{name}"][:, 1:]),
                "terminal": _stats(merged[f"oracle_{name}"][:, -1]),
            }
            for name in ("dx", "translation_m", "rotation_rad", "joint_rmse_over_travel")
        },
        "geometric_gate_vs_gt_contact": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "contact_prevalence": (tp + fn) / (tp + fp + tn + fn),
            "active_fraction": (tp + fp) / (tp + fp + tn + fn),
        },
        "by_object": by_object,
    }
    merged["object_labels"] = np.asarray(list(by_object))
    return result, merged


def _plot(output: Path, results: dict[str, Any], arrays: dict[str, dict[str, np.ndarray]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for column, split in enumerate(("val", "test")):
        values = arrays[split]
        steps = np.arange(33)
        axes[0, column].plot(steps, values["baseline_dx"].mean(axis=0), label="geometric gate", linewidth=2)
        axes[0, column].plot(steps, values["oracle_dx"].mean(axis=0), label="GT contact oracle", linewidth=2)
        axes[0, column].set_xlabel("closure step k")
        axes[0, column].set_ylabel(r"mean $d_X(k)$")
        axes[0, column].set_title(f"{split}: autoregressive H32")
        axes[0, column].grid(alpha=0.25)
        axes[0, column].legend()

        labels = values["object_labels"].tolist()
        object_index = values["object_index"]
        baseline = [values["baseline_dx"][object_index == i, -1].mean() for i in range(len(labels))]
        oracle = [values["oracle_dx"][object_index == i, -1].mean() for i in range(len(labels))]
        x = np.arange(len(labels))
        width = 0.38
        axes[1, column].bar(x - width / 2, baseline, width, label="geometric gate")
        axes[1, column].bar(x + width / 2, oracle, width, label="GT contact oracle")
        axes[1, column].set_xticks(x, [str(label)[:22] for label in labels], rotation=18)
        axes[1, column].set_ylabel(r"terminal $d_X^{H32}$")
        gate = results[split]["geometric_gate_vs_gt_contact"]
        axes[1, column].set_title(
            f"precision={gate['precision']:.3f}, recall={gate['recall']:.3f}"
        )
        axes[1, column].legend()
    fig.savefig(output / "oracle_gate_h32.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    manifest = DatasetManifest.load(config.paths.manifest)
    model = _build_model(config, manifest, device)
    checkpoint = load_checkpoint(args.checkpoint, model=model, map_location=device)
    if checkpoint["manifest_sha256"] != manifest.sha256():
        raise ValueError("checkpoint manifest hash mismatch")
    model.eval()

    results: dict[str, Any] = {
        "definition": {
            "oracle": "GT contact_count>0 chooses exact free bypass or frozen neural cell; predicted state remains autoregressive",
            "dx": "sqrt((||dp||/L)^2 + theta(R,R*)^2 + mean(((r-r*)/travel)^2))",
        },
        "configuration": {
            "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "manifest_sha256": manifest.sha256(),
            "delta_gate_m": manifest.delta_gate_m,
            "contact_offset_sum_m": manifest.contact_offset_sum_m,
            "no_retraining": True,
        },
        "reference_terminal_dx": {"val": 0.17167461395263672, "test": 0.20409706115722656},
    }
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split in ("val", "test"):
        split_result, split_arrays = _evaluate_split(model, manifest, config, split, device)
        reference = results["reference_terminal_dx"][split]
        split_result["baseline_reference_delta"] = (
            split_result["baseline_terminal_dx"]["mean"] - reference
        )
        results[split] = split_result
        arrays[split] = split_arrays

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    for split, values in arrays.items():
        np.savez_compressed(args.output / f"{split}_trajectories.npz", **values)
    _plot(args.output, results, arrays)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
