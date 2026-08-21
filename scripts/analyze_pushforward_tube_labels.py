#!/usr/bin/env python3
"""Diagnose geometry and label regularity in a PhysX pushforward tube."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from srno.data.dataset import H5ObjectDataset
from srno.data.index import ActiveIndex
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import quaternion_xyzw_to_matrix, rotation_geodesic_angle
from srno.training.checkpoint import load_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.types import PoseState

from run_pushforward_tube_ablation import TubeLabels


QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
BANDS = ((1, 8), (9, 16), (17, 24), (25, 31))


def _summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"count": 0}
    result: dict[str, float | int] = {"count": int(len(finite)), "mean": float(finite.mean())}
    result.update(
        {
            f"q{int(round(quantile * 100)):02d}": float(value)
            for quantile, value in zip(QUANTILES, np.quantile(finite, QUANTILES), strict=True)
        }
    )
    return result


def _state_components(
    first: PoseState,
    second: PoseState,
    *,
    length_scale: float,
    joint_scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    translation = torch.linalg.vector_norm(first.position - second.position, dim=-1) / length_scale
    rotation = rotation_geodesic_angle(first.rotation, second.rotation)
    joints = torch.sqrt(
        (((first.joint_position - second.joint_position) / joint_scale).square()).mean(dim=-1)
    )
    return {
        "translation": translation,
        "rotation": rotation,
        "joints": joints,
        "dx": torch.sqrt(translation.square() + rotation.square() + joints.square()),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    base = ExperimentConfig.load(args.config)
    config = base if base.device == args.device else ExperimentConfig.load(args.config)
    device = torch.device(args.device)
    manifest = DatasetManifest.load(config.paths.manifest)
    active = ActiveIndex.load(config.paths.active_index)
    dataset = H5ObjectDataset(manifest, split="train", active_index=active, active_only=True)
    try:
        tube = TubeLabels(args.labels, manifest, dataset)
        model = _build_model(config, manifest, device)
        checkpoint = load_checkpoint(args.checkpoint, model=model, map_location=device)
        if checkpoint["manifest_sha256"] != manifest.sha256():
            raise ValueError("checkpoint and manifest hashes differ")
        model.eval()

        rows: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "current_gap_m", "predicted_gap_m", "oracle_step_dx",
                "oracle_error_dx", "model_nominal_error_dx",
                "nominal_label_error_dx", "pushforward_dx",
                "current_step", "contact_count", "settling_control_steps",
            )
        }
        object_rows: list[np.ndarray] = []
        batches = tube.object_batches(
            objects_per_batch=base.loader.objects_per_batch,
            samples_per_object=8,
            seed=base.seed + 919,
        )
        arrays = tube.arrays
        for raw in batches:
            batch = raw.to(device)
            with torch.no_grad():
                predicted = model.forward_step(batch.current, batch.next_command, batch.sdf)
                current_gap = model.query_geometric_gap(batch.current, batch.sdf).amin(dim=-1)
                predicted_gap = model.query_geometric_gap(predicted, batch.sdf).amin(dim=-1)
                oracle_step = _state_components(
                    batch.current, batch.target,
                    length_scale=model.length_scale,
                    joint_scale=model.joint_travel_range,
                )["dx"]
                oracle_error = _state_components(
                    predicted, batch.target,
                    length_scale=model.length_scale,
                    joint_scale=model.joint_travel_range,
                )["dx"]
            keys = zip(
                (raw.object_ids[index] for index in raw.sdf.sample_to_object.tolist()),
                raw.trajectory_index.tolist(), raw.step_index.tolist(), strict=True,
            )
            indices = []
            for object_id, trajectory, step in keys:
                matches = np.flatnonzero(
                    (tube.object_ids == object_id)
                    & (arrays["trajectory"] == trajectory)
                    & (arrays["current_step"] == step)
                )
                if len(matches) != 1:
                    raise ValueError("tube sample key is not unique")
                indices.append(int(matches[0]))
            selected = np.asarray(indices)
            nominal = PoseState(
                quaternion_xyzw_to_matrix(torch.from_numpy(
                    arrays["nominal_successor_quaternion_xyzw"][selected].astype(np.float32)
                )).to(device),
                torch.from_numpy(arrays["nominal_successor_position"][selected].astype(np.float32)).to(device),
                torch.from_numpy(arrays["nominal_successor_joint"][selected].astype(np.float32)).to(device),
            )
            nominal_error = _state_components(
                batch.target, nominal,
                length_scale=model.length_scale,
                joint_scale=model.joint_travel_range,
            )["dx"]
            model_nominal_error = _state_components(
                predicted, nominal,
                length_scale=model.length_scale,
                joint_scale=model.joint_travel_range,
            )["dx"]
            tensors = {
                "current_gap_m": current_gap,
                "predicted_gap_m": predicted_gap,
                "oracle_step_dx": oracle_step,
                "oracle_error_dx": oracle_error,
                "model_nominal_error_dx": model_nominal_error,
                "nominal_label_error_dx": nominal_error,
            }
            for name, value in tensors.items():
                rows[name].append(value.float().cpu().numpy())
            for name in (
                "pushforward_dx", "current_step", "contact_count", "settling_control_steps",
            ):
                rows[name].append(arrays[name][selected])
            object_rows.append(tube.object_ids[selected])

        values = {name: np.concatenate(chunks) for name, chunks in rows.items()}
        object_id = np.concatenate(object_rows).astype(str)
        result: dict[str, Any] = {
            "definition": {
                "oracle_step_dx": "d(x_hat_k, R_PhysX(x_hat_k,u_{k+1}))",
                "oracle_error_dx": "d(R_theta(x_hat_k,u_{k+1}), R_PhysX(x_hat_k,u_{k+1}))",
                "model_nominal_error_dx": "d(R_theta(x_hat_k,u_{k+1}), x^*_{k+1})",
                "nominal_label_error_dx": "d(x^*_{k+1}, R_PhysX(x_hat_k,u_{k+1}))",
                "gap": "minimum cooked SDF gap over gripper surface points",
            },
            "all": {name: _summary(value.astype(np.float64)) for name, value in values.items()},
            "correlations": {
                "pushforward_vs_oracle_step": float(np.corrcoef(values["pushforward_dx"], values["oracle_step_dx"])[0, 1]),
                "gap_vs_oracle_step": float(np.corrcoef(values["current_gap_m"], values["oracle_step_dx"])[0, 1]),
                "settling_vs_oracle_step": float(np.corrcoef(values["settling_control_steps"], values["oracle_step_dx"])[0, 1]),
            },
            "bands": {},
            "objects": {},
        }
        for first, last in BANDS:
            mask = (values["current_step"] >= first) & (values["current_step"] <= last)
            result["bands"][f"{first:02d}-{last:02d}"] = {
                name: _summary(value[mask].astype(np.float64))
                for name, value in values.items()
            }
        for name in sorted(set(object_id)):
            mask = object_id == name
            result["objects"][name] = {
                field: _summary(value[mask].astype(np.float64))
                for field, value in values.items()
            }
        _write_json(args.output, result)
        print(json.dumps({
            "samples": len(object_id),
            "current_gap_m": result["all"]["current_gap_m"],
            "oracle_step_dx": result["all"]["oracle_step_dx"],
            "oracle_error_dx": result["all"]["oracle_error_dx"],
            "model_nominal_error_dx": result["all"]["model_nominal_error_dx"],
            "nominal_label_error_dx": result["all"]["nominal_label_error_dx"],
            "correlations": result["correlations"],
        }, indent=2, sort_keys=True), flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
