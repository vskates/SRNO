#!/usr/bin/env python3
"""Project a recurrent rollout onto a library of complete physical paths.

The recurrent model supplies only a contact-mode signature.  The final output
is the metric projection of that signature onto a geometry-conditioned set of
recorded solution paths, so late states are not obtained by further recurrent
composition.  Validation selects the BV/state path metric and projection rule;
test is evaluated once after selection.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from evaluate_branch_defect_selector import _candidate_paths
from evaluate_loading_profile_kernel_operator import (
    _load_split,
    _metrics,
    _nearest,
    _predict,
    _profile_embedding,
)
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import rotation_geodesic_angle
from srno.training.checkpoint import load_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _autocast, _build_model
from srno.types import PoseState, SDFBatch


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


@torch.inference_mode()
def _rollout_split(
    model: torch.nn.Module,
    config: ExperimentConfig,
    manifest: DatasetManifest,
    data: dict[str, np.ndarray],
    split: str,
    device: torch.device,
) -> PoseState:
    locations = manifest.object_locations()
    schedule = torch.tensor(
        manifest.commanded_aperture_m, dtype=torch.float32, device=device
    )
    predictions = []
    for object_id in manifest.splits[split]:
        rows = np.flatnonzero(data["object_id"] == object_id)
        shard, group_name = locations[object_id]
        with h5py.File(shard, "r", swmr=True) as handle:
            group = handle[group_name]
            values = torch.from_numpy(
                np.asarray(group["sdf"], dtype=np.float32)
            ).unsqueeze(0).to(device)
            origin = torch.from_numpy(
                np.asarray(group.attrs["grid_origin"], dtype=np.float32)
            ).unsqueeze(0).to(device)
            voxel = torch.from_numpy(
                np.asarray(group.attrs["voxel_size"], dtype=np.float32)
            ).unsqueeze(0).to(device)
        initial = PoseState(
            torch.from_numpy(data["rotation"][rows, 0]).to(device),
            torch.from_numpy(data["position"][rows, 0]).to(device),
            torch.from_numpy(data["joint"][rows, 0]).to(device),
        )
        sdf = SDFBatch(
            values,
            origin,
            voxel,
            torch.zeros(len(rows), dtype=torch.long, device=device),
            manifest.sdf_scale_m,
        )
        with _autocast(config, device):
            prediction = model.rollout(initial, schedule[1:], sdf)
        predictions.append(prediction.to("cpu"))
        print(f"[ROLLOUT] split={split} object={object_id}", flush=True)
    return PoseState(
        torch.cat([value.rotation for value in predictions], dim=0),
        torch.cat([value.position for value in predictions], dim=0),
        torch.cat([value.joint_position for value in predictions], dim=0),
    )


def _path_scores(
    candidates: PoseState,
    rollout: PoseState,
    *,
    length_scale: float,
    joint_scale: np.ndarray,
) -> dict[str, np.ndarray]:
    joint_tensor = torch.from_numpy(joint_scale)
    rollout_position = rollout.position[:, None]
    rollout_rotation = rollout.rotation[:, None]
    rollout_joint = rollout.joint_position[:, None]
    translation = torch.linalg.vector_norm(
        candidates.position - rollout_position, dim=-1
    ) / length_scale
    rotation = rotation_geodesic_angle(
        candidates.rotation,
        rollout_rotation.expand_as(candidates.rotation),
    )
    joints = torch.sqrt(
        (((candidates.joint_position - rollout_joint) / joint_tensor).square()).mean(
            dim=-1
        )
    )
    state_dx = torch.sqrt(
        translation.square() + rotation.square() + joints.square()
    )

    candidate_translation_increment = (
        candidates.position[:, :, 1:] - candidates.position[:, :, :-1]
    ) / length_scale
    rollout_translation_increment = (
        rollout.position[:, 1:] - rollout.position[:, :-1]
    )[:, None] / length_scale
    increment_translation = torch.linalg.vector_norm(
        candidate_translation_increment - rollout_translation_increment, dim=-1
    )
    candidate_rotation_increment = (
        candidates.rotation[:, :, 1:]
        @ candidates.rotation[:, :, :-1].transpose(-1, -2)
    )
    rollout_rotation_increment = (
        rollout.rotation[:, 1:] @ rollout.rotation[:, :-1].transpose(-1, -2)
    )[:, None]
    increment_rotation = rotation_geodesic_angle(
        candidate_rotation_increment,
        rollout_rotation_increment.expand_as(candidate_rotation_increment),
    )
    candidate_joint_increment = (
        candidates.joint_position[:, :, 1:]
        - candidates.joint_position[:, :, :-1]
    ) / joint_tensor
    rollout_joint_increment = (
        (rollout.joint_position[:, 1:] - rollout.joint_position[:, :-1])
        / joint_tensor
    )[:, None]
    increment_joints = torch.sqrt(
        ((candidate_joint_increment - rollout_joint_increment).square()).mean(
            dim=-1
        )
    )
    increment_dx = torch.sqrt(
        increment_translation.square()
        + increment_rotation.square()
        + increment_joints.square()
    )
    increment_pose = torch.sqrt(
        increment_translation.square() + increment_rotation.square()
    )
    return {
        "state": state_dx.numpy(),
        "increment": increment_dx.numpy(),
        "increment_pose": increment_pose.numpy(),
        "increment_joints": increment_joints.numpy(),
    }


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(len(values))[:, None]
    ranks[rows, order] = np.arange(values.shape[1])[None, :]
    return ranks.astype(np.float32) / max(values.shape[1] - 1, 1)


def _projection_score(
    path_scores: dict[str, np.ndarray],
    distances: np.ndarray,
    *,
    metric: str,
    horizon: int,
    pool: int,
    profile_weight: float,
) -> np.ndarray:
    values = path_scores[metric][:, :pool]
    if metric == "state":
        values = values[..., 1 : horizon + 1]
    else:
        values = values[..., :horizon]
    if metric.endswith("_max"):
        path_value = values.max(axis=-1)
    else:
        path_value = values.mean(axis=-1)
    return _ranks(path_value) + profile_weight * _ranks(distances[:, :pool])


def _evaluate_spec(
    train: dict[str, np.ndarray],
    query: dict[str, np.ndarray],
    neighbours: np.ndarray,
    distances: np.ndarray,
    path_scores: dict[str, np.ndarray],
    spec: dict[str, Any],
    *,
    length_scale: float,
    joint_scale: np.ndarray,
) -> dict[str, float]:
    score = _projection_score(
        path_scores,
        distances,
        metric=spec["metric"],
        horizon=spec["horizon"],
        pool=spec["pool"],
        profile_weight=spec["profile_weight"],
    )
    local = np.argsort(score, axis=1, kind="stable")[:, : spec["average"]]
    selected = np.take_along_axis(
        neighbours[:, : spec["pool"]], local, axis=1
    )
    prediction = _predict(train, query, selected, spec["average"])
    return _metrics(
        prediction,
        query,
        length_scale=length_scale,
        joint_scale=joint_scale,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile-metric", default="hinge_050")
    parser.add_argument("--maximum-candidates", type=int, default=64)
    args = parser.parse_args()
    device = torch.device(args.device)
    config = replace(ExperimentConfig.load(args.config), device=str(device))
    manifest = DatasetManifest.load(config.paths.manifest)
    model = _build_model(config, manifest, device)
    saved = load_checkpoint(args.checkpoint, model=model, map_location=device)
    if saved["manifest_sha256"] != manifest.sha256():
        raise ValueError("checkpoint manifest mismatch")
    model.eval()
    data = {
        split: _load_split(manifest, model, split, device)
        for split in ("train", "val", "test")
    }
    rollout = {
        split: _rollout_split(model, config, manifest, data[split], split, device)
        for split in ("val", "test")
    }
    train_embedding = _profile_embedding(
        data["train"]["descriptor"], args.profile_metric
    )
    nearest = {}
    candidates = {}
    scores = {}
    for split in ("val", "test"):
        nearest[split] = _nearest(
            train_embedding,
            _profile_embedding(data[split]["descriptor"], args.profile_metric),
            maximum_k=args.maximum_candidates,
            device=device,
            chunk_size=32,
        )
        candidates[split] = _candidate_paths(
            data["train"], data[split], nearest[split][0]
        )
    joint_scale = model.joint_travel_range.detach().cpu().numpy().astype(np.float32)
    for split in ("val", "test"):
        scores[split] = _path_scores(
            candidates[split],
            rollout[split],
            length_scale=manifest.length_scale_m,
            joint_scale=joint_scale,
        )
    production = {
        split: _metrics(
            rollout[split],
            data[split],
            length_scale=manifest.length_scale_m,
            joint_scale=joint_scale,
        )
        for split in ("val", "test")
    }
    validation = []
    for metric in ("state", "increment", "increment_pose", "increment_joints"):
        for horizon in (4, 8, 12, 16, 24, 32):
            for pool in (16, 32, 64):
                if pool > args.maximum_candidates:
                    continue
                for profile_weight in (0.0, 0.25, 1.0, 4.0):
                    for average in (1, 2, 4, 8):
                        if average > pool:
                            continue
                        spec = {
                            "metric": metric,
                            "horizon": horizon,
                            "pool": pool,
                            "profile_weight": profile_weight,
                            "average": average,
                        }
                        metrics = _evaluate_spec(
                            data["train"],
                            data["val"],
                            nearest["val"][0],
                            nearest["val"][1],
                            scores["val"],
                            spec,
                            length_scale=manifest.length_scale_m,
                            joint_scale=joint_scale,
                        )
                        validation.append({"spec": spec, "metrics": metrics})
    selected = min(
        validation, key=lambda value: value["metrics"]["terminal_dx"]
    )
    test = _evaluate_spec(
        data["train"],
        data["test"],
        nearest["test"][0],
        nearest["test"][1],
        scores["test"],
        selected["spec"],
        length_scale=manifest.length_scale_m,
        joint_scale=joint_scale,
    )
    result = {
        "definition": {
            "coarse_path": "frozen production recurrent rollout",
            "candidate_manifold": "nearest complete physical train paths in loading-profile metric",
            "operator": "validation-selected metric projection of the coarse path onto the candidate manifold",
            "test_used_for_selection": False,
        },
        "contract": {
            "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "manifest_sha256": manifest.sha256(),
            "profile_metric": args.profile_metric,
            "maximum_candidates": args.maximum_candidates,
            "validation_grid_size": len(validation),
        },
        "production": production,
        "selected": selected,
        "test": test,
        "validation_grid": validation,
    }
    _write_json(args.output, result)
    print(
        json.dumps(
            {"production": production, "selected": selected, "test": test},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
