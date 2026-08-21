#!/usr/bin/env python3
"""Select a loading-profile path branch by a local-resolvent defect.

This is a multiple-shooting/collocation experiment.  The candidate set is
formed without target states by nearest loading-clearance profiles.  A frozen
local implicit resolvent then scores whether each complete candidate path is a
discrete solution for the *query* SDF.  Selector hyperparameters are fixed on
validation before the untouched test metrics are evaluated.
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
from srno.training.engine import _build_model
from srno.types import PoseState, SDFBatch


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _query_sdf(
    manifest: DatasetManifest,
    query: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    locations = manifest.object_locations()
    object_ids = list(dict.fromkeys(query["object_id"].tolist()))
    values = []
    origins = []
    voxels = []
    for object_id in object_ids:
        shard, group_name = locations[object_id]
        with h5py.File(shard, "r", swmr=True) as handle:
            group = handle[group_name]
            values.append(np.asarray(group["sdf"], dtype=np.float32))
            origins.append(np.asarray(group.attrs["grid_origin"], dtype=np.float32))
            voxels.append(np.asarray(group.attrs["voxel_size"], dtype=np.float32))
    object_index = {value: index for index, value in enumerate(object_ids)}
    mapping = np.asarray(
        [object_index[value] for value in query["object_id"]], dtype=np.int64
    )
    return (
        torch.from_numpy(np.stack(values)).to(device),
        torch.from_numpy(np.stack(origins)).to(device),
        torch.from_numpy(np.stack(voxels)).to(device),
        torch.from_numpy(mapping).to(device),
    )


def _candidate_paths(
    train: dict[str, np.ndarray],
    query: dict[str, np.ndarray],
    neighbours: np.ndarray,
) -> PoseState:
    paths = [
        _predict(train, query, neighbours[:, rank : rank + 1], 1)
        for rank in range(neighbours.shape[1])
    ]
    return PoseState(
        torch.stack([value.rotation for value in paths], dim=1),
        torch.stack([value.position for value in paths], dim=1),
        torch.stack([value.joint_position for value in paths], dim=1),
    )


def _slice_state(
    candidates: PoseState,
    start: int,
    stop: int,
    step: int,
    device: torch.device,
) -> PoseState:
    return PoseState(
        candidates.rotation[:, start:stop, step].reshape(-1, 3, 3).to(device),
        candidates.position[:, start:stop, step].reshape(-1, 3).to(device),
        candidates.joint_position[:, start:stop, step].reshape(-1, 6).to(device),
    )


def _state_dx(
    prediction: PoseState,
    target: PoseState,
    *,
    length_scale: float,
    joint_scale: torch.Tensor,
) -> torch.Tensor:
    translation = torch.linalg.vector_norm(
        prediction.position - target.position, dim=-1
    ) / length_scale
    rotation = rotation_geodesic_angle(prediction.rotation, target.rotation)
    joints = torch.sqrt(
        (((prediction.joint_position - target.joint_position) / joint_scale).square()).mean(
            dim=-1
        )
    )
    return torch.sqrt(translation.square() + rotation.square() + joints.square())


@torch.inference_mode()
def _collocation_features(
    model: torch.nn.Module,
    manifest: DatasetManifest,
    query: dict[str, np.ndarray],
    candidates: PoseState,
    *,
    steps: tuple[int, ...],
    candidate_chunk: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    values, origins, voxels, query_mapping = _query_sdf(manifest, query, device)
    query_count, candidate_count = candidates.joint_position.shape[:2]
    defect = np.empty((query_count, candidate_count, len(steps)), dtype=np.float32)
    feasibility = np.empty_like(defect)
    joint_scale = model.joint_travel_range
    schedule = torch.tensor(
        manifest.commanded_aperture_m, dtype=torch.float32, device=device
    )
    for step_column, step in enumerate(steps):
        for start in range(0, candidate_count, candidate_chunk):
            stop = min(start + candidate_chunk, candidate_count)
            width = stop - start
            current = _slice_state(candidates, start, stop, step, device)
            previous = _slice_state(candidates, start, stop, max(step - 1, 0), device)
            target = _slice_state(candidates, start, stop, step + 1, device)
            mapping = query_mapping.repeat_interleave(width)
            sdf = SDFBatch(
                values, origins, voxels, mapping, manifest.sdf_scale_m
            )
            prediction = model.forward_step(
                current,
                schedule[step + 1],
                sdf,
                previous_state=previous,
            )
            assert isinstance(prediction, PoseState)
            local_defect = _state_dx(
                prediction,
                target,
                length_scale=manifest.length_scale_m,
                joint_scale=joint_scale,
            )
            geometric_gap = model.query_geometric_gap(target, sdf)
            contact_gap = geometric_gap - model.contact_offset_sum
            penetration = torch.sqrt(
                (torch.relu(-geometric_gap / model.sdf_scale).square()).mean(dim=-1)
            )
            free_joint = model.free_joint_configuration(schedule[step + 1])
            stalled = torch.sqrt(
                (((target.joint_position - free_joint) / joint_scale).square()).mean(
                    dim=-1
                )
            )
            unsupported = stalled * torch.relu(
                contact_gap.amin(dim=-1) / model.sdf_scale
            )
            physical_residual = penetration + unsupported
            defect[:, start:stop, step_column] = local_defect.reshape(
                query_count, width
            ).cpu().numpy()
            feasibility[:, start:stop, step_column] = physical_residual.reshape(
                query_count, width
            ).cpu().numpy()
        print(f"[COLLOCATION] step={step + 1}/{len(schedule) - 1}", flush=True)
    loading_weights = np.asarray([step + 1 for step in steps], dtype=np.float32)
    loading_weights /= loading_weights.sum()
    return {
        "defect_mean": defect.mean(axis=-1),
        "defect_late": (defect * loading_weights).sum(axis=-1),
        "defect_terminal": defect[..., -1],
        "defect_max": defect.max(axis=-1),
        "feasibility_mean": feasibility.mean(axis=-1),
        "feasibility_terminal": feasibility[..., -1],
        "raw_defect": defect,
        "raw_feasibility": feasibility,
    }


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(len(values))[:, None]
    ranks[rows, order] = np.arange(values.shape[1])[None, :]
    return ranks.astype(np.float32) / max(values.shape[1] - 1, 1)


def _selected_neighbours(
    neighbours: np.ndarray,
    distances: np.ndarray,
    features: dict[str, np.ndarray],
    *,
    criterion: str,
    profile_weight: float,
    feasibility_weight: float,
    pool: int,
    average: int,
) -> np.ndarray:
    profile_rank = _ranks(distances[:, :pool])
    defect_rank = _ranks(features[criterion][:, :pool])
    feasibility_rank = _ranks(features["feasibility_mean"][:, :pool])
    score = (
        defect_rank
        + profile_weight * profile_rank
        + feasibility_weight * feasibility_rank
    )
    selected_local = np.argsort(score, axis=1, kind="stable")[:, :average]
    return np.take_along_axis(neighbours[:, :pool], selected_local, axis=1)


def _evaluate_spec(
    train: dict[str, np.ndarray],
    query: dict[str, np.ndarray],
    neighbours: np.ndarray,
    distances: np.ndarray,
    features: dict[str, np.ndarray],
    spec: dict[str, Any],
    *,
    length_scale: float,
    joint_scale: np.ndarray,
) -> dict[str, float]:
    selected = _selected_neighbours(
        neighbours,
        distances,
        features,
        criterion=spec["criterion"],
        profile_weight=spec["profile_weight"],
        feasibility_weight=spec["feasibility_weight"],
        pool=spec["pool"],
        average=spec["average"],
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile-metric", default="hinge_050")
    parser.add_argument("--maximum-candidates", type=int, default=64)
    parser.add_argument("--candidate-chunk", type=int, default=8)
    parser.add_argument("--steps", default="0,3,7,15,23,31")
    args = parser.parse_args()
    steps = tuple(sorted({int(value) for value in args.steps.split(",")}))
    if not steps or min(steps) < 0 or max(steps) >= 32:
        parser.error("collocation steps must lie in 0,...,31")

    device = torch.device(args.device)
    manifest = DatasetManifest.load(args.manifest)
    config = replace(ExperimentConfig.load(args.config), device=str(device))
    model = _build_model(config, manifest, device)
    saved = load_checkpoint(args.checkpoint, model=model, map_location=device)
    if saved["manifest_sha256"] != manifest.sha256():
        raise ValueError("checkpoint manifest mismatch")
    model.eval()
    data = {
        split: _load_split(manifest, model, split, device)
        for split in ("train", "val", "test")
    }
    train_embedding = _profile_embedding(
        data["train"]["descriptor"], args.profile_metric
    )
    nearest: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    candidates: dict[str, PoseState] = {}
    features: dict[str, dict[str, np.ndarray]] = {}
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
        features[split] = _collocation_features(
            model,
            manifest,
            data[split],
            candidates[split],
            steps=steps,
            candidate_chunk=args.candidate_chunk,
            device=device,
        )
        print(f"[SPLIT] collocation complete={split}", flush=True)

    joint_scale = model.joint_travel_range.detach().cpu().numpy().astype(np.float32)
    grid = []
    for criterion in (
        "defect_mean",
        "defect_late",
        "defect_terminal",
        "defect_max",
    ):
        for profile_weight in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            for feasibility_weight in (0.0, 0.25, 1.0):
                for pool in (16, 32, 64):
                    if pool > args.maximum_candidates:
                        continue
                    for average in (1, 2, 4, 8):
                        if average > pool:
                            continue
                        spec = {
                            "criterion": criterion,
                            "profile_weight": profile_weight,
                            "feasibility_weight": feasibility_weight,
                            "pool": pool,
                            "average": average,
                        }
                        metrics = _evaluate_spec(
                            data["train"],
                            data["val"],
                            nearest["val"][0],
                            nearest["val"][1],
                            features["val"],
                            spec,
                            length_scale=manifest.length_scale_m,
                            joint_scale=joint_scale,
                        )
                        grid.append({"spec": spec, "val": metrics})
    selected = min(grid, key=lambda value: value["val"]["terminal_dx"])
    test = _evaluate_spec(
        data["train"],
        data["test"],
        nearest["test"][0],
        nearest["test"][1],
        features["test"],
        selected["spec"],
        length_scale=manifest.length_scale_m,
        joint_scale=joint_scale,
    )
    result = {
        "definition": {
            "candidate_set": "nearest complete train solution paths in a contact-profile metric",
            "causal_selector": "validation-selected local-resolvent collocation defect plus feasibility and profile ranks",
            "target_used_by_selector_at_inference": False,
            "test_used_for_selection": False,
        },
        "contract": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest.sha256(),
            "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "profile_metric": args.profile_metric,
            "maximum_candidates": args.maximum_candidates,
            "collocation_steps": steps,
            "validation_grid_size": len(grid),
        },
        "selected": selected,
        "test": test,
        "feature_summary": {
            split: {
                name: {
                    "mean": float(value.mean()),
                    "median": float(np.median(value)),
                }
                for name, value in features[split].items()
                if not name.startswith("raw_")
            }
            for split in ("val", "test")
        },
    }
    _write_json(args.output, result)
    print(json.dumps({"selected": selected, "test": test}, indent=2), flush=True)


if __name__ == "__main__":
    main()
