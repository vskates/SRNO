#!/usr/bin/env python3
"""Non-parametric audit of a loading-profile solution operator.

The input function is the SDF clearance pulled back to every gripper surface
sample along the complete free monotone closure.  A kernel operator transfers
the full equilibrium correction path of neighbouring train trajectories.  This
is an identifiability/headroom experiment, not a proposed production model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from srno.data.schema import DatasetManifest
from srno.geometry.gripper import GripperAsset
from srno.geometry.se3 import quaternion_xyzw_to_matrix, rotation_geodesic_angle
from srno.model import SRNOModel
from srno.types import PoseState, SDFBatch


SPLITS = ("train", "val", "test")
DEFAULT_PROFILE_METRICS = (
    "raw",
    "violation",
    "violation_late",
    "violation_terminal",
    "hinge_025",
    "hinge_050",
    "hinge_100",
    "hinge_200",
    "hinge_050_late",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _matrix_from_quaternion(values: np.ndarray) -> np.ndarray:
    return quaternion_xyzw_to_matrix(
        torch.from_numpy(values.astype(np.float32))
    ).numpy()


def _project_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(matrix)
    result = u @ vh
    negative = np.linalg.det(result) < 0.0
    if np.any(negative):
        u[negative, :, -1] *= -1.0
        result[negative] = u[negative] @ vh[negative]
    return result.astype(np.float32)


@torch.inference_mode()
def _load_split(
    manifest: DatasetManifest,
    model: SRNOModel,
    split: str,
    device: torch.device,
) -> dict[str, np.ndarray]:
    descriptors: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    joints: list[np.ndarray] = []
    object_ids: list[str] = []
    trajectory_ids: list[int] = []
    locations = manifest.object_locations()
    schedule = torch.tensor(
        manifest.commanded_aperture_m, dtype=torch.float32, device=device
    )
    for object_id in manifest.splits[split]:
        shard, group_name = locations[object_id]
        with h5py.File(shard, "r", swmr=True) as handle:
            group = handle[group_name]
            position = np.asarray(group["position"], dtype=np.float32)
            quaternion = np.asarray(group["quaternion_xyzw"], dtype=np.float32)
            joint = np.asarray(group["joint_position"], dtype=np.float32)
            sdf_values = torch.from_numpy(
                np.asarray(group["sdf"], dtype=np.float32)
            ).unsqueeze(0).to(device)
            origin = torch.from_numpy(
                np.asarray(group.attrs["grid_origin"], dtype=np.float32)
            ).unsqueeze(0).to(device)
            voxel = torch.from_numpy(
                np.asarray(group.attrs["voxel_size"], dtype=np.float32)
            ).unsqueeze(0).to(device)
        count = len(position)
        initial_rotation = quaternion_xyzw_to_matrix(
            torch.from_numpy(quaternion[:, 0]).to(device)
        )
        initial_position = torch.from_numpy(position[:, 0]).to(device)
        sdf = SDFBatch(
            sdf_values,
            origin,
            voxel,
            torch.zeros(count, dtype=torch.long, device=device),
            manifest.sdf_scale_m,
        )
        gaps = []
        for command in schedule:
            free_joint = model.free_joint_configuration(command).unsqueeze(0).expand(
                count, -1
            )
            state = PoseState(initial_rotation, initial_position, free_joint)
            gaps.append(model.query_gap(state, sdf))
        descriptor = torch.stack(gaps, dim=1) / manifest.sdf_scale_m
        descriptors.append(descriptor.float().cpu().numpy())
        positions.append(position)
        rotations.append(_matrix_from_quaternion(quaternion))
        joints.append(joint)
        object_ids.extend([object_id] * count)
        trajectory_ids.extend(range(count))
        print(
            f"[PROFILE] split={split} object={object_id} trajectories={count}",
            flush=True,
        )
    return {
        "descriptor": np.concatenate(descriptors),
        "position": np.concatenate(positions),
        "rotation": np.concatenate(rotations),
        "joint": np.concatenate(joints),
        "object_id": np.asarray(object_ids),
        "trajectory": np.asarray(trajectory_ids, dtype=np.int32),
    }


def _nearest(
    train: np.ndarray,
    query: np.ndarray,
    *,
    maximum_k: int,
    device: torch.device,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_tensor = torch.from_numpy(train.reshape(len(train), -1)).to(device)
    indices: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    for start in range(0, len(query), chunk_size):
        values = torch.from_numpy(
            query[start : start + chunk_size].reshape(
                min(chunk_size, len(query) - start), -1
            )
        ).to(device)
        distance = torch.cdist(values, train_tensor) / np.sqrt(train_tensor.shape[1])
        selected_distance, selected = torch.topk(
            distance, k=maximum_k, dim=1, largest=False, sorted=True
        )
        indices.append(selected.cpu().numpy())
        distances.append(selected_distance.cpu().numpy())
    return np.concatenate(indices), np.concatenate(distances)


def _profile_embedding(profile: np.ndarray, name: str) -> np.ndarray:
    """Embed clearance profiles in a contact-sensitive Hilbert metric.

    ``profile`` is the contact-onset gap divided by the SDF scale.  Hence the
    hinge radii below are dimensionless fractions of that documented scale.
    Multiplication by sqrt(weight) realizes a weighted L2 metric in loading
    time without changing the meaning of the embedded contact field.
    """

    late = name.endswith("_late")
    base_name = name.removesuffix("_late")
    if base_name == "raw":
        result = profile
    elif base_name in {"violation", "violation_terminal"}:
        result = np.maximum(-profile, 0.0)
    elif base_name.startswith("hinge_"):
        radius = int(base_name.split("_")[1]) / 100.0
        result = np.maximum(radius - profile, 0.0) / radius
    else:
        raise ValueError(f"unknown profile metric {name!r}")
    if base_name == "violation_terminal":
        return result[:, -1:, :]
    if late:
        loading_time = np.linspace(
            1.0 / result.shape[1], 1.0, result.shape[1], dtype=np.float32
        )
        result = result * loading_time[None, :, None]
    return np.ascontiguousarray(result, dtype=np.float32)


def _predict(
    train: dict[str, np.ndarray],
    query: dict[str, np.ndarray],
    neighbours: np.ndarray,
    count: int,
) -> PoseState:
    selected = neighbours[:, :count]
    train_position_delta = train["position"] - train["position"][:, :1]
    position_delta = train_position_delta[selected].mean(axis=1)
    position = query["position"][:, :1] + position_delta

    train_rotation_delta = (
        train["rotation"]
        @ np.swapaxes(train["rotation"][:, :1], -1, -2)
    )
    mean_delta = train_rotation_delta[selected].mean(axis=1)
    rotation_delta = _project_rotation(mean_delta.reshape(-1, 3, 3)).reshape(
        len(query["position"]), 33, 3, 3
    )
    rotation = rotation_delta @ query["rotation"][:, :1]

    joint = train["joint"][selected].mean(axis=1)
    # The initial state is observed exactly and must not be replaced by a
    # neighbour average.
    position[:, 0] = query["position"][:, 0]
    rotation[:, 0] = query["rotation"][:, 0]
    joint[:, 0] = query["joint"][:, 0]
    return PoseState(
        torch.from_numpy(rotation),
        torch.from_numpy(position),
        torch.from_numpy(joint.astype(np.float32)),
    )


def _metrics(
    prediction: PoseState,
    target: dict[str, np.ndarray],
    *,
    length_scale: float,
    joint_scale: np.ndarray,
) -> dict[str, float]:
    target_position = torch.from_numpy(target["position"])
    target_rotation = torch.from_numpy(target["rotation"])
    target_joint = torch.from_numpy(target["joint"])
    translation = torch.linalg.vector_norm(
        prediction.position - target_position, dim=-1
    ) / length_scale
    rotation = rotation_geodesic_angle(prediction.rotation, target_rotation)
    joints = torch.sqrt(
        (((prediction.joint_position - target_joint) / torch.from_numpy(joint_scale)).square()).mean(dim=-1)
    )
    dx = torch.sqrt(translation.square() + rotation.square() + joints.square())
    result: dict[str, float] = {}
    for name, value in (
        ("dx", dx),
        ("translation_over_length", translation),
        ("rotation_rad", rotation),
        ("joints", joints),
    ):
        result[f"path_{name}"] = float(value[:, 1:].mean())
        result[f"terminal_{name}"] = float(value[:, -1].mean())
    result["terminal_translation_m"] = float(
        torch.linalg.vector_norm(
            prediction.position[:, -1] - target_position[:, -1], dim=-1
        ).mean()
    )
    return result


def _candidate_oracle(
    train: dict[str, np.ndarray],
    query: dict[str, np.ndarray],
    neighbours: np.ndarray,
    *,
    length_scale: float,
    joint_scale: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    """Non-causal upper bound from the finite set of nearest path branches."""

    candidates = [
        _predict(train, query, neighbours[:, rank : rank + 1], 1)
        for rank in range(neighbours.shape[1])
    ]
    target_position = torch.from_numpy(query["position"][:, -1])
    target_rotation = torch.from_numpy(query["rotation"][:, -1])
    target_joint = torch.from_numpy(query["joint"][:, -1])
    errors = []
    for candidate in candidates:
        translation = torch.linalg.vector_norm(
            candidate.position[:, -1] - target_position, dim=-1
        ) / length_scale
        rotation = rotation_geodesic_angle(
            candidate.rotation[:, -1], target_rotation
        )
        joints = torch.sqrt(
            (
                (
                    (candidate.joint_position[:, -1] - target_joint)
                    / torch.from_numpy(joint_scale)
                ).square()
            ).mean(dim=-1)
        )
        errors.append(
            torch.sqrt(translation.square() + rotation.square() + joints.square())
        )
    selected_rank = torch.stack(errors, dim=1).argmin(dim=1)
    row = torch.arange(len(query["position"]))
    prediction = PoseState(
        torch.stack([value.rotation for value in candidates], dim=1)[
            row, selected_rank
        ],
        torch.stack([value.position for value in candidates], dim=1)[
            row, selected_rank
        ],
        torch.stack([value.joint_position for value in candidates], dim=1)[
            row, selected_rank
        ],
    )
    return (
        _metrics(
            prediction,
            query,
            length_scale=length_scale,
            joint_scale=joint_scale,
        ),
        selected_rank.numpy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--neighbours", default="1,2,4,8,16")
    parser.add_argument(
        "--profile-metrics", default=",".join(DEFAULT_PROFILE_METRICS)
    )
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()
    counts = tuple(sorted({int(value) for value in args.neighbours.split(",")}))
    profile_metrics = tuple(
        dict.fromkeys(value.strip() for value in args.profile_metrics.split(","))
    )
    if not counts or min(counts) <= 0 or args.chunk_size <= 0:
        parser.error("neighbour and chunk counts must be positive")
    if not profile_metrics or any(not value for value in profile_metrics):
        parser.error("profile metrics must be non-empty")

    manifest = DatasetManifest.load(args.manifest)
    gripper = GripperAsset.load(manifest.gripper_path)
    device = torch.device(args.device)
    model = SRNOModel(
        gripper,
        sdf_scale=manifest.sdf_scale_m,
        delta_gate=manifest.delta_gate_m,
        contact_offset_sum=manifest.contact_offset_sum_m,
    ).to(device)
    model.eval()
    data = {
        split: _load_split(manifest, model, split, device) for split in SPLITS
    }
    maximum_k = max(counts)
    nearest: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for metric in profile_metrics:
        train_embedding = _profile_embedding(data["train"]["descriptor"], metric)
        nearest[metric] = {}
        for split in ("val", "test"):
            nearest[metric][split] = _nearest(
                train_embedding,
                _profile_embedding(data[split]["descriptor"], metric),
                maximum_k=maximum_k,
                device=device,
                chunk_size=args.chunk_size,
            )
        print(f"[METRIC] completed={metric}", flush=True)
    joint_scale = gripper.joint_travel_range.numpy().astype(np.float32)
    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        "val": {},
        "test": {},
    }
    for metric in profile_metrics:
        for split in ("val", "test"):
            indices, distances = nearest[metric][split]
            results[split][metric] = {}
            for count in counts:
                prediction = _predict(data["train"], data[split], indices, count)
                values = _metrics(
                    prediction,
                    data[split],
                    length_scale=manifest.length_scale_m,
                    joint_scale=joint_scale,
                )
                values["mean_profile_distance"] = float(
                    distances[:, :count].mean()
                )
                results[split][metric][str(count)] = values
    selected_metric, selected_count = min(
        ((metric, count) for metric in profile_metrics for count in counts),
        key=lambda item: results["val"][item[0]][str(item[1])]["terminal_dx"],
    )
    candidate_oracles: dict[str, dict[str, dict[str, float] | list[int]]] = {}
    for metric in profile_metrics:
        candidate_oracles[metric] = {}
        for split in ("val", "test"):
            oracle_metrics, selected_ranks = _candidate_oracle(
                data["train"],
                data[split],
                nearest[metric][split][0],
                length_scale=manifest.length_scale_m,
                joint_scale=joint_scale,
            )
            candidate_oracles[metric][split] = {
                **oracle_metrics,
                "selected_rank_histogram": np.bincount(
                    selected_ranks, minlength=maximum_k
                ).tolist(),
            }
    result = {
        "definition": {
            "input": "g_{phi,x0}(a,s): full free-closure SDF clearance function",
            "output": "full equilibrium correction path relative to observed x0",
            "kernel": "unweighted k-nearest trajectories in a validation-selected contact-profile Hilbert metric",
            "rotation_average": "chordal mean projected to SO(3)",
            "test_used_for_selection": False,
        },
        "contract": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest.sha256(),
            "train_trajectories": len(data["train"]["position"]),
            "val_trajectories": len(data["val"]["position"]),
            "test_trajectories": len(data["test"]["position"]),
            "profile_shape": list(data["train"]["descriptor"].shape[1:]),
            "neighbour_grid": counts,
            "profile_metric_grid": profile_metrics,
        },
        "selected_profile_metric": selected_metric,
        "selected_neighbours": selected_count,
        "results": results,
        "selected_test": results["test"][selected_metric][str(selected_count)],
        "noncausal_candidate_oracles": candidate_oracles,
    }
    _write_json(args.output, result)
    print(json.dumps({
        "selected_profile_metric": selected_metric,
        "selected_neighbours": selected_count,
        "selected_val": results["val"][selected_metric][str(selected_count)],
        "selected_test": result["selected_test"],
        "selected_metric_test_candidate_oracle": candidate_oracles[
            selected_metric
        ]["test"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
