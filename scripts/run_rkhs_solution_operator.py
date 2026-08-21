#!/usr/bin/env python3
"""Fit a vector-valued RKHS operator from loading profiles to solution paths.

Unlike the previous cumulative SRNO arm, the input is the complete clearance
function over load and gripper surface, and the output is the complete path in
product-manifold coordinates.  Kernel bandwidth and Tikhonov regularization
are selected on validation terminal dX; test is evaluated only once.
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

from evaluate_loading_profile_kernel_operator import (
    _load_split,
    _metrics,
    _profile_embedding,
)
from srno.data.schema import DatasetManifest
from srno.geometry.gripper import GripperAsset
from srno.geometry.se3 import so3_exp, so3_log_vector
from srno.model import SRNOModel
from srno.types import PoseState


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _output_coordinates(
    data: dict[str, np.ndarray],
    model: SRNOModel,
    manifest: DatasetManifest,
) -> np.ndarray:
    position = (data["position"] - data["position"][:, :1]) / manifest.length_scale_m
    rotation_delta = (
        data["rotation"]
        @ np.swapaxes(data["rotation"][:, :1], -1, -2)
    )
    rotation = so3_log_vector(torch.from_numpy(rotation_delta)).numpy()
    schedule = torch.tensor(manifest.commanded_aperture_m, dtype=torch.float32)
    free_joint = model.free_joint_configuration(schedule).cpu().numpy()
    joint_scale = model.joint_travel_range.cpu().numpy()
    joint = (data["joint"] - free_joint[None]) / joint_scale[None, None]
    path = np.concatenate((position, rotation, joint), axis=-1)
    return path[:, 1:].reshape(len(path), -1).astype(np.float32)


def _decode(
    coordinates: np.ndarray,
    query: dict[str, np.ndarray],
    model: SRNOModel,
    manifest: DatasetManifest,
) -> PoseState:
    path = coordinates.reshape(len(coordinates), 32, 12)
    position = (
        query["position"][:, :1]
        + path[..., :3] * manifest.length_scale_m
    )
    rotation_delta = so3_exp(torch.from_numpy(path[..., 3:6])).numpy()
    rotation = rotation_delta @ query["rotation"][:, :1]
    schedule = torch.tensor(manifest.commanded_aperture_m[1:], dtype=torch.float32)
    free_joint = model.free_joint_configuration(schedule).cpu().numpy()
    joint_scale = model.joint_travel_range.cpu().numpy()
    joint = free_joint[None] + path[..., 6:] * joint_scale[None, None]
    return PoseState(
        torch.from_numpy(
            np.concatenate((query["rotation"][:, :1], rotation), axis=1)
        ),
        torch.from_numpy(
            np.concatenate((query["position"][:, :1], position), axis=1)
        ),
        torch.from_numpy(
            np.concatenate((query["joint"][:, :1], joint), axis=1).astype(
                np.float32
            )
        ),
    )


def _objectwise_metrics(
    prediction: PoseState,
    target: dict[str, np.ndarray],
    manifest: DatasetManifest,
    joint_scale: np.ndarray,
) -> dict[str, dict[str, float]]:
    result = {}
    for object_id in dict.fromkeys(target["object_id"].tolist()):
        rows = np.flatnonzero(target["object_id"] == object_id)
        subset_prediction = PoseState(
            prediction.rotation[rows],
            prediction.position[rows],
            prediction.joint_position[rows],
        )
        subset_target = {
            name: value[rows]
            for name, value in target.items()
            if name in {"position", "rotation", "joint"}
        }
        result[object_id] = _metrics(
            subset_prediction,
            subset_target,
            length_scale=manifest.length_scale_m,
            joint_scale=joint_scale,
        )
    return result


def _flat_embedding(data: dict[str, np.ndarray], metric: str) -> np.ndarray:
    values = _profile_embedding(data["descriptor"], metric)
    return np.ascontiguousarray(values.reshape(len(values), -1), dtype=np.float32)


def _squared_distance(
    left: np.ndarray,
    right: np.ndarray,
    device: torch.device,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    right_tensor = torch.from_numpy(right).to(device)
    result = []
    for start in range(0, len(left), chunk_size):
        left_tensor = torch.from_numpy(left[start : start + chunk_size]).to(device)
        distance = torch.cdist(left_tensor, right_tensor) / np.sqrt(right.shape[1])
        result.append(distance.square())
    return torch.cat(result, dim=0)


def _solve(
    kernel: torch.Tensor,
    centered_output: torch.Tensor,
    regularization: float,
) -> torch.Tensor:
    identity = torch.eye(
        len(kernel), dtype=kernel.dtype, device=kernel.device
    )
    return torch.linalg.solve(
        kernel + regularization * identity, centered_output
    )


def _global_shape_distances(
    manifest: DatasetManifest,
    data: dict[str, dict[str, np.ndarray]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Return squared complete-SDF distances from each split to train."""

    locations = manifest.object_locations()
    object_ids = list(
        dict.fromkeys(
            object_id
            for split in ("train", "val", "test")
            for object_id in manifest.splits[split]
        )
    )
    fields = []
    extents = []
    for object_id in object_ids:
        shard, group_name = locations[object_id]
        with h5py.File(shard, "r", swmr=True) as handle:
            group = handle[group_name]
            sdf = np.asarray(group["sdf"], dtype=np.float32)
            voxel = np.asarray(group.attrs["voxel_size"], dtype=np.float32)
        fields.append(np.clip(sdf / manifest.sdf_scale_m, -2.0, 2.0).reshape(-1))
        extents.append(voxel * (np.asarray(sdf.shape, dtype=np.float32) - 1.0))
    field_tensor = torch.from_numpy(np.stack(fields)).to(device)
    extent_tensor = torch.from_numpy(np.stack(extents)).to(device)
    field_distance = torch.cdist(field_tensor, field_tensor) / np.sqrt(
        field_tensor.shape[1]
    )
    extent_distance = torch.cdist(
        extent_tensor / manifest.length_scale_m,
        extent_tensor / manifest.length_scale_m,
    ) / np.sqrt(3.0)
    object_distance = torch.sqrt(
        field_distance.square() + extent_distance.square()
    )
    object_index = {value: index for index, value in enumerate(object_ids)}
    train_object_rows = torch.tensor(
        [object_index[value] for value in data["train"]["object_id"]],
        dtype=torch.long,
        device=device,
    )
    train_unique = torch.tensor(
        [object_index[value] for value in manifest.splits["train"]],
        dtype=torch.long,
        device=device,
    )
    train_object_distance = object_distance.index_select(
        0, train_unique
    ).index_select(1, train_unique)
    nonzero = train_object_distance[train_object_distance > 0.0]
    shape_median = nonzero.median()
    result = {}
    for split in ("train", "val", "test"):
        query_rows = torch.tensor(
            [object_index[value] for value in data[split]["object_id"]],
            dtype=torch.long,
            device=device,
        )
        result[split] = (
            object_distance.index_select(0, query_rows).index_select(
                1, train_object_rows
            )
            / shape_median
        ).square()
    return result


def _object_mass_distances(
    manifest: DatasetManifest,
    data: dict[str, dict[str, np.ndarray]],
    device: torch.device,
    catalog_path: Path,
) -> dict[str, torch.Tensor]:
    payload = json.loads(catalog_path.read_text())
    masses = {
        str(record["id"]): float(record["mass_kg"])
        for record in payload["objects"]
    }
    train_object_mass = torch.tensor(
        [masses[value] for value in manifest.splits["train"]],
        dtype=torch.float32,
        device=device,
    )
    object_scale = train_object_mass.log().std().clamp_min(1e-6)
    train_trajectory_mass = torch.tensor(
        [masses[value] for value in data["train"]["object_id"]],
        dtype=torch.float32,
        device=device,
    ).log()
    result = {}
    for split in ("train", "val", "test"):
        query_mass = torch.tensor(
            [masses[value] for value in data[split]["object_id"]],
            dtype=torch.float32,
            device=device,
        ).log()
        result[split] = (
            (query_mass[:, None] - train_trajectory_mass[None, :]) / object_scale
        ).square()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile-metrics", default="hinge_025,hinge_050,hinge_100,raw")
    parser.add_argument("--bandwidth-factors", default="0.25,0.5,1,2,4")
    parser.add_argument("--regularizations", default="0.0001,0.001,0.01,0.1,1")
    parser.add_argument("--shape-weights", default="0")
    parser.add_argument("--mass-weights", default="0")
    parser.add_argument("--catalog", type=Path, default=Path("assets/catalog.json"))
    args = parser.parse_args()
    metrics = tuple(dict.fromkeys(args.profile_metrics.split(",")))
    bandwidth_factors = tuple(float(value) for value in args.bandwidth_factors.split(","))
    regularizations = tuple(float(value) for value in args.regularizations.split(","))
    shape_weights = tuple(float(value) for value in args.shape_weights.split(","))
    mass_weights = tuple(float(value) for value in args.mass_weights.split(","))
    device = torch.device(args.device)
    manifest = DatasetManifest.load(args.manifest)
    gripper = GripperAsset.load(manifest.gripper_path)
    model = SRNOModel(
        gripper,
        sdf_scale=manifest.sdf_scale_m,
        delta_gate=manifest.delta_gate_m,
        contact_offset_sum=manifest.contact_offset_sum_m,
    )
    model.eval()
    geometry_model = model.to(device)
    data = {
        split: _load_split(manifest, geometry_model, split, device)
        for split in ("train", "val", "test")
    }
    model = model.to("cpu")
    train_output = _output_coordinates(data["train"], model, manifest)
    output_mean = train_output.mean(axis=0, keepdims=True)
    centered_output = torch.from_numpy(train_output - output_mean).to(device)
    joint_scale = gripper.joint_travel_range.numpy().astype(np.float32)
    shape_distance = _global_shape_distances(manifest, data, device)
    mass_distance = _object_mass_distances(
        manifest, data, device, args.catalog
    )
    validation = []
    metric_cache: dict[str, dict[str, Any]] = {}
    for profile_metric in metrics:
        train_embedding = _flat_embedding(data["train"], profile_metric)
        val_embedding = _flat_embedding(data["val"], profile_metric)
        train_distance = _squared_distance(train_embedding, train_embedding, device)
        val_distance = _squared_distance(val_embedding, train_embedding, device)
        off_diagonal = train_distance[
            ~torch.eye(len(train_distance), dtype=torch.bool, device=device)
        ]
        median_distance = float(torch.sqrt(off_diagonal.median()).cpu())
        metric_cache[profile_metric] = {
            "train_embedding": train_embedding,
            "median_distance": median_distance,
        }
        for shape_weight in shape_weights:
            shape_scale = (shape_weight * median_distance) ** 2
            for mass_weight in mass_weights:
                mass_scale = (mass_weight * median_distance) ** 2
                combined_train_distance = (
                    train_distance
                    + shape_scale * shape_distance["train"]
                    + mass_scale * mass_distance["train"]
                )
                combined_val_distance = (
                    val_distance
                    + shape_scale * shape_distance["val"]
                    + mass_scale * mass_distance["val"]
                )
                for factor in bandwidth_factors:
                    bandwidth = factor * median_distance
                    train_kernel = torch.exp(
                        -combined_train_distance / (2.0 * bandwidth**2)
                    )
                    val_kernel = torch.exp(
                        -combined_val_distance / (2.0 * bandwidth**2)
                    )
                    for regularization in regularizations:
                        coefficients = _solve(
                            train_kernel, centered_output, regularization
                        )
                        prediction_coordinates = (
                            val_kernel @ coefficients
                        ).cpu().numpy() + output_mean
                        prediction = _decode(
                            prediction_coordinates.astype(np.float32),
                            data["val"],
                            model,
                            manifest,
                        )
                        values = _metrics(
                            prediction,
                            data["val"],
                            length_scale=manifest.length_scale_m,
                            joint_scale=joint_scale,
                        )
                        validation.append(
                            {
                                "profile_metric": profile_metric,
                                "shape_weight": shape_weight,
                                "mass_weight": mass_weight,
                                "bandwidth_factor": factor,
                                "bandwidth": bandwidth,
                                "regularization": regularization,
                                "metrics": values,
                            }
                        )
            best_so_far = min(
                validation, key=lambda value: value["metrics"]["terminal_dx"]
            )
            print(
                f"[RKHS] metric={profile_metric} shape={shape_weight:g} "
                f"mass={mass_weight:g} "
                f"factor={factor:g} "
                f"best_val={best_so_far['metrics']['terminal_dx']:.8f}",
                flush=True,
            )
        del train_distance, val_distance
        torch.cuda.empty_cache() if device.type == "cuda" else None
    selected = min(
        validation, key=lambda value: value["metrics"]["terminal_dx"]
    )
    profile_metric = selected["profile_metric"]
    train_embedding = metric_cache[profile_metric]["train_embedding"]
    test_embedding = _flat_embedding(data["test"], profile_metric)
    train_distance = _squared_distance(train_embedding, train_embedding, device)
    test_distance = _squared_distance(test_embedding, train_embedding, device)
    shape_scale = (
        selected["shape_weight"]
        * metric_cache[profile_metric]["median_distance"]
    ) ** 2
    mass_scale = (
        selected["mass_weight"]
        * metric_cache[profile_metric]["median_distance"]
    ) ** 2
    train_distance = train_distance + shape_scale * shape_distance["train"]
    test_distance = test_distance + shape_scale * shape_distance["test"]
    train_distance = train_distance + mass_scale * mass_distance["train"]
    test_distance = test_distance + mass_scale * mass_distance["test"]
    bandwidth = selected["bandwidth"]
    train_kernel = torch.exp(-train_distance / (2.0 * bandwidth**2))
    test_kernel = torch.exp(-test_distance / (2.0 * bandwidth**2))
    coefficients = _solve(
        train_kernel, centered_output, selected["regularization"]
    )
    test_coordinates = (test_kernel @ coefficients).cpu().numpy() + output_mean
    test_prediction = _decode(
        test_coordinates.astype(np.float32), data["test"], model, manifest
    )
    test = _metrics(
        test_prediction,
        data["test"],
        length_scale=manifest.length_scale_m,
        joint_scale=joint_scale,
    )
    identity_coordinates = np.zeros_like(test_coordinates, dtype=np.float32)
    identity_prediction = _decode(
        identity_coordinates, data["test"], model, manifest
    )
    identity = _metrics(
        identity_prediction,
        data["test"],
        length_scale=manifest.length_scale_m,
        joint_scale=joint_scale,
    )
    result = {
        "definition": {
            "input": "complete free-loading clearance function g(a,s)",
            "output": "complete equilibrium path in relative product-manifold coordinates",
            "operator": "vector-valued Gaussian RKHS regression with Tikhonov regularization",
            "recurrent_state_feedback": False,
            "test_used_for_selection": False,
        },
        "contract": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest.sha256(),
            "train_trajectories": len(data["train"]["position"]),
            "val_trajectories": len(data["val"]["position"]),
            "test_trajectories": len(data["test"]["position"]),
            "profile_metric_grid": metrics,
            "bandwidth_factor_grid": bandwidth_factors,
            "regularization_grid": regularizations,
            "shape_weight_grid": shape_weights,
            "mass_weight_grid": mass_weights,
            "catalog": str(args.catalog.resolve()),
            "validation_grid_size": len(validation),
        },
        "selected": selected,
        "test": test,
        "test_objectwise": _objectwise_metrics(
            test_prediction, data["test"], manifest, joint_scale
        ),
        "identity_pose_free_joint_control": identity,
        "identity_control_objectwise": _objectwise_metrics(
            identity_prediction, data["test"], manifest, joint_scale
        ),
        "validation_grid": validation,
    }
    _write_json(args.output, result)
    print(json.dumps({"selected": selected, "test": test}, indent=2), flush=True)


if __name__ == "__main__":
    main()
