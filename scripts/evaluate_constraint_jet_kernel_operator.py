#!/usr/bin/env python3
"""Audit a solution operator indexed by the contact-potential first jet.

The input is not a clearance trace at one pose.  It is a Galerkin sampling of
the local configuration-space constraint functional along the complete loading
program: a smooth unilateral potential and its derivative in dimensionless
SE(3) x joint coordinates.  A non-parametric path transfer tests whether this
mathematical argument has sufficient cross-geometry identifiability before a
parametric operator is trained.
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

from evaluate_loading_profile_kernel_operator import _metrics, _nearest, _predict
from srno.data.schema import DatasetManifest
from srno.geometry.gripper import GripperAsset
from srno.geometry.se3 import quaternion_xyzw_to_matrix
from srno.model import SRNOModel
from srno.types import PoseState, SDFBatch


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


@torch.inference_mode()
def _load_split(
    manifest: DatasetManifest,
    model: SRNOModel,
    split: str,
    device: torch.device,
    load_indices: tuple[int, ...],
) -> dict[str, np.ndarray]:
    gaps = []
    jacobians = []
    positions = []
    rotations = []
    joints = []
    object_ids = []
    trajectory_ids = []
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
        object_gaps = []
        object_jacobians = []
        for load_index in load_indices:
            free_joint = model.free_joint_configuration(
                schedule[load_index]
            ).unsqueeze(0).expand(count, -1)
            state = PoseState(initial_rotation, initial_position, free_joint)
            contact_gap, _, jacobian = model._contact_gap_and_full_jacobian(
                state, sdf
            )
            geometric_gap = contact_gap + model.contact_offset_sum
            object_gaps.append(geometric_gap / model.sdf_scale)
            object_jacobians.append(jacobian)
        gaps.append(torch.stack(object_gaps, dim=1).half().cpu().numpy())
        jacobians.append(
            torch.stack(object_jacobians, dim=1).half().cpu().numpy()
        )
        positions.append(position)
        rotations.append(
            quaternion_xyzw_to_matrix(torch.from_numpy(quaternion)).numpy()
        )
        joints.append(joint)
        object_ids.extend([object_id] * count)
        trajectory_ids.extend(range(count))
        print(
            f"[JET] split={split} object={object_id} trajectories={count}",
            flush=True,
        )
    return {
        "gap": np.concatenate(gaps),
        "jacobian": np.concatenate(jacobians),
        "position": np.concatenate(positions),
        "rotation": np.concatenate(rotations),
        "joint": np.concatenate(joints),
        "object_id": np.asarray(object_ids),
        "trajectory": np.asarray(trajectory_ids, dtype=np.int32),
    }


def _jet_embedding(
    data: dict[str, np.ndarray],
    *,
    tau: float,
    gradient_weight: float,
    coordinates: str,
) -> np.ndarray:
    gap = data["gap"].astype(np.float32)
    potential = tau * np.logaddexp(0.0, -gap / tau)
    if coordinates == "potential":
        return np.ascontiguousarray(potential)
    activation = 1.0 / (1.0 + np.exp(np.clip(gap / tau, -30.0, 30.0)))
    if coordinates == "pose":
        jacobian = data["jacobian"][..., :6].astype(np.float32)
    elif coordinates == "all":
        jacobian = data["jacobian"].astype(np.float32)
    else:
        raise ValueError(f"unknown coordinates {coordinates!r}")
    gradient = activation[..., None] * jacobian
    return np.ascontiguousarray(
        np.concatenate(
            (potential[..., None], gradient_weight * gradient), axis=-1
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-stride", type=int, default=2)
    parser.add_argument("--neighbours", default="1,2,4,8,16,32,64")
    parser.add_argument("--chunk-size", type=int, default=16)
    args = parser.parse_args()
    if args.load_stride <= 0:
        parser.error("load stride must be positive")
    load_indices = tuple(range(0, 33, args.load_stride))
    if load_indices[-1] != 32:
        load_indices = (*load_indices, 32)
    counts = tuple(sorted({int(value) for value in args.neighbours.split(",")}))
    device = torch.device(args.device)
    manifest = DatasetManifest.load(args.manifest)
    gripper = GripperAsset.load(manifest.gripper_path)
    model = SRNOModel(
        gripper,
        sdf_scale=manifest.sdf_scale_m,
        delta_gate=manifest.delta_gate_m,
        contact_offset_sum=manifest.contact_offset_sum_m,
    ).to(device)
    model.eval()
    data = {
        split: _load_split(manifest, model, split, device, load_indices)
        for split in ("train", "val", "test")
    }
    joint_scale = gripper.joint_travel_range.numpy().astype(np.float32)
    variants = [
        {"tau": tau, "gradient_weight": 0.0, "coordinates": "potential"}
        for tau in (0.25, 0.5, 1.0)
    ]
    variants.extend(
        {
            "tau": tau,
            "gradient_weight": weight,
            "coordinates": coordinates,
        }
        for tau in (0.25, 0.5, 1.0)
        for coordinates in ("pose", "all")
        for weight in (0.01, 0.025, 0.05, 0.1, 0.2)
    )
    validation = []
    for variant in variants:
        train_embedding = _jet_embedding(data["train"], **variant)
        val_embedding = _jet_embedding(data["val"], **variant)
        neighbours, distances = _nearest(
            train_embedding,
            val_embedding,
            maximum_k=max(counts),
            device=device,
            chunk_size=args.chunk_size,
        )
        del train_embedding, val_embedding
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for count in counts:
            prediction = _predict(
                data["train"], data["val"], neighbours, count
            )
            metrics = _metrics(
                prediction,
                data["val"],
                length_scale=manifest.length_scale_m,
                joint_scale=joint_scale,
            )
            validation.append(
                {
                    "variant": variant,
                    "neighbours": count,
                    "metrics": metrics,
                    "mean_distance": float(distances[:, :count].mean()),
                }
            )
        best_so_far = min(
            validation, key=lambda value: value["metrics"]["terminal_dx"]
        )
        print(
            f"[VARIANT] {variant} best_val="
            f"{best_so_far['metrics']['terminal_dx']:.8f}",
            flush=True,
        )
    selected = min(
        validation, key=lambda value: value["metrics"]["terminal_dx"]
    )
    train_embedding = _jet_embedding(data["train"], **selected["variant"])
    test_embedding = _jet_embedding(data["test"], **selected["variant"])
    test_neighbours, test_distances = _nearest(
        train_embedding,
        test_embedding,
        maximum_k=selected["neighbours"],
        device=device,
        chunk_size=args.chunk_size,
    )
    test_prediction = _predict(
        data["train"],
        data["test"],
        test_neighbours,
        selected["neighbours"],
    )
    test = _metrics(
        test_prediction,
        data["test"],
        length_scale=manifest.length_scale_m,
        joint_scale=joint_scale,
    )
    test["mean_distance"] = float(test_distances.mean())
    result = {
        "definition": {
            "input": "first jet (Psi_tau(g), gradient_z Psi_tau(g)) of a smooth unilateral contact potential along loading",
            "output": "complete equilibrium path relative to observed x0",
            "kernel": "unweighted nearest paths in the validation-selected first-jet L2 metric",
            "test_used_for_selection": False,
        },
        "contract": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest.sha256(),
            "load_indices": load_indices,
            "variant_grid_size": len(variants),
            "neighbour_grid": counts,
            "train_trajectories": len(data["train"]["position"]),
            "val_trajectories": len(data["val"]["position"]),
            "test_trajectories": len(data["test"]["position"]),
        },
        "selected": selected,
        "test": test,
        "validation_grid": validation,
    }
    _write_json(args.output, result)
    print(json.dumps({"selected": selected, "test": test}, indent=2), flush=True)


if __name__ == "__main__":
    main()
