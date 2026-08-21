#!/usr/bin/env python3
"""Check whether observed successors belong to the current tangent contact set."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from srno.data.dataset import H5ObjectDataset, LocalTransitionBatch, make_dataloader
from srno.data.index import ActiveIndex
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import so3_log_vector
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model


def _summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ablation-inertial-implicit-resolvent-local.toml"),
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/resolvent-target-compatibility.json"),
    )
    args = parser.parse_args()

    config = replace(ExperimentConfig.load(args.config), device="cpu")
    manifest = DatasetManifest.load(config.paths.manifest)
    active_index = ActiveIndex.load(config.paths.active_index)
    model = _build_model(config, manifest, torch.device("cpu")).eval()
    dataset = H5ObjectDataset(
        manifest,
        split=args.split,
        active_index=active_index,
        active_only=True,
    )
    loader = make_dataloader(
        dataset,
        mode="local",
        objects_per_batch=1,
        samples_per_object=0,
        workers=config.loader.workers,
        seed=config.seed + 911,
        shuffle=False,
    )

    rows: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "pose_motion",
            "minimum_linear_gap_m",
            "minimum_target_gap_m",
            "maximum_linearization_error_m",
        )
    }
    try:
        with torch.no_grad():
            for raw_batch in loader:
                assert isinstance(raw_batch, LocalTransitionBatch)
                batch = raw_batch.to("cpu")
                contact_gap, _, jacobian = model._contact_gap_and_full_jacobian(
                    batch.current, batch.sdf
                )
                geometric_gap = contact_gap + model.contact_offset_sum
                support = (contact_gap <= model.delta_gate) & (
                    torch.linalg.vector_norm(jacobian, dim=-1) > 1e-8
                )
                z = torch.cat(
                    (
                        (batch.target.position - batch.current.position)
                        / model.length_scale,
                        so3_log_vector(
                            batch.target.rotation
                            @ batch.current.rotation.transpose(-1, -2)
                        ),
                        (batch.target.joint_position - batch.current.joint_position)
                        / model.joint_travel_range,
                    ),
                    dim=-1,
                )
                linear_gap = geometric_gap + model.sdf_scale * torch.einsum(
                    "bmi,bi->bm", jacobian, z
                )
                target_gap = model.query_geometric_gap(batch.target, batch.sdf)
                infinity = torch.full_like(linear_gap, torch.inf)
                minimum_linear = torch.where(support, linear_gap, infinity).amin(dim=-1)
                no_support = ~support.any(dim=-1)
                minimum_linear = torch.where(
                    no_support, geometric_gap.amin(dim=-1), minimum_linear
                )
                maximum_error = torch.where(
                    support,
                    (target_gap - linear_gap).abs(),
                    torch.zeros_like(linear_gap),
                ).amax(dim=-1)
                pose_motion = torch.sqrt(
                    z[:, :3].square().sum(dim=-1)
                    + z[:, 3:6].square().sum(dim=-1)
                )
                tensors = {
                    "pose_motion": pose_motion,
                    "minimum_linear_gap_m": minimum_linear,
                    "minimum_target_gap_m": target_gap.amin(dim=-1),
                    "maximum_linearization_error_m": maximum_error,
                }
                for name, value in tensors.items():
                    rows[name].append(value.cpu().numpy())
    finally:
        dataset.close()

    arrays = {name: np.concatenate(parts) for name, parts in rows.items()}
    subsets = {
        "all": np.ones(len(arrays["pose_motion"]), dtype=bool),
        "smooth": arrays["pose_motion"] <= 0.05,
        "jump": arrays["pose_motion"] > 0.05,
        "top_1_percent_motion": arrays["pose_motion"]
        >= np.quantile(arrays["pose_motion"], 0.99),
    }
    result: dict[str, object] = {
        "format_version": 1,
        "split": args.split,
        "constraint_gap_m": config.model.resolvent_constraint_gap_m,
        "subsets": {},
    }
    for name, mask in subsets.items():
        linear = arrays["minimum_linear_gap_m"][mask]
        nonlinear = arrays["minimum_target_gap_m"][mask]
        result["subsets"][name] = {  # type: ignore[index]
            "count": int(mask.sum()),
            "linear_feasible_fraction_at_zero": float(np.mean(linear >= 0.0)),
            "target_feasible_fraction_at_zero": float(np.mean(nonlinear >= 0.0)),
            "target_feasible_fraction_at_admissible_gap": float(
                np.mean(nonlinear >= config.loss.admissible_gap_m)
            ),
            "minimum_linear_gap_m": _summary(linear),
            "minimum_target_gap_m": _summary(nonlinear),
            "maximum_linearization_error_m": _summary(
                arrays["maximum_linearization_error_m"][mask]
            ),
        }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
