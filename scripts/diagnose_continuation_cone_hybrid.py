#!/usr/bin/env python3
"""Combine a frozen cone model's joints with a causal pose continuation prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from srno.data.dataset import H5ObjectDataset, LocalTransitionBatch, make_dataloader
from srno.data.index import ActiveIndex
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import rotation_geodesic_angle, so3_exp, so3_log_vector
from srno.training.checkpoint import load_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.types import PoseState


def _evaluate(
    config: ExperimentConfig,
    checkpoint: Path,
    split: str,
    factors: tuple[float, ...],
) -> dict[str, dict[str, float]]:
    device = torch.device(config.device)
    manifest = DatasetManifest.load(config.paths.manifest)
    active_index = ActiveIndex.load(config.paths.active_index)
    model = _build_model(config, manifest, device)
    load_checkpoint(checkpoint, model=model, map_location=device)
    model.eval()
    dataset = H5ObjectDataset(
        manifest,
        split=split,  # type: ignore[arg-type]
        active_index=active_index,
        active_only=True,
    )
    rows = {
        factor: {name: [] for name in ("dx", "translation_m", "rotation_rad", "joints")}
        for factor in factors
    }
    loader = make_dataloader(
        dataset,
        mode="local",
        objects_per_batch=1,
        samples_per_object=0,
        workers=0,
        seed=config.seed + 1701,
        shuffle=False,
    )
    try:
        with torch.no_grad():
            for batch in loader:
                assert isinstance(batch, LocalTransitionBatch)
                assert batch.previous is not None
                batch = batch.to(device)
                prediction = model.forward_step(
                    batch.current,
                    batch.next_command,
                    batch.sdf,
                    previous_state=batch.previous,
                )
                assert isinstance(prediction, PoseState)
                incoming_rotation = (
                    batch.current.rotation
                    @ batch.previous.rotation.transpose(-1, -2)
                )
                incoming_vector = so3_log_vector(incoming_rotation)
                incoming_translation = batch.current.position - batch.previous.position
                joints = torch.sqrt(
                    (((prediction.joint_position - batch.target.joint_position) / model.joint_travel_range) ** 2)
                    .mean(dim=-1)
                )
                for factor in factors:
                    position = batch.current.position + factor * incoming_translation
                    rotation = so3_exp(factor * incoming_vector) @ batch.current.rotation
                    translation_m = torch.linalg.vector_norm(
                        position - batch.target.position, dim=-1
                    )
                    rotation_error = rotation_geodesic_angle(
                        rotation, batch.target.rotation
                    )
                    dx = torch.sqrt(
                        (translation_m / model.length_scale).square()
                        + rotation_error.square()
                        + joints.square()
                    )
                    for name, value in (
                        ("dx", dx),
                        ("translation_m", translation_m),
                        ("rotation_rad", rotation_error),
                        ("joints", joints),
                    ):
                        rows[factor][name].append(value.cpu().numpy())
    finally:
        dataset.close()
    return {
        f"{factor:g}": {
            name: float(np.concatenate(chunks).mean())
            for name, chunks in values.items()
        }
        for factor, values in rows.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--factors", type=float, nargs="+", default=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    factors = tuple(float(value) for value in args.factors)
    config = ExperimentConfig.load(args.config)
    validation = _evaluate(config, args.checkpoint, "val", factors)
    selected = min(validation, key=lambda key: validation[key]["dx"])
    # Test only the validation-selected factor and the identity-pose control.
    test_factors = tuple(dict.fromkeys((0.0, float(selected))))
    test = _evaluate(config, args.checkpoint, "test", test_factors)
    output = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "selection_metric": "validation mean d_X",
        "validation": validation,
        "selected_factor": float(selected),
        "test": test,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
