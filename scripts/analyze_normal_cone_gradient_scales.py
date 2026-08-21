#!/usr/bin/env python3
"""Measure per-objective gradient scales in a trained local model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from srno.data.dataset import H5ObjectDataset, LocalTransitionBatch, make_dataloader
from srno.data.index import ActiveIndex
from srno.data.schema import DatasetManifest
from srno.losses import combined_loss
from srno.training.checkpoint import load_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.types import PoseState


def _gradient_vector(
    value: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        value,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return torch.cat(
        [
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.reshape(-1)
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    manifest = DatasetManifest.load(config.paths.manifest)
    active_index = ActiveIndex.load(config.paths.active_index)
    device = torch.device(config.device)
    model = _build_model(config, manifest, device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    dataset = H5ObjectDataset(
        manifest,
        split="train",
        active_index=active_index,
        active_only=True,
    )
    loader = make_dataloader(
        dataset,
        mode="local",
        objects_per_batch=config.loader.objects_per_batch,
        samples_per_object=min(config.loader.local_samples_per_object, 64),
        workers=0,
        seed=config.seed + 911,
        shuffle=True,
    )
    rows: list[dict[str, float]] = []
    try:
        for batch_index, raw_batch in enumerate(loader):
            if batch_index >= args.batches:
                break
            assert isinstance(raw_batch, LocalTransitionBatch)
            batch = raw_batch.to(device)
            prediction = model.forward_step(
                batch.current,
                batch.next_command,
                batch.sdf,
                previous_state=batch.previous,
            )
            assert isinstance(prediction, PoseState)
            gap = model.query_geometric_gap(prediction, batch.sdf)
            terms = combined_loss(
                prediction,
                batch.target,
                gap,
                length_scale=model.length_scale,
                joint_scale=model.joint_travel_range,
                sdf_scale=model.sdf_scale,
                admissible_gap=config.loss.admissible_gap_m,
            )
            translation_gradient = _gradient_vector(
                terms.translation,
                parameters,
                retain_graph=True,
            )
            rotation_gradient = _gradient_vector(
                terms.rotation,
                parameters,
                retain_graph=True,
            )
            pose_gradient = translation_gradient + rotation_gradient
            joint_gradient = _gradient_vector(
                terms.joints,
                parameters,
                retain_graph=True,
            )
            feasibility_gradient = _gradient_vector(
                terms.feasibility,
                parameters,
                retain_graph=False,
            )
            pose_norm = torch.linalg.vector_norm(pose_gradient)
            translation_norm = torch.linalg.vector_norm(translation_gradient)
            rotation_norm = torch.linalg.vector_norm(rotation_gradient)
            joint_norm = torch.linalg.vector_norm(joint_gradient)
            feasibility_norm = torch.linalg.vector_norm(feasibility_gradient)
            cosine = torch.dot(pose_gradient, joint_gradient) / (
                pose_norm * joint_norm
            ).clamp_min(1e-20)
            rows.append(
                {
                    "pose_norm": float(pose_norm),
                    "translation_norm": float(translation_norm),
                    "rotation_norm": float(rotation_norm),
                    "translation_over_rotation": float(
                        translation_norm / rotation_norm.clamp_min(1e-20)
                    ),
                    "joint_norm": float(joint_norm),
                    "feasibility_norm": float(feasibility_norm),
                    "joint_over_pose": float(joint_norm / pose_norm.clamp_min(1e-20)),
                    "feasibility_over_pose": float(
                        feasibility_norm / pose_norm.clamp_min(1e-20)
                    ),
                    "pose_joint_cosine": float(cosine),
                }
            )
    finally:
        dataset.close()

    output = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "batches": len(rows),
        "summary": {
            name: {
                "mean": float(np.mean([row[name] for row in rows])),
                "median": float(np.median([row[name] for row in rows])),
                "min": float(np.min([row[name] for row in rows])),
                "max": float(np.max([row[name] for row in rows])),
            }
            for name in rows[0]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
