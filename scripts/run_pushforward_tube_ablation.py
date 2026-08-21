#!/usr/bin/env python3
"""Paired nominal-control versus PhysX pushforward-tube fine-tuning.

Both arms start from the same rollout checkpoint, see the same number of
examples and take the same optimizer steps.  The control's second half-batch
contains independently sampled nominal transitions.  The treatment replaces
that half by fresh-PhysX successors of states induced by the checkpoint's own
rollout.  Validation and test remain the untouched canonical splits.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Iterable, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from srno.data.dataset import (
    H5ObjectDataset,
    LocalTransitionBatch,
    ObjectBatchCollator,
    TrajectoryBatch,
    make_dataloader,
)
from srno.data.index import ActiveIndex, file_sha256
from srno.data.schema import DatasetManifest
from srno.geometry.gripper import GripperAsset
from srno.geometry.se3 import (
    quaternion_xyzw_to_matrix,
    rotation_geodesic_angle,
    so3_exp,
    so3_log_vector,
)
from srno.training.checkpoint import load_checkpoint, save_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import (
    _autocast,
    _build_model,
    _local_iteration,
    _optimizer,
    _run_epoch,
    _state_at,
)
from srno.training.metrics import MetricAccumulator, accumulate_trajectory_metrics
from srno.types import PoseState, SDFBatch


Arm = Literal["nominal_control", "pushforward_tube"]
ARMS: tuple[Arm, ...] = ("nominal_control", "pushforward_tube")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _mean(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        raise ValueError("no metric records")
    return {
        name: float(np.mean([record[name] for record in records]))
        for name in records[0]
    }


def _trajectory_metrics(
    model: torch.nn.Module,
    loader: Iterable[TrajectoryBatch],
    manifest: DatasetManifest,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    accumulator = MetricAccumulator()
    with torch.no_grad():
        for raw in loader:
            batch = raw.to(device)
            with _autocast(config, device):
                prediction = model.rollout(
                    _state_at(batch.states, 0), batch.command_schedule[1:], batch.sdf
                )
                gaps = torch.stack(
                    [
                        model.query_geometric_gap(_state_at(prediction, step), batch.sdf)
                        for step in range(33)
                    ],
                    dim=1,
                )
            accumulate_trajectory_metrics(
                accumulator,
                prediction,
                batch.states,
                batch.command_schedule,
                model.aperture_from_joints(prediction.joint_position),
                batch.actual_aperture,
                gaps,
                length_scale=model.length_scale,
                joint_scale=model.joint_travel_range,
                lag_threshold=manifest.delta_gate_m,
            )
    return accumulator.compute()


class TubeLabels:
    def __init__(
        self,
        path: Path,
        manifest: DatasetManifest,
        train_dataset: H5ObjectDataset,
        *,
        max_pushforward_dx: float | None = None,
        max_oracle_step_dx: float | None = None,
        cap_oracle_step_dx: float | None = None,
    ) -> None:
        labels = np.load(path, allow_pickle=False)
        if str(labels["manifest_sha256"]) != manifest.sha256():
            raise ValueError("tube labels and manifest hashes differ")
        if str(labels["split"]) != "train":
            raise ValueError("tube labels must be train-only")
        selected = np.asarray(labels["settled"], dtype=bool)
        self.source_sample_count = int(len(selected))
        if max_pushforward_dx is not None:
            selected &= np.asarray(labels["pushforward_dx"]) <= max_pushforward_dx
        oracle_step_dx = self._oracle_step_dx(labels, manifest)
        if max_oracle_step_dx is not None:
            selected &= oracle_step_dx <= max_oracle_step_dx
        if not selected.any():
            raise ValueError("tube labels contain no settled successors")
        self.arrays = {
            name: np.asarray(labels[name])[selected]
            for name in labels.files
            if np.asarray(labels[name]).ndim > 0
            and len(np.asarray(labels[name])) == len(selected)
        }
        self.arrays["oracle_step_dx"] = oracle_step_dx[selected]
        gripper = GripperAsset.load(manifest.gripper_path)
        current_rotation = quaternion_xyzw_to_matrix(torch.from_numpy(
            self.arrays["model_quaternion_xyzw"].astype(np.float32)
        ))
        successor_rotation = quaternion_xyzw_to_matrix(torch.from_numpy(
            self.arrays["successor_quaternion_xyzw"].astype(np.float32)
        ))
        factor = np.ones(len(self.arrays["oracle_step_dx"]), dtype=np.float32)
        if cap_oracle_step_dx is not None:
            factor = np.minimum(
                factor,
                cap_oracle_step_dx / np.maximum(self.arrays["oracle_step_dx"], 1e-8),
            )
        factor_tensor = torch.from_numpy(factor)
        relative_rotation = successor_rotation @ current_rotation.transpose(-1, -2)
        self.target_rotation = (
            so3_exp(so3_log_vector(relative_rotation) * factor_tensor[:, None])
            @ current_rotation
        )
        self.target_position = (
            self.arrays["model_position"].astype(np.float32)
            + factor[:, None]
            * (
                self.arrays["successor_position"].astype(np.float32)
                - self.arrays["model_position"].astype(np.float32)
            )
        )
        self.target_joint = (
            self.arrays["model_joint"].astype(np.float32)
            + factor[:, None]
            * (
                self.arrays["successor_joint"].astype(np.float32)
                - self.arrays["model_joint"].astype(np.float32)
            )
        )
        self.target_aperture = gripper.aperture_from_joints(
            torch.from_numpy(self.target_joint)
        ).numpy()
        self.cap_oracle_step_dx = cap_oracle_step_dx
        self.object_ids = self.arrays["object_id"].astype(str)
        unknown = set(self.object_ids) - set(manifest.splits["train"])
        if unknown:
            raise ValueError(f"tube labels contain non-train objects: {sorted(unknown)}")
        self.object_to_dataset = {
            object_id: index for index, object_id in enumerate(train_dataset.object_ids)
        }
        self.unique_objects = tuple(
            object_id
            for object_id in manifest.splits["train"]
            if np.any(self.object_ids == object_id)
        )
        self.indices = {
            object_id: np.flatnonzero(self.object_ids == object_id)
            for object_id in self.unique_objects
        }
        self.manifest = manifest
        self.dataset = train_dataset

    @staticmethod
    def _oracle_step_dx(labels: Any, manifest: DatasetManifest) -> np.ndarray:
        gripper = GripperAsset.load(manifest.gripper_path)
        current_rotation = quaternion_xyzw_to_matrix(torch.from_numpy(
            np.asarray(labels["model_quaternion_xyzw"], dtype=np.float32)
        ))
        successor_rotation = quaternion_xyzw_to_matrix(torch.from_numpy(
            np.asarray(labels["successor_quaternion_xyzw"], dtype=np.float32)
        ))
        translation = torch.linalg.vector_norm(
            torch.from_numpy(np.asarray(labels["successor_position"], dtype=np.float32))
            - torch.from_numpy(np.asarray(labels["model_position"], dtype=np.float32)),
            dim=-1,
        ) / manifest.length_scale_m
        rotation = rotation_geodesic_angle(current_rotation, successor_rotation)
        joint_scale = gripper.joint_travel_range.float().clamp_min(1e-8)
        joints = torch.sqrt(
            ((
                torch.from_numpy(np.asarray(labels["successor_joint"], dtype=np.float32))
                - torch.from_numpy(np.asarray(labels["model_joint"], dtype=np.float32))
            ) / joint_scale).square().mean(dim=-1)
        )
        return torch.sqrt(
            translation.square() + rotation.square() + joints.square()
        ).numpy()

    def object_batches(
        self,
        *,
        objects_per_batch: int,
        samples_per_object: int,
        seed: int,
    ) -> list[LocalTransitionBatch]:
        generator = np.random.default_rng(seed)
        object_ids = list(self.unique_objects)
        generator.shuffle(object_ids)
        batches: list[LocalTransitionBatch] = []
        schedule = torch.tensor(self.manifest.commanded_aperture_m, dtype=torch.float32)
        for start in range(0, len(object_ids), objects_per_batch):
            selected_objects = object_ids[start : start + objects_per_batch]
            records = [
                self.dataset[self.object_to_dataset[object_id]]
                for object_id in selected_objects
            ]
            chosen: list[np.ndarray] = []
            mapping: list[int] = []
            for object_index, object_id in enumerate(selected_objects):
                candidates = self.indices[object_id]
                values = generator.choice(
                    candidates,
                    size=samples_per_object,
                    replace=len(candidates) < samples_per_object,
                )
                chosen.append(values)
                mapping.extend([object_index] * len(values))
            indices = np.concatenate(chosen)
            steps = torch.from_numpy(self.arrays["current_step"][indices]).long()
            batches.append(
                LocalTransitionBatch(
                    SDFBatch(
                        torch.stack([record.sdf for record in records]),
                        torch.stack([record.origin for record in records]),
                        torch.stack([record.voxel_size for record in records]),
                        torch.tensor(mapping, dtype=torch.long),
                        self.manifest.sdf_scale_m,
                    ),
                    PoseState(
                        quaternion_xyzw_to_matrix(torch.from_numpy(
                            self.arrays["model_quaternion_xyzw"][indices].astype(np.float32)
                        )),
                        torch.from_numpy(self.arrays["model_position"][indices].astype(np.float32)),
                        torch.from_numpy(self.arrays["model_joint"][indices].astype(np.float32)),
                    ),
                    PoseState(
                        self.target_rotation.index_select(0, torch.from_numpy(indices)),
                        torch.from_numpy(self.target_position[indices]),
                        torch.from_numpy(self.target_joint[indices]),
                    ),
                    schedule.index_select(0, steps + 1),
                    torch.from_numpy(self.target_aperture[indices].astype(np.float32)),
                    tuple(selected_objects),
                    torch.from_numpy(self.arrays["trajectory"][indices]).long(),
                    steps,
                    None,
                )
            )
        return batches


def _stochastic_nominal_loader(
    dataset: H5ObjectDataset,
    config: ExperimentConfig,
    *,
    samples_per_object: int,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.loader.objects_per_batch,
        shuffle=True,
        num_workers=0,
        collate_fn=ObjectBatchCollator(
            dataset.manifest,
            mode="local",
            samples_per_object=samples_per_object,
            seed=seed,
            resample=True,
        ),
        generator=torch.Generator().manual_seed(seed),
    )


def _paired_epoch(
    model: torch.nn.Module,
    arm: Arm,
    nominal_a: Iterable[LocalTransitionBatch],
    nominal_b: Iterable[LocalTransitionBatch],
    tube_batches: list[LocalTransitionBatch],
    config: ExperimentConfig,
    tube_config: ExperimentConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    tube_weight: float,
) -> dict[str, float]:
    model.train(True)
    records: list[dict[str, float]] = []
    second = nominal_b if arm == "nominal_control" else tube_batches
    for raw_nominal, raw_second in zip(nominal_a, second, strict=True):
        nominal = raw_nominal.to(device)
        comparison = raw_second.to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(config, device):
            nominal_loss, nominal_metrics = _local_iteration(model, nominal, config)
            comparison_config = config if arm == "nominal_control" else tube_config
            second_loss, second_metrics = _local_iteration(
                model, comparison, comparison_config
            )
            loss = (1.0 - tube_weight) * nominal_loss + tube_weight * second_loss
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.optimizer.gradient_clip
        )
        optimizer.step()
        scheduler.step()
        records.append(
            {
                "loss": float(loss.detach()),
                "nominal_dx": nominal_metrics["dx"],
                "second_dx": second_metrics["dx"],
                "gradient_norm": float(gradient.detach()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
    return _mean(records)


def _tube_metrics(
    model: torch.nn.Module,
    batches: Iterable[LocalTransitionBatch],
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    records: list[dict[str, float]] = []
    with torch.no_grad():
        for raw in batches:
            _, metrics = _local_iteration(model, raw.to(device), config)
            records.append(metrics)
    return _mean(records)


def _train_arm(
    arm: Arm,
    config: ExperimentConfig,
    manifest: DatasetManifest,
    train_dataset: H5ObjectDataset,
    tube: TubeLabels,
    val_rollout_loader: Iterable[TrajectoryBatch],
    val_local_loader: Iterable[LocalTransitionBatch],
    initial: Path,
    output: Path,
    device: torch.device,
    *,
    epochs: int,
    patience: int,
    samples_per_object: int,
    tube_weight: float,
    tube_config: ExperimentConfig,
) -> dict[str, Any]:
    _seed_everything(config.seed)
    model = _build_model(config, manifest, device)
    initial_state = load_checkpoint(initial, model=model, map_location=device)
    if initial_state["manifest_sha256"] != manifest.sha256():
        raise ValueError("initial checkpoint manifest mismatch")
    batches_per_epoch = math.ceil(
        len(tube.unique_objects) / config.loader.objects_per_batch
    )
    optimizer, scheduler = _optimizer(model, config, batches_per_epoch * epochs)
    arm_output = output / arm
    arm_output.mkdir(parents=True, exist_ok=True)
    checkpoint = arm_output / "best-pushforward-tube.pt"
    history: list[dict[str, Any]] = []
    best = float("inf")
    stale = 0
    for epoch in range(epochs):
        started = perf_counter()
        nominal_a = _stochastic_nominal_loader(
            train_dataset, config,
            samples_per_object=samples_per_object,
            seed=config.seed + epoch * 2,
        )
        nominal_b = _stochastic_nominal_loader(
            train_dataset, config,
            samples_per_object=samples_per_object,
            seed=config.seed + epoch * 2 + 1,
        )
        tube_batches = tube.object_batches(
            objects_per_batch=config.loader.objects_per_batch,
            samples_per_object=samples_per_object,
            seed=config.seed + epoch,
        )
        train_metrics = _paired_epoch(
            model, arm, nominal_a, nominal_b, tube_batches,
            config, tube_config, device, optimizer, scheduler, tube_weight,
        )
        val_rollout = _trajectory_metrics(
            model, val_rollout_loader, manifest, config, device
        )
        selection = val_rollout["terminal_dx"]
        tube_eval = _tube_metrics(model, tube_batches, tube_config, device)
        record = {
            "epoch": epoch,
            "seconds": perf_counter() - started,
            "train": train_metrics,
            "tube": tube_eval,
            "val": val_rollout,
            "selection": selection,
        }
        history.append(record)
        with (arm_output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if selection < best:
            best = selection
            stale = 0
            save_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config.to_dict(),
                manifest_sha256=manifest.sha256(),
                gripper_sha256=manifest.gripper_sha256,
                stage=f"pushforward_tube_{arm}",
                epoch=epoch,
                horizon=32,
                horizon_epoch=epoch,
                stale_epochs=0,
                best_metric=best,
                extra_state={
                    "arm": arm,
                    "tube_definition": "fresh PhysX successors under checkpoint-induced state measure",
                },
            )
        else:
            stale += 1
        print(
            f"[TUBE train] arm={arm} epoch={epoch:03d} "
            f"val_H32={selection:.6f} tube_dX={tube_eval['dx']:.6f} "
            f"best={best:.6f} stale={stale}",
            flush=True,
        )
        if stale >= patience:
            break
    load_checkpoint(checkpoint, model=model, map_location=device)
    val_local = _run_epoch(
        model, val_local_loader, config, device,
        mode="local", horizon=1, optimizer=None, scheduler=None,
    )
    return {
        "model": model,
        "checkpoint": checkpoint,
        "history": history,
        "val_rollout": _trajectory_metrics(
            model, val_rollout_loader, manifest, config, device
        ),
        "val_local": val_local,
        "tube": _tube_metrics(
            model,
            tube.object_batches(
                objects_per_batch=config.loader.objects_per_batch,
                samples_per_object=samples_per_object,
                seed=config.seed + 100_000,
            ),
            tube_config,
            device,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--samples-per-object", type=int, default=8)
    parser.add_argument("--tube-weight", type=float, default=0.5)
    parser.add_argument("--max-pushforward-dx", type=float)
    parser.add_argument("--max-oracle-step-dx", type=float)
    parser.add_argument("--cap-oracle-step-dx", type=float)
    parser.add_argument(
        "--tube-pose-penalty", choices=("squared", "huber"), default="squared"
    )
    parser.add_argument("--tube-pose-huber-delta", type=float, default=0.1)
    args = parser.parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.samples_per_object <= 0:
        parser.error("training counts must be positive")
    if not 0.0 < args.tube_weight < 1.0:
        parser.error("--tube-weight must lie strictly between zero and one")
    for name in ("max_pushforward_dx", "max_oracle_step_dx", "cap_oracle_step_dx"):
        value = getattr(args, name)
        if value is not None and value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.tube_pose_huber_delta <= 0.0:
        parser.error("--tube-pose-huber-delta must be positive")
    base = ExperimentConfig.load(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = replace(
        base,
        device=args.device,
        paths=replace(base.paths, output_dir=output),
        loader=replace(base.loader, workers=0),
    )
    tube_config = replace(
        config,
        loss=replace(
            config.loss,
            pose_penalty=args.tube_pose_penalty,
            pose_huber_delta=args.tube_pose_huber_delta,
        ),
    )
    device = torch.device(config.device)
    manifest = DatasetManifest.load(config.paths.manifest)
    active = ActiveIndex.load(config.paths.active_index)
    train_dataset = H5ObjectDataset(
        manifest, split="train", active_index=active, active_only=True
    )
    val_local_dataset = H5ObjectDataset(
        manifest, split="val", active_index=active, active_only=True
    )
    val_dataset = H5ObjectDataset(manifest, split="val")
    test_dataset = H5ObjectDataset(manifest, split="test")
    try:
        tube = TubeLabels(
            args.labels.resolve(), manifest, train_dataset,
            max_pushforward_dx=args.max_pushforward_dx,
            max_oracle_step_dx=args.max_oracle_step_dx,
            cap_oracle_step_dx=args.cap_oracle_step_dx,
        )
        val_rollout_loader = make_dataloader(
            val_dataset, mode="rollout", objects_per_batch=1,
            samples_per_object=0, workers=0, seed=config.seed + 1, shuffle=False,
        )
        test_loader = make_dataloader(
            test_dataset, mode="rollout", objects_per_batch=1,
            samples_per_object=0, workers=0, seed=config.seed + 2, shuffle=False,
        )
        val_local_loader = make_dataloader(
            val_local_dataset, mode="local", objects_per_batch=1,
            samples_per_object=0, workers=0, seed=config.seed + 1, shuffle=False,
        )
        initial_model = _build_model(config, manifest, device)
        load_checkpoint(args.initial, model=initial_model, map_location=device)
        baseline = {
            "val": _trajectory_metrics(
                initial_model, val_rollout_loader, manifest, config, device
            ),
            "test": _trajectory_metrics(
                initial_model, test_loader, manifest, config, device
            ),
            "local_val": _run_epoch(
                initial_model, val_local_loader, config, device,
                mode="local", horizon=1, optimizer=None, scheduler=None,
            ),
        }
        del initial_model
        runs: dict[str, Any] = {}
        for arm in ARMS:
            trained = _train_arm(
                arm, config, manifest, train_dataset, tube,
                val_rollout_loader, val_local_loader,
                args.initial.resolve(), output, device,
                epochs=args.epochs, patience=args.patience,
                samples_per_object=args.samples_per_object,
                tube_weight=args.tube_weight,
                tube_config=tube_config,
            )
            model = trained.pop("model")
            checkpoint = trained["checkpoint"]
            trained["checkpoint"] = str(checkpoint)
            trained["checkpoint_sha256"] = file_sha256(checkpoint)
            trained["test"] = _trajectory_metrics(
                model, test_loader, manifest, config, device
            )
            runs[arm] = trained
            del model
        control = runs["nominal_control"]["test"]["terminal_dx"]
        treatment = runs["pushforward_tube"]["test"]["terminal_dx"]
        result = {
            "definition": {
                "risk": "E_{z~mu_theta} d(R_theta(z,u), R_PhysX(z,u))^2",
                "nominal_control": "two independent nominal half-batches",
                "pushforward_tube": "one nominal anchor plus one fresh-PhysX pushforward half-batch",
                "same_initial_checkpoint": True,
                "same_optimizer_steps": True,
                "architecture_changed": False,
                "canonical_val_test_changed": False,
            },
            "contract": {
                "config": str(args.config.resolve()),
                "manifest_sha256": manifest.sha256(),
                "initial": str(args.initial.resolve()),
                "initial_sha256": file_sha256(args.initial),
                "labels": str(args.labels.resolve()),
                "labels_sha256": _sha256(args.labels),
                "device": str(device),
                "epochs": args.epochs,
                "patience": args.patience,
                "samples_per_object": args.samples_per_object,
                "tube_weight": args.tube_weight,
                "max_pushforward_dx": args.max_pushforward_dx,
                "max_oracle_step_dx": args.max_oracle_step_dx,
                "cap_oracle_step_dx": args.cap_oracle_step_dx,
                "tube_pose_penalty": args.tube_pose_penalty,
                "tube_pose_huber_delta": args.tube_pose_huber_delta,
                "source_tube_samples": tube.source_sample_count,
                "settled_tube_samples": int(len(tube.object_ids)),
            },
            "baseline": baseline,
            "runs": runs,
            "paired_test": {
                "control_terminal_dx": control,
                "pushforward_tube_terminal_dx": treatment,
                "relative_change": float((treatment - control) / control),
            },
        }
        _write_json(output / "results.json", result)
        print(json.dumps(result["paired_test"], indent=2, sort_keys=True), flush=True)
    finally:
        train_dataset.close()
        val_local_dataset.close()
        val_dataset.close()
        test_dataset.close()


if __name__ == "__main__":
    main()
