#!/usr/bin/env python3
"""Read-only diagnostics from item 16 of the SRNO stability review."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from srno.data.dataset import H5ObjectDataset, ObjectRecord
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import (
    quaternion_xyzw_to_matrix,
    rotation_geodesic_angle,
    so3_exp,
)
from srno.losses import feasibility_loss, state_error
from srno.training.checkpoint import load_checkpoint
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.types import PoseState, SDFBatch


SPLITS = ("train", "val", "test")
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


@dataclass
class Bundle:
    split: str
    object_id: str
    record: ObjectRecord
    states: PoseState


@dataclass
class Group:
    bundle_index: int
    step: int
    active: np.ndarray
    near_contact: np.ndarray
    stall: np.ndarray
    sliding: np.ndarray


def _stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    result: dict[str, float | int] = {
        "count": int(values.size),
        "finite_count": int(finite.size),
    }
    if finite.size:
        result.update(
            {
                "mean": float(finite.mean()),
                "std": float(finite.std()),
                **{
                    f"q{int(q * 100):02d}": float(np.quantile(finite, q))
                    for q in QUANTILES
                },
            }
        )
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _state_at(states: PoseState, step: int, indices: Tensor | None = None) -> PoseState:
    state = PoseState(states.rotation[:, step], states.position[:, step], states.aperture[:, step])
    return state if indices is None else state.index_select(0, indices)


def _sdf_for(record: ObjectRecord, count: int, device: torch.device) -> SDFBatch:
    return SDFBatch(
        record.sdf.unsqueeze(0).to(device),
        record.origin.unsqueeze(0).to(device),
        record.voxel_size.unsqueeze(0).to(device),
        torch.zeros(count, dtype=torch.long, device=device),
        0.02,
    )


def _rotation_vector(relative: Tensor) -> Tensor:
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1)
    skew_vector = 0.5 * torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sine = torch.linalg.vector_norm(skew_vector, dim=-1)
    angle = torch.atan2(sine, cosine)
    scale = torch.where(sine > 1e-7, angle / sine, torch.ones_like(sine))
    return skew_vector * scale.unsqueeze(-1)


def _correction_vector(current: PoseState, target: PoseState, command: Tensor, length: float) -> Tensor:
    relative = target.rotation @ current.rotation.transpose(-1, -2)
    return torch.cat(
        (
            (target.position - current.position) / length,
            _rotation_vector(relative),
            ((target.aperture - command) / length).unsqueeze(-1),
        ),
        dim=-1,
    )


def _gradient_geometry(gradients: tuple[Tensor | None, ...]) -> tuple[Tensor, Tensor]:
    norm2 = None
    for gradient in gradients:
        if gradient is None:
            continue
        value = gradient.float().square().sum()
        norm2 = value if norm2 is None else norm2 + value
    if norm2 is None:
        zero = torch.zeros((), device="cuda" if torch.cuda.is_available() else "cpu")
        return zero, zero
    return norm2.sqrt(), norm2


def _gradient_pair(
    left: tuple[Tensor | None, ...], right: tuple[Tensor | None, ...]
) -> tuple[float, float, float, float]:
    left_norm, _ = _gradient_geometry(left)
    right_norm, _ = _gradient_geometry(right)
    dot = left_norm.new_zeros(())
    for first, second in zip(left, right):
        if first is not None and second is not None:
            dot = dot + (first.float() * second.float()).sum()
    denominator = left_norm * right_norm
    cosine = dot / denominator if float(denominator) > 1e-20 else dot.new_tensor(float("nan"))
    ratio = right_norm / left_norm if float(left_norm) > 1e-12 else left_norm.new_tensor(float("nan"))
    return float(cosine), float(ratio), float(left_norm), float(right_norm)


def _load_bundles(manifest: DatasetManifest) -> list[Bundle]:
    bundles: list[Bundle] = []
    for split in SPLITS:
        dataset = H5ObjectDataset(manifest, split=split)
        try:
            for index in range(len(dataset)):
                record = dataset[index]
                states = PoseState(
                    quaternion_xyzw_to_matrix(record.quaternion_xyzw.float()),
                    record.position.float(),
                    record.aperture.float(),
                )
                bundles.append(Bundle(split, record.object_id, record, states))
        finally:
            dataset.close()
    return bundles


def _main_pass(model, manifest: DatasetManifest, bundles: list[Bundle], device: torch.device):
    gt_gap = {split: [] for split in SPLITS}
    gt_feasibility = {split: [] for split in SPLITS}
    teacher = {split: [] for split in SPLITS}
    pushforward = {split: [] for split in SPLITS}
    teacher_contact = {split: [] for split in SPLITS}
    pushforward_contact = {split: [] for split in SPLITS}
    teacher_by_step = {split: [[] for _ in range(31)] for split in SPLITS}
    pushforward_by_step = {split: [[] for _ in range(31)] for split in SPLITS}
    rollout_terminal = {split: [] for split in SPLITS}
    rollout_by_step = {split: [[] for _ in range(33)] for split in SPLITS}
    object_terminal: list[dict[str, Any]] = []
    suff_input: list[np.ndarray] = []
    suff_output: list[np.ndarray] = []
    suff_contact: list[np.ndarray] = []
    suff_refs: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    groups: list[Group] = []

    schedule = torch.tensor(manifest.commanded_aperture_m, dtype=torch.float32, device=device)
    with torch.no_grad():
        for bundle_index, bundle in enumerate(bundles):
            states = bundle.states.to(device)
            count = states.shape[0]
            sdf = _sdf_for(bundle.record, count, device)
            split = bundle.split

            rollout = model.rollout(_state_at(states, 0), schedule[1:], sdf)
            object_errors = []
            for step in range(33):
                error = state_error(
                    _state_at(rollout, step),
                    _state_at(states, step),
                    length_scale=model.length_scale,
                )[0].sqrt()
                values = error.cpu().numpy()
                rollout_by_step[split][step].append(values)
                if step == 32:
                    rollout_terminal[split].append(values)
                    object_errors = values
            object_terminal.append(
                {
                    "split": split,
                    "object_id": bundle.object_id,
                    "mean_terminal_dx": float(np.mean(object_errors)),
                    "median_terminal_dx": float(np.median(object_errors)),
                    "p95_terminal_dx": float(np.quantile(object_errors, 0.95)),
                }
            )

            for step in range(33):
                actual = _state_at(states, step)
                gap = model.query_gap(actual, sdf)
                gt_gap[split].append(gap.amin(dim=-1).cpu().numpy())
                per_state = torch.relu(-gap / model.sdf_scale).square().mean(dim=-1)
                gt_feasibility[split].append(per_state.cpu().numpy())

            previous_one_step = None
            for step in range(32):
                current = _state_at(states, step)
                target = _state_at(states, step + 1)
                command = schedule[step + 1]
                prediction, aux = model.forward_step(current, command, sdf, return_aux=True)

                if step > 0:
                    assert previous_one_step is not None
                    pushed = model.forward_step(previous_one_step, command, sdf)
                    teacher_error = state_error(
                        prediction, target, length_scale=model.length_scale
                    )[0].sqrt()
                    pushed_error = state_error(
                        pushed, target, length_scale=model.length_scale
                    )[0].sqrt()
                    contact = bundle.record.diagnostics["contact_count"][:, step].numpy() > 0
                    teacher_values = teacher_error.cpu().numpy()
                    pushed_values = pushed_error.cpu().numpy()
                    teacher[split].append(teacher_values)
                    pushforward[split].append(pushed_values)
                    teacher_by_step[split][step - 1].append(teacher_values)
                    pushforward_by_step[split][step - 1].append(pushed_values)
                    teacher_contact[split].append(teacher_values[contact])
                    pushforward_contact[split].append(pushed_values[contact])
                previous_one_step = prediction

                contact = bundle.record.diagnostics["contact_count"][:, step].to(device) > 0
                lag = target.aperture - command
                translation = torch.linalg.vector_norm(target.position - current.position, dim=-1)
                rotation = rotation_geodesic_angle(target.rotation, current.rotation)
                minimum = aux.trial_gap.amin(dim=-1)
                active = aux.active
                near_contact = active & (minimum >= 0)
                stall = contact & (lag > 0.0005)
                sliding = contact & (
                    (translation > 0.00025) | (rotation > math.radians(0.25))
                )
                groups.append(
                    Group(
                        bundle_index,
                        step,
                        active.cpu().numpy(),
                        near_contact.cpu().numpy(),
                        stall.cpu().numpy(),
                        sliding.cpu().numpy(),
                    )
                )

                features = torch.cat(
                    (
                        aux.trial_gap / model.sdf_scale,
                        (current.aperture / model.length_scale).unsqueeze(-1),
                        (command / model.length_scale).expand(count, 1),
                    ),
                    dim=-1,
                ).float()
                output = _correction_vector(current, target, command, model.length_scale).float()
                distances = torch.cdist(features, features) / math.sqrt(features.shape[-1])
                distances.fill_diagonal_(float("inf"))
                nearest_distance, nearest = distances.min(dim=1)
                output_difference = torch.linalg.vector_norm(
                    output - output.index_select(0, nearest), dim=-1
                )
                suff_input.append(nearest_distance.cpu().numpy())
                suff_output.append(output_difference.cpu().numpy())
                suff_contact.append(contact.cpu().numpy())
                suff_refs.append(
                    (
                        bundle_index,
                        step,
                        np.arange(count, dtype=np.int64),
                        nearest.cpu().numpy(),
                    )
                )

            del sdf, states, rollout

    arrays = {
        "gt_gap": {split: np.concatenate(gt_gap[split]) for split in SPLITS},
        "gt_feasibility": {
            split: np.concatenate(gt_feasibility[split]) for split in SPLITS
        },
        "teacher": {split: np.concatenate(teacher[split]) for split in SPLITS},
        "pushforward": {split: np.concatenate(pushforward[split]) for split in SPLITS},
        "teacher_contact": {
            split: np.concatenate([x for x in teacher_contact[split] if x.size])
            for split in SPLITS
        },
        "pushforward_contact": {
            split: np.concatenate([x for x in pushforward_contact[split] if x.size])
            for split in SPLITS
        },
        "teacher_by_step": {
            split: np.asarray([np.concatenate(values).mean() for values in teacher_by_step[split]])
            for split in SPLITS
        },
        "pushforward_by_step": {
            split: np.asarray(
                [np.concatenate(values).mean() for values in pushforward_by_step[split]]
            )
            for split in SPLITS
        },
        "rollout_terminal": {
            split: np.concatenate(rollout_terminal[split]) for split in SPLITS
        },
        "rollout_by_step": {
            split: np.asarray(
                [np.concatenate(values).mean() for values in rollout_by_step[split]]
            )
            for split in SPLITS
        },
        "suff_input": np.concatenate(suff_input),
        "suff_output": np.concatenate(suff_output),
        "suff_contact": np.concatenate(suff_contact),
    }
    return arrays, groups, object_terminal, suff_refs


def _gradient_diagnostic(
    model,
    manifest: DatasetManifest,
    bundles: list[Bundle],
    groups: list[Group],
    device: torch.device,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    result = {
        category: {"cosine": [], "ratio": [], "state_norm": [], "feasibility_norm": [], "samples": []}
        for category in ("near_contact", "stall", "sliding")
    }
    schedule = torch.tensor(manifest.commanded_aperture_m, dtype=torch.float32, device=device)
    for category in result:
        candidates = [group for group in groups if np.count_nonzero(getattr(group, category)) >= 4]
        rng.shuffle(candidates)
        for group in candidates[:32]:
            indices_np = np.flatnonzero(getattr(group, category))
            if indices_np.size > 64:
                indices_np = rng.choice(indices_np, size=64, replace=False)
            indices = torch.from_numpy(indices_np).long().to(device)
            bundle = bundles[group.bundle_index]
            states = bundle.states.to(device)
            current = _state_at(states, group.step, indices)
            target = _state_at(states, group.step + 1, indices)
            sdf = _sdf_for(bundle.record, len(indices_np), device)
            prediction = model.forward_step(current, schedule[group.step + 1], sdf)
            predicted_gap = model.query_gap(prediction, sdf)
            flow = state_error(prediction, target, length_scale=model.length_scale)[0].mean()
            feasibility = feasibility_loss(predicted_gap, sdf_scale=model.sdf_scale)
            state_grad = torch.autograd.grad(
                flow, parameters, retain_graph=True, allow_unused=True
            )
            feasibility_grad = torch.autograd.grad(
                feasibility, parameters, allow_unused=True
            )
            cosine, ratio, state_norm, feasibility_norm = _gradient_pair(
                state_grad, feasibility_grad
            )
            result[category]["cosine"].append(cosine)
            result[category]["ratio"].append(ratio)
            result[category]["state_norm"].append(state_norm)
            result[category]["feasibility_norm"].append(feasibility_norm)
            result[category]["samples"].append(int(len(indices_np)))
            del sdf, states, prediction, state_grad, feasibility_grad
    return result


def _gt_feasibility_gradient_diagnostic(
    model,
    manifest: DatasetManifest,
    bundles: list[Bundle],
    groups: list[Group],
    device: torch.device,
    seed: int,
) -> dict[str, list[float | str]]:
    """Differentiate GT feasibility with respect to normalized SE(3)+aperture state."""

    rng = np.random.default_rng(seed + 2)
    selected: list[Group] = []
    for split, limit in (("train", 60), ("val", 20), ("test", 20)):
        candidates = [group for group in groups if bundles[group.bundle_index].split == split]
        rng.shuffle(candidates)
        selected.extend(candidates[:limit])
    result: dict[str, list[float | str]] = {"gradient_norm": [], "split": []}
    for group in selected:
        bundle = bundles[group.bundle_index]
        states = bundle.states.to(device)
        actual = _state_at(states, group.step + 1)
        sdf_all = _sdf_for(bundle.record, actual.shape[0], device)
        with torch.no_grad():
            gap = model.query_gap(actual, sdf_all)
            violating = gap.amin(dim=-1) < 0
        indices = torch.nonzero(violating, as_tuple=False).flatten()
        if not len(indices):
            continue
        if len(indices) > 32:
            chosen = rng.choice(indices.cpu().numpy(), size=32, replace=False)
            indices = torch.from_numpy(chosen).long().to(device)
        actual = actual.index_select(0, indices)
        sdf = _sdf_for(bundle.record, len(indices), device)
        delta = torch.zeros(len(indices), 7, device=device, requires_grad=True)
        perturbed = PoseState(
            so3_exp(delta[:, 3:6]) @ actual.rotation,
            actual.position + delta[:, :3] * model.length_scale,
            actual.aperture + delta[:, 6] * model.length_scale,
        )
        per_state = torch.relu(-model.query_gap(perturbed, sdf) / model.sdf_scale).square().mean(dim=-1)
        gradient = torch.autograd.grad(per_state.sum(), delta)[0]
        norms = torch.linalg.vector_norm(gradient, dim=-1).detach().cpu().numpy()
        result["gradient_norm"].extend(map(float, norms))
        result["split"].extend([bundle.split] * len(norms))
    return result


def _perturb_state(
    state: PoseState,
    mode: str,
    rng: np.random.Generator,
    device: torch.device,
) -> PoseState:
    count = state.shape[0]
    translation = torch.zeros(count, 3, device=device)
    rotation_vector = torch.zeros(count, 3, device=device)
    aperture = torch.zeros(count, device=device)

    if mode in ("translation", "combined"):
        values = rng.normal(size=(count, 3))
        values /= np.linalg.norm(values, axis=1, keepdims=True).clip(1e-12)
        translation = torch.from_numpy(values).float().to(device) * 1e-4
    if mode in ("rotation", "combined"):
        values = rng.normal(size=(count, 3))
        values /= np.linalg.norm(values, axis=1, keepdims=True).clip(1e-12)
        rotation_vector = torch.from_numpy(values).float().to(device) * math.radians(0.1)
    if mode in ("aperture", "combined"):
        signs = rng.choice((-1.0, 1.0), size=count)
        aperture = torch.from_numpy(signs).float().to(device) * 1e-4
    return PoseState(
        so3_exp(rotation_vector) @ state.rotation,
        state.position + translation,
        state.aperture + aperture,
    )


def _amplification_diagnostic(
    model,
    manifest: DatasetManifest,
    bundles: list[Bundle],
    groups: list[Group],
    device: torch.device,
    seed: int,
) -> dict[str, dict[str, list[float]]]:
    rng = np.random.default_rng(seed + 1)
    schedule = torch.tensor(manifest.commanded_aperture_m, dtype=torch.float32, device=device)
    selected: list[Group] = []
    for split, limit in (("train", 120), ("val", 30), ("test", 30)):
        candidates = [
            group
            for group in groups
            if bundles[group.bundle_index].split == split and np.count_nonzero(group.active)
        ]
        rng.shuffle(candidates)
        selected.extend(candidates[:limit])
    result = {
        mode: {"amplification": [], "split": [], "gate_switch": []}
        for mode in ("translation", "rotation", "aperture", "combined")
    }
    with torch.no_grad():
        for group in selected:
            indices_np = np.flatnonzero(group.active)
            if indices_np.size > 32:
                indices_np = rng.choice(indices_np, size=32, replace=False)
            indices = torch.from_numpy(indices_np).long().to(device)
            bundle = bundles[group.bundle_index]
            states = bundle.states.to(device)
            current = _state_at(states, group.step, indices)
            sdf = _sdf_for(bundle.record, len(indices_np), device)
            command = schedule[group.step + 1]
            baseline, base_aux = model.forward_step(current, command, sdf, return_aux=True)
            for mode in result:
                perturbed = _perturb_state(current, mode, rng, device)
                response, aux = model.forward_step(perturbed, command, sdf, return_aux=True)
                denominator = state_error(
                    perturbed, current, length_scale=model.length_scale
                )[0].sqrt()
                numerator = state_error(
                    response, baseline, length_scale=model.length_scale
                )[0].sqrt()
                amplification = (numerator / denominator.clamp_min(1e-12)).cpu().numpy()
                result[mode]["amplification"].extend(map(float, amplification))
                result[mode]["split"].extend([bundle.split] * len(amplification))
                switched = (aux.active != base_aux.active).cpu().numpy()
                result[mode]["gate_switch"].extend(map(bool, switched))
            del states, sdf
    return result


def _summaries(
    manifest: DatasetManifest,
    bundles: list[Bundle],
    arrays: dict[str, Any],
    groups: list[Group],
    object_terminal: list[dict[str, Any]],
    suff_refs: list[tuple[int, int, np.ndarray, np.ndarray]],
    gradients: dict[str, Any],
    gt_feasibility_gradients: dict[str, Any],
    amplification: dict[str, Any],
) -> dict[str, Any]:
    gt = {}
    for split in SPLITS:
        gap = arrays["gt_gap"][split]
        feas = arrays["gt_feasibility"][split]
        gt[split] = {
            "min_gap_m": _stats(gap),
            "feasibility_per_state": _stats(feas),
            "violating_state_fraction": float(np.mean(gap < 0)),
            "deeper_than_0_5mm_fraction": float(np.mean(gap < -0.0005)),
            "deeper_than_1mm_fraction": float(np.mean(gap < -0.001)),
            "deeper_than_5mm_fraction": float(np.mean(gap < -0.005)),
        }
    all_gap = np.concatenate([arrays["gt_gap"][split] for split in SPLITS])
    all_feasibility = np.concatenate(
        [arrays["gt_feasibility"][split] for split in SPLITS]
    )
    gt["all"] = {
        "min_gap_m": _stats(all_gap),
        "feasibility_per_state": _stats(all_feasibility),
        "violating_state_fraction": float(np.mean(all_gap < 0)),
        "deeper_than_0_5mm_fraction": float(np.mean(all_gap < -0.0005)),
        "deeper_than_1mm_fraction": float(np.mean(all_gap < -0.001)),
        "deeper_than_5mm_fraction": float(np.mean(all_gap < -0.005)),
    }

    gradient_summary = {}
    for category, values in gradients.items():
        cosine = np.asarray(values["cosine"])
        ratio = np.asarray(values["ratio"])
        feasibility_norm = np.asarray(values["feasibility_norm"])
        valid_cosine = cosine[np.isfinite(cosine)]
        gradient_summary[category] = {
            "batches": int(cosine.size),
            "samples": int(np.sum(values["samples"])),
            "cosine": _stats(cosine),
            "feasibility_to_state_norm_ratio": _stats(ratio),
            "negative_cosine_fraction_of_nonzero": (
                float(np.mean(valid_cosine < 0)) if valid_cosine.size else None
            ),
            "zero_feasibility_gradient_fraction": float(np.mean(feasibility_norm <= 1e-12)),
        }

    teacher_summary = {}
    for split in SPLITS:
        teacher_values = arrays["teacher"][split]
        pushed_values = arrays["pushforward"][split]
        teacher_contact = arrays["teacher_contact"][split]
        pushed_contact = arrays["pushforward_contact"][split]
        teacher_summary[split] = {
            "teacher_forced_dx": _stats(teacher_values),
            "pushforward_dx": _stats(pushed_values),
            "mean_ratio": float(pushed_values.mean() / teacher_values.mean()),
            "pushforward_worse_fraction": float(np.mean(pushed_values > teacher_values)),
            "contact_teacher_forced_dx": _stats(teacher_contact),
            "contact_pushforward_dx": _stats(pushed_contact),
            "contact_mean_ratio": float(pushed_contact.mean() / teacher_contact.mean()),
        }

    amplification_summary = {}
    for mode, values in amplification.items():
        amp = np.asarray(values["amplification"])
        amplification_summary[mode] = {
            "amplification": _stats(amp),
            "fraction_gt_1": float(np.mean(amp > 1)),
            "fraction_gt_1_05": float(np.mean(amp > 1.05)),
            "fraction_gt_1_1": float(np.mean(amp > 1.1)),
            "fraction_gt_2": float(np.mean(amp > 2)),
            "fraction_gt_5": float(np.mean(amp > 5)),
            "gate_switch_fraction": float(np.mean(values["gate_switch"])),
            "by_split": {
                split: _stats(amp[np.asarray(values["split"]) == split]) for split in SPLITS
            },
        }

    h32 = {
        split: {
            "terminal_dx": _stats(arrays["rollout_terminal"][split]),
            "objects": [item for item in object_terminal if item["split"] == split],
        }
        for split in SPLITS
    }

    suff_input = arrays["suff_input"]
    suff_output = arrays["suff_output"]
    contact = arrays["suff_contact"].astype(bool)
    close_threshold = float(np.quantile(suff_input[contact], 0.01))
    close = contact & (suff_input <= close_threshold)
    order = np.argsort(np.where(close, suff_output, -np.inf))[-10:][::-1]
    flattened_refs: list[dict[str, Any]] = []
    offset = 0
    for bundle_index, step, trajectory, nearest in suff_refs:
        for local_index in range(len(trajectory)):
            flattened_refs.append(
                {
                    "bundle_index": bundle_index,
                    "step": step,
                    "trajectory": int(trajectory[local_index]),
                    "nearest_trajectory": int(nearest[local_index]),
                }
            )
        offset += len(trajectory)
    ambiguous = []
    for index in order:
        ref = flattened_refs[int(index)]
        bundle = bundles[ref["bundle_index"]]
        ambiguous.append(
            {
                "split": bundle.split,
                "object_id": bundle.object_id,
                "step": ref["step"],
                "trajectory": ref["trajectory"],
                "nearest_trajectory": ref["nearest_trajectory"],
                "input_rms_distance": float(suff_input[index]),
                "output_correction_distance": float(suff_output[index]),
            }
        )
    sufficiency = {
        "all_nearest_input_rms_distance": _stats(suff_input),
        "all_nearest_output_correction_distance": _stats(suff_output),
        "contact_nearest_input_rms_distance": _stats(suff_input[contact]),
        "contact_nearest_output_correction_distance": _stats(suff_output[contact]),
        "close_contact_definition": "lowest 1% nearest-input RMS distance among simulator-contact transitions",
        "close_contact_input_threshold": close_threshold,
        "close_contact_output_distance": _stats(suff_output[close]),
        "close_contact_fraction_output_gt_0_05": float(np.mean(suff_output[close] > 0.05)),
        "exact_duplicate_contact_count": int(np.count_nonzero(contact & (suff_input == 0))),
        "exact_duplicate_contact_output_distance": _stats(
            suff_output[contact & (suff_input == 0)]
        ),
        "ambiguous_examples": ambiguous,
    }

    gt_gradient_norm = np.asarray(gt_feasibility_gradients["gradient_norm"])
    gt_gradient_split = np.asarray(gt_feasibility_gradients["split"])
    gt_gradient_summary = {
        "normalized_state_gradient_norm": _stats(gt_gradient_norm),
        "nonzero_fraction": float(np.mean(gt_gradient_norm > 1e-8)),
        "by_split": {
            split: _stats(gt_gradient_norm[gt_gradient_split == split]) for split in SPLITS
        },
    }

    sdf_rows = []
    for bundle in bundles:
        values = bundle.record.sdf.float().numpy()
        sdf_rows.append(
            {
                "split": bundle.split,
                "object_id": bundle.object_id,
                "minimum_m": float(values.min()),
                "maximum_m": float(values.max()),
                "minimum_plateau_fraction": float(np.mean(values == values.min())),
                "maximum_plateau_fraction": float(np.mean(values == values.max())),
            }
        )
    return {
        "definitions": {
            "near_contact": "active trial state with 0 <= min(trial gap) <= delta_gate",
            "stall": "simulator contact_count > 0 and target aperture exceeds command by >0.5 mm",
            "sliding": "simulator contact_count > 0 and object moves >0.25 mm or rotates >0.25 deg during the transition",
            "perturbations": {
                "translation_m": 0.0001,
                "rotation_deg": 0.1,
                "aperture_m": 0.0001,
            },
            "state_sufficiency_input": "[trial gaps / sdf_scale, current aperture / length_scale, next command / length_scale]",
            "state_sufficiency_search": "nearest other trajectory at the same object and command step",
        },
        "limitations": {
            "train_shape_holdout": "No within-train-object trajectory holdout exists: the checkpoint saw all 100 trajectories. Train-shape H32 is therefore an in-sample shape comparison, not a held-out-trajectory estimate.",
            "sliding_label": "No tangential slip state is stored; sliding is an operational pose-motion proxy.",
        },
        "gt_feasibility": gt,
        "gt_feasibility_state_gradient": gt_gradient_summary,
        "sdf_storage_check": {
            "appears_hard_clipped": bool(
                any(
                    row["minimum_plateau_fraction"] > 0.01
                    or row["maximum_plateau_fraction"] > 0.01
                    for row in sdf_rows
                )
            ),
            "objects": sdf_rows,
            "training_and_evaluation_use_same_256_gripper_points": True,
        },
        "gradient_interference": gradient_summary,
        "teacher_forced_vs_pushforward": teacher_summary,
        "local_perturbation_amplification": amplification_summary,
        "h32_shape_comparison": h32,
        "state_sufficiency": sufficiency,
        "category_counts": {
            category: int(sum(np.count_nonzero(getattr(group, category)) for group in groups))
            for category in ("active", "near_contact", "stall", "sliding")
        },
    }


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(np.asarray(values))
    return values, np.arange(1, len(values) + 1) / len(values)


def _plots(output: Path, arrays: dict[str, Any], gradients: dict[str, Any], amplification: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    colors = {"train": "#3478bf", "val": "#e67e22", "test": "#2e9d58"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for split in SPLITS:
        axes[0].hist(arrays["gt_gap"][split] * 1000, bins=100, density=True, histtype="step", linewidth=1.5, label=split, color=colors[split])
        x, y = _ecdf(arrays["gt_feasibility"][split])
        axes[1].plot(x, y, label=split, color=colors[split])
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="GT min gap [mm]", ylabel="density", title="GT surrogate feasibility")
    axes[1].set_xscale("symlog", linthresh=1e-8)
    axes[1].set(xlabel="per-state feasibility loss", ylabel="CDF", title="GT feasibility loss CDF")
    axes[0].legend(); axes[1].legend(); fig.tight_layout()
    path = output / "01_gt_feasibility.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    categories = ("near_contact", "stall", "sliding")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cosines = [np.asarray(gradients[c]["cosine"])[np.isfinite(gradients[c]["cosine"])] for c in categories]
    ratios = [np.asarray(gradients[c]["ratio"])[np.isfinite(gradients[c]["ratio"])] for c in categories]
    axes[0].boxplot(cosines, tick_labels=categories, showmeans=True)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set(ylabel="cos(g_state, g_feas)", title="Gradient interference")
    axes[1].boxplot(ratios, tick_labels=categories, showmeans=True)
    axes[1].set_yscale("log")
    axes[1].set(ylabel="||g_feas|| / ||g_state||", title="Gradient norm ratio")
    fig.tight_layout(); path = output / "02_gradient_cosine.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    steps = np.arange(2, 33)
    for split in SPLITS:
        axes[0].plot(steps, arrays["teacher_by_step"][split], linestyle="--", color=colors[split], label=f"{split} teacher")
        axes[0].plot(steps, arrays["pushforward_by_step"][split], color=colors[split], label=f"{split} push")
    ratios = [arrays["pushforward"][s].mean() / arrays["teacher"][s].mean() for s in SPLITS]
    axes[1].bar(SPLITS, ratios, color=[colors[s] for s in SPLITS])
    axes[1].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="target step", ylabel="mean one-step d_X", title="Teacher-forced vs two-step pushforward")
    axes[1].set(ylabel="mean push / teacher", title="Distribution-shift multiplier")
    axes[0].legend(ncol=2, fontsize=8); fig.tight_layout(); path = output / "03_teacher_pushforward.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    modes = tuple(amplification)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for mode in modes:
        values = np.asarray(amplification[mode]["amplification"])
        x, y = _ecdf(np.minimum(values, 10))
        axes[0].plot(x, y, label=mode)
    axes[0].axvline(1, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="directional amplification A (clipped at 10)", ylabel="CDF", title="Local perturbation amplification")
    axes[1].bar(modes, [np.mean(np.asarray(amplification[m]["amplification"]) > 1) for m in modes])
    axes[1].set(ylabel="fraction A > 1", title="Locally expansive active states")
    axes[0].legend(); axes[1].tick_params(axis="x", rotation=20); fig.tight_layout(); path = output / "04_local_amplification.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].boxplot([arrays["rollout_terminal"][s] for s in SPLITS], tick_labels=SPLITS, showmeans=True, showfliers=False)
    for split in SPLITS:
        axes[1].plot(np.arange(33), arrays["rollout_by_step"][split], label=split, color=colors[split])
    axes[0].set(ylabel="terminal d_X", title="H32 by shape split")
    axes[1].set(xlabel="rollout step", ylabel="mean d_X", title="Autoregressive error growth")
    axes[1].legend(); fig.tight_layout(); path = output / "05_h32_shape_comparison.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    contact = arrays["suff_contact"].astype(bool)
    x = arrays["suff_input"][contact]
    y = arrays["suff_output"][contact]
    sample = np.linspace(0, len(x) - 1, min(len(x), 30000), dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hexbin(x[sample], y[sample], gridsize=70, bins="log", mincnt=1, cmap="viridis")
    axes[0].set(xlabel="nearest input RMS distance", ylabel="GT correction distance", title="State sufficiency: contact transitions")
    threshold = summary["state_sufficiency"]["close_contact_input_threshold"]
    close = contact & (arrays["suff_input"] <= threshold)
    axes[1].hist(arrays["suff_output"][close], bins=60, color="#7f4fa3")
    axes[1].axvline(0.05, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="GT correction distance", ylabel="count", title="Closest 1% input neighbours")
    fig.tight_layout(); path = output / "06_state_sufficiency.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = ExperimentConfig.load(args.config)
    manifest = DatasetManifest.load(config.paths.manifest)
    device = torch.device(config.device)
    model = _build_model(config, manifest, device)
    checkpoint = load_checkpoint(args.checkpoint, model=model, map_location=device)
    if checkpoint["manifest_sha256"] != manifest.sha256():
        raise ValueError("checkpoint manifest hash mismatch")
    model.eval()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundles = _load_bundles(manifest)
    arrays, groups, object_terminal, suff_refs = _main_pass(
        model, manifest, bundles, device
    )
    gradients = _gradient_diagnostic(
        model, manifest, bundles, groups, device, args.seed
    )
    gt_feasibility_gradients = _gt_feasibility_gradient_diagnostic(
        model, manifest, bundles, groups, device, args.seed
    )
    amplification = _amplification_diagnostic(
        model, manifest, bundles, groups, device, args.seed
    )
    summary = _summaries(
        manifest,
        bundles,
        arrays,
        groups,
        object_terminal,
        suff_refs,
        gradients,
        gt_feasibility_gradients,
        amplification,
    )
    summary["metadata"] = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_horizon": int(checkpoint["horizon"]),
        "manifest": str(config.paths.manifest),
        "device": str(device),
        "seed": args.seed,
    }
    plot_paths = _plots(output, arrays, gradients, amplification, summary)
    summary["plots"] = plot_paths
    (output / "results.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output / "raw_arrays.npz",
        **{
            f"gt_gap_{split}": arrays["gt_gap"][split] for split in SPLITS
        },
        **{
            f"teacher_{split}": arrays["teacher"][split] for split in SPLITS
        },
        **{
            f"pushforward_{split}": arrays["pushforward"][split] for split in SPLITS
        },
        **{
            f"rollout_terminal_{split}": arrays["rollout_terminal"][split]
            for split in SPLITS
        },
        sufficiency_input=arrays["suff_input"],
        sufficiency_output=arrays["suff_output"],
        sufficiency_contact=arrays["suff_contact"],
    )
    print(json.dumps({"results": str(output / "results.json"), "plots": plot_paths}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
