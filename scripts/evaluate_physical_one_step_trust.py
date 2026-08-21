#!/usr/bin/env python3
"""Evaluate the seed-0 true one-step physics constrained BPTT ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ARMS = ("frozen_local", "old_bptt", "frozen_output_trust", "physical_trust")
CONTROL_SAMPLES = Path("runs/ablation-pushforward-stopgrad/samples.npz")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _state_at(states: Any, step: int) -> Any:
    from srno.types import PoseState

    return PoseState(
        states.rotation[:, step],
        states.position[:, step],
        states.joint_position[:, step],
    )


def _load_model(config: Any, manifest: Any, path: Path) -> tuple[Any, dict[str, Any]]:
    import torch

    from srno.training.checkpoint import load_checkpoint
    from srno.training.engine import _build_model

    model = _build_model(config, manifest, torch.device(config.device))
    checkpoint = load_checkpoint(path, model=model, map_location=torch.device(config.device))
    if checkpoint["manifest_sha256"] != manifest.sha256():
        raise ValueError(f"manifest hash mismatch for {path}")
    if checkpoint["gripper_sha256"] != manifest.gripper_sha256:
        raise ValueError(f"gripper hash mismatch for {path}")
    model.eval()
    return model, checkpoint


def _evaluate_candidate_split(
    model: Any,
    config: Any,
    manifest: Any,
    *,
    split: str,
) -> dict[str, np.ndarray]:
    import torch

    try:
        from contact_composition_diagnostics import _pose_log_error
    except ModuleNotFoundError:
        from scripts.contact_composition_diagnostics import _pose_log_error
    from srno.data.dataset import H5ObjectDataset, TrajectoryBatch, make_dataloader
    from srno.geometry.se3 import rotation_geodesic_angle
    from srno.losses import state_error
    from srno.training.engine import _autocast
    from srno.types import PoseState

    device = torch.device(config.device)
    dataset = H5ObjectDataset(manifest, split=split)
    loader = make_dataloader(
        dataset,
        mode="rollout",
        objects_per_batch=1,
        samples_per_object=0,
        workers=0,
        seed=0,
        shuffle=False,
    )
    rows: dict[str, list[np.ndarray]] = {}
    labels: list[str] = []
    try:
        with torch.no_grad():
            for object_index, raw in enumerate(loader):
                assert isinstance(raw, TrajectoryBatch)
                batch = raw.to(device)
                with _autocast(config, device):
                    rollout = model.rollout(
                        _state_at(batch.states, 0), batch.command_schedule[1:], batch.sdf
                    )
                metrics: dict[str, list[torch.Tensor]] = {
                    name: []
                    for name in (
                        "dx",
                        "translation_m",
                        "rotation_rad",
                        "joint_rmse_over_travel",
                        "aperture_m",
                        "penetration_mean_m",
                        "penetration_max_m",
                    )
                }
                tf_log: list[torch.Tensor] = []
                ar_log: list[torch.Tensor] = []
                for step in range(33):
                    predicted = _state_at(rollout, step)
                    target = _state_at(batch.states, step)
                    state_sq, _, _, _ = state_error(
                        predicted,
                        target,
                        length_scale=model.length_scale,
                        joint_scale=model.joint_travel_range,
                    )
                    metrics["dx"].append(state_sq.sqrt())
                    metrics["translation_m"].append(
                        torch.linalg.vector_norm(
                            predicted.position - target.position, dim=-1
                        )
                    )
                    metrics["rotation_rad"].append(
                        rotation_geodesic_angle(predicted.rotation, target.rotation)
                    )
                    metrics["joint_rmse_over_travel"].append(
                        torch.sqrt(
                            (
                                (predicted.joint_position - target.joint_position)
                                / model.joint_travel_range
                            )
                            .square()
                            .mean(dim=-1)
                        )
                    )
                    metrics["aperture_m"].append(
                        (
                            model.aperture_from_joints(predicted.joint_position)
                            - batch.actual_aperture[:, step]
                        ).abs()
                    )
                    gap = model.query_geometric_gap(predicted, batch.sdf)
                    penetration = torch.relu(-gap)
                    metrics["penetration_mean_m"].append(penetration.mean(dim=-1))
                    metrics["penetration_max_m"].append(penetration.max(dim=-1).values)
                    if step:
                        ar_log.append(_pose_log_error(target, predicted))
                for step in range(32):
                    current = _state_at(batch.states, step)
                    target = _state_at(batch.states, step + 1)
                    with _autocast(config, device):
                        prediction = model.forward_step(
                            current, batch.command_schedule[step + 1], batch.sdf
                        )
                    assert isinstance(prediction, PoseState)
                    tf_log.append(_pose_log_error(target, prediction))
                for name, values in metrics.items():
                    rows.setdefault(name, []).append(
                        torch.stack(values, dim=1).cpu().numpy().astype(np.float32)
                    )
                rows.setdefault("tf_log", []).append(
                    torch.stack(tf_log, dim=1).cpu().numpy().astype(np.float32)
                )
                rows.setdefault("ar_log", []).append(
                    torch.stack(ar_log, dim=1).cpu().numpy().astype(np.float32)
                )
                count = len(raw.trajectory_index)
                rows.setdefault("object_index", []).append(
                    np.full(count, object_index, dtype=np.int32)
                )
                rows.setdefault("trajectory_index", []).append(
                    raw.trajectory_index.cpu().numpy().astype(np.int32)
                )
                labels.append(raw.object_ids[0])
    finally:
        dataset.close()
    result = {name: np.concatenate(values, axis=0) for name, values in rows.items()}
    result["object_labels"] = np.asarray(labels)
    return result


def _evaluate_active_physics(
    model: Any,
    config: Any,
    manifest: Any,
    active_index: Any,
    *,
    split: str,
) -> dict[str, float]:
    import torch

    from srno.data.dataset import H5ObjectDataset, LocalTransitionBatch, make_dataloader
    from srno.losses import state_error
    from srno.training.engine import _autocast

    device = torch.device(config.device)
    dataset = H5ObjectDataset(
        manifest, split=split, active_index=active_index, active_only=True
    )
    loader = make_dataloader(
        dataset,
        mode="local",
        objects_per_batch=min(4, len(dataset)),
        samples_per_object=256,
        workers=config.loader.workers,
        seed=0,
        shuffle=False,
    )
    sums = torch.zeros(5, dtype=torch.float64)
    count = 0
    try:
        with torch.no_grad():
            for raw in loader:
                assert isinstance(raw, LocalTransitionBatch)
                batch = raw.to(device, non_blocking=True)
                with _autocast(config, device):
                    prediction = model.forward_step(
                        batch.current, batch.next_command, batch.sdf
                    )
                    total, translation, rotation, joints = state_error(
                        prediction,
                        batch.target,
                        length_scale=model.length_scale,
                        joint_scale=model.joint_travel_range,
                    )
                count += total.numel()
                sums += torch.tensor(
                    [
                        float(total.sum()),
                        float(total.sqrt().sum()),
                        float(translation.sum()),
                        float(rotation.sum()),
                        float(joints.sum()),
                    ],
                    dtype=torch.float64,
                )
    finally:
        dataset.close()
    mean = sums / count
    return {
        "transitions": count,
        "loss": float(mean[0]),
        "mean_dx": float(mean[1]),
        "translation": float(mean[2]),
        "rotation": float(mean[3]),
        "joints": float(mean[4]),
    }


def _summary(values: dict[str, np.ndarray]) -> dict[str, Any]:
    object_index = values["object_index"]
    labels = values["object_labels"].tolist()
    summary: dict[str, Any] = {
        "trajectories": int(len(object_index)),
        "terminal_dx": float(values["dx"][:, -1].mean()),
        "terminal_translation_m": float(values["translation_m"][:, -1].mean()),
        "terminal_rotation_rad": float(values["rotation_rad"][:, -1].mean()),
        "terminal_joint_rmse_over_travel": float(
            values["joint_rmse_over_travel"][:, -1].mean()
        ),
        "terminal_aperture_m": float(values["aperture_m"][:, -1].mean()),
        "max_penetration_m": float(values["penetration_max_m"].max()),
        "tf_b32_v_m_w_rad": values["tf_log"].sum(axis=1).mean(axis=0).tolist(),
        "terminal_ar_bias_v_m_w_rad": values["ar_log"][:, -1].mean(axis=0).tolist(),
        "per_object": {},
    }
    for index, label in enumerate(labels):
        selected = object_index == index
        summary["per_object"][label] = {
            "trajectories": int(selected.sum()),
            "terminal_dx": float(values["dx"][selected, -1].mean()),
            "terminal_translation_m": float(
                values["translation_m"][selected, -1].mean()
            ),
            "terminal_rotation_rad": float(
                values["rotation_rad"][selected, -1].mean()
            ),
            "terminal_joint_rmse_over_travel": float(
                values["joint_rmse_over_travel"][selected, -1].mean()
            ),
        }
    return summary


def _resampled_mean(
    values: np.ndarray,
    object_index: np.ndarray,
    rng: np.random.Generator,
) -> float:
    objects = np.unique(object_index)
    selected_objects = rng.choice(objects, size=len(objects), replace=True)
    means = []
    for object_id in selected_objects:
        rows = np.flatnonzero(object_index == object_id)
        sample = rng.choice(rows, size=len(rows), replace=True)
        means.append(float(values[sample].mean()))
    return float(np.mean(means))


def _bootstrap_difference(
    candidate: np.ndarray,
    baseline: np.ndarray,
    object_index: np.ndarray,
    *,
    replicates: int = 10_000,
) -> dict[str, float]:
    rng = np.random.default_rng(0)
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(
        baseline, dtype=np.float64
    )
    samples = np.asarray(
        [_resampled_mean(delta, object_index, rng) for _ in range(replicates)]
    )
    return {
        "mean": float(delta.mean()),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
    }


def _bootstrap_bias_distance(
    candidate: np.ndarray,
    local: np.ndarray,
    old: np.ndarray,
    object_index: np.ndarray,
    *,
    replicates: int = 10_000,
) -> dict[str, float]:
    rng = np.random.default_rng(0)
    direct = abs(candidate.mean() - local.mean()) - abs(old.mean() - local.mean())
    samples = np.empty(replicates, dtype=np.float64)
    objects = np.unique(object_index)
    for replicate in range(replicates):
        candidate_object_means = []
        local_object_means = []
        old_object_means = []
        for object_id in rng.choice(objects, size=len(objects), replace=True):
            rows = np.flatnonzero(object_index == object_id)
            selected = rng.choice(rows, size=len(rows), replace=True)
            candidate_object_means.append(float(candidate[selected].mean()))
            local_object_means.append(float(local[selected].mean()))
            old_object_means.append(float(old[selected].mean()))
        candidate_mean = float(np.mean(candidate_object_means))
        local_mean = float(np.mean(local_object_means))
        old_mean = float(np.mean(old_object_means))
        samples[replicate] = abs(candidate_mean - local_mean) - abs(
            old_mean - local_mean
        )
    return {
        "mean": float(direct),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
    }


def _control_values(
    archive: Any, arm: str, split: str
) -> dict[str, np.ndarray]:
    prefix = f"{arm}_seed0_{split}_"
    return {
        name: np.asarray(archive[prefix + name])
        for name in (
            "dx",
            "translation_m",
            "rotation_rad",
            "joint_rmse_over_travel",
            "aperture_m",
            "penetration_mean_m",
            "penetration_max_m",
            "tf_log",
            "ar_log",
            "object_index",
            "trajectory_index",
            "object_labels",
        )
    }


def _plot(output: Path, results: dict[str, Any], candidate: dict[str, Any]) -> None:
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ("local", "old BPTT", "output trust", "physical trust")
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    x = np.arange(len(ARMS))
    for offset, split in ((-0.18, "val"), (0.18, "test")):
        axes[0, 0].bar(
            x + offset,
            [results["rollout"][arm][split]["terminal_dx"] for arm in ARMS],
            0.36,
            label=split,
        )
    axes[0, 0].set_xticks(x, labels, rotation=15)
    axes[0, 0].set_title("H32 terminal state error")
    axes[0, 0].set_ylabel(r"terminal $d_X$")
    axes[0, 0].legend()

    splits = ("train", "val", "test")
    for arm in ARMS:
        axes[0, 1].plot(
            splits,
            [results["physical_one_step"][arm][split]["loss"] for split in splits],
            marker="o",
            label=labels[ARMS.index(arm)],
        )
    axes[0, 1].set_title("True one-step physics loss")
    axes[0, 1].set_ylabel(r"$L_{phys}=E[d_X^2]$")
    axes[0, 1].legend(fontsize=8)

    width = 0.2
    for index, arm in enumerate(ARMS):
        axes[1, 0].bar(
            np.arange(3) + (index - 1.5) * width,
            [results["signed_tf_b32_z_mm"][arm][split] for split in splits],
            width,
            label=labels[index],
        )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_xticks(np.arange(3), splits)
    axes[1, 0].set_title("Accumulated teacher-forced z bias")
    axes[1, 0].set_ylabel(r"$B_{32,z}^{TF}$, mm")
    axes[1, 0].legend(fontsize=8)

    for split in ("val", "test"):
        axes[1, 1].plot(
            np.arange(33), candidate[split]["dx"].mean(axis=0), label=split
        )
    axes[1, 1].set_title("Physical-trust rollout")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].set_ylabel(r"$d_X(k)$")
    axes[1, 1].legend()

    component_names = ("translation_m", "rotation_rad", "joint_rmse_over_travel")
    component_labels = ("translation, m", "rotation, rad", "joint RMS/travel")
    component_x = np.arange(len(component_names))
    component_width = 0.18
    for index, arm in enumerate(ARMS):
        summary = results["rollout"][arm]["test"]
        axes[0, 2].bar(
            component_x + (index - 1.5) * component_width,
            [summary[f"terminal_{name}"] for name in component_names],
            component_width,
            label=labels[index],
        )
    axes[0, 2].set_xticks(component_x, component_labels, rotation=12)
    axes[0, 2].set_title("Test terminal T/R/J components")
    axes[0, 2].legend(fontsize=8)

    per_object = results["comparison"]["per_object_test_new_minus_local"]
    object_labels = list(per_object)
    object_y = np.arange(len(object_labels))
    axes[1, 2].barh(
        object_y,
        [per_object[label] for label in object_labels],
    )
    axes[1, 2].axvline(0.0, color="black", linewidth=0.8)
    axes[1, 2].set_yticks(
        object_y,
        [textwrap.fill(label.replace("-", " "), 31) for label in object_labels],
    )
    axes[1, 2].set_title("Per-object test: physical trust - local")
    axes[1, 2].set_xlabel(r"difference in terminal $d_X$")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.savefig(output / "physical_one_step_trust_ablation.png", dpi=180)
    plt.close(fig)


def _training_history(output: Path) -> list[dict[str, Any]]:
    history = []
    for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        train = row.get("train", {})
        history.append(
            {
                "epoch": int(row["epoch"]),
                "horizon": int(row["horizon"]),
                "kind": str(row.get("kind", "epoch")),
                "validation_terminal_dx": float(row["val"]["terminal_dx"]),
                "physical_loss_exact": float(
                    train["physical_one_step_sq_exact"]
                ),
                "constraint_ratio_exact": float(train["constraint_ratio_exact"]),
                "multiplier": (
                    None
                    if row.get("kind") == "horizon_initial"
                    else float(train["multiplier_after_update"])
                ),
            }
        )
    return history


def _plot_training(output: Path, history: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.asarray([row["epoch"] for row in history])
    regular = np.asarray([row["kind"] == "epoch" for row in history])
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(
        epochs, [row["validation_terminal_dx"] for row in history], marker="."
    )
    axes[0, 0].set_ylabel(r"validation terminal $d_X$")
    axes[0, 1].plot(
        epochs, [row["physical_loss_exact"] for row in history], marker="."
    )
    axes[0, 1].axhline(
        history[0]["physical_loss_exact"], color="black", linestyle="--"
    )
    axes[0, 1].set_ylabel(r"exact train $L_{phys}$")
    axes[1, 0].plot(
        epochs, [row["constraint_ratio_exact"] for row in history], marker="."
    )
    axes[1, 0].axhline(1.0, color="black", linestyle="--")
    axes[1, 0].set_ylabel(r"$L_{phys}/L_{phys}(\theta_0)$")
    axes[1, 1].plot(
        epochs[regular],
        [row["multiplier"] for row in history if row["kind"] == "epoch"],
        marker=".",
    )
    axes[1, 1].set_ylabel(r"HPR multiplier $\mu$")
    for axis in axes.flat:
        axis.set_xlabel("global epoch")
        axis.grid(alpha=0.2)
        for row in history:
            if row["kind"] == "horizon_initial":
                axis.axvline(row["epoch"], color="grey", alpha=0.25)
    figure.savefig(output / "physical_one_step_trust_training.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ablation-physical-one-step-trust.toml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/ablation-physical-one-step-trust/seed-0/best-trust-rollout-h32.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/ablation-physical-one-step-trust/seed-0"),
    )
    args = parser.parse_args()

    import torch

    from srno.data.index import ActiveIndex
    from srno.data.schema import DatasetManifest
    from srno.training.config import ExperimentConfig

    config = ExperimentConfig.load(args.config)
    if config.trust_region.constraint_target != "physical_one_step":
        raise ValueError("evaluator requires physical_one_step constraint")
    manifest = DatasetManifest.load(config.paths.manifest)
    active_index = ActiveIndex.load(config.paths.active_index)
    args.output.mkdir(parents=True, exist_ok=True)

    checkpoints = {
        "frozen_local": Path(
            "runs/ablation-jq-local/baseline/seed-0/best-local.pt"
        ),
        "old_bptt": Path(
            "runs/ablation-actuator-rollout/aperture/seed-0/best-rollout-h32.pt"
        ),
        "frozen_output_trust": Path(
            "runs/ablation-local-trust-region/seed-0/best-trust-rollout-h32.pt"
        ),
        "physical_trust": args.checkpoint,
    }
    models: dict[str, Any] = {}
    checkpoint_payloads: dict[str, Any] = {}
    for arm, path in checkpoints.items():
        models[arm], checkpoint_payloads[arm] = _load_model(
            config, manifest, path
        )
    trust_state = checkpoint_payloads["physical_trust"]["extra_state"][
        "trust_region"
    ]
    if trust_state.get("constraint_target") != "physical_one_step":
        raise ValueError("candidate checkpoint has the wrong constraint target")
    if int(checkpoint_payloads["physical_trust"]["horizon"]) != 32:
        raise ValueError("candidate checkpoint is not H32")
    baseline_path = args.output / "physical-baseline-contract.json"
    if _sha256(baseline_path) != trust_state.get("physical_baseline_sha256"):
        raise ValueError("candidate physical baseline sidecar hash mismatch")
    baseline_contract = json.loads(baseline_path.read_text(encoding="utf-8"))
    expected_baseline_contract = {
        "manifest_sha256": manifest.sha256(),
        "gripper_sha256": manifest.gripper_sha256,
        "active_index_sha256": _sha256(config.paths.active_index),
        "local_checkpoint_sha256": _sha256(checkpoints["frozen_local"]),
    }
    for key, value in expected_baseline_contract.items():
        if baseline_contract.get(key) != value:
            raise ValueError(f"physical baseline {key} mismatch")
    control_results = json.loads(
        Path("runs/ablation-pushforward-stopgrad/results.json").read_text(
            encoding="utf-8"
        )
    )
    for arm, control_name in (
        ("frozen_local", "frozen_local"),
        ("old_bptt", "old_bptt_h32"),
    ):
        metadata = control_results["contract"]["checkpoints"][control_name]["0"]
        if metadata["sha256"] != _sha256(checkpoints[arm]):
            raise ValueError(f"control sample checkpoint mismatch for {arm}")
    if control_results["contract"]["manifest_sha256"] != manifest.sha256():
        raise ValueError("control samples manifest mismatch")

    evaluated = {
        arm: {
            split: _evaluate_candidate_split(
                models[arm], config, manifest, split=split
            )
            for split in ("train", "val", "test")
        }
        for arm in ("frozen_output_trust", "physical_trust")
    }
    candidate = evaluated["physical_trust"]
    with np.load(CONTROL_SAMPLES, allow_pickle=False) as archive:
        local = {
            split: _control_values(archive, "frozen_local", split)
            for split in ("train", "val", "test")
        }
        old = {
            split: _control_values(archive, "old_bptt_h32", split)
            for split in ("train", "val", "test")
        }
    for split in ("train", "val", "test"):
        for control in (local, old, evaluated["frozen_output_trust"]):
            if not np.array_equal(
                candidate[split]["trajectory_index"], control[split]["trajectory_index"]
            ) or not np.array_equal(
                candidate[split]["object_labels"], control[split]["object_labels"]
            ):
                raise ValueError(f"paired sample order mismatch on {split}")

    physical: dict[str, Any] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for split in ("train", "val", "test"):
            physical[arm][split] = _evaluate_active_physics(
                models[arm], config, manifest, active_index, split=split
            )

    rollout: dict[str, Any] = {
        "frozen_local": {
            split: _summary(local[split]) for split in ("train", "val", "test")
        },
        "old_bptt": {
            split: _summary(old[split]) for split in ("train", "val", "test")
        },
        "physical_trust": {
            split: _summary(candidate[split])
            for split in ("train", "val", "test")
        },
        "frozen_output_trust": {
            split: _summary(evaluated["frozen_output_trust"][split])
            for split in ("train", "val", "test")
        },
    }

    signed: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    for split in ("train", "val", "test"):
        signed["frozen_local"][split] = float(
            1000.0 * local[split]["tf_log"].sum(axis=1)[:, 2].mean()
        )
        signed["old_bptt"][split] = float(
            1000.0 * old[split]["tf_log"].sum(axis=1)[:, 2].mean()
        )
        signed["physical_trust"][split] = float(
            1000.0 * candidate[split]["tf_log"].sum(axis=1)[:, 2].mean()
        )
        signed["frozen_output_trust"][split] = float(
            1000.0
            * evaluated["frozen_output_trust"][split]["tf_log"]
            .sum(axis=1)[:, 2]
            .mean()
        )

    horizon_checkpoints: dict[str, Any] = {}
    for horizon in (4, 8, 16, 32):
        path = args.output / f"best-trust-rollout-h{horizon:02d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["extra_state"]["trust_region"]
        horizon_checkpoints[str(horizon)] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "validation_terminal_dx": float(payload["best_metric"]),
            "constraint_ratio": float(state["best_constraint_ratio"]),
            "epoch": int(payload["epoch"]),
            "horizon_epoch": int(payload["horizon_epoch"]),
        }

    test_objects = candidate["test"]["object_index"]
    terminal_new = candidate["test"]["dx"][:, -1]
    terminal_local = local["test"]["dx"][:, -1]
    terminal_old = old["test"]["dx"][:, -1]
    comparison = {
        "terminal_test_new_minus_local": _bootstrap_difference(
            terminal_new, terminal_local, test_objects
        ),
        "terminal_test_new_minus_old": _bootstrap_difference(
            terminal_new, terminal_old, test_objects
        ),
        "bias_distance": {},
        "per_object_test_new_minus_local": {
            object_id: (
                rollout["physical_trust"]["test"]["per_object"][object_id][
                    "terminal_dx"
                ]
                - rollout["frozen_local"]["test"]["per_object"][object_id][
                    "terminal_dx"
                ]
            )
            for object_id in rollout["physical_trust"]["test"]["per_object"]
        },
        "per_object_test_new_minus_old": {
            object_id: (
                rollout["physical_trust"]["test"]["per_object"][object_id][
                    "terminal_dx"
                ]
                - rollout["old_bptt"]["test"]["per_object"][object_id][
                    "terminal_dx"
                ]
            )
            for object_id in rollout["physical_trust"]["test"]["per_object"]
        },
    }
    for split in ("val", "test"):
        comparison["bias_distance"][split] = _bootstrap_bias_distance(
            candidate[split]["tf_log"].sum(axis=1)[:, 2],
            local[split]["tf_log"].sum(axis=1)[:, 2],
            old[split]["tf_log"].sum(axis=1)[:, 2],
            candidate[split]["object_index"],
        )

    history = _training_history(args.output)
    result = {
        "definition": {
            "physical_loss": "E_active d_X^2(R_theta(x*_k,u_{k+1}),x*_{k+1})",
            "constraint": "L_phys(theta) <= L_phys(theta_0)",
            "hpr": "(max(0,mu+rho*c)^2-mu^2)/(2*rho)",
            "dual_update": "mu <- max(0,mu+rho*c_epoch)",
            "teacher_forced_bias": "sum_k E[Log(inv(q*_{k+1}) R_theta(x*_k,u_{k+1})_q)]",
        },
        "contract": {
            "manifest_sha256": manifest.sha256(),
            "gripper_sha256": manifest.gripper_sha256,
            "active_index_sha256": _sha256(config.paths.active_index),
            "constraint_target": trust_state["constraint_target"],
            "constraint_baseline": trust_state["constraint_baseline"],
            "rho": trust_state["rho"],
            "seed": config.seed,
            "bootstrap_replicates": 10_000,
            "checkpoints": {
                arm: {"path": str(path.resolve()), "sha256": _sha256(path)}
                for arm, path in checkpoints.items()
            },
        },
        "horizon_checkpoints": horizon_checkpoints,
        "training_history": history,
        "physical_one_step": physical,
        "rollout": rollout,
        "signed_tf_b32_z_mm": signed,
        "comparison": comparison,
        "decision": {
            "final_constraint_satisfied": physical["physical_trust"]["train"][
                "loss"
            ]
            <= trust_state["constraint_baseline"] + 1e-8,
            "test_better_than_local": comparison[
                "terminal_test_new_minus_local"
            ]["ci95_upper"]
            < 0.0,
            "test_better_than_old_bptt": comparison[
                "terminal_test_new_minus_old"
            ]["ci95_upper"]
            < 0.0,
            "bias_preserved_better_on_val": comparison["bias_distance"]["val"][
                "ci95_upper"
            ]
            < 0.0,
            "bias_preserved_better_on_test": comparison["bias_distance"]["test"][
                "ci95_upper"
            ]
            < 0.0,
        },
    }

    _write_json(args.output / "results.json", result)
    arrays: dict[str, np.ndarray] = {}
    for arm, split_values in evaluated.items():
        for split, values in split_values.items():
            for name, value in values.items():
                arrays[f"{arm}_seed0_{split}_{name}"] = value
    temporary = args.output / "samples.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, args.output / "samples.npz")
    _plot(args.output, result, candidate)
    _plot_training(args.output, history)
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
