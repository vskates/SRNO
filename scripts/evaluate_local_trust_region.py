#!/usr/bin/env python3
"""Evaluate the seed-0 local-resolvent trust-region BPTT ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", "utf-8")
    os.replace(temporary, path)


def _state_at(states: Any, step: int) -> Any:
    from srno.types import PoseState

    return PoseState(
        states.rotation[:, step],
        states.position[:, step],
        states.joint_position[:, step],
    )


def _evaluate_trust_checkpoint(
    config: Any,
    manifest: Any,
    checkpoint_path: Path,
    *,
    split: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import torch

    try:
        from contact_composition_diagnostics import _pose_log_error
    except ModuleNotFoundError:  # Imported as ``scripts.*`` by tests.
        from scripts.contact_composition_diagnostics import _pose_log_error
    from srno.data.dataset import H5ObjectDataset, TrajectoryBatch, make_dataloader
    from srno.geometry.se3 import rotation_geodesic_angle
    from srno.losses import state_error
    from srno.training.checkpoint import load_checkpoint
    from srno.training.engine import _autocast, _build_model
    from srno.types import PoseState

    device = torch.device(config.device)
    model = _build_model(config, manifest, device)
    checkpoint = load_checkpoint(checkpoint_path, model=model, map_location=device)
    if checkpoint.get("stage") != "trust_rollout" or int(checkpoint.get("horizon")) != 32:
        raise ValueError("candidate must be a trust_rollout H32 checkpoint")
    if checkpoint.get("manifest_sha256") != manifest.sha256():
        raise ValueError("candidate manifest hash mismatch")
    if checkpoint.get("gripper_sha256") != manifest.gripper_sha256:
        raise ValueError("candidate gripper hash mismatch")
    model.eval()
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
    object_tf: list[np.ndarray] = []
    object_ar_dx: list[np.ndarray] = []
    object_ar_translation: list[np.ndarray] = []
    object_ar_rotation: list[np.ndarray] = []
    object_ar_joints: list[np.ndarray] = []
    object_ids: list[str] = []
    try:
        with torch.no_grad():
            for raw_batch in loader:
                assert isinstance(raw_batch, TrajectoryBatch)
                batch = raw_batch.to(device)
                tf: list[torch.Tensor] = []
                for step in range(32):
                    current = _state_at(batch.states, step)
                    target = _state_at(batch.states, step + 1)
                    with _autocast(config, device):
                        prediction = model.forward_step(
                            current,
                            batch.command_schedule[step + 1],
                            batch.sdf,
                        )
                    assert isinstance(prediction, PoseState)
                    tf.append(_pose_log_error(target, prediction))
                tf_values = torch.stack(tf, dim=1)
                object_tf.append(tf_values.mean(dim=0).cpu().numpy())
                object_ids.append(raw_batch.object_ids[0])

                if split in {"val", "test"}:
                    with _autocast(config, device):
                        rollout = model.rollout(
                            _state_at(batch.states, 0),
                            batch.command_schedule[1:],
                            batch.sdf,
                        )
                    dx_rows: list[torch.Tensor] = []
                    translation_rows: list[torch.Tensor] = []
                    rotation_rows: list[torch.Tensor] = []
                    joint_rows: list[torch.Tensor] = []
                    for step in range(33):
                        predicted = _state_at(rollout, step)
                        target = _state_at(batch.states, step)
                        state_sq, _, _, _ = state_error(
                            predicted,
                            target,
                            length_scale=model.length_scale,
                            joint_scale=model.joint_travel_range,
                        )
                        dx_rows.append(state_sq.sqrt().mean())
                        translation_rows.append(
                            torch.linalg.vector_norm(
                                predicted.position - target.position, dim=-1
                            ).mean()
                        )
                        rotation_rows.append(
                            rotation_geodesic_angle(
                                predicted.rotation, target.rotation
                            ).mean()
                        )
                        joint_rows.append(
                            torch.sqrt(
                                (
                                    (predicted.joint_position - target.joint_position)
                                    / model.joint_travel_range
                                )
                                .square()
                                .mean(dim=-1)
                            ).mean()
                        )
                    object_ar_dx.append(torch.stack(dx_rows).cpu().numpy())
                    object_ar_translation.append(
                        torch.stack(translation_rows).cpu().numpy()
                    )
                    object_ar_rotation.append(torch.stack(rotation_rows).cpu().numpy())
                    object_ar_joints.append(torch.stack(joint_rows).cpu().numpy())
    finally:
        dataset.close()

    tf_curve = np.mean(np.stack(object_tf), axis=0)
    arrays: dict[str, np.ndarray] = {"tf_mean_curve": tf_curve}
    summary: dict[str, Any] = {
        "objects": object_ids,
        "trajectories": 100 * len(object_ids),
        "tf_b32_v_m_w_rad": tf_curve.sum(axis=0).tolist(),
        "tf_b32_z_m": float(tf_curve[:, 2].sum()),
    }
    if object_ar_dx:
        curves = {
            "ar_dx_curve": np.mean(np.stack(object_ar_dx), axis=0),
            "ar_translation_m_curve": np.mean(
                np.stack(object_ar_translation), axis=0
            ),
            "ar_rotation_rad_curve": np.mean(np.stack(object_ar_rotation), axis=0),
            "ar_joint_rmse_curve": np.mean(np.stack(object_ar_joints), axis=0),
        }
        arrays.update(curves)
        summary.update(
            {
                "terminal_dx_equal_object": float(curves["ar_dx_curve"][-1]),
                "terminal_translation_m_equal_object": float(
                    curves["ar_translation_m_curve"][-1]
                ),
                "terminal_rotation_rad_equal_object": float(
                    curves["ar_rotation_rad_curve"][-1]
                ),
                "terminal_joint_rmse_equal_object": float(
                    curves["ar_joint_rmse_curve"][-1]
                ),
            }
        )
    return summary, arrays


def _cached_tf_bz(path: Path) -> float:
    with np.load(path, allow_pickle=False) as archive:
        values = np.asarray(archive["tf_log"], dtype=np.float64)
    return float(values[:, :, 2].mean(axis=0).sum())


def _plot(output: Path, results: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = ("frozen_local", "trust_region", "old_bptt")
    labels = ("frozen local", "trust-region", "old BPTT")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    x = np.arange(3)
    width = 0.36
    val = [results["evaluation"][arm]["val"]["terminal_dx"] for arm in arms]
    test = [results["evaluation"][arm]["test"]["terminal_dx"] for arm in arms]
    axes[0].bar(x - width / 2, val, width, label="val")
    axes[0].bar(x + width / 2, test, width, label="test")
    axes[0].set_xticks(x, labels, rotation=15)
    axes[0].set(ylabel=r"terminal $d_X$", title="H32 rollout")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    splits = ("train", "val", "test")
    width = 0.25
    for index, arm in enumerate(arms):
        axes[1].bar(
            np.arange(3) + (index - 1) * width,
            [results["signed_tf_b32_z_mm"][arm][split] for split in splits],
            width,
            label=labels[index],
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(np.arange(3), splits)
    axes[1].set(ylabel=r"$B_{32,z}^{TF}$, mm", title="Signed local bias")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)

    steps = np.arange(33)
    for split, color in (("val", "#4472C4"), ("test", "#C00000")):
        axes[2].plot(steps, arrays[f"{split}_ar_dx_curve"], label=split, color=color)
    axes[2].set(xlabel="step", ylabel=r"$d_X(k)$", title="Trust-region rollout")
    axes[2].grid(alpha=0.2)
    axes[2].legend()
    fig.savefig(output / "trust_region_ablation.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/ablation-local-trust-region.toml")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/ablation-local-trust-region/seed-0/best-trust-rollout-h32.pt"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/ablation-local-trust-region/seed-0")
    )
    args = parser.parse_args()

    from srno.data.schema import DatasetManifest
    from srno.training.checkpoint import load_checkpoint
    from srno.training.config import ExperimentConfig
    from srno.training.engine import _build_model
    import torch

    config = ExperimentConfig.load(args.config)
    manifest = DatasetManifest.load(config.paths.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    trust: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        trust[split], split_arrays = _evaluate_trust_checkpoint(
            config, manifest, args.checkpoint, split=split
        )
        arrays.update({f"{split}_{key}": value for key, value in split_arrays.items()})

    evaluation: dict[str, Any] = {}
    paths = {
        "frozen_local": "local",
        "trust_region": None,
        "old_bptt": "old-bptt",
    }
    for arm, stem in paths.items():
        evaluation[arm] = {}
        for split in ("val", "test"):
            path = (
                args.output / f"eval-{split}.json"
                if stem is None
                else args.output / f"eval-{stem}-{split}.json"
            )
            evaluation[arm][split] = json.loads(path.read_text("utf-8"))

    signed: dict[str, dict[str, float]] = {
        "frozen_local": {},
        "trust_region": {},
        "old_bptt": {},
    }
    for split in ("train", "val", "test"):
        signed["frozen_local"][split] = 1000.0 * _cached_tf_bz(
            Path("runs/local-vs-h32-signed-bias-v1/local")
            / f"seed-0-{split}.npz"
        )
        signed["old_bptt"][split] = 1000.0 * _cached_tf_bz(
            Path("runs/quasistatic-refinement-bias-v1/bias")
            / f"seed-0-{split}.npz"
        )
        signed["trust_region"][split] = 1000.0 * trust[split]["tf_b32_z_m"]

    device = torch.device(config.device)
    model = _build_model(config, manifest, device)
    checkpoint = load_checkpoint(args.checkpoint, model=model, map_location=device)
    trust_state = checkpoint["extra_state"]["trust_region"]
    horizon_checkpoints: dict[str, Any] = {}
    for horizon in (4, 8, 16, 32):
        path = args.output / f"best-trust-rollout-h{horizon:02d}.pt"
        raw = torch.load(path, map_location="cpu", weights_only=False)
        horizon_checkpoints[str(horizon)] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "validation_terminal_dx": float(raw["best_metric"]),
            "constraint_ratio": float(
                raw["extra_state"]["trust_region"]["best_constraint_ratio"]
            ),
            "epoch": int(raw["epoch"]),
            "horizon_epoch": int(raw["horizon_epoch"]),
        }

    result = {
        "definition": {
            "rollout_objective": "mean_{k=1..H} d_X^2(xhat_k,x*_k) + lambda_K L_K",
            "functional_drift": "E_active d_X^2(R_theta(x*_k,u_{k+1}), stopgrad(R_theta0(x*_k,u_{k+1})))",
            "constraint": "D_loc <= epsilon_dx^2",
            "hpr_penalty": "(max(0,mu+rho*c)^2-mu^2)/(2*rho), c=D_loc-epsilon_dx^2",
            "dual_update": "mu <- max(0,mu+rho*c_epoch)",
            "teacher_forced_bias": "B_32^TF=sum_k E[Log(inv(q*_{k+1}) R_theta(x*_k,u_{k+1})_q)]",
        },
        "contract": {
            "manifest_sha256": manifest.sha256(),
            "gripper_sha256": manifest.gripper_sha256,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "local_checkpoint_sha256": trust_state["local_checkpoint_sha256"],
            "reference_sha256": trust_state["reference_sha256"],
            "epsilon_dx": trust_state["epsilon_dx"],
            "rho": trust_state["rho"],
            "seed": config.seed,
            "architecture": "gap + aperture (unchanged)",
        },
        "horizon_checkpoints": horizon_checkpoints,
        "evaluation": evaluation,
        "trust_region_diagnostics": trust,
        "signed_tf_b32_z_mm": signed,
        "decision": {
            "feasible_h16_update_found": horizon_checkpoints["16"]["horizon_epoch"] >= 0,
            "feasible_h32_update_found": horizon_checkpoints["32"]["horizon_epoch"] >= 0,
            "final_constraint_satisfied": horizon_checkpoints["32"]["constraint_ratio"] <= 1.0,
        },
    }
    _write_json(args.output / "results.json", result)
    temporary = args.output / "samples.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, args.output / "samples.npz")
    _plot(args.output, result, arrays)
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
