#!/usr/bin/env python3
"""Measure continuous repeatability, fresh-reset sensitivity and model gap.

This is the focused material-v2 diagnostic from the architecture decision
note.  It intentionally does not run the older strong-friction, SAT or warm
preconditioning ablations.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from contact_memory_diagnostics import (
    _continuous_replay,
    _errors,
    _reset_successor,
)
from markov_state_sufficiency import (
    TransitionSet,
    _model_errors,
    _select_transitions,
    _stats,
)


def _finite(*values: np.ndarray) -> np.ndarray:
    valid = np.ones(len(values[0]), dtype=bool)
    for value in values:
        valid &= np.isfinite(value).reshape(len(value), -1).all(axis=1)
    return valid


def _replayed_transition(selected: TransitionSet, replay: Any, model: Any) -> TransitionSet:
    """Use the exact continuous-A observable state and successor for E_model."""

    target_joint = replay.successor_joint
    with __import__("torch").no_grad():
        aperture = (
            model.aperture_from_joints(
                __import__("torch").from_numpy(target_joint).to(model.surface_local_points.device)
            )
            .cpu()
            .numpy()
        )
    return replace(
        selected,
        current_position=replay.current_position,
        current_quaternion_xyzw=replay.current_quaternion_wxyz[:, (1, 2, 3, 0)],
        current_joint=replay.current_joint,
        preserve_position=replay.successor_position,
        preserve_quaternion_xyzw=replay.successor_quaternion_wxyz[:, (1, 2, 3, 0)],
        preserve_joint=target_joint,
        preserve_aperture=aperture,
    )


def _cat(rows: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.concatenate(values, axis=0) for name, values in rows.items()}


def _append(rows: dict[str, list[np.ndarray]], prefix: str, values: dict[str, np.ndarray]) -> None:
    for name, value in values.items():
        rows.setdefault(f"{prefix}_{name}", []).append(np.asarray(value))


def _summary(rows: dict[str, np.ndarray], prefix: str, valid: np.ndarray) -> dict[str, Any]:
    return {
        name: _stats(rows[f"{prefix}_{name}"][valid])
        for name in ("dx", "translation_m", "rotation_rad", "joint_rmse_over_travel")
    }


def _plot(output: Path, rows: dict[str, np.ndarray], common: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = rows["object_labels"].tolist()
    object_index = rows["object_index"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    values = [rows[f"{name}_dx"][common] for name in ("continuous", "reset", "model")]
    axes[0, 0].boxplot(values, tick_labels=[r"$D_{cont}$", r"$D_{reset}$", r"$E_{model}$"])
    axes[0, 0].set_ylabel(r"$d_X$")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Same material-v2 transitions")

    upper = float(np.quantile(np.concatenate((values[0], values[2])), 0.99))
    axes[0, 1].scatter(values[0], values[2], c=rows["current_step"][common], cmap="viridis", s=32)
    axes[0, 1].plot((0, upper), (0, upper), "k--", linewidth=1)
    axes[0, 1].set_xlim(0, upper)
    axes[0, 1].set_ylim(0, upper)
    axes[0, 1].set_xlabel(r"simulator floor $D_{cont}$")
    axes[0, 1].set_ylabel(r"one-step model error $E_{model}$")
    axes[0, 1].set_title("colour = command step")

    grouped = [rows["continuous_dx"][common & (object_index == i)] for i in range(len(labels))]
    axes[1, 0].boxplot(grouped, tick_labels=[str(x)[:25] for x in labels])
    axes[1, 0].tick_params(axis="x", rotation=18)
    axes[1, 0].set_ylabel(r"$D_{cont}$")
    axes[1, 0].set_yscale("log")

    bins = np.linspace(0.0, float(np.quantile(np.concatenate(values), 0.99)), 32)
    for value, label in zip(values, (r"$D_{cont}$", r"$D_{reset}$", r"$E_{model}$"), strict=True):
        axes[1, 1].hist(value, bins=bins, alpha=0.6, label=label)
    axes[1, 1].set_xlabel(r"$d_X$")
    axes[1, 1].set_ylabel("transitions")
    axes[1, 1].legend()

    fig.savefig(output / "material_v2_simulator_floor.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sim-config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--samples-per-object", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.samples_per_object <= 0:
        parser.error("--samples-per-object must be positive")

    from isaaclab.app import AppLauncher
    from srno.sim.memory_guard import MemoryWatchdog

    watchdog = MemoryWatchdog(14.0, 0.25)
    watchdog.start()
    app = AppLauncher({"headless": True, "device": "cuda:0"}).app
    try:
        import torch

        from srno.data.schema import DatasetManifest
        from srno.geometry.gripper import GripperAsset
        from srno.sim.assets import SimulatorAssetCatalog
        from srno.sim.config import SimulatorConfig
        from srno.training.checkpoint import load_checkpoint
        from srno.training.config import ExperimentConfig
        from srno.training.engine import _build_model

        manifest = DatasetManifest.load(args.manifest)
        sim_config = SimulatorConfig.load(args.sim_config)
        train_config = ExperimentConfig.load(args.train_config)
        catalog = SimulatorAssetCatalog.load(sim_config.catalog)
        gripper = GripperAsset.load(manifest.gripper_path)
        if gripper.sha256() != manifest.gripper_sha256:
            raise ValueError("manifest gripper hash mismatch")
        device = torch.device(sim_config.device)
        model = _build_model(train_config, manifest, device)
        checkpoint = load_checkpoint(args.checkpoint, model=model, map_location=device)
        if checkpoint["manifest_sha256"] != manifest.sha256():
            raise ValueError("checkpoint was trained on a different manifest")
        model.eval()

        objects = args.objects or list(manifest.splits["test"])
        unknown = set(objects) - set(manifest.object_locations())
        if unknown:
            raise ValueError(f"objects are absent from manifest: {sorted(unknown)}")

        collected: dict[str, list[np.ndarray]] = {}
        object_summaries: dict[str, Any] = {}
        for object_index, object_id in enumerate(objects):
            selected = _select_transitions(
                manifest,
                object_id,
                count=args.samples_per_object,
                seed=args.seed + object_index,
                joint_names=gripper.joint_names,
            )
            print(f"[FLOOR] {object_id}: independent continuous replay A", flush=True)
            replay_a = _continuous_replay(
                app, sim_config, catalog, gripper, selected,
                collision_system="PCM", paired_explicit_zero=False,
            )
            print(f"[FLOOR] {object_id}: independent continuous replay B", flush=True)
            replay_b = _continuous_replay(
                app, sim_config, catalog, gripper, selected,
                collision_system="PCM", paired_explicit_zero=False,
            )
            print(f"[FLOOR] {object_id}: fresh reset from continuous-A x_k", flush=True)
            reset = _reset_successor(
                app, sim_config, catalog, gripper, selected,
                current_position=replay_a.current_position,
                current_quaternion_wxyz=replay_a.current_quaternion_wxyz,
                current_joint=replay_a.current_joint,
                collision_system="PCM", precondition=False,
            )

            continuous = _errors(
                replay_a.successor_position, replay_a.successor_quaternion_wxyz, replay_a.successor_joint,
                replay_b.successor_position, replay_b.successor_quaternion_wxyz, replay_b.successor_joint,
                length_scale=gripper.length_scale,
                joint_scale=gripper.joint_travel_range,
                device=device,
            )
            reset_errors = _errors(
                replay_a.successor_position, replay_a.successor_quaternion_wxyz, replay_a.successor_joint,
                reset.successor_position, reset.successor_quaternion_wxyz, reset.successor_joint,
                length_scale=gripper.length_scale,
                joint_scale=gripper.joint_travel_range,
                device=device,
            )
            replayed = _replayed_transition(selected, replay_a, model)
            with torch.no_grad():
                model_errors = _model_errors(replayed, manifest, gripper, model, device=device)

            valid = (
                replay_a.current_settled & replay_a.successor_settled
                & replay_b.current_settled & replay_b.successor_settled
                & reset.successor_settled
                & _finite(
                    replay_a.successor_position, replay_a.successor_quaternion_wxyz, replay_a.successor_joint,
                    replay_b.successor_position, replay_b.successor_quaternion_wxyz, replay_b.successor_joint,
                    reset.successor_position, reset.successor_quaternion_wxyz, reset.successor_joint,
                    model_errors["dx"],
                )
            )
            collected.setdefault("object_index", []).append(np.full(len(valid), object_index, dtype=np.int32))
            collected.setdefault("current_step", []).append(selected.current_step.astype(np.int32))
            collected.setdefault("valid", []).append(valid)
            _append(collected, "continuous", continuous)
            _append(collected, "reset", reset_errors)
            _append(collected, "model", model_errors)
            object_summaries[object_id] = {
                "requested": len(valid),
                "common_valid": int(valid.sum()),
                "continuous": {name: _stats(value[valid]) for name, value in continuous.items()},
                "reset": {name: _stats(value[valid]) for name, value in reset_errors.items()},
                "model": {name: _stats(value[valid]) for name, value in model_errors.items() if name in continuous},
            }

        rows = _cat(collected)
        rows["object_labels"] = np.asarray(objects)
        common = rows["valid"].astype(bool)
        if not common.any():
            raise RuntimeError("no transition was valid in all three branches")
        floor = float(np.mean(rows["continuous_dx"][common]))
        model_error = float(np.mean(rows["model_dx"][common]))
        result = {
            "definition": {
                "dx": "sqrt((||dp||/L)^2 + theta(R,R*)^2 + mean(((r-r*)/travel)^2))",
                "D_cont": "dX(continuous_A_successor, continuous_B_successor)",
                "D_reset": "dX(continuous_A_successor, fresh_successor_from_A_current)",
                "E_model": "dX(R_theta(continuous_A_current), continuous_A_successor)",
            },
            "configuration": {
                "manifest": str(args.manifest.resolve()),
                "manifest_sha256": manifest.sha256(),
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "objects": objects,
                "samples_per_object": args.samples_per_object,
                "friction": {"static": 2.4, "dynamic": 2.0, "combine": "min"},
            },
            "requested": int(len(common)),
            "common_valid": int(common.sum()),
            "E_floor": floor,
            "E_model": model_error,
            "gamma": model_error / floor if floor > 0.0 else float("inf"),
            "continuous": _summary(rows, "continuous", common),
            "reset": _summary(rows, "reset", common),
            "model": _summary(rows, "model", common),
            "by_object": object_summaries,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        np.savez_compressed(args.output / "samples.npz", **rows)
        _plot(args.output, rows, common)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    finally:
        app.close()
        watchdog.stop()


if __name__ == "__main__":
    main()
