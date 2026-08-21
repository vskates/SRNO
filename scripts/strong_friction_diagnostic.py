#!/usr/bin/env python3
"""Run only the P0-B strong-friction ablation on paired SRNO transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from contact_memory_diagnostics import (
    _continuous_replay,
    _errors,
    _reset_successor,
    _summarize_errors,
    _take_transitions,
)
from markov_state_sufficiency import _select_transitions, _stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sim-config", type=Path, required=True)
    parser.add_argument("--baseline-samples", type=Path, required=True)
    parser.add_argument("--native-material-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--samples-per-object", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

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

        manifest = DatasetManifest.load(args.manifest)
        config = SimulatorConfig.load(args.sim_config)
        catalog = SimulatorAssetCatalog.load(config.catalog)
        gripper = GripperAsset.load(manifest.gripper_path)
        baseline = np.load(args.baseline_samples, allow_pickle=False)
        labels = baseline["object_labels"].tolist()
        objects = args.objects or list(manifest.splits["val"] + manifest.splits["test"])
        device = torch.device(config.device)

        parts: dict[str, list[np.ndarray]] = {}
        baseline_parts: list[np.ndarray] = []
        for object_index, object_id in enumerate(objects):
            baseline_index = labels.index(object_id)
            mask = baseline["object_index"] == baseline_index
            available = int(mask.sum())
            selected_full = _select_transitions(
                manifest,
                object_id,
                count=available,
                seed=args.seed + baseline_index,
                joint_names=gripper.joint_names,
            )
            if not np.array_equal(
                baseline["trajectory"][mask], selected_full.trajectory
            ) or not np.array_equal(
                baseline["current_step"][mask], selected_full.current_step
            ):
                raise ValueError(f"{object_id}: selection differs from baseline")
            selected = _take_transitions(selected_full, args.samples_per_object)
            baseline_parts.append(
                baseline["reset_dx"][mask][: args.samples_per_object]
            )

            print(f"[P0-B] {object_id}: continuous", flush=True)
            continuous = _continuous_replay(
                app,
                config,
                catalog,
                gripper,
                selected,
                collision_system="PCM",
                paired_explicit_zero=False,
                disable_strong_friction=True,
                native_material_library=args.native_material_library,
            )
            print(f"[P0-B] {object_id}: fresh A", flush=True)
            fresh = _reset_successor(
                app,
                config,
                catalog,
                gripper,
                selected,
                current_position=continuous.current_position,
                current_quaternion_wxyz=continuous.current_quaternion_wxyz,
                current_joint=continuous.current_joint,
                collision_system="PCM",
                precondition=False,
                disable_strong_friction=True,
                native_material_library=args.native_material_library,
            )
            print(f"[P0-B] {object_id}: fresh B", flush=True)
            repeat = _reset_successor(
                app,
                config,
                catalog,
                gripper,
                selected,
                current_position=continuous.current_position,
                current_quaternion_wxyz=continuous.current_quaternion_wxyz,
                current_joint=continuous.current_joint,
                collision_system="PCM",
                precondition=False,
                disable_strong_friction=True,
                native_material_library=args.native_material_library,
            )
            errors = _errors(
                fresh.successor_position,
                fresh.successor_quaternion_wxyz,
                fresh.successor_joint,
                continuous.successor_position,
                continuous.successor_quaternion_wxyz,
                continuous.successor_joint,
                length_scale=gripper.length_scale,
                joint_scale=gripper.joint_travel_range,
                device=device,
            )
            repeat_errors = _errors(
                repeat.successor_position,
                repeat.successor_quaternion_wxyz,
                repeat.successor_joint,
                fresh.successor_position,
                fresh.successor_quaternion_wxyz,
                fresh.successor_joint,
                length_scale=gripper.length_scale,
                joint_scale=gripper.joint_travel_range,
                device=device,
            )
            for name, value in errors.items():
                parts.setdefault(f"sf_off_{name}", []).append(value)
            for name, value in repeat_errors.items():
                parts.setdefault(f"repeat_{name}", []).append(value)
            parts.setdefault("valid", []).append(
                continuous.current_settled
                & continuous.successor_settled
                & fresh.successor_settled
                & repeat.successor_settled
            )
            parts.setdefault("object_index", []).append(
                np.full(len(selected.current_step), object_index, dtype=np.int16)
            )
            parts.setdefault("trajectory", []).append(selected.trajectory)
            parts.setdefault("current_step", []).append(selected.current_step)

        arrays = {name: np.concatenate(value) for name, value in parts.items()}
        arrays["object_labels"] = np.asarray(objects)
        baseline_dx = np.concatenate(baseline_parts)
        valid = arrays["valid"].astype(bool)
        summary = {
            "definition": "continuous versus cold-fresh successor; same q_k,r_k and command",
            "changed": "PxMaterialFlag::eDISABLE_STRONG_FRICTION=true",
            "unchanged": "friction coefficients, combine modes, PCM, scene and collector",
            "objects": objects,
            "samples_per_object": args.samples_per_object,
            "settled_samples": int(valid.sum()),
            "baseline_cold_pcm_dx": _stats(baseline_dx[valid]),
            "strong_friction_off": _summarize_errors(arrays, "sf_off", valid),
            "fresh_repeatability": _summarize_errors(arrays, "repeat", valid),
            "mean_dx_ratio_to_baseline": float(
                arrays["sf_off_dx"][valid].mean() / baseline_dx[valid].mean()
            ),
            "fraction_improved_over_baseline": float(
                np.mean(arrays["sf_off_dx"][valid] < baseline_dx[valid])
            ),
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "results.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        np.savez_compressed(
            args.output / "samples.npz", **arrays, baseline_cold_pcm_dx=baseline_dx
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    finally:
        app.close()
        watchdog.stop()


if __name__ == "__main__":
    main()
