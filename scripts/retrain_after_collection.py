#!/usr/bin/env python3
"""Wait for a collector, calibrate its dataset, and run SRNO-r retraining."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import h5py

from srno.data.schema import DatasetManifest
from srno.sim.assets import SimulatorAssetCatalog
from srno.sim.config import SimulatorConfig


def _run(command: list[str], *, capture: bool = False) -> str:
    print("[SRNO pipeline] $ " + " ".join(command), flush=True)
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    if capture:
        assert result.stdout is not None
        print(result.stdout, end="", flush=True)
        return result.stdout
    return ""


def _systemd_value(unit: str, property_name: str) -> str:
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            f"--property={property_name}",
            "--value",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cannot inspect collector unit {unit}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _wait_for_collector(unit: str, interval_s: float) -> None:
    while _systemd_value(unit, "ActiveState") in {"active", "activating", "reloading"}:
        print("[SRNO pipeline] collector is still running", flush=True)
        time.sleep(interval_s)
    result = _systemd_value(unit, "Result")
    if result != "success":
        raise RuntimeError(f"collector unit {unit} ended with Result={result!r}")
    print("[SRNO pipeline] collector completed successfully", flush=True)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _audit_trajectory_identity(
    manifest: DatasetManifest, *, trajectories_per_object: int
) -> None:
    expected_physics = manifest.physics.canonical_json()
    for shard in manifest.shards:
        with h5py.File(manifest.shard_path(shard), "r") as handle:
            if str(handle.attrs.get("physics_metadata_json", "")) != expected_physics:
                raise RuntimeError(f"physics fingerprint mismatch in {shard.path}")
            for group_name in handle["objects"]:
                group = handle[f"objects/{group_name}"]
                if group["position"].shape[0] != trajectories_per_object:
                    raise RuntimeError(f"trajectory count mismatch in {shard.path}")
                if "joint_position" not in group:
                    raise RuntimeError(f"actual joints are missing in {shard.path}")
                indices = group["source_pose_index"][...]
                if len(set(map(int, indices))) != trajectories_per_object:
                    raise RuntimeError(f"seed poses are not unique in {shard.path}")


def _training_toml(
    *, manifest: Path, active_index: Path, output_dir: Path, admissible_gap_m: float
) -> str:
    return f'''seed = 0
device = "cuda"

[paths]
manifest = "{manifest}"
active_index = "{active_index}"
output_dir = "{output_dir}"

[model]
hidden_dim = 64

[loss]
lambda_rotation = 1.0
lambda_joints = 1.0
lambda_feasibility = 1.0
# Q99.5 raw cooked-SDF residual at GT pose and actual joints. The 2.56 mm
# PhysX contact envelope is intentionally excluded from geometric feasibility.
admissible_gap_m = {admissible_gap_m:.17g}

[optimizer]
learning_rate = 3e-4
weight_decay = 1e-4
gradient_clip = 1.0
warmup_fraction = 0.05

[loader]
objects_per_batch = 4
local_samples_per_object = 256
rollout_trajectories_per_object = 8
workers = 4

[training]
local_epochs = 100
rollout_epochs_per_horizon = 25
rollout_horizons = [4, 8, 16, 32]
early_stopping_patience = 10
use_bfloat16 = true
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collector-unit", required=True)
    parser.add_argument("--sim-config", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--poll-interval", type=float, default=30.0)
    args = parser.parse_args()
    if not 1.0 <= args.poll_interval <= 60.0:
        raise ValueError("poll interval must be in [1, 60] seconds")

    config = SimulatorConfig.load(args.sim_config)
    catalog = SimulatorAssetCatalog.load(config.catalog)
    manifest_path = config.output_dir / "manifest.json"
    active_index_path = config.output_dir / "active-index.npz"
    failure_path = config.output_dir / "collection-failure.txt"
    training_config_path = args.training_config.resolve()
    run_dir = args.run_dir.resolve()

    _wait_for_collector(args.collector_unit, args.poll_interval)
    if failure_path.exists():
        raise RuntimeError(f"collector left a failure report: {failure_path}")

    validation_raw = _run(
        ["srno", "dataset", "validate", str(manifest_path)], capture=True
    )
    validation = json.loads(validation_raw)
    expected_objects = len(catalog.object_ids)
    expected_trajectories = expected_objects * config.trajectories_per_object
    if validation != {
        "objects": expected_objects,
        "trajectories": expected_trajectories,
        "diagnostics": [
            "actuator_effort",
            "angular_velocity",
            "contact_count",
            "linear_velocity",
            "settling_substeps",
        ],
    }:
        raise RuntimeError(f"unexpected dataset validation report: {validation}")
    manifest = DatasetManifest.load(manifest_path)
    _audit_trajectory_identity(
        manifest, trajectories_per_object=config.trajectories_per_object
    )

    calibration_raw = _run(
        [
            "srno",
            "dataset",
            "calibrate-gate",
            str(manifest_path),
            "--target-recall",
            "0.995",
            "--device",
            "cuda:0",
        ],
        capture=True,
    )
    calibration = json.loads(calibration_raw)
    if float(calibration["contact_recall"]) < 0.995:
        raise RuntimeError(f"contact calibration missed its recall target: {calibration}")
    calibration["source_manifest_sha256"] = manifest.sha256()
    calibration["physics"] = manifest.physics.to_dict()
    _atomic_json(config.output_dir / "calibration-material-v2.json", calibration)

    manifest_payload = manifest.to_dict()
    manifest_payload["delta_gate_m"] = float(calibration["recommended_delta_gate_m"])
    _atomic_json(manifest_path, manifest_payload)
    _run(["srno", "dataset", "validate", str(manifest_path)])
    _run(
        [
            "srno",
            "dataset",
            "build-active-index",
            str(manifest_path),
            "--output",
            str(active_index_path),
            "--device",
            "cuda:0",
        ]
    )

    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"refusing to mix retraining outputs in non-empty {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    training_config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_config = training_config_path.with_suffix(".toml.tmp")
    temporary_config.write_text(
        _training_toml(
            manifest=manifest_path.resolve(),
            active_index=active_index_path.resolve(),
            output_dir=run_dir,
            admissible_gap_m=float(calibration["recommended_admissible_gap_m"]),
        ),
        encoding="utf-8",
    )
    os.replace(temporary_config, training_config_path)

    _run(["srno", "train", "--config", str(training_config_path), "--stage", "local"])
    best_local = run_dir / "best-local.pt"
    _run(
        [
            "srno",
            "train",
            "--config",
            str(training_config_path),
            "--stage",
            "rollout",
            "--resume",
            str(best_local),
        ]
    )
    best_rollout = run_dir / "best-rollout.pt"
    _run(
        [
            "srno",
            "evaluate",
            "--config",
            str(training_config_path),
            "--checkpoint",
            str(best_rollout),
            "--split",
            "val",
            "--output",
            str(run_dir / "evaluation-val.json"),
        ]
    )
    _run(
        [
            "srno",
            "evaluate",
            "--config",
            str(training_config_path),
            "--checkpoint",
            str(best_rollout),
            "--split",
            "test",
            "--output",
            str(run_dir / "evaluation-test.json"),
        ]
    )
    _atomic_json(
        run_dir / "pipeline-complete.json",
        {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": DatasetManifest.load(manifest_path).sha256(),
            "calibration": calibration,
            "best_local": str(best_local),
            "best_rollout": str(best_rollout),
        },
    )
    print("[SRNO pipeline] retraining and evaluation completed", flush=True)


if __name__ == "__main__":
    main()
