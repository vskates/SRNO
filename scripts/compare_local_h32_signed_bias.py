#!/usr/bin/env python3
"""Compare frozen local and H32 checkpoints by signed teacher-forced pose bias.

This is deliberately an inference-only diagnostic.  It neither trains a model
nor launches Isaac Sim.  The H32 predictions are reused from the frozen
quasistatic-refinement diagnostic after their hashes and sample ordering have
been verified; only the local checkpoints are evaluated here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from contact_composition_diagnostics import (
        _checkpoint_config,
        _load_model,
        _pose_log_error,
    )
    from quasistatic_refinement_bias import (
        BIAS_STEP_BANDS,
        BOOTSTRAP_SEED,
        EXPECTED_SPLITS,
        MODEL_SEEDS,
        _atomic_savez,
        _hierarchical_bias_bootstrap,
        _mean_by_object,
        _persistent_bias,
        _sha256,
        _write_json,
    )
except ModuleNotFoundError:  # Imported as ``scripts.*`` by pytest.
    from scripts.contact_composition_diagnostics import (
        _checkpoint_config,
        _load_model,
        _pose_log_error,
    )
    from scripts.quasistatic_refinement_bias import (
        BIAS_STEP_BANDS,
        BOOTSTRAP_SEED,
        EXPECTED_SPLITS,
        MODEL_SEEDS,
        _atomic_savez,
        _hierarchical_bias_bootstrap,
        _mean_by_object,
        _persistent_bias,
        _sha256,
        _write_json,
    )


def _tf_feature_vector(tf_log: np.ndarray) -> np.ndarray:
    """Return four band means followed by cumulative B32, shape [..., 30]."""

    values = np.asarray(tf_log, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (32, 6):
        raise ValueError("tf_log must have shape [trajectory, 32, 6]")
    bands = np.concatenate(
        [
            values[:, lower - 1 : upper].mean(axis=1)
            for lower, upper in BIAS_STEP_BANDS
        ],
        axis=-1,
    )
    return np.concatenate((bands, values.sum(axis=1)), axis=-1)


def _classify_split_vz(
    local: dict[str, np.ndarray],
    h32: dict[str, np.ndarray],
    delta: dict[str, np.ndarray],
) -> str:
    """Classify whether positive cumulative z-bias predates rollout training."""

    index = 24 + 2
    local_lower = float(local["ci95_lower"][index])
    local_upper = float(local["ci95_upper"][index])
    h32_lower = float(h32["ci95_lower"][index])
    delta_lower = float(delta["ci95_lower"][index])
    if local_lower > 0.0:
        return "bias_already_in_local"
    if local_lower <= 0.0 <= local_upper and h32_lower > 0.0 and delta_lower > 0.0:
        return "bias_introduced_by_rollout"
    if local_lower <= 0.0 <= local_upper and h32_lower > 0.0:
        return "h32_only_but_paired_difference_inconclusive"
    return "inconclusive"


def _evaluate_local_tf_split(
    model: Any,
    config: Any,
    manifest: Any,
    *,
    split: str,
    device: Any,
    global_object_index: dict[str, int],
) -> dict[str, np.ndarray]:
    import torch

    from srno.data.dataset import H5ObjectDataset, TrajectoryBatch, make_dataloader
    from srno.training.engine import _autocast
    from srno.types import PoseState

    rows: dict[str, list[np.ndarray]] = {}
    dataset = H5ObjectDataset(manifest, split=split)
    loader = make_dataloader(
        dataset,
        mode="rollout",
        objects_per_batch=1,
        samples_per_object=0,
        workers=0,
        seed=config.seed,
        shuffle=False,
    )
    try:
        with torch.no_grad():
            for raw_batch in loader:
                assert isinstance(raw_batch, TrajectoryBatch)
                object_id = raw_batch.object_ids[0]
                batch = raw_batch.to(device)
                tf_log: list[Any] = []
                tf_spatial: list[Any] = []
                for step in range(32):
                    current = PoseState(
                        batch.states.rotation[:, step],
                        batch.states.position[:, step],
                        batch.states.joint_position[:, step],
                    )
                    target = PoseState(
                        batch.states.rotation[:, step + 1],
                        batch.states.position[:, step + 1],
                        batch.states.joint_position[:, step + 1],
                    )
                    with _autocast(config, device):
                        prediction = model.forward_step(
                            current,
                            batch.command_schedule[step + 1],
                            batch.sdf,
                        )
                    tf_log.append(_pose_log_error(target, prediction))
                    tf_spatial.append(prediction.position - target.position)
                values = {
                    "tf_log": torch.stack(tf_log, dim=1).cpu().numpy(),
                    "tf_spatial": torch.stack(tf_spatial, dim=1).cpu().numpy(),
                    "object_index": np.full(
                        len(raw_batch.trajectory_index),
                        global_object_index[object_id],
                        dtype=np.int32,
                    ),
                    "trajectory": raw_batch.trajectory_index.cpu().numpy().astype(
                        np.int32
                    ),
                }
                for name, value in values.items():
                    rows.setdefault(name, []).append(np.asarray(value))
                print(
                    f"[LOCAL-TF] seed={config.seed} split={split} "
                    f"object={object_id} trajectories={len(raw_batch.trajectory_index)}",
                    flush=True,
                )
    finally:
        dataset.close()
    return {name: np.concatenate(values, axis=0) for name, values in rows.items()}


def _load_h32_contract(path: Path, manifest: Any) -> dict[str, Any]:
    contract_path = path / "checkpoint-contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("manifest_sha256") != manifest.sha256():
        raise ValueError("H32 bias manifest hash mismatch")
    if contract.get("gripper_sha256") != manifest.gripper_sha256:
        raise ValueError("H32 bias gripper hash mismatch")
    if contract.get("production_model") != {
        "contact_features": "gap",
        "global_conditioning": "aperture",
    }:
        raise ValueError("H32 bias does not use production gap+aperture model")
    for seed in MODEL_SEEDS:
        metadata = contract["checkpoints"][str(seed)]
        checkpoint = Path(metadata["path"])
        if not checkpoint.is_file() or _sha256(checkpoint) != metadata["sha256"]:
            raise ValueError(f"H32 checkpoint hash mismatch for seed {seed}")
    return contract


def _run_local_inference(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from srno.data.schema import DatasetManifest
    from srno.training.config import ExperimentConfig

    manifest = DatasetManifest.load(args.manifest)
    base = ExperimentConfig.load(args.train_config)
    if base.paths.manifest.resolve() != args.manifest.resolve():
        raise ValueError("train config and diagnostic manifest differ")
    h32_contract = _load_h32_contract(args.h32_bias_root, manifest)
    object_order = tuple(
        object_id
        for split in EXPECTED_SPLITS
        for object_id in manifest.splits[split]
    )
    if len(object_order) != 28 or len(set(object_order)) != 28:
        raise ValueError("expected the frozen 22/3/3 material-v2 object split")
    global_index = {object_id: index for index, object_id in enumerate(object_order)}
    output = args.output / "local"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints: dict[str, Any] = {}
    for seed in MODEL_SEEDS:
        config = _checkpoint_config(base, arm="aperture", seed=seed)
        checkpoint_path = args.local_root / f"seed-{seed}" / "best-local.pt"
        model, checkpoint = _load_model(
            config,
            manifest,
            checkpoint_path,
            device=device,
            stage="local",
            horizon=None,
        )
        checkpoint_hash = _sha256(checkpoint_path.resolve())
        checkpoints[str(seed)] = {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_hash,
            "best_metric": float(checkpoint["best_metric"]),
            "contact_features": config.model.contact_features,
            "global_conditioning": config.model.global_conditioning,
        }
        for split in EXPECTED_SPLITS:
            destination = output / f"seed-{seed}-{split}.npz"
            if args.resume and destination.is_file():
                print(f"[LOCAL-TF] resume: keeping {destination}", flush=True)
                continue
            values = _evaluate_local_tf_split(
                model,
                config,
                manifest,
                split=split,
                device=device,
                global_object_index=global_index,
            )
            _atomic_savez(
                destination,
                **values,
                split=np.asarray(split),
                seed=np.asarray(seed, dtype=np.int32),
                manifest_sha256=np.asarray(manifest.sha256()),
                gripper_sha256=np.asarray(manifest.gripper_sha256),
                checkpoint_sha256=np.asarray(checkpoint_hash),
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    contract = {
        "manifest_sha256": manifest.sha256(),
        "gripper_sha256": manifest.gripper_sha256,
        "objects": list(object_order),
        "local_checkpoints": checkpoints,
        "h32_checkpoint_contract": h32_contract,
        "model": {
            "contact_features": "gap",
            "global_conditioning": "aperture",
        },
    }
    _write_json(args.output / "contract.json", contract)
    return contract


def _bootstrap_summary(
    features: np.ndarray,
    object_index: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    values = _hierarchical_bias_bootstrap(
        features,
        object_index,
        replicates=replicates,
        seed=seed,
    )
    mean = values["mean"]
    lower = values["ci95_lower"]
    upper = values["ci95_upper"]
    return {
        "mean": mean,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "json": {
            "band_mean_v_m_w_rad": mean[:24].reshape(4, 6).tolist(),
            "band_ci95_lower": lower[:24].reshape(4, 6).tolist(),
            "band_ci95_upper": upper[:24].reshape(4, 6).tolist(),
            "cumulative_b32_mean_v_m_w_rad": mean[24:30].tolist(),
            "cumulative_b32_ci95_lower": lower[24:30].tolist(),
            "cumulative_b32_ci95_upper": upper[24:30].tolist(),
            "persistent_components_vx_vy_vz_wx_wy_wz": _persistent_bias(
                mean, lower, upper
            ),
        },
    }


def _summarize(args: argparse.Namespace, contract: dict[str, Any]) -> None:
    from srno.data.schema import DatasetManifest

    manifest = DatasetManifest.load(args.manifest)
    results: dict[str, Any] = {}
    packed: dict[str, np.ndarray] = {}
    classifications: dict[str, str] = {}
    per_seed_bz: dict[str, dict[str, list[float]]] = {}
    for split_index, split in enumerate(EXPECTED_SPLITS):
        local_rows: list[np.ndarray] = []
        h32_rows: list[np.ndarray] = []
        object_index: np.ndarray | None = None
        trajectory: np.ndarray | None = None
        for seed in MODEL_SEEDS:
            local_path = args.output / "local" / f"seed-{seed}-{split}.npz"
            h32_path = args.h32_bias_root / f"seed-{seed}-{split}.npz"
            with np.load(local_path, allow_pickle=False) as local, np.load(
                h32_path, allow_pickle=False
            ) as h32:
                for archive, label in ((local, "local"), (h32, "H32")):
                    if str(archive["manifest_sha256"].item()) != manifest.sha256():
                        raise ValueError(f"{label} {split} manifest hash mismatch")
                expected_h32_hash = contract["h32_checkpoint_contract"][
                    "checkpoints"
                ][str(seed)]["sha256"]
                if str(h32["checkpoint_sha256"].item()) != expected_h32_hash:
                    raise ValueError(f"H32 raw checkpoint hash mismatch for seed {seed}")
                expected_local_hash = contract["local_checkpoints"][str(seed)][
                    "sha256"
                ]
                if str(local["checkpoint_sha256"].item()) != expected_local_hash:
                    raise ValueError(f"local raw checkpoint hash mismatch for seed {seed}")
                if not np.array_equal(local["object_index"], h32["object_index"]):
                    raise ValueError(f"{split}: local/H32 object ordering differs")
                if not np.array_equal(local["trajectory"], h32["trajectory"]):
                    raise ValueError(f"{split}: local/H32 trajectory ordering differs")
                if object_index is None:
                    object_index = np.asarray(local["object_index"])
                    trajectory = np.asarray(local["trajectory"])
                local_rows.append(np.asarray(local["tf_log"], dtype=np.float64))
                h32_rows.append(np.asarray(h32["tf_log"], dtype=np.float64))
        assert object_index is not None and trajectory is not None
        local_tf = np.stack(local_rows)
        h32_tf = np.stack(h32_rows)
        local_features = np.stack([_tf_feature_vector(value) for value in local_tf])
        h32_features = np.stack([_tf_feature_vector(value) for value in h32_tf])
        delta_features = h32_features - local_features
        local_summary = _bootstrap_summary(
            local_features,
            object_index,
            replicates=args.bootstrap_replicates,
            seed=BOOTSTRAP_SEED + 3 * split_index,
        )
        h32_summary = _bootstrap_summary(
            h32_features,
            object_index,
            replicates=args.bootstrap_replicates,
            seed=BOOTSTRAP_SEED + 3 * split_index + 1,
        )
        delta_summary = _bootstrap_summary(
            delta_features,
            object_index,
            replicates=args.bootstrap_replicates,
            seed=BOOTSTRAP_SEED + 3 * split_index + 2,
        )
        classification = _classify_split_vz(
            local_summary, h32_summary, delta_summary
        )
        classifications[split] = classification
        local_curve = np.mean(
            [_mean_by_object(value, object_index) for value in local_tf], axis=0
        )
        h32_curve = np.mean(
            [_mean_by_object(value, object_index) for value in h32_tf], axis=0
        )
        seed_local_bz = [
            float(_mean_by_object(value.sum(axis=1), object_index)[2])
            for value in local_tf
        ]
        seed_h32_bz = [
            float(_mean_by_object(value.sum(axis=1), object_index)[2])
            for value in h32_tf
        ]
        per_seed_bz[split] = {"local": seed_local_bz, "h32": seed_h32_bz}
        per_object: dict[str, Any] = {}
        labels = contract["objects"]
        for object_id in np.unique(object_index):
            mask = object_index == object_id
            local_object = local_tf[:, mask].sum(axis=2).mean(axis=(0, 1))
            h32_object = h32_tf[:, mask].sum(axis=2).mean(axis=(0, 1))
            per_object[labels[int(object_id)]] = {
                "local_b32_v_m_w_rad": local_object.tolist(),
                "h32_b32_v_m_w_rad": h32_object.tolist(),
                "h32_minus_local_b32_v_m_w_rad": (
                    h32_object - local_object
                ).tolist(),
            }
        results[split] = {
            "trajectories": int(local_tf.shape[1]),
            "objects": int(len(np.unique(object_index))),
            "local": local_summary["json"],
            "h32": h32_summary["json"],
            "h32_minus_local": delta_summary["json"],
            "classification_vz": classification,
            "per_seed_b32_vz_m": per_seed_bz[split],
            "per_object": per_object,
        }
        prefix = f"{split}"
        packed.update(
            {
                f"{prefix}_local_tf_log": local_tf,
                f"{prefix}_h32_tf_log": h32_tf,
                f"{prefix}_object_index": object_index,
                f"{prefix}_trajectory": trajectory,
                f"{prefix}_local_mean_curve": local_curve,
                f"{prefix}_h32_mean_curve": h32_curve,
            }
        )
    values = tuple(classifications.values())
    if all(value == "bias_already_in_local" for value in values):
        overall = "bias_already_in_local_all_splits"
    elif all(value == "bias_introduced_by_rollout" for value in values):
        overall = "bias_introduced_by_rollout_all_splits"
    else:
        overall = "mixed_or_inconclusive"
    result = {
        "definition": {
            "teacher_forced_error": "Log(inv(q*_{k+1}) R_theta(x*_k,a_bar_{k+1})_q)",
            "b_k_tf": "equal-object E[e_k_tf]",
            "B_32_tf": "sum_{k=0..31} b_k_tf",
            "paired_delta": "B_32_tf(H32)-B_32_tf(local)",
        },
        "configuration": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest.sha256(),
            "gripper_sha256": manifest.gripper_sha256,
            "model_seeds": list(MODEL_SEEDS),
            "step_bands": [list(value) for value in BIAS_STEP_BANDS],
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "contract": contract,
            "retrained": False,
            "architecture_changed": False,
        },
        "splits": results,
        "decision": {
            "classification": overall,
            "by_split": classifications,
        },
    }
    _write_json(args.output / "results.json", result)
    _atomic_savez(args.output / "samples.npz", **packed)
    _plot(args.output, results, packed, per_seed_bz)
    print(f"[LOCAL-vs-H32] decision={overall}", flush=True)


def _plot(
    output: Path,
    results: dict[str, Any],
    arrays: dict[str, np.ndarray],
    per_seed_bz: dict[str, dict[str, list[float]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"train": "#4472C4", "val": "#70AD47", "test": "#C00000"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for column, split in enumerate(EXPECTED_SPLITS):
        local = np.cumsum(arrays[f"{split}_local_mean_curve"][:, 2]) * 1000.0
        h32 = np.cumsum(arrays[f"{split}_h32_mean_curve"][:, 2]) * 1000.0
        steps = np.arange(1, 33)
        axes[0, column].plot(steps, local, label="frozen local", color="#4472C4")
        axes[0, column].plot(steps, h32, label="H32", color="#C00000")
        axes[0, column].axhline(0.0, color="black", linewidth=0.8)
        axes[0, column].set(
            title=split,
            xlabel="step",
            ylabel=r"cumulative $B_z^{TF}$, mm",
        )
        axes[0, column].grid(alpha=0.2)
        axes[0, column].legend()

        local_seed = np.asarray(per_seed_bz[split]["local"]) * 1000.0
        h32_seed = np.asarray(per_seed_bz[split]["h32"]) * 1000.0
        for seed in range(len(MODEL_SEEDS)):
            axes[1, column].plot(
                (0, 1),
                (local_seed[seed], h32_seed[seed]),
                marker="o",
                color=colors[split],
                alpha=0.8,
            )
        axes[1, column].axhline(0.0, color="black", linewidth=0.8)
        axes[1, column].set_xticks((0, 1), ("local", "H32"))
        axes[1, column].set(
            ylabel=r"$B_{32,z}^{TF}$, mm",
            title=f"paired seeds; {results[split]['classification_vz']}",
        )
        axes[1, column].grid(axis="y", alpha=0.2)
    fig.suptitle("Frozen local vs H32: signed teacher-forced z-bias", fontsize=15)
    fig.savefig(output / "local_vs_h32_signed_bias.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/simulator-r-v1/manifest.json")
    )
    parser.add_argument(
        "--train-config", type=Path, default=Path("configs/srno-r-material-v2.toml")
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path("runs/ablation-jq-local/baseline"),
    )
    parser.add_argument(
        "--h32-bias-root",
        type=Path,
        default=Path("runs/quasistatic-refinement-bias-v1/bias"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/local-vs-h32-signed-bias-v1"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_replicates <= 0:
        parser.error("--bootstrap-replicates must be positive")
    if args.summarize_only:
        contract = json.loads((args.output / "contract.json").read_text("utf-8"))
    else:
        contract = _run_local_inference(args)
    _summarize(args, contract)


if __name__ == "__main__":
    main()
