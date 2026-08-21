#!/usr/bin/env python3
"""Learn an energy over a set-valued loading-path solution operator.

The candidate branches are complete recorded train paths selected in the
clearance-profile Hilbert metric.  The scalar energy is trained from
leave-one-object-out train pairs, so it cannot identify a branch by seeing the
same geometry.  Validation fixes the estimator and argmin/averaging rule;
test is evaluated once after that selection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_loading_profile_kernel_operator import (
    _load_split,
    _metrics,
    _nearest,
    _predict,
    _profile_embedding,
)
from srno.data.schema import DatasetManifest
from srno.geometry.gripper import GripperAsset
from srno.geometry.se3 import rotation_geodesic_angle, so3_log_vector
from srno.model import SRNOModel


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _nearest_excluding_same_object(
    embedding: np.ndarray,
    object_ids: np.ndarray,
    *,
    maximum_k: int,
    device: torch.device,
    chunk_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    train = torch.from_numpy(embedding.reshape(len(embedding), -1)).to(device)
    indices = []
    distances = []
    for start in range(0, len(embedding), chunk_size):
        stop = min(start + chunk_size, len(embedding))
        query = train[start:stop]
        distance = torch.cdist(query, train) / np.sqrt(train.shape[1])
        same_object = torch.from_numpy(
            object_ids[start:stop, None] == object_ids[None, :]
        ).to(device)
        distance.masked_fill_(same_object, torch.inf)
        selected_distance, selected = torch.topk(
            distance, k=maximum_k, dim=1, largest=False, sorted=True
        )
        indices.append(selected.cpu().numpy())
        distances.append(selected_distance.cpu().numpy())
    return np.concatenate(indices), np.concatenate(distances)


def _response_coordinates(
    data: dict[str, np.ndarray],
    *,
    length_scale: float,
    joint_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position = (data["position"] - data["position"][:, :1]) / length_scale
    rotation_delta = (
        data["rotation"]
        @ np.swapaxes(data["rotation"][:, :1], -1, -2)
    )
    rotation = so3_log_vector(torch.from_numpy(rotation_delta)).numpy()
    joints = data["joint"] / joint_scale[None, None, :]
    path = np.concatenate((position, rotation, joints), axis=-1)
    return path.reshape(len(path), -1).astype(np.float32), path[:, -1].astype(
        np.float32
    )


def _candidate_terminal_error(
    train: dict[str, np.ndarray],
    query: dict[str, np.ndarray],
    neighbours: np.ndarray,
    *,
    length_scale: float,
    joint_scale: np.ndarray,
) -> np.ndarray:
    train_position_delta = train["position"][:, -1] - train["position"][:, 0]
    candidate_position = query["position"][:, None, 0] + train_position_delta[
        neighbours
    ]
    translation = np.linalg.norm(
        candidate_position - query["position"][:, None, -1], axis=-1
    ) / length_scale

    train_rotation_delta = (
        train["rotation"][:, -1]
        @ np.swapaxes(train["rotation"][:, 0], -1, -2)
    )
    candidate_rotation = (
        train_rotation_delta[neighbours] @ query["rotation"][:, None, 0]
    )
    target_rotation = np.broadcast_to(
        query["rotation"][:, None, -1], candidate_rotation.shape
    )
    rotation = rotation_geodesic_angle(
        torch.from_numpy(candidate_rotation), torch.from_numpy(target_rotation.copy())
    ).numpy()

    candidate_joint = train["joint"][neighbours, -1]
    joints = np.sqrt(
        np.mean(
            (
                (candidate_joint - query["joint"][:, None, -1])
                / joint_scale[None, None, :]
            )
            ** 2,
            axis=-1,
        )
    )
    return np.sqrt(translation**2 + rotation**2 + joints**2).astype(np.float32)


def _pair_features(
    query_profile: np.ndarray,
    train_profile: np.ndarray,
    train_response: np.ndarray,
    train_terminal: np.ndarray,
    neighbours: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    query = np.broadcast_to(
        query_profile[:, None, :],
        (len(query_profile), neighbours.shape[1], query_profile.shape[1]),
    )
    source = train_profile[neighbours]
    difference = query - source
    features = np.concatenate(
        (
            query,
            source,
            difference,
            np.abs(difference),
            train_response[neighbours],
            train_terminal[neighbours],
            distances[..., None],
        ),
        axis=-1,
    )
    return features.reshape(-1, features.shape[-1]).astype(np.float32)


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(len(values))[:, None]
    ranks[rows, order] = np.arange(values.shape[1])[None, :]
    return ranks.astype(np.float32) / max(values.shape[1] - 1, 1)


def _select(
    neighbours: np.ndarray,
    distances: np.ndarray,
    energy: np.ndarray,
    *,
    pool: int,
    profile_weight: float,
    average: int,
) -> np.ndarray:
    score = _ranks(energy[:, :pool]) + profile_weight * _ranks(
        distances[:, :pool]
    )
    selected_local = np.argsort(score, axis=1, kind="stable")[:, :average]
    return np.take_along_axis(neighbours[:, :pool], selected_local, axis=1)


def _evaluate_rule(
    train: dict[str, np.ndarray],
    query: dict[str, np.ndarray],
    neighbours: np.ndarray,
    distances: np.ndarray,
    energy: np.ndarray,
    rule: dict[str, Any],
    *,
    length_scale: float,
    joint_scale: np.ndarray,
) -> dict[str, float]:
    selected = _select(
        neighbours,
        distances,
        energy,
        pool=rule["pool"],
        profile_weight=rule["profile_weight"],
        average=rule["average"],
    )
    prediction = _predict(train, query, selected, rule["average"])
    return _metrics(
        prediction,
        query,
        length_scale=length_scale,
        joint_scale=joint_scale,
    )


def _estimators(seed: int) -> list[tuple[str, Any, str]]:
    return [
        (
            "ridge_1",
            make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "identity",
        ),
        (
            "hgb_l15",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.06,
                max_iter=150,
                max_leaf_nodes=15,
                min_samples_leaf=40,
                l2_regularization=0.1,
                early_stopping=True,
                random_state=seed,
            ),
            "identity",
        ),
        (
            "hgb_l31",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_iter=180,
                max_leaf_nodes=31,
                min_samples_leaf=40,
                l2_regularization=0.3,
                early_stopping=True,
                random_state=seed,
            ),
            "identity",
        ),
        (
            "hgb_log_l31",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_iter=180,
                max_leaf_nodes=31,
                min_samples_leaf=40,
                l2_regularization=0.3,
                early_stopping=True,
                random_state=seed,
            ),
            "log",
        ),
        (
            "extra_trees",
            ExtraTreesRegressor(
                n_estimators=256,
                min_samples_leaf=20,
                max_features=0.7,
                n_jobs=-1,
                random_state=seed,
            ),
            "identity",
        ),
        (
            "hgb_centered_l31",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_iter=180,
                max_leaf_nodes=31,
                min_samples_leaf=40,
                l2_regularization=0.3,
                early_stopping=True,
                random_state=seed,
            ),
            "centered",
        ),
        (
            "hgb_rank_l31",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_iter=180,
                max_leaf_nodes=31,
                min_samples_leaf=40,
                l2_regularization=0.3,
                early_stopping=True,
                random_state=seed,
            ),
            "rank",
        ),
        (
            "extra_trees_centered",
            ExtraTreesRegressor(
                n_estimators=256,
                min_samples_leaf=20,
                max_features=0.7,
                n_jobs=-1,
                random_state=seed,
            ),
            "centered",
        ),
        (
            "extra_trees_rank",
            ExtraTreesRegressor(
                n_estimators=256,
                min_samples_leaf=20,
                max_features=0.7,
                n_jobs=-1,
                random_state=seed,
            ),
            "rank",
        ),
    ]


def _energy_target(error: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        result = error
    elif transform == "log":
        result = np.log(error.clip(min=1e-6))
    elif transform == "centered":
        result = error - error.mean(axis=1, keepdims=True)
    elif transform == "rank":
        result = _ranks(error)
    else:
        raise ValueError(f"unknown target transform {transform!r}")
    return result.reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile-metric", default="hinge_050")
    parser.add_argument("--maximum-candidates", type=int, default=64)
    parser.add_argument("--profile-components", type=int, default=32)
    parser.add_argument("--response-components", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device)
    manifest = DatasetManifest.load(args.manifest)
    gripper = GripperAsset.load(manifest.gripper_path)
    model = SRNOModel(
        gripper,
        sdf_scale=manifest.sdf_scale_m,
        delta_gate=manifest.delta_gate_m,
        contact_offset_sum=manifest.contact_offset_sum_m,
    ).to(device)
    model.eval()
    data = {
        split: _load_split(manifest, model, split, device)
        for split in ("train", "val", "test")
    }
    joint_scale = gripper.joint_travel_range.numpy().astype(np.float32)
    embeddings = {
        split: _profile_embedding(data[split]["descriptor"], args.profile_metric)
        for split in ("train", "val", "test")
    }
    profile_pca = PCA(
        n_components=args.profile_components,
        svd_solver="randomized",
        random_state=args.seed,
    )
    train_profile = profile_pca.fit_transform(
        embeddings["train"].reshape(len(embeddings["train"]), -1)
    ).astype(np.float32)
    profile_coordinates = {
        "train": train_profile,
        "val": profile_pca.transform(
            embeddings["val"].reshape(len(embeddings["val"]), -1)
        ).astype(np.float32),
        "test": profile_pca.transform(
            embeddings["test"].reshape(len(embeddings["test"]), -1)
        ).astype(np.float32),
    }
    path_response, terminal_response = _response_coordinates(
        data["train"],
        length_scale=manifest.length_scale_m,
        joint_scale=joint_scale,
    )
    response_pca = PCA(
        n_components=args.response_components,
        svd_solver="randomized",
        random_state=args.seed,
    )
    response_coordinates = response_pca.fit_transform(path_response).astype(np.float32)

    train_nearest = _nearest_excluding_same_object(
        embeddings["train"],
        data["train"]["object_id"],
        maximum_k=args.maximum_candidates,
        device=device,
    )
    nearest = {"train": train_nearest}
    for split in ("val", "test"):
        nearest[split] = _nearest(
            embeddings["train"],
            embeddings[split],
            maximum_k=args.maximum_candidates,
            device=device,
            chunk_size=32,
        )
    train_x = _pair_features(
        profile_coordinates["train"],
        train_profile,
        response_coordinates,
        terminal_response,
        nearest["train"][0],
        nearest["train"][1],
    )
    train_error = _candidate_terminal_error(
        data["train"],
        data["train"],
        nearest["train"][0],
        length_scale=manifest.length_scale_m,
        joint_scale=joint_scale,
    )
    train_y = train_error.reshape(-1)
    val_x = _pair_features(
        profile_coordinates["val"],
        train_profile,
        response_coordinates,
        terminal_response,
        nearest["val"][0],
        nearest["val"][1],
    )
    print(
        f"[ENERGY] train_pairs={len(train_x)} features={train_x.shape[1]}",
        flush=True,
    )

    validation_results = []
    fitted: dict[str, tuple[Any, str]] = {}
    for name, estimator, transform in _estimators(args.seed):
        target = _energy_target(train_error, transform)
        estimator.fit(train_x, target)
        prediction = estimator.predict(val_x).reshape(
            len(data["val"]["position"]), args.maximum_candidates
        )
        if transform == "log":
            prediction = np.exp(prediction)
        fitted[name] = (estimator, transform)
        best_for_estimator = None
        for pool in (16, 32, 64):
            if pool > args.maximum_candidates:
                continue
            for profile_weight in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
                for average in (1, 2, 4, 8):
                    if average > pool:
                        continue
                    rule = {
                        "pool": pool,
                        "profile_weight": profile_weight,
                        "average": average,
                    }
                    metrics = _evaluate_rule(
                        data["train"],
                        data["val"],
                        nearest["val"][0],
                        nearest["val"][1],
                        prediction,
                        rule,
                        length_scale=manifest.length_scale_m,
                        joint_scale=joint_scale,
                    )
                    record = {
                        "estimator": name,
                        "target_transform": transform,
                        "rule": rule,
                        "val": metrics,
                    }
                    validation_results.append(record)
                    if (
                        best_for_estimator is None
                        or metrics["terminal_dx"]
                        < best_for_estimator["val"]["terminal_dx"]
                    ):
                        best_for_estimator = record
        assert best_for_estimator is not None
        print(
            f"[ENERGY] estimator={name} val_terminal_dx="
            f"{best_for_estimator['val']['terminal_dx']:.8f}",
            flush=True,
        )
    selected = min(
        validation_results, key=lambda value: value["val"]["terminal_dx"]
    )
    estimator, transform = fitted[selected["estimator"]]
    test_x = _pair_features(
        profile_coordinates["test"],
        train_profile,
        response_coordinates,
        terminal_response,
        nearest["test"][0],
        nearest["test"][1],
    )
    test_energy = estimator.predict(test_x).reshape(
        len(data["test"]["position"]), args.maximum_candidates
    )
    if transform == "log":
        test_energy = np.exp(test_energy)
    test = _evaluate_rule(
        data["train"],
        data["test"],
        nearest["test"][0],
        nearest["test"][1],
        test_energy,
        selected["rule"],
        length_scale=manifest.length_scale_m,
        joint_scale=joint_scale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_path = args.output.with_suffix(".joblib")
    joblib.dump(
        {
            "estimator": estimator,
            "target_transform": transform,
            "profile_pca": profile_pca,
            "response_pca": response_pca,
            "selection": selected,
        },
        model_path,
    )
    result = {
        "definition": {
            "operator": "argmin of a learned scalar branch energy over complete train solution paths",
            "training_pairs": "leave-one-object-out candidates for every train query",
            "target_used_at_inference": False,
            "test_used_for_selection": False,
        },
        "contract": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest.sha256(),
            "profile_metric": args.profile_metric,
            "maximum_candidates": args.maximum_candidates,
            "profile_components": args.profile_components,
            "response_components": args.response_components,
            "train_pairs": len(train_x),
            "energy_model": str(model_path.resolve()),
        },
        "train_candidate_error": {
            "mean": float(train_y.mean()),
            "median": float(np.median(train_y)),
            "oracle_mean": float(train_error.min(axis=1).mean()),
        },
        "selected": selected,
        "test": test,
        "best_validation_by_estimator": {
            name: min(
                (value for value in validation_results if value["estimator"] == name),
                key=lambda value: value["val"]["terminal_dx"],
            )
            for name, _, _ in _estimators(args.seed)
        },
    }
    _write_json(args.output, result)
    print(json.dumps({"selected": selected, "test": test}, indent=2), flush=True)


if __name__ == "__main__":
    main()
