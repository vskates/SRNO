#!/usr/bin/env python3
"""Learn a single-valued operator by stratifying the physical path manifold.

Complete train paths are clustered into contact-response strata.  A classifier
maps the loading-profile function (and the documented object mass) to a stratum;
the final path is a local kernel average restricted to the predicted strata.
Validation selects the number of strata and local averaging rule before one
test evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

from evaluate_loading_profile_kernel_operator import (
    _load_split,
    _metrics,
    _nearest,
    _predict,
    _profile_embedding,
)
from run_rkhs_solution_operator import _output_coordinates, _objectwise_metrics
from srno.data.schema import DatasetManifest
from srno.geometry.gripper import GripperAsset
from srno.model import SRNOModel


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _mass_feature(
    data: dict[str, np.ndarray], catalog: dict[str, Any]
) -> np.ndarray:
    records = {str(value["id"]): value for value in catalog["objects"]}
    values = []
    for object_id in data["object_id"]:
        record = records[str(object_id)]
        bbox = np.asarray(record["bbox_size_m"], dtype=np.float32)
        values.append(
            np.concatenate(
                (
                    np.asarray([np.log(float(record["mass_kg"]))], dtype=np.float32),
                    bbox,
                    np.sort(bbox),
                )
            )
        )
    return np.stack(values)


def _select_by_stratum(
    neighbours: np.ndarray,
    train_labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    top_strata: int,
    count: int,
) -> np.ndarray:
    preferred = np.argsort(-probabilities, axis=1, kind="stable")[:, :top_strata]
    selected = np.empty((len(neighbours), count), dtype=np.int64)
    for row in range(len(neighbours)):
        allowed = np.isin(train_labels[neighbours[row]], preferred[row])
        choices = neighbours[row, allowed]
        if len(choices) < count:
            choices = neighbours[row]
        selected[row] = choices[:count]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("assets/catalog.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile-metric", default="hinge_100")
    parser.add_argument("--profile-components", type=int, default=64)
    parser.add_argument("--maximum-neighbours", type=int, default=512)
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
    model = model.to("cpu")
    joint_scale = gripper.joint_travel_range.numpy().astype(np.float32)
    embedding = {
        split: _profile_embedding(data[split]["descriptor"], args.profile_metric)
        for split in ("train", "val", "test")
    }
    pca = PCA(
        n_components=args.profile_components,
        svd_solver="randomized",
        random_state=args.seed,
    )
    train_profile = pca.fit_transform(
        embedding["train"].reshape(len(embedding["train"]), -1)
    ).astype(np.float32)
    profile_coordinate = {
        "train": train_profile,
        "val": pca.transform(
            embedding["val"].reshape(len(embedding["val"]), -1)
        ).astype(np.float32),
        "test": pca.transform(
            embedding["test"].reshape(len(embedding["test"]), -1)
        ).astype(np.float32),
    }
    catalog = json.loads(args.catalog.read_text())
    mass = {split: _mass_feature(data[split], catalog) for split in ("train", "val", "test")}
    mass_mean = mass["train"].mean(axis=0, keepdims=True)
    mass_scale = mass["train"].std(axis=0, keepdims=True).clip(min=1e-6)
    classifier_input = {
        split: np.concatenate(
            (profile_coordinate[split], (mass[split] - mass_mean) / mass_scale),
            axis=-1,
        )
        for split in ("train", "val", "test")
    }
    nearest = {
        split: _nearest(
            embedding["train"],
            embedding[split],
            maximum_k=args.maximum_neighbours,
            device=device,
            chunk_size=16,
        )
        for split in ("val", "test")
    }
    output = _output_coordinates(data["train"], model, manifest).reshape(-1, 32, 12)
    terminal = output[:, -1].copy()
    terminal[:, 6:] /= np.sqrt(6.0)
    validation = []
    fitted = {}
    for strata in (4, 8, 16, 32):
        clustering = KMeans(
            n_clusters=strata,
            n_init=20,
            random_state=args.seed,
        ).fit(terminal)
        labels = clustering.labels_
        classifier = ExtraTreesClassifier(
            n_estimators=512,
            min_samples_leaf=8,
            max_features=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.seed,
        ).fit(classifier_input["train"], labels)
        val_probability = classifier.predict_proba(classifier_input["val"])
        fitted[strata] = (clustering, classifier, labels)
        for top_strata in (1, 2, 4):
            if top_strata > strata:
                continue
            for count in (1, 2, 4, 8, 16):
                selected = _select_by_stratum(
                    nearest["val"][0],
                    labels,
                    val_probability,
                    top_strata=top_strata,
                    count=count,
                )
                prediction = _predict(data["train"], data["val"], selected, count)
                values = _metrics(
                    prediction,
                    data["val"],
                    length_scale=manifest.length_scale_m,
                    joint_scale=joint_scale,
                )
                validation.append(
                    {
                        "strata": strata,
                        "top_strata": top_strata,
                        "neighbours": count,
                        "metrics": values,
                    }
                )
        best = min(
            (value for value in validation if value["strata"] == strata),
            key=lambda value: value["metrics"]["terminal_dx"],
        )
        print(
            f"[STRATA] count={strata} val_terminal_dx="
            f"{best['metrics']['terminal_dx']:.8f}",
            flush=True,
        )
    selected_rule = min(
        validation, key=lambda value: value["metrics"]["terminal_dx"]
    )
    _, classifier, labels = fitted[selected_rule["strata"]]
    test_probability = classifier.predict_proba(classifier_input["test"])
    selected_neighbours = _select_by_stratum(
        nearest["test"][0],
        labels,
        test_probability,
        top_strata=selected_rule["top_strata"],
        count=selected_rule["neighbours"],
    )
    test_prediction = _predict(
        data["train"],
        data["test"],
        selected_neighbours,
        selected_rule["neighbours"],
    )
    test = _metrics(
        test_prediction,
        data["test"],
        length_scale=manifest.length_scale_m,
        joint_scale=joint_scale,
    )
    result = {
        "definition": {
            "solution_concept": "single-valued branch-stratified complete path operator",
            "branch_labels": "KMeans strata of train terminal product-manifold response",
            "branch_selector_input": "complete loading profile plus documented mass and bounding box",
            "test_used_for_selection": False,
        },
        "contract": {
            "manifest_sha256": manifest.sha256(),
            "catalog": str(args.catalog.resolve()),
            "profile_metric": args.profile_metric,
            "maximum_neighbours": args.maximum_neighbours,
            "validation_grid_size": len(validation),
        },
        "selected": selected_rule,
        "test": test,
        "test_objectwise": _objectwise_metrics(
            test_prediction, data["test"], manifest, joint_scale
        ),
        "validation_grid": validation,
    }
    _write_json(args.output, result)
    print(json.dumps({"selected": selected_rule, "test": test}, indent=2), flush=True)


if __name__ == "__main__":
    main()
