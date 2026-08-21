from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from srno.data.tools import build_active_index, calibrate_gate, validate_dataset
from srno.geometry.gripper import preprocess_urdf
from srno.sim import SimulatorAssetCatalog
from srno.sim.config import SimulatorConfig
from srno.sim.pose_seeds import import_validation_pose_directory
from srno.training.config import ExperimentConfig
from srno.training.engine import evaluate_checkpoint, train


def _joint_mapping(values: list[str]) -> dict[str, tuple[float, float]]:
    mapping: dict[str, tuple[float, float]] = {}
    for value in values:
        try:
            name, coefficients = value.split("=", 1)
            offset, multiplier = map(float, coefficients.split(",", 1))
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid joint map {value!r}; expected NAME=OFFSET,MULTIPLIER"
            ) from error
        mapping[name] = (offset, multiplier)
    if not mapping:
        raise argparse.ArgumentTypeError("at least one --joint-map is required")
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="srno", description="SRNO contact-resolvent tools")
    commands = parser.add_subparsers(dest="command", required=True)

    gripper = commands.add_parser("gripper", help="gripper asset tools")
    gripper_commands = gripper.add_subparsers(dest="gripper_command", required=True)
    preprocess = gripper_commands.add_parser("preprocess", help="sample an affine gripper URDF")
    preprocess.add_argument("--urdf", type=Path, required=True)
    preprocess.add_argument("--finger-link", action="append", required=True)
    preprocess.add_argument(
        "--joint-map",
        action="append",
        required=True,
        metavar="NAME=OFFSET,MULTIPLIER",
    )
    preprocess.add_argument("--aperture-min", type=float, required=True)
    preprocess.add_argument("--aperture-max", type=float, required=True)
    preprocess.add_argument("--length-scale", type=float, required=True)
    preprocess.add_argument("--samples-per-link", type=int, default=128)
    preprocess.add_argument("--seed", type=int, default=0)
    preprocess.add_argument("--affine-tolerance", type=float, default=1e-6)
    preprocess.add_argument("--output", type=Path, required=True)

    dataset = commands.add_parser("dataset", help="dataset tools")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    validate = dataset_commands.add_parser("validate", help="validate manifest and shards")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--allow-nonstandard-resolution", action="store_true")

    index = dataset_commands.add_parser(
        "build-active-index", help="index transitions routed through the contact cell"
    )
    index.add_argument("manifest", type=Path)
    index.add_argument("--output", type=Path, required=True)
    index.add_argument("--device", default="cpu")

    calibrate = dataset_commands.add_parser(
        "calibrate-gate", help="calibrate the contact gate from simulator diagnostics"
    )
    calibrate.add_argument("manifest", type=Path)
    calibrate.add_argument("--target-recall", type=float, default=0.995)
    calibrate.add_argument("--device", default="cpu")

    assets = commands.add_parser("assets", help="vendored simulator asset tools")
    assets_commands = assets.add_subparsers(dest="assets_command", required=True)
    assets_validate = assets_commands.add_parser(
        "validate", help="validate the self-contained gripper/object catalog"
    )
    assets_validate.add_argument("--catalog", type=Path)
    assets_validate.add_argument("--verify-hashes", action="store_true")
    import_poses = assets_commands.add_parser(
        "import-validation-poses",
        help="convert validation-gen grasp JSON files into compact local pose seeds",
    )
    import_poses.add_argument("--source-directory", type=Path, required=True)
    import_poses.add_argument("--catalog", type=Path)

    simulator = commands.add_parser("sim", help="Isaac Lab dataset collection")
    simulator_commands = simulator.add_subparsers(dest="sim_command", required=True)
    collect = simulator_commands.add_parser(
        "collect", help="collect zero-gravity 32-step quasistatic trajectories"
    )
    collect.add_argument("--config", type=Path, required=True)
    collect.add_argument("--object", dest="objects", action="append")
    collect.add_argument("--max-objects", type=int)
    collect.add_argument("--trajectories-per-object", type=int)
    collect.add_argument(
        "--pose-index",
        dest="pose_indices",
        action="append",
        type=int,
        help="explicit index in the selected object's validation pose file (repeatable)",
    )
    collect.add_argument("--overwrite", action="store_true", default=None)
    collect.add_argument(
        "--resume",
        action="store_true",
        help="retain complete shards and replace only missing or mismatched shards",
    )

    training = commands.add_parser(
        "train", help="train local, rollout, or local-resolvent trust-region stage"
    )
    training.add_argument("--config", type=Path, required=True)
    training.add_argument(
        "--stage", choices=("local", "rollout", "trust-rollout"), required=True
    )
    training.add_argument("--resume", type=Path)
    training.add_argument(
        "--advance-horizon",
        action="store_true",
        help="resume a rollout checkpoint at the next configured curriculum horizon",
    )

    evaluate = commands.add_parser("evaluate", help="evaluate a rollout checkpoint")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "gripper" and args.gripper_command == "preprocess":
        if len(args.finger_link) != 2:
            parser.error("gripper preprocess requires exactly two --finger-link values")
        try:
            mapping = _joint_mapping(args.joint_map)
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
        asset = preprocess_urdf(
            args.urdf,
            finger_links=tuple(args.finger_link),
            joint_map=mapping,
            aperture_min=args.aperture_min,
            aperture_max=args.aperture_max,
            length_scale=args.length_scale,
            samples_per_link=args.samples_per_link,
            seed=args.seed,
            affine_tolerance=args.affine_tolerance,
        )
        asset.save(args.output)
        print(json.dumps({"output": str(args.output), "sha256": asset.sha256(), "points": asset.point_count}))
        return 0
    if args.command == "dataset":
        if args.dataset_command == "validate":
            report = validate_dataset(
                args.manifest, strict_resolution=not args.allow_nonstandard_resolution
            )
            print(json.dumps(report.__dict__, indent=2))
            return 0
        if args.dataset_command == "build-active-index":
            active = build_active_index(args.manifest, args.output, device=args.device)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "objects": len(active.object_ids),
                        "active_transitions": int(len(active.step_index)),
                    },
                    indent=2,
                )
            )
            return 0
        if args.dataset_command == "calibrate-gate":
            print(
                json.dumps(
                    calibrate_gate(
                        args.manifest,
                        target_recall=args.target_recall,
                        device=args.device,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
    if args.command == "assets" and args.assets_command == "validate":
        catalog = SimulatorAssetCatalog.load(args.catalog)
        catalog.validate_files(verify_hashes=args.verify_hashes)
        print(
            json.dumps(
                {
                    "catalog": str(catalog.catalog_path),
                    "objects": len(catalog.object_ids),
                    "gripper_usd": str(catalog.gripper.runtime_usd),
                    "approach_axis_local": catalog.approach_axis_local,
                    "hashes_verified": args.verify_hashes,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "assets" and args.assets_command == "import-validation-poses":
        catalog = SimulatorAssetCatalog.load(args.catalog)
        result = import_validation_pose_directory(
            args.source_directory,
            catalog.pose_seed_directory,
            catalog.object_ids,
        )
        print(
            json.dumps(
                {
                    "output": str(catalog.pose_seed_directory),
                    "objects": len(result),
                    "poses": sum(result.values()),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "sim" and args.sim_command == "collect":
        from srno.sim.runner import run_simulator_collection

        manifest = run_simulator_collection(
            SimulatorConfig.load(args.config),
            object_ids=args.objects,
            max_objects=args.max_objects,
            trajectories_per_object=args.trajectories_per_object,
            pose_indices=args.pose_indices,
            overwrite=args.overwrite,
            resume=args.resume,
        )
        print(json.dumps({"manifest": None if manifest is None else str(manifest)}, indent=2))
        return 0
    if args.command == "train":
        config = ExperimentConfig.load(args.config)
        stage = "trust_rollout" if args.stage == "trust-rollout" else args.stage
        checkpoint = train(
            config,
            stage=stage,
            resume=args.resume,
            advance_horizon=args.advance_horizon,
        )
        print(json.dumps({"best_checkpoint": str(checkpoint)}))
        return 0
    if args.command == "evaluate":
        config = ExperimentConfig.load(args.config)
        metrics = evaluate_checkpoint(config, args.checkpoint, split=args.split)
        payload = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    parser.error("unknown command")
    return 2
