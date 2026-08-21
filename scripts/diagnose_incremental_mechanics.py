#!/usr/bin/env python3
"""Test energetic and static-equilibrium necessities on settled transitions.

The collector records an equilibrium only after a fixed aperture command has
been held until the object and all joints settle.  This script therefore tests
two properties that any incremental variational formulation must satisfy:

1. the known implicit-PD spring releases (rather than creates) energy; and
2. the target generalized spring force can be balanced by target-state contact
   reactions after eliminating their unknown magnitudes with NNLS.

The contact-force fit is only a compatibility diagnostic.  It is not a force
label: cooked-SDF normals and the 256 canonical gripper samples approximate,
rather than reproduce, PhysX's collision manifold.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
import torch

from srno.data.dataset import H5ObjectDataset, LocalTransitionBatch, make_dataloader
from srno.data.index import ActiveIndex
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import rotation_geodesic_angle
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.types import PoseState, SDFBatch


GRIPPER_STIFFNESS_NM_PER_RAD = 14.0


def _take_state(state: PoseState, indices: torch.Tensor) -> PoseState:
    return PoseState(
        state.rotation.index_select(0, indices),
        state.position.index_select(0, indices),
        state.joint_position.index_select(0, indices),
    )


def _repeat_sdf(sdf: SDFBatch, count: int) -> SDFBatch:
    if sdf.values.shape[0] != 1:
        raise ValueError("diagnostic expects one object per batch")
    return SDFBatch(
        sdf.values,
        sdf.origin,
        sdf.voxel_size,
        torch.zeros(count, dtype=torch.long),
        sdf.outside_value,
    )


def _stats(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "q10": float(np.quantile(finite, 0.10)),
        "q90": float(np.quantile(finite, 0.90)),
        "q99": float(np.quantile(finite, 0.99)),
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 2 or np.std(left[finite]) == 0 or np.std(right[finite]) == 0:
        return float("nan")
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def _relative_nnls_residual(design: np.ndarray, target: np.ndarray) -> float:
    norm = float(np.linalg.norm(target))
    if norm <= 1e-12:
        return float("nan")
    if design.shape[1] == 0:
        return 1.0
    _, residual = nnls(design, target, maxiter=max(5000, 5 * design.shape[1]))
    return float(residual / norm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/srno-r-material-v2.toml")
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--samples-per-object", type=int, default=100)
    parser.add_argument(
        "--support-margin-mm", type=float, nargs="+", default=(0.0, 0.5, 2.0, 8.0)
    )
    parser.add_argument("--friction", type=float, default=2.4)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/incremental-mechanics-test.json")
    )
    args = parser.parse_args()

    config = replace(ExperimentConfig.load(args.config), device="cpu")
    manifest = DatasetManifest.load(config.paths.manifest)
    active_index = ActiveIndex.load(config.paths.active_index)
    model = _build_model(config, manifest, torch.device("cpu")).eval()
    model.requires_grad_(False)
    dataset = H5ObjectDataset(
        manifest, split=args.split, active_index=active_index, active_only=True
    )
    loader = make_dataloader(
        dataset,
        mode="local",
        objects_per_batch=1,
        samples_per_object=0,
        workers=0,
        seed=config.seed,
        shuffle=False,
    )

    arrays: dict[str, list[np.ndarray]] = {}
    residuals: dict[str, list[float]] = {}
    try:
        with torch.no_grad():
            for batch in loader:
                assert isinstance(batch, LocalTransitionBatch)
                count = batch.current.position.shape[0]
                selected = torch.from_numpy(
                    np.linspace(
                        0,
                        count - 1,
                        min(args.samples_per_object, count),
                        dtype=np.int64,
                    )
                )
                current = _take_state(batch.current, selected)
                target = _take_state(batch.target, selected)
                command = batch.next_command.index_select(0, selected)
                trial_joint = model.free_joint_configuration(command)
                sdf = _repeat_sdf(batch.sdf, len(selected))

                current_error = current.joint_position - trial_joint
                target_error = target.joint_position - trial_joint
                current_energy = 0.5 * GRIPPER_STIFFNESS_NM_PER_RAD * (
                    current_error.square().sum(dim=-1)
                )
                target_energy = 0.5 * GRIPPER_STIFFNESS_NM_PER_RAD * (
                    target_error.square().sum(dim=-1)
                )
                released = current_energy - target_energy
                pose_motion = torch.sqrt(
                    (
                        torch.linalg.vector_norm(
                            target.position - current.position, dim=-1
                        )
                        / model.length_scale
                    ).square()
                    + rotation_geodesic_angle(
                        target.rotation, current.rotation
                    ).square()
                )
                joint_motion = torch.sqrt(
                    (
                        (target.joint_position - current.joint_position)
                        / model.joint_travel_range
                    ).square().mean(dim=-1)
                )
                arrays.setdefault("spring_energy_current_j", []).append(
                    current_energy.numpy()
                )
                arrays.setdefault("spring_energy_target_j", []).append(
                    target_energy.numpy()
                )
                arrays.setdefault("spring_energy_released_j", []).append(released.numpy())
                arrays.setdefault("pose_motion", []).append(pose_motion.numpy())
                arrays.setdefault("joint_motion", []).append(joint_motion.numpy())

                gap, points, jacobian = model._contact_gap_and_full_jacobian(
                    target, sdf
                )
                directions = model._friction_cone_directions(
                    target, points, jacobian
                )
                # Gradient of 1/2*k*||r-r_cmd||^2 with respect to normalized
                # z_r=(r-r_k)/travel.  At equilibrium J^T lambda balances it.
                spring_gradient = torch.zeros(len(selected), 12)
                spring_gradient[:, 6:] = (
                    GRIPPER_STIFFNESS_NM_PER_RAD
                    * model.joint_travel_range
                    * target_error
                )
                valid_normal = torch.linalg.vector_norm(jacobian, dim=-1) > 1e-8
                for row in range(len(selected)):
                    target_force = spring_gradient[row].numpy().astype(np.float64)
                    for margin_mm in args.support_margin_mm:
                        support = valid_normal[row] & (gap[row] <= margin_mm * 1e-3)
                        if not bool(support.any()):
                            support[gap[row].argmin()] = True
                        normal_design = (
                            jacobian[row, support].T.numpy().astype(np.float64)
                        )
                        basis = directions[row, support]
                        normal = basis[..., 0]
                        tangent_one = basis[..., 1]
                        tangent_two = basis[..., 2]
                        rays = torch.stack(
                            (
                                normal + args.friction * tangent_one,
                                normal - args.friction * tangent_one,
                                normal + args.friction * tangent_two,
                                normal - args.friction * tangent_two,
                            ),
                            dim=-1,
                        ).reshape(12, -1)
                        key = f"normal_margin_{margin_mm:g}mm"
                        residuals.setdefault(key, []).append(
                            _relative_nnls_residual(normal_design, target_force)
                        )
                        key = f"friction_margin_{margin_mm:g}mm"
                        residuals.setdefault(key, []).append(
                            _relative_nnls_residual(
                                rays.numpy().astype(np.float64), target_force
                            )
                        )
    finally:
        dataset.close()

    joined = {name: np.concatenate(parts) for name, parts in arrays.items()}
    released = joined["spring_energy_released_j"]
    pose = joined["pose_motion"]
    jump = pose > 0.05
    output = {
        "config": str(args.config),
        "split": args.split,
        "samples_per_object": args.samples_per_object,
        "sample_count": int(len(released)),
        "spring_energy": {
            "current_j": _stats(joined["spring_energy_current_j"]),
            "target_j": _stats(joined["spring_energy_target_j"]),
            "released_j": _stats(released),
            "fraction_nonnegative_release": float(np.mean(released >= -1e-10)),
            "fraction_strictly_positive_release": float(np.mean(released > 1e-10)),
        },
        "motion": {
            "pose": _stats(pose),
            "joint": _stats(joined["joint_motion"]),
            "jump_count": int(jump.sum()),
            "jump_fraction": float(jump.mean()),
            "released_energy_pose_correlation": _correlation(released, pose),
            "released_energy_joint_correlation": _correlation(
                released, joined["joint_motion"]
            ),
            "smooth_released_energy_j": _stats(released[~jump]),
            "jump_released_energy_j": _stats(released[jump]),
        },
        "target_static_balance": {
            "definition": "relative NNLS residual ||G lambda-grad(E_act)||/||grad(E_act)||",
            "contact_sampling_caveat": (
                "cooked SDF/canonical samples approximate the PhysX contact manifold"
            ),
            **{
                key: {
                    **_stats(np.asarray(values, dtype=np.float64)),
                    "fraction_below_0.1": float(
                        np.mean(np.asarray(values, dtype=np.float64) < 0.1)
                    ),
                    "fraction_below_0.25": float(
                        np.mean(np.asarray(values, dtype=np.float64) < 0.25)
                    ),
                }
                for key, values in residuals.items()
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
