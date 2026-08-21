#!/usr/bin/env python3
"""Screen a true current-state metric contact resolvent without learning.

The older projection diagnostic linearized at the fully closed free-command
trial state, which can already be several millimetres inside the object.  A
quasi-dynamic contact step instead starts at the observed, nearly feasible
state and treats the actuator displacement as the query of a metric
projection:

    min_z  0.5 (z - z_free)^T Q (z - z_free)
    s.t.   h(x_k) + J(x_k) z >= h_admissible.

Here ``z = [delta_p/L, delta_rotation, delta_r/travel]`` and ``z_free`` is
zero for the object and the commanded free displacement for the six joints.
This script contains no learned parameter and uses validation only to select
the relative pose resistance.  It is a representational screen, not a final
model evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import osqp
import scipy.sparse as sparse
import torch

from diagnose_linearized_contact_projection import _errors, _repeat_sdf, _select_state
from srno.data.dataset import H5ObjectDataset, LocalTransitionBatch, make_dataloader
from srno.data.index import ActiveIndex
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import so3_exp, so3_log_vector
from srno.training.config import ExperimentConfig
from srno.training.engine import _build_model
from srno.types import PoseState


def _state_from_increment(
    model: torch.nn.Module, current: PoseState, z: np.ndarray
) -> PoseState:
    value = torch.from_numpy(z.astype(np.float32))[None]
    return PoseState(
        so3_exp(value[:, 3:6]) @ current.rotation,
        current.position + value[:, :3] * model.length_scale,
        current.joint_position + value[:, 6:] * model.joint_travel_range,
    )


def _solve(
    jacobian: np.ndarray,
    rhs: np.ndarray,
    query: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    *,
    pose_weight: float,
) -> tuple[np.ndarray | None, str]:
    metric = np.full(12, 1.0 / 6.0, dtype=np.float64)
    metric[:6] = pose_weight
    joint_selector = np.zeros((6, 12), dtype=np.float64)
    joint_selector[:, 6:] = np.eye(6)
    constraint = sparse.vstack(
        (sparse.csc_matrix(jacobian), sparse.csc_matrix(joint_selector)),
        format="csc",
    )
    lower = np.concatenate((rhs, joint_lower))
    upper = np.concatenate((np.full(len(rhs), np.inf), joint_upper))
    solver = osqp.OSQP()
    solver.setup(
        P=sparse.diags(metric, format="csc"),
        q=-metric * query,
        A=constraint,
        l=lower,
        u=upper,
        eps_abs=1e-7,
        eps_rel=1e-6,
        max_iter=20_000,
        polish=True,
        verbose=False,
    )
    result = solver.solve()
    status = str(result.info.status)
    if result.x is None or not status.startswith("solved"):
        return None, status
    return np.asarray(result.x, dtype=np.float64), status


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    names = ("dx", "translation_m", "rotation_rad", "joint_rmse_over_travel")
    return {name: float(np.mean([row[name] for row in rows])) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/srno-r-material-v2.toml")
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--samples-per-object", type=int, default=100)
    parser.add_argument(
        "--pose-weights",
        type=float,
        nargs="+",
        default=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
    )
    parser.add_argument(
        "--friction-coefficients",
        type=float,
        nargs="+",
        default=(0.0,),
        help="Anitescu polyhedral-friction coefficients (four tangent facets).",
    )
    parser.add_argument(
        "--pose-query-factors",
        type=float,
        nargs="+",
        default=(0.0,),
        help="Scale applied to the previous observed pose increment in the QP query.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/current-state-resolvent-val.json"),
    )
    parser.add_argument(
        "--constraint-gap-m",
        type=float,
        default=None,
        help="Geometric gap boundary; defaults to config admissible_gap_m.",
    )
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    # The diagnostic integrates z with the product retraction in
    # _state_from_increment, so its analytic rotational Jacobian must use the
    # same retraction.  Leaving the legacy SE(3)-left default here silently
    # changes the moment arm from (point-position) to point.
    config = replace(
        config,
        device="cpu",
        model=replace(config.model, pose_update="decoupled"),
    )
    manifest = DatasetManifest.load(config.paths.manifest)
    active_index = ActiveIndex.load(config.paths.active_index)
    model = _build_model(config, manifest, torch.device("cpu")).eval()
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
    weights = [float(value) for value in args.pose_weights]
    friction_coefficients = [float(value) for value in args.friction_coefficients]
    pose_query_factors = [float(value) for value in args.pose_query_factors]
    if any(value < 0.0 for value in friction_coefficients):
        raise ValueError("friction coefficients cannot be negative")
    arm_keys = [
        (weight, friction, pose_query_factor)
        for weight in weights
        for friction in friction_coefficients
        for pose_query_factor in pose_query_factors
    ]
    constraint_gap_m = (
        config.loss.admissible_gap_m
        if args.constraint_gap_m is None
        else float(args.constraint_gap_m)
    )
    rows: dict[tuple[float, float, float], list[dict[str, float]]] = {
        key: [] for key in arm_keys
    }
    baseline_rows: list[dict[str, float]] = []
    statuses: dict[tuple[float, float, float], dict[str, int]] = {
        key: {} for key in arm_keys
    }
    nonlinear_feasible: dict[tuple[float, float, float], list[bool]] = {
        key: [] for key in arm_keys
    }
    contact_counts: list[int] = []
    current_feasible: list[bool] = []
    try:
        with torch.no_grad():
            for batch in loader:
                assert isinstance(batch, LocalTransitionBatch)
                count = batch.current.position.shape[0]
                selected = np.linspace(
                    0,
                    count - 1,
                    min(args.samples_per_object, count),
                    dtype=np.int64,
                )
                for batch_index in selected:
                    i = int(batch_index)
                    current = _select_state(batch.current, i)
                    assert batch.previous is not None
                    previous = _select_state(batch.previous, i)
                    target = _select_state(batch.target, i)
                    sdf = _repeat_sdf(batch.sdf, 1)
                    command = batch.next_command[i : i + 1]
                    free_joint = model.free_joint_configuration(command)
                    free_state = PoseState(
                        current.rotation, current.position, free_joint
                    )
                    baseline_rows.append(_errors(model, free_state, target))

                    contact_gap, contact_points, full_jacobian = (
                        model._contact_gap_and_full_jacobian(current, sdf)
                    )
                    geometric_gap = contact_gap[0] + model.contact_offset_sum
                    jacobian = full_jacobian[0]
                    support = (
                        contact_gap[0] <= model.delta_gate
                    ) & (torch.linalg.vector_norm(jacobian, dim=-1) > 1e-8)
                    if not bool(support.any()):
                        support[torch.argmin(contact_gap[0])] = True
                    selected_gap = geometric_gap[support].numpy().astype(np.float64)
                    selected_jacobian = jacobian[support].numpy().astype(np.float64)
                    normal_rhs = (
                        constraint_gap_m - selected_gap
                    ) / model.sdf_scale
                    contact_directions = model._friction_cone_directions(
                        current, contact_points, full_jacobian
                    )[0, support]
                    contact_counts.append(int(support.sum()))
                    current_feasible.append(
                        bool(geometric_gap.amin() >= constraint_gap_m)
                    )

                    base_query = np.zeros(12, dtype=np.float64)
                    base_query[:3] = (
                        (current.position - previous.position) / model.length_scale
                    )[0].numpy()
                    base_query[3:6] = so3_log_vector(
                        current.rotation @ previous.rotation.transpose(-1, -2)
                    )[0].numpy()
                    base_query[6:] = (
                        (free_joint - current.joint_position)
                        / model.joint_travel_range
                    )[0].numpy()
                    joint_min = model.free_joint_knots.amin(dim=0)
                    joint_max = model.free_joint_knots.amax(dim=0)
                    joint_lower = (
                        (joint_min - current.joint_position[0])
                        / model.joint_travel_range
                    ).numpy().astype(np.float64)
                    joint_upper = (
                        (joint_max - current.joint_position[0])
                        / model.joint_travel_range
                    ).numpy().astype(np.float64)

                    for pose_weight, friction, pose_query_factor in arm_keys:
                        query = base_query.copy()
                        query[:6] *= pose_query_factor
                        if friction == 0.0:
                            constraint_jacobian = selected_jacobian
                            rhs = normal_rhs
                        else:
                            tangent_one = contact_directions[..., 1]
                            tangent_two = contact_directions[..., 2]
                            tangents = torch.stack(
                                (tangent_one, -tangent_one, tangent_two, -tangent_two),
                                dim=1,
                            ).numpy().astype(np.float64)
                            constraint_jacobian = (
                                selected_jacobian[:, None, :] - friction * tangents
                            ).reshape(-1, 12)
                            rhs = np.repeat(normal_rhs, 4)
                        prediction_z, status = _solve(
                            constraint_jacobian,
                            rhs,
                            query,
                            joint_lower,
                            joint_upper,
                            pose_weight=pose_weight,
                        )
                        key = (pose_weight, friction, pose_query_factor)
                        status_counts = statuses[key]
                        status_counts[status] = status_counts.get(status, 0) + 1
                        if prediction_z is None:
                            prediction = free_state
                            nonlinear_feasible[key].append(False)
                        else:
                            prediction = _state_from_increment(
                                model, current, prediction_z
                            )
                            nonlinear_gap = model.query_geometric_gap(
                                prediction, sdf
                            ).amin()
                            nonlinear_feasible[key].append(
                                bool(nonlinear_gap >= constraint_gap_m)
                            )
                        rows[key].append(_errors(model, prediction, target))
    finally:
        dataset.close()

    baseline = _mean(baseline_rows)
    arms: dict[str, object] = {}
    for pose_weight, friction, pose_query_factor in arm_keys:
        key = (pose_weight, friction, pose_query_factor)
        metrics = _mean(rows[key])
        arms[
            f"pose_weight={pose_weight:g},friction={friction:g},"
            f"pose_query={pose_query_factor:g}"
        ] = {
            "metrics": metrics,
            "change_percent_vs_free": {
                key: 100.0 * (metrics[key] / baseline[key] - 1.0)
                for key in metrics
            },
            "solver_statuses": statuses[key],
            "nonlinear_feasible_fraction": float(
                np.mean(nonlinear_feasible[key])
            ),
        }
    output = {
        "format_version": 1,
        "config": str(args.config),
        "split": args.split,
        "samples_per_object": args.samples_per_object,
        "coordinates": "[delta_position/L, left_rotation, delta_joint/travel]",
        "query": (
            "[alpha*(p_current-p_previous)/L, "
            "alpha*Log(R_current R_previous^T), "
            "(r_free(command_next)-r_current)/travel]"
        ),
        "linearization_state": "current observed state",
        "constraint_gap_m": constraint_gap_m,
        "friction_coefficients": friction_coefficients,
        "pose_query_factors": pose_query_factors,
        "current_nonlinear_feasible_fraction": float(np.mean(current_feasible)),
        "mean_contact_constraints": float(np.mean(contact_counts)),
        "free_baseline": baseline,
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
