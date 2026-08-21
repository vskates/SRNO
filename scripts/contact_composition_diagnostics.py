#!/usr/bin/env python3
"""Diagnose local amplification, signed bias and hidden contact state.

This runner is intentionally diagnostic-only.  It freezes the material-v2
dataset and simulator contract, evaluates the existing gap+aperture H32
models, and uses drive-error checkpoints only as contextual controls.  It does
not train a model or write to the dataset.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    from contact_memory_diagnostics import (
        ResetStates,
        _close_stage,
        _errors,
        _open_stage,
        _reset_successor,
    )
    from markov_state_sufficiency import (
        TransitionSet,
        _decode_names,
        _model_errors,
        _stats,
    )
except ModuleNotFoundError:  # Imported as ``scripts.*`` by pytest.
    from scripts.contact_memory_diagnostics import (
        ResetStates,
        _close_stage,
        _errors,
        _open_stage,
        _reset_successor,
    )
    from scripts.markov_state_sufficiency import (
        TransitionSet,
        _decode_names,
        _model_errors,
        _stats,
    )


SEEDS = (0, 1, 2)
STEP_BANDS = ((1, 8), (9, 16), (17, 24), (25, 31))
PERTURBATION_MODES = ("translation", "rotation", "joints")
PERTURBATION_SCALES = (0.005, 0.01)
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class PerturbationDirections:
    translation: np.ndarray
    rotation: np.ndarray
    joints: np.ndarray


@dataclass(frozen=True)
class AmplificationReplay:
    branch_labels: tuple[str, ...]
    current_position: np.ndarray
    current_quaternion_wxyz: np.ndarray
    current_joint: np.ndarray
    current_contact_count: np.ndarray
    successor_position: np.ndarray
    successor_quaternion_wxyz: np.ndarray
    successor_joint: np.ndarray
    successor_contact_count: np.ndarray
    current_settled: np.ndarray
    successor_settled: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _band_index(step: int) -> int:
    for index, (lower, upper) in enumerate(STEP_BANDS):
        if lower <= step <= upper:
            return index
    raise ValueError(f"contact state step is outside 1..31: {step}")


def _choose_stratified_indices(
    steps: np.ndarray,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    """Choose contact candidates evenly across the four fixed step bands."""

    steps = np.asarray(steps, dtype=np.int64)
    if steps.ndim != 1 or count <= 0 or len(steps) < count:
        raise ValueError("invalid candidates/count for stratified selection")
    rng = np.random.default_rng(seed)
    quota = np.full(len(STEP_BANDS), count // len(STEP_BANDS), dtype=np.int64)
    quota[: count % len(STEP_BANDS)] += 1
    chosen: list[int] = []
    deficits: list[int] = []
    for band, requested in zip(STEP_BANDS, quota, strict=True):
        lower, upper = band
        eligible = np.flatnonzero((steps >= lower) & (steps <= upper))
        take = min(int(requested), len(eligible))
        if take:
            chosen.extend(rng.choice(eligible, size=take, replace=False).tolist())
        deficits.extend([len(deficits)] * (int(requested) - take))
    remaining = count - len(chosen)
    if remaining:
        pool = np.setdiff1d(
            np.arange(len(steps)), np.asarray(chosen, dtype=np.int64)
        )
        # Prefer candidates closest to a band that missed its quota.  Random
        # jitter gives deterministic tie-breaking without favouring trajectory
        # order in the HDF5 shard.
        missing_centres = [
            0.5 * sum(STEP_BANDS[index])
            for index, requested in enumerate(quota)
            if np.count_nonzero(
                np.isin(
                    np.asarray(chosen, dtype=np.int64),
                    np.flatnonzero(
                        (steps >= STEP_BANDS[index][0])
                        & (steps <= STEP_BANDS[index][1])
                    ),
                )
            )
            < requested
        ]
        if missing_centres:
            distance = np.min(
                np.abs(steps[pool, None] - np.asarray(missing_centres)[None]),
                axis=1,
            )
        else:
            distance = np.zeros(len(pool), dtype=np.float64)
        order = np.lexsort((rng.random(len(pool)), distance))
        chosen.extend(pool[order[:remaining]].tolist())
    result = np.asarray(chosen, dtype=np.int64)
    rng.shuffle(result)
    if len(np.unique(result)) != count:
        raise RuntimeError("stratified selection produced duplicate candidates")
    return result


def _select_contact_transitions(
    manifest: Any,
    object_id: str,
    *,
    count: int,
    seed: int,
    joint_names: tuple[str, ...],
) -> TransitionSet:
    """Select sustained-contact transitions using exactly four step strata."""

    shard, group_name = manifest.object_locations()[object_id]
    with h5py.File(shard, "r") as handle:
        group = handle[group_name]
        position = np.asarray(group["position"], dtype=np.float32)
        quaternion = np.asarray(group["quaternion_xyzw"], dtype=np.float32)
        joint = np.asarray(group["joint_position"], dtype=np.float32)
        aperture = np.asarray(group["actual_aperture"], dtype=np.float32)
        contact = np.asarray(group["diagnostics/contact_count"], dtype=np.float32)
        source_pose_index = np.asarray(group["source_pose_index"], dtype=np.int64)
        names = _decode_names(group["joint_position"].attrs["joint_names"])
    if names != joint_names:
        raise ValueError(f"{object_id}: joint order does not match gripper asset")
    schedule = np.asarray(manifest.commanded_aperture_m, dtype=np.float32)
    lag = aperture[:, 1:32] - schedule[None, 1:32]
    sustained = (
        (contact[:, :31] > 0.0)
        & (contact[:, 1:32] > 0.0)
        & (lag > 1e-4)
    )
    candidates = np.argwhere(sustained)
    if len(candidates) < count:
        raise ValueError(
            f"{object_id}: only {len(candidates)} sustained-contact transitions, "
            f"requested {count}"
        )
    state_steps = candidates[:, 1] + 1
    selected = _choose_stratified_indices(state_steps, count=count, seed=seed)
    trajectory = candidates[selected, 0]
    current_step = state_steps[selected]
    return TransitionSet(
        object_id=object_id,
        trajectory=trajectory,
        source_pose_index=source_pose_index[trajectory],
        current_step=current_step,
        current_position=position[trajectory, current_step],
        current_quaternion_xyzw=quaternion[trajectory, current_step],
        current_joint=joint[trajectory, current_step],
        preserve_position=position[trajectory, current_step + 1],
        preserve_quaternion_xyzw=quaternion[trajectory, current_step + 1],
        preserve_joint=joint[trajectory, current_step + 1],
        preserve_aperture=aperture[trajectory, current_step + 1],
        current_contact_count=contact[trajectory, current_step - 1],
        next_contact_count=contact[trajectory, current_step],
        current_lag_m=lag[trajectory, current_step - 1],
    )


def _unit_vectors(rng: np.random.Generator, count: int, dimension: int) -> np.ndarray:
    values = rng.normal(size=(count, dimension))
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise RuntimeError("random perturbation direction has zero norm")
    return values / norm


def _make_perturbation_directions(
    current_joint: np.ndarray,
    *,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    joint_travel: np.ndarray,
    max_scale: float,
    seed: int,
) -> PerturbationDirections:
    """Create deterministic unit directions and reflect joints at limits."""

    joint = np.asarray(current_joint, dtype=np.float64)
    lower = np.asarray(joint_lower, dtype=np.float64)
    upper = np.asarray(joint_upper, dtype=np.float64)
    travel = np.asarray(joint_travel, dtype=np.float64)
    if joint.ndim != 2 or joint.shape[1] != 6:
        raise ValueError("current_joint must have shape [samples, 6]")
    if lower.shape != (6,) or upper.shape != (6,) or travel.shape != (6,):
        raise ValueError("joint limits/travel must have shape [6]")
    rng = np.random.default_rng(seed)
    translation = _unit_vectors(rng, len(joint), 3)
    rotation = _unit_vectors(rng, len(joint), 3)
    joints = _unit_vectors(rng, len(joint), 6) * math.sqrt(6.0)
    delta = max_scale * travel[None] * joints
    proposed = joint + delta
    flip = (proposed < lower[None] - 1e-7) | (proposed > upper[None] + 1e-7)
    joints = np.where(flip, -joints, joints)
    proposed = joint + max_scale * travel[None] * joints
    if np.any(proposed < lower[None] - 1e-6) or np.any(
        proposed > upper[None] + 1e-6
    ):
        raise ValueError("cannot construct a joint perturbation inside limits")
    rms = np.sqrt(np.mean(joints**2, axis=1))
    if not np.allclose(rms, 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("joint perturbation is not normalized in d_X")
    return PerturbationDirections(translation, rotation, joints)


def _so3_log(rotation: Any) -> Any:
    """Stable diagnostic SO(3) logarithm returning an axis-angle vector."""

    import torch

    matrix = rotation.to(dtype=torch.float64)
    cosine = ((matrix.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
        -1.0, 1.0
    )
    skew_vector = 0.5 * torch.stack(
        (
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ),
        dim=-1,
    )
    sine = torch.linalg.vector_norm(skew_vector, dim=-1)
    theta = torch.atan2(sine, cosine)
    theta2 = theta.square()
    small = theta < 1e-7
    factor = torch.where(
        small,
        1.0 + theta2 / 6.0 + 7.0 * theta2.square() / 360.0,
        theta / sine.clamp_min(1e-15),
    )
    omega = factor[..., None] * skew_vector

    near_pi = (math.pi - theta).abs() < 1e-5
    if bool(torch.any(near_pi)):
        selected = matrix[near_pi]
        diagonal = torch.diagonal(selected, dim1=-2, dim2=-1)
        axis = torch.sqrt(((diagonal + 1.0) * 0.5).clamp_min(0.0))
        signs = torch.sign(
            torch.stack(
                (
                    selected[:, 2, 1] - selected[:, 1, 2],
                    selected[:, 0, 2] - selected[:, 2, 0],
                    selected[:, 1, 0] - selected[:, 0, 1],
                ),
                dim=-1,
            )
        )
        axis = torch.where(signs == 0.0, axis, axis * signs)
        axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(
            1e-15
        )
        omega = omega.clone()
        omega[near_pi] = theta[near_pi, None] * axis
    return omega


def _se3_log(rotation: Any, translation: Any) -> Any:
    """Logarithm of an SE(3) transform, ordered as ``[v, omega]``."""

    import torch

    omega = _so3_log(rotation)
    theta2 = (omega * omega).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-30))
    small = theta2 < 1e-10
    coefficient = torch.where(
        small,
        1.0 / 12.0 + theta2 / 720.0 + theta2.square() / 30240.0,
        (1.0 - 0.5 * theta / torch.tan(0.5 * theta).clamp_min(1e-15))
        / theta2.clamp_min(1e-30),
    )
    from srno.geometry.se3 import skew

    omega_hat = skew(omega)
    identity = torch.eye(3, dtype=torch.float64, device=omega.device).expand(
        omega.shape[:-1] + (3, 3)
    )
    inverse_jacobian = (
        identity
        - 0.5 * omega_hat
        + coefficient[..., None] * (omega_hat @ omega_hat)
    )
    value = translation.to(dtype=torch.float64)
    velocity = (inverse_jacobian @ value.unsqueeze(-1)).squeeze(-1)
    return torch.cat((velocity, omega), dim=-1)


def _pose_log_error(target: Any, prediction: Any) -> Any:
    relative_rotation = target.rotation.transpose(-1, -2) @ prediction.rotation
    relative_translation = (
        target.rotation.transpose(-1, -2)
        @ (prediction.position - target.position).unsqueeze(-1)
    ).squeeze(-1)
    return _se3_log(relative_rotation, relative_translation)


def _state_metrics(left: Any, right: Any, *, length_scale: float, joint_scale: Any) -> dict[str, Any]:
    import torch

    from srno.geometry.se3 import rotation_geodesic_angle

    translation_m = torch.linalg.vector_norm(left.position - right.position, dim=-1)
    rotation = rotation_geodesic_angle(left.rotation, right.rotation)
    joints = torch.sqrt(
        (((left.joint_position - right.joint_position) / joint_scale) ** 2).mean(
            dim=-1
        )
    )
    dx = torch.sqrt((translation_m / length_scale) ** 2 + rotation**2 + joints**2)
    return {
        "dx": dx,
        "translation_m": translation_m,
        "rotation_rad": rotation,
        "joint_rmse_over_travel": joints,
    }


def _amplification_ratio(output_distance: Any, input_distance: Any) -> Any:
    return output_distance / input_distance.clamp_min(1e-12)


def _continuous_amplification_replay(
    app: Any,
    config: Any,
    catalog: Any,
    gripper: Any,
    selected: TransitionSet,
    directions: PerturbationDirections,
) -> AmplificationReplay:
    """Replay paired histories and branch at x_k into six perturbations."""

    import torch
    from isaaclab.utils.math import quat_apply, quat_mul

    from srno.sim.pose_seeds import PoseSeeds

    perturb_labels = tuple(
        f"{mode}:{scale:g}"
        for mode in PERTURBATION_MODES
        for scale in PERTURBATION_SCALES
    )
    labels = ("preserve", "repeat", "amplification_base") + perturb_labels
    sample_count = len(selected.current_step)
    branch_count = len(labels)
    source_index = np.tile(selected.source_pose_index, branch_count)
    current_step_np = np.tile(selected.current_step, branch_count)
    count = len(source_index)
    simulation = scene = collector = None
    try:
        simulation, scene, collector = _open_stage(
            app,
            config,
            catalog,
            gripper,
            selected.object_id,
            env_count=count,
            collision_system="PCM",
        )
        device = torch.device(scene.device)
        record = catalog.object(selected.object_id)
        seeds = PoseSeeds.load(record.pose_seed_path)
        seed_position = torch.from_numpy(seeds.position_m[source_index]).to(device)
        seed_quaternion = torch.from_numpy(seeds.quaternion_wxyz[source_index]).to(
            device
        )
        base_position, base_quaternion = collector._reset_batch(
            seed_position, seed_quaternion
        )
        required = torch.ones(count, dtype=torch.bool, device=device)
        initial = collector._settle_command(
            collector.open_joint_target, 0, required_mask=required
        )
        alive = initial.settled_mask.clone()
        current_step = torch.from_numpy(current_step_np).to(device=device)
        successor_step = current_step + 1

        current_position = torch.full((count, 3), torch.nan, device=device)
        current_quaternion = torch.full((count, 4), torch.nan, device=device)
        current_joint = torch.full_like(collector.robot.data.joint_pos, torch.nan)
        successor_position = torch.full((count, 3), torch.nan, device=device)
        successor_quaternion = torch.full((count, 4), torch.nan, device=device)
        successor_joint = torch.full_like(collector.robot.data.joint_pos, torch.nan)
        current_contact_count = torch.full((count,), torch.nan, device=device)
        successor_contact_count = torch.full((count,), torch.nan, device=device)
        current_settled = torch.zeros(count, dtype=torch.bool, device=device)
        successor_settled = torch.zeros(count, dtype=torch.bool, device=device)

        translation_direction = torch.from_numpy(directions.translation).to(
            device=device, dtype=torch.float32
        )
        rotation_direction = torch.from_numpy(directions.rotation).to(
            device=device, dtype=torch.float32
        )
        joint_direction = torch.from_numpy(directions.joints).to(
            device=device, dtype=torch.float32
        )
        joint_travel = gripper.joint_travel_range.to(device=device)
        perturb_specs: list[tuple[str, float | None]] = [("baseline", None)] + [
            (mode, scale)
            for mode in PERTURBATION_MODES
            for scale in PERTURBATION_SCALES
        ]

        max_step = int(successor_step.max().item())
        for command_index in range(1, max_step + 1):
            required = alive & (successor_step >= command_index)
            if not bool(torch.any(required)):
                break
            target = collector.close_joint_target * (float(command_index) / 32.0)
            result = collector._settle_command(
                target, command_index, required_mask=required
            )
            failed = required & ~result.settled_mask
            alive &= ~failed
            relative_position, relative_quaternion, _ = collector._record_state(
                base_position, base_quaternion, result
            )

            at_current = alive & (current_step == command_index)
            if bool(torch.any(at_current)):
                current_position[at_current] = relative_position[at_current]
                current_quaternion[at_current] = relative_quaternion[at_current]
                current_joint[at_current] = result.joint_position[at_current]
                current_contact_count[at_current] = result.contact_count[at_current]
                current_settled[at_current] = True

                for offset, (mode, scale) in enumerate(perturb_specs, start=2):
                    begin = offset * sample_count
                    end = begin + sample_count
                    branch_mask = at_current[begin:end]
                    local_ids = torch.nonzero(branch_mask).flatten()
                    if not len(local_ids):
                        continue
                    env_ids = local_ids + begin
                    sample_ids = local_ids
                    reference_ids = sample_ids
                    perturbed_position = relative_position[reference_ids].clone()
                    perturbed_quaternion = relative_quaternion[reference_ids].clone()
                    perturbed_joint = result.joint_position[reference_ids].clone()
                    if mode == "translation":
                        assert scale is not None
                        perturbed_position = (
                            perturbed_position
                            + float(scale)
                            * float(gripper.length_scale)
                            * translation_direction[sample_ids]
                        )
                    elif mode == "rotation":
                        assert scale is not None
                        rotation_vector = float(scale) * rotation_direction[sample_ids]
                        half = 0.5 * torch.linalg.vector_norm(
                            rotation_vector, dim=-1, keepdim=True
                        )
                        axis = rotation_vector / (2.0 * half).clamp_min(1e-12)
                        delta_quaternion = torch.cat(
                            (torch.cos(half), axis * torch.sin(half)), dim=-1
                        )
                        perturbed_quaternion = quat_mul(
                            delta_quaternion, perturbed_quaternion
                        )
                    elif mode == "joints":
                        assert scale is not None
                        perturbed_joint = (
                            perturbed_joint
                            + float(scale)
                            * joint_travel[None]
                            * joint_direction[sample_ids]
                        )
                    current_position[env_ids] = perturbed_position
                    current_quaternion[env_ids] = perturbed_quaternion
                    current_joint[env_ids] = perturbed_joint
                    object_state = collector.object.data.root_state_w[env_ids].clone()
                    object_state[:, :3] = base_position[env_ids] + quat_apply(
                        base_quaternion[env_ids], perturbed_position
                    )
                    object_state[:, 3:7] = quat_mul(
                        base_quaternion[env_ids], perturbed_quaternion
                    )
                    object_state[:, 7:13] = 0.0
                    collector.object.write_root_state_to_sim(
                        object_state, env_ids=env_ids
                    )
                    collector.robot.write_joint_state_to_sim(
                        perturbed_joint,
                        torch.zeros_like(perturbed_joint),
                        env_ids=env_ids,
                    )

            at_successor = alive & (successor_step == command_index)
            if bool(torch.any(at_successor)):
                successor_position[at_successor] = relative_position[at_successor]
                successor_quaternion[at_successor] = relative_quaternion[at_successor]
                successor_joint[at_successor] = result.joint_position[at_successor]
                successor_contact_count[at_successor] = result.contact_count[at_successor]
                successor_settled[at_successor] = True

        def shaped(value: Any) -> np.ndarray:
            array = value.detach().cpu().numpy()
            return array.reshape((branch_count, sample_count) + array.shape[1:])

        return AmplificationReplay(
            labels,
            shaped(current_position),
            shaped(current_quaternion),
            shaped(current_joint),
            shaped(current_contact_count),
            shaped(successor_position),
            shaped(successor_quaternion),
            shaped(successor_joint),
            shaped(successor_contact_count),
            shaped(current_settled),
            shaped(successor_settled),
        )
    finally:
        if simulation is not None:
            _close_stage(app, simulation, scene, collector)


def _state_from_numpy(
    position: np.ndarray,
    quaternion_wxyz: np.ndarray,
    joint: np.ndarray,
    *,
    device: Any,
) -> Any:
    import torch

    from srno.geometry.se3 import quaternion_xyzw_to_matrix
    from srno.types import PoseState

    quaternion_xyzw = torch.from_numpy(quaternion_wxyz[:, (1, 2, 3, 0)]).to(device)
    return PoseState(
        quaternion_xyzw_to_matrix(quaternion_xyzw),
        torch.from_numpy(position).to(device),
        torch.from_numpy(joint).to(device),
    )


def _sdf_for_selected(manifest: Any, selected: TransitionSet, *, device: Any) -> Any:
    import torch

    from srno.types import SDFBatch

    shard, group_name = manifest.object_locations()[selected.object_id]
    with h5py.File(shard, "r") as handle:
        group = handle[group_name]
        sdf = torch.from_numpy(np.asarray(group["sdf"], dtype=np.float32)).to(device)
        origin = torch.from_numpy(
            np.asarray(group.attrs["grid_origin"], dtype=np.float32)
        ).to(device)
        voxel = torch.from_numpy(
            np.asarray(group.attrs["voxel_size"], dtype=np.float32)
        ).to(device)
    return SDFBatch(
        sdf[None],
        origin[None],
        voxel[None],
        torch.zeros(len(selected.current_step), dtype=torch.long, device=device),
        manifest.sdf_scale_m,
    )


def _evaluate_model_amplification(
    model: Any,
    manifest: Any,
    selected: TransitionSet,
    replay: AmplificationReplay,
    *,
    device: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    reference = _state_from_numpy(
        replay.current_position[2],
        replay.current_quaternion_wxyz[2],
        replay.current_joint[2],
        device=device,
    )
    sdf = _sdf_for_selected(manifest, selected, device=device)
    schedule = torch.as_tensor(
        np.asarray(manifest.commanded_aperture_m, dtype=np.float32), device=device
    )
    command = schedule[
        torch.from_numpy(selected.current_step + 1).to(device=device)
    ]
    with torch.no_grad():
        reference_output, reference_aux = model.forward_step(
            reference, command, sdf, return_aux=True
        )
    amplification = np.full(
        (len(PERTURBATION_MODES), len(PERTURBATION_SCALES), len(command)),
        np.nan,
        dtype=np.float64,
    )
    input_distance = np.full_like(amplification, np.nan)
    gate_switch = np.zeros_like(amplification, dtype=bool)
    for mode_index, _ in enumerate(PERTURBATION_MODES):
        for scale_index, _ in enumerate(PERTURBATION_SCALES):
            branch = 3 + mode_index * len(PERTURBATION_SCALES) + scale_index
            perturbed = _state_from_numpy(
                replay.current_position[branch],
                replay.current_quaternion_wxyz[branch],
                replay.current_joint[branch],
                device=device,
            )
            with torch.no_grad():
                response, aux = model.forward_step(
                    perturbed, command, sdf, return_aux=True
                )
            denominator = _state_metrics(
                perturbed,
                reference,
                length_scale=model.length_scale,
                joint_scale=model.joint_travel_range,
            )["dx"]
            numerator = _state_metrics(
                response,
                reference_output,
                length_scale=model.length_scale,
                joint_scale=model.joint_travel_range,
            )["dx"]
            amplification[mode_index, scale_index] = (
                _amplification_ratio(numerator, denominator).cpu().numpy()
            )
            input_distance[mode_index, scale_index] = denominator.cpu().numpy()
            gate_switch[mode_index, scale_index] = (
                aux.active != reference_aux.active
            ).cpu().numpy()
    return amplification, input_distance, gate_switch


def _simulator_amplification(
    replay: AmplificationReplay,
    *,
    length_scale: float,
    joint_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    joint_scale_tensor = torch.as_tensor(
        joint_scale, dtype=torch.float32, device="cuda:0"
    )
    sample_count = replay.current_position.shape[1]
    amplification = np.full(
        (len(PERTURBATION_MODES), len(PERTURBATION_SCALES), sample_count),
        np.nan,
    )
    noise = np.full_like(amplification, np.nan)
    input_distance = np.full_like(amplification, np.nan)
    valid = np.zeros_like(amplification, dtype=bool)
    reference_successor = (
        replay.successor_position[2],
        replay.successor_quaternion_wxyz[2],
        replay.successor_joint[2],
    )
    repeat_error = _errors(
        replay.successor_position[0],
        replay.successor_quaternion_wxyz[0],
        replay.successor_joint[0],
        replay.successor_position[1],
        replay.successor_quaternion_wxyz[1],
        replay.successor_joint[1],
        length_scale=length_scale,
        joint_scale=joint_scale_tensor,
        device="cuda:0",
    )["dx"]
    for mode_index, _ in enumerate(PERTURBATION_MODES):
        for scale_index, scale in enumerate(PERTURBATION_SCALES):
            branch = 3 + mode_index * len(PERTURBATION_SCALES) + scale_index
            incoming = _errors(
                replay.current_position[2],
                replay.current_quaternion_wxyz[2],
                replay.current_joint[2],
                replay.current_position[branch],
                replay.current_quaternion_wxyz[branch],
                replay.current_joint[branch],
                length_scale=length_scale,
                joint_scale=joint_scale_tensor,
                device="cuda:0",
            )["dx"]
            outgoing = _errors(
                *reference_successor,
                replay.successor_position[branch],
                replay.successor_quaternion_wxyz[branch],
                replay.successor_joint[branch],
                length_scale=length_scale,
                joint_scale=joint_scale_tensor,
                device="cuda:0",
            )["dx"]
            input_distance[mode_index, scale_index] = incoming
            amplification[mode_index, scale_index] = outgoing / np.clip(
                incoming, 1e-12, None
            )
            noise[mode_index, scale_index] = repeat_error / float(scale)
            valid[mode_index, scale_index] = (
                replay.current_settled[0]
                & replay.successor_settled[0]
                & replay.successor_settled[1]
                & replay.current_settled[2]
                & replay.successor_settled[2]
                & replay.current_settled[branch]
                & replay.successor_settled[branch]
                & np.isfinite(incoming)
                & np.isfinite(outgoing)
            )
    return amplification, noise, input_distance, valid


def _replayed_transition(
    selected: TransitionSet,
    replay: AmplificationReplay,
    model: Any,
) -> TransitionSet:
    import torch

    target_joint = replay.successor_joint[0]
    with torch.no_grad():
        aperture = (
            model.aperture_from_joints(
                torch.from_numpy(target_joint).to(model.surface_local_points.device)
            )
            .cpu()
            .numpy()
        )
    return replace(
        selected,
        current_position=replay.current_position[0],
        current_quaternion_xyzw=replay.current_quaternion_wxyz[0][
            :, (1, 2, 3, 0)
        ],
        current_joint=replay.current_joint[0],
        preserve_position=replay.successor_position[0],
        preserve_quaternion_xyzw=replay.successor_quaternion_wxyz[0][
            :, (1, 2, 3, 0)
        ],
        preserve_joint=target_joint,
        preserve_aperture=aperture,
    )


def _checkpoint_config(config: Any, *, arm: str, seed: int) -> Any:
    if arm != "aperture":
        raise ValueError(
            "drive_error checkpoints use a removed model contract and are "
            "historical artifacts only"
        )
    return replace(
        config,
        seed=seed,
        model=replace(
            config.model,
            contact_features="gap",
            pose_update="se3_left",
            history_conditioning="none",
        ),
    )


def _normalized_saved_model(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    normalized.setdefault("contact_features", "gap")
    legacy_conditioning = normalized.pop("global_conditioning", "aperture")
    if legacy_conditioning != "aperture":
        normalized["removed_global_conditioning"] = legacy_conditioning
    normalized.setdefault("operator_layers", 1)
    normalized.setdefault("pose_update", "se3_left")
    normalized.setdefault("history_conditioning", "none")
    return normalized


def _verify_checkpoint_contract(
    checkpoint: dict[str, Any],
    *,
    config: Any,
    manifest: Any,
    stage: str,
    horizon: int | None,
) -> None:
    if checkpoint.get("stage") != stage:
        raise ValueError(f"checkpoint stage must be {stage!r}")
    if horizon is not None and int(checkpoint.get("horizon", -1)) != horizon:
        raise ValueError(f"checkpoint horizon must be {horizon}")
    if checkpoint.get("manifest_sha256") != manifest.sha256():
        raise ValueError("checkpoint manifest hash mismatch")
    if checkpoint.get("gripper_sha256") != manifest.gripper_sha256:
        raise ValueError("checkpoint gripper hash mismatch")
    saved = checkpoint.get("config", {})
    expected = config.to_dict()
    for key in ("model", "loss", "optimizer", "loader", "training", "seed"):
        actual = saved.get(key)
        if key == "model":
            actual = _normalized_saved_model(actual)
        if actual != expected.get(key):
            raise ValueError(f"checkpoint {key} violates frozen diagnostic config")


def _load_model(
    config: Any,
    manifest: Any,
    checkpoint_path: Path,
    *,
    device: Any,
    stage: str,
    horizon: int | None,
) -> tuple[Any, dict[str, Any]]:
    import torch

    from srno.training.checkpoint import load_checkpoint
    from srno.training.engine import _build_model

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    model = _build_model(config, manifest, device)
    checkpoint = load_checkpoint(checkpoint_path, model=model, map_location=device)
    _verify_checkpoint_contract(
        checkpoint,
        config=config,
        manifest=manifest,
        stage=stage,
        horizon=horizon,
    )
    model.eval()
    return model, checkpoint


def _finite(*arrays: np.ndarray) -> np.ndarray:
    result = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        result &= np.isfinite(array).reshape(len(array), -1).all(axis=1)
    return result


def _append(rows: dict[str, list[np.ndarray]], name: str, value: Any) -> None:
    rows.setdefault(name, []).append(np.asarray(value))


def _concatenate(rows: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.concatenate(values, axis=0) for name, values in rows.items()}


def _equal_object_stats(values: np.ndarray, object_index: np.ndarray) -> dict[str, Any]:
    by_object = {
        str(index): _stats(values[object_index == index])
        for index in np.unique(object_index)
    }
    means = np.asarray([entry["mean"] for entry in by_object.values()])
    return {
        "pooled": _stats(values),
        "equal_object_mean": float(means.mean()),
        "by_object_index": by_object,
    }


def _bootstrap_log_amplification_ratio(
    model: np.ndarray,
    simulator: np.ndarray,
    object_index: np.ndarray,
    valid: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Hierarchical seed -> object -> transition bootstrap."""

    model = np.asarray(model, dtype=np.float64)
    simulator = np.asarray(simulator, dtype=np.float64)
    if not np.any(valid):
        return {
            "replicates": 0,
            "seed": seed,
            "mean_log_model_over_sim": None,
            "ci95_lower": None,
            "ci95_upper": None,
            "model_systematically_more_expansive": False,
        }
    objects = np.unique(object_index)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_seeds = rng.choice(model.shape[0], size=model.shape[0], replace=True)
        seed_means: list[float] = []
        for model_seed in sampled_seeds:
            sampled_objects = rng.choice(objects, size=len(objects), replace=True)
            object_means: list[float] = []
            for object_id in sampled_objects:
                candidates = np.flatnonzero(valid & (object_index == object_id))
                if not len(candidates):
                    continue
                sampled = rng.choice(candidates, size=len(candidates), replace=True)
                ratio = np.log(
                    np.clip(model[int(model_seed), sampled], 1e-12, None)
                    / np.clip(simulator[sampled], 1e-12, None)
                )
                object_means.append(float(np.mean(ratio)))
            if object_means:
                seed_means.append(float(np.mean(object_means)))
        samples[replicate] = float(np.mean(seed_means))
    return {
        "replicates": replicates,
        "seed": seed,
        "mean_log_model_over_sim": float(samples.mean()),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "model_systematically_more_expansive": bool(
            np.quantile(samples, 0.025) > 0.0
        ),
    }


def _hierarchical_curve_bootstrap(
    values: np.ndarray,
    object_index: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Bootstrap a [seed, sample, step, component] signed curve."""

    data = np.asarray(values, dtype=np.float64)
    objects = np.unique(object_index)
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates,) + data.shape[2:], dtype=np.float64)
    for replicate in range(replicates):
        sampled_seeds = rng.choice(data.shape[0], size=data.shape[0], replace=True)
        seed_means: list[np.ndarray] = []
        for model_seed in sampled_seeds:
            sampled_objects = rng.choice(objects, size=len(objects), replace=True)
            object_means: list[np.ndarray] = []
            for object_id in sampled_objects:
                candidates = np.flatnonzero(object_index == object_id)
                selected = rng.choice(candidates, size=len(candidates), replace=True)
                object_means.append(data[int(model_seed), selected].mean(axis=0))
            seed_means.append(np.mean(object_means, axis=0))
        samples[replicate] = np.mean(seed_means, axis=0)
    return {
        "mean": data.reshape((-1,) + data.shape[2:]).mean(axis=0),
        "ci95_lower": np.quantile(samples, 0.025, axis=0),
        "ci95_upper": np.quantile(samples, 0.975, axis=0),
    }


def _bootstrap_history_ratio(
    warm: np.ndarray,
    model_error: np.ndarray,
    object_index: np.ndarray,
    valid: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    objects = np.unique(object_index)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_seeds = rng.choice(
            model_error.shape[0], size=model_error.shape[0], replace=True
        )
        seed_ratios: list[float] = []
        for model_seed in sampled_seeds:
            sampled_objects = rng.choice(objects, size=len(objects), replace=True)
            warm_means: list[float] = []
            model_means: list[float] = []
            for object_id in sampled_objects:
                candidates = np.flatnonzero(valid & (object_index == object_id))
                selected = rng.choice(candidates, size=len(candidates), replace=True)
                warm_means.append(float(warm[selected].mean()))
                model_means.append(
                    float(model_error[int(model_seed), selected].mean())
                )
            seed_ratios.append(float(np.mean(warm_means) / np.mean(model_means)))
        samples[replicate] = float(np.mean(seed_ratios))
    point = float(
        np.mean(
            [
                warm[valid & (object_index == object_id)].mean()
                for object_id in objects
            ]
        )
        / np.mean(
            [
                model_error[:, valid & (object_index == object_id)].mean()
                for object_id in objects
            ]
        )
    )
    lower = float(np.quantile(samples, 0.025))
    upper = float(np.quantile(samples, 0.975))
    if upper < 0.25:
        classification = "negligible"
    elif 0.5 <= point <= 2.0 and lower >= 0.25:
        classification = "comparable"
    else:
        classification = "intermediate"
    return {
        "replicates": replicates,
        "seed": seed,
        "gamma_history": point,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "classification": classification,
    }


def _runtime_contract_audit(
    app: Any,
    sim_config: Any,
    catalog: Any,
    gripper: Any,
    object_id: str,
) -> dict[str, Any]:
    from srno.sim.actuator_audit import verify_runtime_actuator

    simulation = scene = collector = None
    try:
        simulation, scene, collector = _open_stage(
            app,
            sim_config,
            catalog,
            gripper,
            object_id,
            env_count=1,
            collision_system="PCM",
        )
        audit = verify_runtime_actuator(
            collector.robot,
            expected_joint_names=gripper.joint_names,
            relaxation=sim_config.relaxation,
            position_target=collector.open_joint_target,
        )
        limits = (
            collector.robot.root_physx_view.get_dof_limits()[0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        if limits.shape != (6, 2) or np.any(limits[:, 0] >= limits[:, 1]):
            raise RuntimeError(f"invalid runtime PhysX joint limits: {limits.tolist()}")
        audit["physx_position_limits_rad"] = limits.tolist()
        return audit
    finally:
        if simulation is not None:
            _close_stage(app, simulation, scene, collector)


def _verify_frozen_contract(manifest: Any, sim_config: Any) -> None:
    from srno.sim.physx_material import expected_physics_metadata

    expected = expected_physics_metadata(sim_config)
    if expected.canonical_json() != manifest.physics.canonical_json():
        raise ValueError("simulator physics metadata differs from material-v2 manifest")
    if sim_config.material.static_friction != 2.4:
        raise ValueError("frozen static friction must be 2.4")
    if sim_config.material.dynamic_friction != 2.0:
        raise ValueError("frozen dynamic friction must be 2.0")
    if sim_config.material.friction_combine_mode != "min":
        raise ValueError("frozen friction combine mode must be min")
    if not sim_config.material.strong_friction_enabled:
        raise ValueError("frozen material-v2 contract requires strong friction")
    if manifest.physics.contact_generation != "PCM":
        raise ValueError("frozen contact generation must be PCM")
    for name, actual, intended in (
        ("delta_gate_m", sim_config.dataset.delta_gate_m, manifest.delta_gate_m),
        (
            "contact_offset_sum_m",
            sim_config.dataset.contact_offset_sum_m,
            manifest.contact_offset_sum_m,
        ),
        ("sdf_scale_m", sim_config.dataset.sdf_scale_m, manifest.sdf_scale_m),
    ):
        if not np.isclose(actual, intended, rtol=0.0, atol=1e-12):
            raise ValueError(f"frozen {name} mismatch: {actual} != {intended}")
    if not sim_config.headless:
        raise ValueError("composition diagnostics require headless=true")
    if not np.isclose(sim_config.memory_limit_gib, 14.0, rtol=0.0, atol=1e-12):
        raise ValueError("composition diagnostics require the 14 GiB watchdog")


def _evaluate_bias_for_model(
    model: Any,
    config: Any,
    manifest: Any,
    *,
    device: Any,
    object_order: tuple[str, ...],
) -> dict[str, np.ndarray]:
    import torch

    from srno.data.dataset import H5ObjectDataset, TrajectoryBatch, make_dataloader
    from srno.training.engine import _autocast
    from srno.types import PoseState

    rows: dict[str, list[np.ndarray]] = {}
    global_object = {object_id: index for index, object_id in enumerate(object_order)}
    for split in ("val", "test"):
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
                    if object_id not in global_object:
                        continue
                    batch = raw_batch.to(device)
                    initial = PoseState(
                        batch.states.rotation[:, 0],
                        batch.states.position[:, 0],
                        batch.states.joint_position[:, 0],
                    )
                    with _autocast(config, device):
                        autoregressive = model.rollout(
                            initial, batch.command_schedule[1:33], batch.sdf
                        )
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
                    target_rollout = PoseState(
                        batch.states.rotation[:, 1:33],
                        batch.states.position[:, 1:33],
                        batch.states.joint_position[:, 1:33],
                    )
                    predicted_rollout = PoseState(
                        autoregressive.rotation[:, 1:33],
                        autoregressive.position[:, 1:33],
                        autoregressive.joint_position[:, 1:33],
                    )
                    tf_log_tensor = torch.stack(tf_log, dim=1)
                    tf_spatial_tensor = torch.stack(tf_spatial, dim=1)
                    ar_log_tensor = _pose_log_error(
                        target_rollout, predicted_rollout
                    )
                    ar_spatial_tensor = (
                        predicted_rollout.position - target_rollout.position
                    )
                    shard, group_name = manifest.object_locations()[object_id]
                    with h5py.File(shard, "r") as handle:
                        outgoing_contact = np.asarray(
                            handle[group_name]["diagnostics/contact_count"],
                            dtype=np.float32,
                        ) > 0.0
                    trajectory = raw_batch.trajectory_index.cpu().numpy()
                    _append(rows, "tf_log", tf_log_tensor.cpu().numpy())
                    _append(rows, "ar_log", ar_log_tensor.cpu().numpy())
                    _append(rows, "tf_spatial", tf_spatial_tensor.cpu().numpy())
                    _append(rows, "ar_spatial", ar_spatial_tensor.cpu().numpy())
                    _append(rows, "contact", outgoing_contact[trajectory])
                    _append(
                        rows,
                        "object_index",
                        np.full(
                            len(trajectory), global_object[object_id], dtype=np.int32
                        ),
                    )
                    _append(rows, "trajectory", trajectory.astype(np.int32))
                    _append(
                        rows,
                        "split",
                        np.full(len(trajectory), 0 if split == "val" else 1, dtype=np.int8),
                    )
        finally:
            dataset.close()
    return _concatenate(rows)


def _bias_summary(
    values: np.ndarray,
    object_index: np.ndarray,
    *,
    replicates: int,
) -> dict[str, Any]:
    tf = _hierarchical_curve_bootstrap(
        values[:, :, :, :6], object_index, replicates=replicates, seed=BOOTSTRAP_SEED
    )
    cumulative_values = np.cumsum(values[:, :, :, :6], axis=2)
    cumulative = _hierarchical_curve_bootstrap(
        cumulative_values,
        object_index,
        replicates=replicates,
        seed=BOOTSTRAP_SEED + 1,
    )
    return {
        "mean": tf["mean"].tolist(),
        "ci95_lower": tf["ci95_lower"].tolist(),
        "ci95_upper": tf["ci95_upper"].tolist(),
        "cumulative_mean": cumulative["mean"].tolist(),
        "cumulative_ci95_lower": cumulative["ci95_lower"].tolist(),
        "cumulative_ci95_upper": cumulative["ci95_upper"].tolist(),
    }


def _persistent_bias(
    tf_values: np.ndarray,
    object_index: np.ndarray,
    *,
    replicates: int,
) -> dict[str, Any]:
    band_values = np.stack(
        [
            tf_values[:, :, lower - 1 : upper, :].mean(axis=2)
            for lower, upper in STEP_BANDS
        ],
        axis=2,
    )
    summary = _hierarchical_curve_bootstrap(
        band_values,
        object_index,
        replicates=replicates,
        seed=BOOTSTRAP_SEED + 2,
    )
    mean = summary["mean"]
    lower = summary["ci95_lower"]
    upper = summary["ci95_upper"]
    signs = np.sign(mean)
    excludes_zero = (lower > 0.0) | (upper < 0.0)
    persistent: list[bool] = []
    for component in range(mean.shape[-1]):
        valid_signs = signs[:, component][excludes_zero[:, component]]
        persistent.append(
            bool(
                len(valid_signs) >= 3
                and max(
                    np.count_nonzero(valid_signs > 0),
                    np.count_nonzero(valid_signs < 0),
                )
                >= 3
            )
        )
    return {
        "band_mean": mean.tolist(),
        "band_ci95_lower": lower.tolist(),
        "band_ci95_upper": upper.tolist(),
        "persistent_components_vx_vy_vz_wx_wy_wz": persistent,
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-15:
        return None
    return float(np.dot(left, right) / denominator)


def _jsonable_stats(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return {"count": 0}
    return _stats(finite)


def _plot_amplification(output: Path, arrays: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = arrays["amplification_model"]
    simulator = arrays["amplification_sim"]
    noise = arrays["amplification_noise"]
    valid = arrays["amplification_valid"].astype(bool)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for mode_index, mode in enumerate(PERTURBATION_MODES):
        for scale_index, scale in enumerate(PERTURBATION_SCALES):
            ax = axes[scale_index, mode_index]
            mask = valid[:, mode_index, scale_index]
            sim = simulator[mask, mode_index, scale_index]
            mod = model[:, mask, mode_index, scale_index].reshape(-1)
            repeated_sim = np.tile(sim, model.shape[0])
            ax.scatter(repeated_sim, mod, s=14, alpha=0.55, label="model vs sim")
            upper = max(
                1.0,
                float(np.nanquantile(np.concatenate((repeated_sim, mod)), 0.98)),
            )
            ax.plot((0, upper), (0, upper), "k--", linewidth=1)
            ax.axvline(
                float(np.nanmedian(noise[mask, mode_index, scale_index])),
                color="#C00000",
                linestyle=":",
                label="median noise",
            )
            ax.set(
                xlabel=r"$A_{sim}$",
                ylabel=r"$A_{model}$",
                title=f"{mode}, epsilon={scale:g}",
            )
            ax.grid(alpha=0.2)
            if mode_index == 0 and scale_index == 0:
                ax.legend()
    fig.savefig(output / "amplification.png", dpi=180)
    plt.close(fig)


def _plot_bias(output: Path, bias: dict[str, dict[int, dict[str, np.ndarray]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ("vx", "vy", "vz", "wx", "wy", "wz")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for arm, color in (("aperture", "#4472C4"), ("drive_error", "#E67E22")):
        tf = np.stack([bias[arm][seed]["tf_log"] for seed in SEEDS])
        ar = np.stack([bias[arm][seed]["ar_log"] for seed in SEEDS])
        tf_mean = tf.mean(axis=(0, 1))
        ar_mean = ar.mean(axis=(0, 1))
        cumulative = np.cumsum(tf_mean, axis=0)
        for component, ax in enumerate(axes.flat):
            step = np.arange(1, 33)
            ax.plot(step, cumulative[:, component], color=color, label=f"{arm} sum TF")
            ax.plot(
                step,
                ar_mean[:, component],
                color=color,
                linestyle="--",
                label=f"{arm} AR",
            )
            ax.axhline(0.0, color="black", linewidth=0.7)
            ax.set_title(labels[component])
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output / "signed_pose_bias.png", dpi=180)
    plt.close(fig)


def _plot_history(output: Path, arrays: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = arrays["history_valid"].astype(bool)
    values = [
        arrays["history_repeat_dx"][valid],
        arrays["history_cold_dx"][valid],
        arrays["history_warm_dx"][valid],
        arrays["history_precondition_dx"][valid],
        arrays["history_model_drive_dx"][:, valid].reshape(-1),
        arrays["history_model_aperture_dx"][:, valid].reshape(-1),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].boxplot(
        values,
        tick_labels=("repeat", "cold", "warm", "precond", "drive local", "aperture local"),
        showfliers=False,
    )
    axes[0].set_yscale("log")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].set_ylabel(r"$d_X$")
    axes[0].set_title("History discrepancy and model error")
    axes[1].scatter(
        arrays["history_warm_dx"][valid],
        arrays["history_model_drive_dx"][:, valid].mean(axis=0),
        c=arrays["selection_step"][valid],
        cmap="viridis",
        s=32,
    )
    upper = float(
        np.quantile(
            np.concatenate(
                (
                    arrays["history_warm_dx"][valid],
                    arrays["history_model_drive_dx"][:, valid].reshape(-1),
                )
            ),
            0.98,
        )
    )
    axes[1].plot((0, upper), (0, upper), "k--", linewidth=1)
    axes[1].set(
        xlabel=r"$D_{warm}$",
        ylabel=r"$E_{local}^{drive}$",
        title="Same transition; colour = closure step",
    )
    fig.savefig(output / "history_markov.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/simulator-r-v1/manifest.json")
    )
    parser.add_argument(
        "--sim-config", type=Path, default=Path("configs/simulator-r.toml")
    )
    parser.add_argument(
        "--train-config", type=Path, default=Path("configs/srno-r-material-v2.toml")
    )
    parser.add_argument(
        "--aperture-rollout-root",
        type=Path,
        default=Path("runs/ablation-actuator-rollout/aperture"),
    )
    parser.add_argument(
        "--drive-rollout-root",
        type=Path,
        default=Path("runs/ablation-actuator-rollout/drive_error"),
    )
    parser.add_argument(
        "--aperture-local-root",
        type=Path,
        default=Path("runs/ablation-jq-local/baseline"),
    )
    parser.add_argument(
        "--drive-local-root",
        type=Path,
        default=Path("runs/ablation-actuator-local/drive_error"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/contact-composition-diagnostics-v2"),
    )
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--samples-per-object", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--skip-bias", action="store_true")
    args = parser.parse_args()
    if args.samples_per_object <= 0:
        parser.error("--samples-per-object must be positive")
    if args.bootstrap_replicates <= 0:
        parser.error("--bootstrap-replicates must be positive")

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
        from srno.training.config import ExperimentConfig

        manifest = DatasetManifest.load(args.manifest)
        sim_config = SimulatorConfig.load(args.sim_config)
        train_config = ExperimentConfig.load(args.train_config)
        _verify_frozen_contract(manifest, sim_config)
        if train_config.paths.manifest.resolve() != args.manifest.resolve():
            raise ValueError("train config and diagnostic manifest differ")
        gripper = GripperAsset.load(manifest.gripper_path)
        if gripper.sha256() != manifest.gripper_sha256:
            raise ValueError("manifest gripper hash mismatch")
        catalog = SimulatorAssetCatalog.load(sim_config.catalog)
        objects = tuple(
            args.objects
            or (tuple(manifest.splits["val"]) + tuple(manifest.splits["test"]))
        )
        expected_objects = set(manifest.splits["val"]) | set(manifest.splits["test"])
        if not set(objects) <= expected_objects:
            raise ValueError("diagnostic objects must belong to val/test splits")
        object_split = {
            object_id: "val" if object_id in manifest.splits["val"] else "test"
            for object_id in objects
        }
        device = torch.device(sim_config.device)

        models: dict[str, dict[str, dict[int, Any]]] = {
            "rollout": {"aperture": {}, "drive_error": {}},
            "local": {"aperture": {}, "drive_error": {}},
        }
        checkpoint_metadata: dict[str, Any] = {"rollout": {}, "local": {}}
        for arm, root in (
            ("aperture", args.aperture_rollout_root),
            ("drive_error", args.drive_rollout_root),
        ):
            checkpoint_metadata["rollout"][arm] = {}
            for seed in SEEDS:
                config = _checkpoint_config(train_config, arm=arm, seed=seed)
                path = root / f"seed-{seed}" / "best-rollout-h32.pt"
                model, checkpoint = _load_model(
                    config,
                    manifest,
                    path,
                    device=device,
                    stage="rollout",
                    horizon=32,
                )
                models["rollout"][arm][seed] = model
                checkpoint_metadata["rollout"][arm][str(seed)] = {
                    "path": str(path.resolve()),
                    "epoch": int(checkpoint["epoch"]),
                    "best_metric": float(checkpoint["best_metric"]),
                }
        for arm, root in (
            ("aperture", args.aperture_local_root),
            ("drive_error", args.drive_local_root),
        ):
            checkpoint_metadata["local"][arm] = {}
            for seed in SEEDS:
                config = _checkpoint_config(train_config, arm=arm, seed=seed)
                path = root / f"seed-{seed}" / "best-local.pt"
                model, checkpoint = _load_model(
                    config,
                    manifest,
                    path,
                    device=device,
                    stage="local",
                    horizon=None,
                )
                models["local"][arm][seed] = model
                checkpoint_metadata["local"][arm][str(seed)] = {
                    "path": str(path.resolve()),
                    "epoch": int(checkpoint["epoch"]),
                    "best_metric": float(checkpoint["best_metric"]),
                }

        print("[COMPOSITION] runtime material + actuator contract audit", flush=True)
        actuator_audit = _runtime_contract_audit(
            app, sim_config, catalog, gripper, objects[0]
        )

        rows: dict[str, list[np.ndarray]] = {}
        model_amp_rows: dict[int, list[np.ndarray]] = {seed: [] for seed in SEEDS}
        model_gate_rows: dict[int, list[np.ndarray]] = {seed: [] for seed in SEEDS}
        history_components = (
            "dx",
            "translation_m",
            "rotation_rad",
            "joint_rmse_over_travel",
        )
        history_model_rows: dict[
            str, dict[int, dict[str, list[np.ndarray]]]
        ] = {
            arm: {
                seed: {component: [] for component in history_components}
                for seed in SEEDS
            }
            for arm in ("aperture", "drive_error")
        }
        runtime_joint_limits = np.asarray(
            actuator_audit["physx_position_limits_rad"], dtype=np.float64
        )
        joint_lower = runtime_joint_limits[:, 0]
        joint_upper = runtime_joint_limits[:, 1]
        joint_travel = gripper.joint_travel_range.cpu().numpy()
        joint_travel_tensor = gripper.joint_travel_range.to(device)

        for object_index, object_id in enumerate(objects):
            print(f"[COMPOSITION] select/replay {object_id}", flush=True)
            selected = _select_contact_transitions(
                manifest,
                object_id,
                count=args.samples_per_object,
                seed=args.seed + object_index,
                joint_names=gripper.joint_names,
            )
            directions = _make_perturbation_directions(
                selected.current_joint,
                joint_lower=joint_lower,
                joint_upper=joint_upper,
                joint_travel=joint_travel,
                max_scale=max(PERTURBATION_SCALES),
                seed=args.seed + 1000 + object_index,
            )
            replay = _continuous_amplification_replay(
                app, sim_config, catalog, gripper, selected, directions
            )
            sim_amp, sim_noise, sim_input, sim_valid = _simulator_amplification(
                replay,
                length_scale=gripper.length_scale,
                joint_scale=joint_travel,
            )
            _append(rows, "amplification_sim", np.moveaxis(sim_amp, -1, 0))
            _append(rows, "amplification_noise", np.moveaxis(sim_noise, -1, 0))
            _append(rows, "amplification_input_sim", np.moveaxis(sim_input, -1, 0))
            _append(rows, "amplification_valid", np.moveaxis(sim_valid, -1, 0))
            branch_contact = replay.successor_contact_count[3:].reshape(
                len(PERTURBATION_MODES),
                len(PERTURBATION_SCALES),
                len(selected.current_step),
            )
            contact_switch = branch_contact != replay.successor_contact_count[2][
                None, None, :
            ]
            _append(
                rows,
                "amplification_contact_switch",
                np.moveaxis(contact_switch, -1, 0),
            )

            for seed in SEEDS:
                model_amp, _, gate_switch = _evaluate_model_amplification(
                    models["rollout"]["aperture"][seed],
                    manifest,
                    selected,
                    replay,
                    device=device,
                )
                model_amp_rows[seed].append(np.moveaxis(model_amp, -1, 0))
                model_gate_rows[seed].append(np.moveaxis(gate_switch, -1, 0))

            print(f"[COMPOSITION] cold/warm fresh branches {object_id}", flush=True)
            reset: dict[str, ResetStates] = {}
            for label, precondition in (
                ("cold_a", False),
                ("cold_b", False),
                ("warm_a", True),
                ("warm_b", True),
            ):
                reset[label] = _reset_successor(
                    app,
                    sim_config,
                    catalog,
                    gripper,
                    selected,
                    current_position=replay.current_position[0],
                    current_quaternion_wxyz=replay.current_quaternion_wxyz[0],
                    current_joint=replay.current_joint[0],
                    collision_system="PCM",
                    precondition=precondition,
                )
            reference = (
                replay.successor_position[0],
                replay.successor_quaternion_wxyz[0],
                replay.successor_joint[0],
            )
            history: dict[str, dict[str, np.ndarray]] = {
                "repeat": _errors(
                    *reference,
                    replay.successor_position[1],
                    replay.successor_quaternion_wxyz[1],
                    replay.successor_joint[1],
                    length_scale=gripper.length_scale,
                    joint_scale=joint_travel_tensor,
                    device=device,
                ),
                "cold": _errors(
                    *reference,
                    reset["cold_a"].successor_position,
                    reset["cold_a"].successor_quaternion_wxyz,
                    reset["cold_a"].successor_joint,
                    length_scale=gripper.length_scale,
                    joint_scale=joint_travel_tensor,
                    device=device,
                ),
                "cold_repeat": _errors(
                    reset["cold_a"].successor_position,
                    reset["cold_a"].successor_quaternion_wxyz,
                    reset["cold_a"].successor_joint,
                    reset["cold_b"].successor_position,
                    reset["cold_b"].successor_quaternion_wxyz,
                    reset["cold_b"].successor_joint,
                    length_scale=gripper.length_scale,
                    joint_scale=joint_travel_tensor,
                    device=device,
                ),
                "warm": _errors(
                    *reference,
                    reset["warm_a"].successor_position,
                    reset["warm_a"].successor_quaternion_wxyz,
                    reset["warm_a"].successor_joint,
                    length_scale=gripper.length_scale,
                    joint_scale=joint_travel_tensor,
                    device=device,
                ),
                "warm_repeat": _errors(
                    reset["warm_a"].successor_position,
                    reset["warm_a"].successor_quaternion_wxyz,
                    reset["warm_a"].successor_joint,
                    reset["warm_b"].successor_position,
                    reset["warm_b"].successor_quaternion_wxyz,
                    reset["warm_b"].successor_joint,
                    length_scale=gripper.length_scale,
                    joint_scale=joint_travel_tensor,
                    device=device,
                ),
                "precondition": _errors(
                    replay.current_position[0],
                    replay.current_quaternion_wxyz[0],
                    replay.current_joint[0],
                    reset["warm_a"].warm_position,
                    reset["warm_a"].warm_quaternion_wxyz,
                    reset["warm_a"].warm_joint,
                    length_scale=gripper.length_scale,
                    joint_scale=joint_travel_tensor,
                    device=device,
                ),
            }
            for prefix, metrics in history.items():
                for name, value in metrics.items():
                    _append(rows, f"history_{prefix}_{name}", value)

            replayed = _replayed_transition(
                selected, replay, models["local"]["aperture"][0]
            )
            for arm in ("aperture", "drive_error"):
                for seed in SEEDS:
                    with torch.no_grad():
                        error = _model_errors(
                            replayed,
                            manifest,
                            gripper,
                            models["local"][arm][seed],
                            device=device,
                        )
                    for component in history_components:
                        history_model_rows[arm][seed][component].append(
                            error[component]
                        )

            valid_history = (
                replay.current_settled[0]
                & replay.successor_settled[0]
                & replay.successor_settled[1]
                & reset["cold_a"].successor_settled
                & reset["cold_b"].successor_settled
                & reset["warm_a"].successor_settled
                & reset["warm_b"].successor_settled
                & reset["warm_a"].warm_settled
                & reset["warm_b"].warm_settled
                & _finite(
                    replay.successor_position[0],
                    reset["cold_a"].successor_position,
                    reset["warm_a"].successor_position,
                )
            )
            _append(rows, "history_valid", valid_history)
            _append(
                rows,
                "selection_object_index",
                np.full(len(selected.current_step), object_index, dtype=np.int32),
            )
            _append(rows, "selection_step", selected.current_step.astype(np.int32))
            _append(
                rows,
                "selection_step_band",
                np.asarray([_band_index(int(step)) for step in selected.current_step], dtype=np.int8),
            )
            _append(rows, "selection_trajectory", selected.trajectory.astype(np.int32))
            _append(
                rows,
                "selection_split",
                np.full(
                    len(selected.current_step),
                    0 if object_split[object_id] == "val" else 1,
                    dtype=np.int8,
                ),
            )

        arrays = _concatenate(rows)
        arrays["amplification_model"] = np.stack(
            [np.concatenate(model_amp_rows[seed], axis=0) for seed in SEEDS]
        )
        arrays["amplification_gate_switch"] = np.stack(
            [np.concatenate(model_gate_rows[seed], axis=0) for seed in SEEDS]
        )
        for arm in ("aperture", "drive_error"):
            short_arm = "drive" if arm == "drive_error" else "aperture"
            for component in history_components:
                arrays[f"history_model_{short_arm}_{component}"] = np.stack(
                    [
                        np.concatenate(history_model_rows[arm][seed][component])
                        for seed in SEEDS
                    ]
                )
        arrays["object_labels"] = np.asarray(objects)

        bias: dict[str, dict[int, dict[str, np.ndarray]]] = {
            "aperture": {},
            "drive_error": {},
        }
        if not args.skip_bias:
            for arm in ("aperture", "drive_error"):
                for seed in SEEDS:
                    print(f"[COMPOSITION] signed bias arm={arm} seed={seed}", flush=True)
                    config = _checkpoint_config(train_config, arm=arm, seed=seed)
                    bias[arm][seed] = _evaluate_bias_for_model(
                        models["rollout"][arm][seed],
                        config,
                        manifest,
                        device=device,
                        object_order=objects,
                    )
                    for name, value in bias[arm][seed].items():
                        arrays[f"bias_{arm}_seed{seed}_{name}"] = value

        object_index = arrays["selection_object_index"]
        amplification_summary: dict[str, Any] = {}
        for mode_index, mode in enumerate(PERTURBATION_MODES):
            amplification_summary[mode] = {}
            for scale_index, scale in enumerate(PERTURBATION_SCALES):
                valid = arrays["amplification_valid"][:, mode_index, scale_index].astype(bool)
                sim = arrays["amplification_sim"][:, mode_index, scale_index]
                noise = arrays["amplification_noise"][:, mode_index, scale_index]
                model_values = arrays["amplification_model"][:, :, mode_index, scale_index]
                late = valid & (arrays["selection_step"] >= 17)
                amplification_summary[mode][str(scale)] = {
                    "simulator": _equal_object_stats(sim[valid], object_index[valid]),
                    "noise": _equal_object_stats(noise[valid], object_index[valid]),
                    "model": _equal_object_stats(
                        model_values[:, valid].reshape(-1),
                        np.tile(object_index[valid], len(SEEDS)),
                    ),
                    "input_dx_sim": _jsonable_stats(
                        arrays["amplification_input_sim"][:, mode_index, scale_index][valid]
                    ),
                    "gate_switch_fraction": float(
                        arrays["amplification_gate_switch"][:, valid, mode_index, scale_index].mean()
                    ),
                    "simulator_contact_count_switch_fraction": float(
                        arrays["amplification_contact_switch"][
                            valid, mode_index, scale_index
                        ].mean()
                    ),
                    "bootstrap_all": _bootstrap_log_amplification_ratio(
                        model_values,
                        sim,
                        object_index,
                        valid,
                        replicates=args.bootstrap_replicates,
                        seed=BOOTSTRAP_SEED,
                    ),
                    "bootstrap_late_steps_17_31": _bootstrap_log_amplification_ratio(
                        model_values,
                        sim,
                        object_index,
                        late,
                        replicates=args.bootstrap_replicates,
                        seed=BOOTSTRAP_SEED + 1,
                    ),
                    "by_step_band": {
                        str(band): {
                            "simulator": _jsonable_stats(
                                sim[valid & (arrays["selection_step_band"] == band)]
                            ),
                            "model": _jsonable_stats(
                                model_values[
                                    :, valid & (arrays["selection_step_band"] == band)
                                ]
                            ),
                        }
                        for band in range(4)
                    },
                }

        history_valid = arrays["history_valid"].astype(bool)
        history_summary = {
            name: {
                component: _equal_object_stats(
                    arrays[f"history_{name}_{component}"][history_valid],
                    object_index[history_valid],
                )
                for component in (
                    "dx",
                    "translation_m",
                    "rotation_rad",
                    "joint_rmse_over_travel",
                )
            }
            for name in (
                "repeat",
                "cold",
                "cold_repeat",
                "warm",
                "warm_repeat",
                "precondition",
            )
        }
        history_summary["local_model"] = {}
        for arm in ("aperture", "drive_error"):
            short_arm = "drive" if arm == "drive_error" else "aperture"
            history_summary["local_model"][arm] = {
                component: _equal_object_stats(
                    arrays[f"history_model_{short_arm}_{component}"][
                        :, history_valid
                    ].reshape(-1),
                    np.tile(object_index[history_valid], len(SEEDS)),
                )
                for component in history_components
            }
        history_summary["gamma_history"] = _bootstrap_history_ratio(
            arrays["history_warm_dx"],
            arrays["history_model_drive_dx"],
            object_index,
            history_valid,
            replicates=args.bootstrap_replicates,
            seed=BOOTSTRAP_SEED,
        )
        warm_over_repeat = _bootstrap_history_ratio(
            arrays["history_warm_dx"],
            np.broadcast_to(
                arrays["history_repeat_dx"][None, :],
                arrays["history_model_drive_dx"].shape,
            ),
            object_index,
            history_valid,
            replicates=args.bootstrap_replicates,
            seed=BOOTSTRAP_SEED + 5,
        )
        history_summary["warm_over_continuous_repeat"] = {
            "ratio": warm_over_repeat["gamma_history"],
            "ci95_lower": warm_over_repeat["ci95_lower"],
            "ci95_upper": warm_over_repeat["ci95_upper"],
            "resolved_above_repeatability_floor": bool(
                warm_over_repeat["ci95_lower"] > 1.0
            ),
        }
        history_summary["interpretation"] = (
            "resolved_history_effect"
            if warm_over_repeat["ci95_lower"] > 1.0
            else "noise_limited"
        )

        bias_summary: dict[str, Any] = {}
        if not args.skip_bias:
            for arm in ("aperture", "drive_error"):
                tf = np.stack([bias[arm][seed]["tf_log"] for seed in SEEDS])
                ar = np.stack([bias[arm][seed]["ar_log"] for seed in SEEDS])
                spatial_tf = np.stack(
                    [bias[arm][seed]["tf_spatial"] for seed in SEEDS]
                )
                bias_object = bias[arm][0]["object_index"]
                tf_summary = _bias_summary(
                    tf, bias_object, replicates=args.bootstrap_replicates
                )
                ar_curve = _hierarchical_curve_bootstrap(
                    ar,
                    bias_object,
                    replicates=args.bootstrap_replicates,
                    seed=BOOTSTRAP_SEED + 3,
                )
                spatial_curve = _hierarchical_curve_bootstrap(
                    spatial_tf,
                    bias_object,
                    replicates=args.bootstrap_replicates,
                    seed=BOOTSTRAP_SEED + 4,
                )
                tf_final = np.asarray(tf_summary["cumulative_mean"])[-1]
                ar_final = ar_curve["mean"][-1]
                bias_summary[arm] = {
                    "teacher_forced": tf_summary,
                    "autoregressive": {
                        "mean": ar_curve["mean"].tolist(),
                        "ci95_lower": ar_curve["ci95_lower"].tolist(),
                        "ci95_upper": ar_curve["ci95_upper"].tolist(),
                    },
                    "teacher_forced_spatial_translation_m": {
                        "mean": spatial_curve["mean"].tolist(),
                        "ci95_lower": spatial_curve["ci95_lower"].tolist(),
                        "ci95_upper": spatial_curve["ci95_upper"].tolist(),
                    },
                    "persistent_teacher_forced_bias": _persistent_bias(
                        tf, bias_object, replicates=args.bootstrap_replicates
                    ),
                    "final_alignment": {
                        "translation_cosine_sum_tf_vs_ar": _cosine(
                            tf_final[:3], ar_final[:3]
                        ),
                        "rotation_cosine_sum_tf_vs_ar": _cosine(
                            tf_final[3:], ar_final[3:]
                        ),
                    },
                }

        result = {
            "definition": {
                "dx": "sqrt((||dp||/L)^2 + theta(R,R*)^2 + mean(((r-r*)/travel)^2))",
                "A_model": "dX(R_theta(x+delta,u),R_theta(x,u))/dX(x+delta,x)",
                "A_sim": "dX(F_sim(x+delta,u),F_sim(x,u))/dX(x+delta,x)",
                "signed_tf_bias": "E[Log(inv(q*_{k+1}) qhat^{TF}_{k+1})]",
                "signed_ar_bias": "E[Log(inv(q*_k) qhat^{AR}_k)]",
                "D_cold": "dX(x_preserve_{k+1},x_fresh_cold_{k+1})",
                "D_warm": "dX(x_preserve_{k+1},x_fresh_warm_{k+1})",
                "gamma_history": "E[D_warm]/E[E_local_drive]",
                "warm_over_continuous_repeat": "E[D_warm]/E[D_repeat]",
            },
            "configuration": {
                "manifest": str(args.manifest.resolve()),
                "manifest_sha256": manifest.sha256(),
                "gripper_sha256": manifest.gripper_sha256,
                "sim_config": str(args.sim_config.resolve()),
                "sim_config_sha256": _sha256(args.sim_config.resolve()),
                "train_config": str(args.train_config.resolve()),
                "objects": list(objects),
                "object_split": object_split,
                "samples_per_object": args.samples_per_object,
                "selection_seed": args.seed,
                "step_bands": [list(value) for value in STEP_BANDS],
                "perturbation_scales_dx": list(PERTURBATION_SCALES),
                "perturbation_modes": list(PERTURBATION_MODES),
                "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "production_model": {
                    "contact_features": "gap",
                    "global_conditioning": "aperture",
                },
                "diagnostic_only_controls": ["drive_error"],
                "delta_gate_m": manifest.delta_gate_m,
                "contact_offset_sum_m": manifest.contact_offset_sum_m,
                "sdf_scale_m": manifest.sdf_scale_m,
                "physics": json.loads(manifest.physics.canonical_json()),
                "runtime_material_audit": {
                    "verified": True,
                    "live_readback": {
                        "static_friction": manifest.physics.static_friction,
                        "dynamic_friction": manifest.physics.dynamic_friction,
                        "friction_combine_mode": (
                            manifest.physics.friction_combine_mode
                        ),
                        "restitution": manifest.physics.restitution,
                        "contact_damping": manifest.physics.contact_damping,
                        "strong_friction_enabled": (
                            manifest.physics.strong_friction_enabled
                        ),
                    },
                },
                "checkpoints": checkpoint_metadata,
                "runtime_actuator_audit": actuator_audit,
            },
            "sample_counts": {
                "requested": len(objects) * args.samples_per_object,
                "history_common_valid": int(history_valid.sum()),
                "bias_trajectories": 0
                if args.skip_bias
                else int(len(bias["aperture"][0]["object_index"])),
            },
            "amplification": amplification_summary,
            "signed_pose_bias": bias_summary,
            "history_markov": history_summary,
            "decision": {
                "production_remains": "gap+aperture",
                "architecture_or_physics_changed": False,
                "implicit_solver_implemented": False,
            },
        }

        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "results.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        np.savez_compressed(args.output / "samples.npz", **arrays)
        _plot_amplification(args.output, arrays)
        if not args.skip_bias:
            _plot_bias(args.output, bias)
        _plot_history(args.output, arrays)
        print(
            "[COMPOSITION] completed: "
            f"results={args.output / 'results.json'}, "
            f"valid_history={int(history_valid.sum())}/{len(history_valid)}",
            flush=True,
        )
    except BaseException:
        # SimulationApp.close() may terminate Kit before Python prints an
        # uncaught exception, so emit the diagnostic traceback first.
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()
        watchdog.stop()


if __name__ == "__main__":
    main()
