from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor

from srno.data.dataset import LocalTransitionBatch
from srno.data.index import ActiveIndex, file_sha256
from srno.losses import state_error
from srno.model import SRNOModel
from srno.types import PoseState


REFERENCE_FORMAT_VERSION = 1
PHYSICAL_BASELINE_FORMAT_VERSION = 1
STEPS_PER_TRAJECTORY = 32


def effective_rho(epsilon_dx: float, configured_rho: float) -> float:
    """Return a scale-aware HPR penalty coefficient.

    With the default ``rho=1/epsilon**2``, the derivative of the quadratic
    term with respect to the squared functional drift is order one after a
    one-budget violation.  This avoids hiding a dimensionless state metric of
    order 1e-5 behind an ineffective order-one penalty coefficient.
    """

    if epsilon_dx <= 0:
        raise ValueError("epsilon_dx must be positive")
    if configured_rho < 0:
        raise ValueError("configured_rho cannot be negative")
    return configured_rho if configured_rho > 0 else 1.0 / (epsilon_dx * epsilon_dx)


def effective_physical_rho(baseline_loss: float, configured_rho: float) -> float:
    """Return the scale-aware HPR coefficient for a physical-loss bound."""

    if baseline_loss <= 0:
        raise ValueError("baseline_loss must be positive")
    if configured_rho < 0:
        raise ValueError("configured_rho cannot be negative")
    return configured_rho if configured_rho > 0 else 1.0 / baseline_loss


def hpr_inequality_penalty(constraint: Tensor, multiplier: float, rho: float) -> Tensor:
    r"""Projected Hestenes--Powell--Rockafellar inequality term.

    For ``c(theta) <= 0`` this is

    .. math::

       \frac{[\max(0,\mu+\rho c)]^2-\mu^2}{2\rho}.

    Unlike ``mu*c + rho/2*[c]_+^2``, its gradient vanishes for a sufficiently
    inactive constraint instead of rewarding unnecessary over-satisfaction.
    """

    if multiplier < 0:
        raise ValueError("multiplier cannot be negative")
    if rho <= 0:
        raise ValueError("rho must be positive")
    shifted = torch.clamp_min(constraint * rho + multiplier, 0.0)
    return (shifted.square() - multiplier * multiplier) / (2.0 * rho)


def update_multiplier(multiplier: float, constraint: float, rho: float) -> float:
    if multiplier < 0 or rho <= 0:
        raise ValueError("invalid HPR multiplier or penalty coefficient")
    return max(0.0, multiplier + rho * constraint)


def functional_drift(
    prediction: PoseState,
    frozen_reference: PoseState,
    *,
    length_scale: float,
    joint_scale: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Mean squared function-space distance and its physical components."""

    total, translation, rotation, joints = state_error(
        prediction,
        frozen_reference,
        length_scale=length_scale,
        joint_scale=joint_scale,
        lambda_rotation=1.0,
        lambda_joints=1.0,
    )
    return total.mean(), translation.mean(), rotation.mean(), joints.mean()


def physical_one_step_error(
    prediction: PoseState,
    target: PoseState,
    *,
    length_scale: float,
    joint_scale: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Mean true one-step state error and its T/R/J components."""

    return functional_drift(
        prediction,
        target,
        length_scale=length_scale,
        joint_scale=joint_scale,
    )


@dataclass(frozen=True)
class PhysicalBaseline:
    """Hashed scalar contract for the frozen local one-step physics loss."""

    manifest_sha256: str
    gripper_sha256: str
    active_index_sha256: str
    local_checkpoint_sha256: str
    loss: float
    translation: float
    rotation: float
    joints: float
    transitions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": PHYSICAL_BASELINE_FORMAT_VERSION,
            "constraint_target": "physical_one_step",
            "manifest_sha256": self.manifest_sha256,
            "gripper_sha256": self.gripper_sha256,
            "active_index_sha256": self.active_index_sha256,
            "local_checkpoint_sha256": self.local_checkpoint_sha256,
            "loss": self.loss,
            "translation": self.translation,
            "rotation": self.rotation,
            "joints": self.joints,
            "transitions": self.transitions,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        manifest_sha256: str,
        gripper_sha256: str,
        active_index_sha256: str,
        local_checkpoint_sha256: str,
    ) -> "PhysicalBaseline":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("format_version") != PHYSICAL_BASELINE_FORMAT_VERSION:
            raise ValueError("unsupported physical baseline version")
        if raw.get("constraint_target") != "physical_one_step":
            raise ValueError("physical baseline constraint_target mismatch")
        expected = {
            "manifest_sha256": manifest_sha256,
            "gripper_sha256": gripper_sha256,
            "active_index_sha256": active_index_sha256,
            "local_checkpoint_sha256": local_checkpoint_sha256,
        }
        for key, value in expected.items():
            if str(raw.get(key)) != value:
                raise ValueError(f"physical baseline {key} mismatch")
        numeric = {
            name: float(raw[name])
            for name in ("loss", "translation", "rotation", "joints")
        }
        transitions = int(raw["transitions"])
        if not all(math.isfinite(value) for value in numeric.values()):
            raise ValueError("physical baseline contains invalid metrics")
        if numeric["loss"] <= 0 or transitions <= 0:
            raise ValueError("physical baseline contains invalid metrics")
        if not math.isclose(
            numeric["loss"],
            numeric["translation"] + numeric["rotation"] + numeric["joints"],
            rel_tol=1e-7,
            abs_tol=1e-12,
        ):
            raise ValueError("physical baseline components do not sum to loss")
        return cls(**expected, **numeric, transitions=transitions)


@dataclass(frozen=True)
class FrozenLocalReference:
    """Dense CPU cache of frozen-local outputs on active physical states.

    Each object tensor is indexed by ``trajectory * 32 + step``.  Only active
    entries are valid; the boolean mask makes accidental use of an uncomputed
    transition a fail-fast error.
    """

    manifest_sha256: str
    gripper_sha256: str
    active_index_sha256: str
    local_checkpoint_sha256: str
    states: dict[str, PoseState]
    active_masks: dict[str, Tensor]

    @property
    def transition_count(self) -> int:
        return sum(int(mask.sum()) for mask in self.active_masks.values())

    def lookup(
        self,
        batch: LocalTransitionBatch,
        *,
        device: torch.device,
        non_blocking: bool = True,
    ) -> PoseState:
        sample_to_object = batch.sdf.sample_to_object.detach().cpu()
        trajectory = batch.trajectory_index.detach().cpu()
        step = batch.step_index.detach().cpu()
        flat_index = trajectory * STEPS_PER_TRAJECTORY + step
        count = int(flat_index.numel())
        rotation = torch.empty((count, 3, 3), dtype=torch.float32)
        position = torch.empty((count, 3), dtype=torch.float32)
        joints = torch.empty((count, 6), dtype=torch.float32)
        filled = torch.zeros(count, dtype=torch.bool)

        for object_index, object_id in enumerate(batch.object_ids):
            selected = torch.nonzero(sample_to_object == object_index, as_tuple=False).flatten()
            if not len(selected):
                continue
            if object_id not in self.states:
                raise KeyError(f"frozen reference has no object {object_id!r}")
            indices = flat_index.index_select(0, selected)
            mask = self.active_masks[object_id].index_select(0, indices)
            if not bool(mask.all()):
                raise ValueError(f"frozen reference misses an active transition for {object_id!r}")
            state = self.states[object_id]
            rotation[selected] = state.rotation.index_select(0, indices)
            position[selected] = state.position.index_select(0, indices)
            joints[selected] = state.joint_position.index_select(0, indices)
            filled[selected] = True
        if not bool(filled.all()):
            raise ValueError("frozen-reference lookup did not fill every batch sample")
        return PoseState(rotation, position, joints).to(
            device=device, non_blocking=non_blocking
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(
            {
                "format_version": REFERENCE_FORMAT_VERSION,
                "manifest_sha256": self.manifest_sha256,
                "gripper_sha256": self.gripper_sha256,
                "active_index_sha256": self.active_index_sha256,
                "local_checkpoint_sha256": self.local_checkpoint_sha256,
                "states": {
                    object_id: {
                        "rotation": state.rotation,
                        "position": state.position,
                        "joint_position": state.joint_position,
                        "active_mask": self.active_masks[object_id],
                    }
                    for object_id, state in self.states.items()
                },
            },
            temporary,
        )
        os.replace(temporary, destination)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        manifest_sha256: str,
        gripper_sha256: str,
        active_index_sha256: str,
        local_checkpoint_sha256: str,
    ) -> "FrozenLocalReference":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("format_version") != REFERENCE_FORMAT_VERSION:
            raise ValueError("unsupported frozen-local reference version")
        expected = {
            "manifest_sha256": manifest_sha256,
            "gripper_sha256": gripper_sha256,
            "active_index_sha256": active_index_sha256,
            "local_checkpoint_sha256": local_checkpoint_sha256,
        }
        for key, value in expected.items():
            if str(payload.get(key)) != value:
                raise ValueError(f"frozen-local reference {key} mismatch")
        states: dict[str, PoseState] = {}
        masks: dict[str, Tensor] = {}
        for object_id, raw in payload["states"].items():
            states[str(object_id)] = PoseState(
                raw["rotation"].float(),
                raw["position"].float(),
                raw["joint_position"].float(),
            )
            masks[str(object_id)] = raw["active_mask"].bool()
        return cls(**expected, states=states, active_masks=masks)


def build_frozen_local_reference(
    model: SRNOModel,
    loader: Iterable[LocalTransitionBatch],
    *,
    device: torch.device,
    autocast_context,
    active_index: ActiveIndex,
    active_index_path: str | Path,
    manifest_sha256: str,
    gripper_sha256: str,
    local_checkpoint_path: str | Path,
) -> FrozenLocalReference:
    """Evaluate ``theta_0`` once and cache all active training outputs."""

    states: dict[str, PoseState] = {}
    masks: dict[str, Tensor] = {}
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = raw_batch.to(device, non_blocking=True)
            with autocast_context():
                prediction = model.forward_step(batch.current, batch.next_command, batch.sdf)
            if not isinstance(prediction, PoseState):
                raise TypeError("SRNO forward_step did not return PoseState")
            prediction = prediction.to(device="cpu", dtype=torch.float32)
            sample_to_object = batch.sdf.sample_to_object.detach().cpu()
            flat_index = (
                batch.trajectory_index.detach().cpu() * STEPS_PER_TRAJECTORY
                + batch.step_index.detach().cpu()
            )
            for object_index, object_id in enumerate(batch.object_ids):
                selected = torch.nonzero(
                    sample_to_object == object_index, as_tuple=False
                ).flatten()
                if not len(selected):
                    continue
                if object_id not in states:
                    size = STEPS_PER_TRAJECTORY * (
                        int(active_index.pairs_for(object_id)[:, 0].max()) + 1
                    )
                    states[object_id] = PoseState(
                        torch.full((size, 3, 3), torch.nan),
                        torch.full((size, 3), torch.nan),
                        torch.full((size, 6), torch.nan),
                    )
                    masks[object_id] = torch.zeros(size, dtype=torch.bool)
                indices = flat_index.index_select(0, selected)
                if bool(masks[object_id].index_select(0, indices).any()):
                    raise ValueError(f"duplicate frozen reference entry for {object_id!r}")
                state = states[object_id]
                state.rotation[indices] = prediction.rotation.index_select(0, selected)
                state.position[indices] = prediction.position.index_select(0, selected)
                state.joint_position[indices] = prediction.joint_position.index_select(0, selected)
                masks[object_id][indices] = True

    expected_count = sum(
        len(active_index.pairs_for(object_id)) for object_id in states
    )
    actual_count = sum(int(mask.sum()) for mask in masks.values())
    if actual_count != expected_count:
        raise ValueError(
            f"frozen reference is incomplete: {actual_count} != {expected_count}"
        )
    if any(not torch.isfinite(state.position[mask]).all() for object_id, state in states.items() for mask in (masks[object_id],)):
        raise ValueError("frozen reference contains non-finite active entries")
    return FrozenLocalReference(
        manifest_sha256=manifest_sha256,
        gripper_sha256=gripper_sha256,
        active_index_sha256=file_sha256(active_index_path),
        local_checkpoint_sha256=file_sha256(local_checkpoint_path),
        states=states,
        active_masks=masks,
    )


def reference_contract(reference: FrozenLocalReference, path: str | Path) -> dict[str, Any]:
    return {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(path),
        "manifest_sha256": reference.manifest_sha256,
        "gripper_sha256": reference.gripper_sha256,
        "active_index_sha256": reference.active_index_sha256,
        "local_checkpoint_sha256": reference.local_checkpoint_sha256,
        "active_transitions": reference.transition_count,
    }
