"""Quasistatic 32-step trajectory collection inside an initialized Isaac app."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from isaaclab.utils.math import (
    quat_apply,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

from srno.geometry.gripper import GripperAsset
from srno.sim.assets import ObjectAssetRecord, SimulatorAssetCatalog
from srno.sim.config import SimulatorConfig
from srno.sim.isaac_scene import CONTROL_DECIMATION, SIM_DT
from srno.sim.pose_seeds import PoseSeeds


ROOT_TO_BASE_POSITION = (0.0, -0.045, 0.0)
ROOT_TO_BASE_QUATERNION_WXYZ = (0.7071055, 0.7071081, 0.0, 0.0)
ALIGNMENT_QUATERNION_WXYZ = (0.70710678, 0.70710678, 0.0, 0.0)
ALIGNMENT_POSITION = (0.0, 0.0, 0.045)


@dataclass(frozen=True)
class CollectedTrajectories:
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    actual_aperture: np.ndarray
    diagnostics: dict[str, np.ndarray]
    source_pose_index: np.ndarray


@dataclass(frozen=True)
class SettlingResult:
    steps: int
    environment_steps: torch.Tensor
    object_position: torch.Tensor
    object_quaternion_wxyz: torch.Tensor
    joint_position: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor
    contact_count: torch.Tensor
    actuator_effort: torch.Tensor
    settled_mask: torch.Tensor


class QuasistaticCollector:
    def __init__(
        self,
        simulation: object,
        scene: object,
        catalog: SimulatorAssetCatalog,
        record: ObjectAssetRecord,
        config: SimulatorConfig,
        gripper_asset: GripperAsset,
    ) -> None:
        self.simulation = simulation
        self.scene = scene
        self.catalog = catalog
        self.record = record
        self.config = config
        self.gripper_asset = gripper_asset.to(scene.device)
        self.robot = scene["robot"]
        self.object = scene["object"]
        self.device = torch.device(scene.device)
        self.env_count = scene.num_envs
        self.env_ids = torch.arange(self.env_count, device=self.device)
        self.commanded_aperture = torch.flip(self.gripper_asset.aperture_knots, dims=(0,))
        self.close_joint_target = self._ordered_joint_target(
            catalog.gripper.close_joint_position_rad
        )
        self.open_joint_target = torch.zeros_like(self.close_joint_target)
        self._root_position = torch.zeros((self.env_count, 3), device=self.device)
        self._root_quaternion = torch.zeros((self.env_count, 4), device=self.device)

    def _ordered_joint_target(self, values: object) -> torch.Tensor:
        names = self.robot.data.joint_names
        try:
            positions = [float(values[name]) for name in names]
        except KeyError as error:
            raise RuntimeError(f"runtime gripper has unexpected joint {error.args[0]!r}") from error
        return torch.tensor(positions, device=self.device, dtype=torch.float32)

    def _base_to_root(
        self, base_position: torch.Tensor, base_quaternion: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        correction = torch.tensor(
            ALIGNMENT_QUATERNION_WXYZ, device=self.device, dtype=torch.float32
        ).expand(len(base_position), -1)
        base_to_root = quat_inv(
            torch.tensor(
                ROOT_TO_BASE_QUATERNION_WXYZ, device=self.device, dtype=torch.float32
            ).expand(len(base_position), -1)
        )
        effective_rotation = quat_mul(base_to_root, correction)
        root_quaternion = quat_mul(base_quaternion, effective_rotation)
        root_to_base = torch.tensor(
            ROOT_TO_BASE_POSITION, device=self.device, dtype=torch.float32
        ).expand(len(base_position), -1)
        position_correction = torch.tensor(
            ALIGNMENT_POSITION, device=self.device, dtype=torch.float32
        ).expand(len(base_position), -1)
        effective_position = quat_apply(correction, root_to_base) + position_correction
        root_position = base_position - quat_apply(root_quaternion, effective_position)
        return root_position, root_quaternion

    def _hold_root(self) -> None:
        state = self.robot.data.root_state_w.clone()
        state[:, :3] = self._root_position
        state[:, 3:7] = self._root_quaternion
        state[:, 7:13] = 0.0
        self.robot.write_root_state_to_sim(state)

    def _control_step(self, joint_target: torch.Tensor) -> torch.Tensor:
        target = joint_target.expand(self.env_count, -1)
        self._hold_root()
        self.robot.set_joint_position_target(target)
        self.scene.write_data_to_sim()
        peak_contact_count = torch.zeros(
            self.env_count, device=self.device, dtype=torch.float32
        )
        for substep in range(CONTROL_DECIMATION):
            # In GUI mode render the final physics substep of every control
            # step.  Besides drawing the viewport, Kit uses this call to pump
            # window events; running every step with render=False makes the OS
            # label an otherwise healthy collection process as unresponsive.
            render = not self.config.headless and substep + 1 == CONTROL_DECIMATION
            self.simulation.step(render=render)
            self.scene.update(SIM_DT)
            peak_contact_count = torch.maximum(
                peak_contact_count, self._contact_count()
            )
        return peak_contact_count

    def _reset_batch(
        self,
        base_position_object: torch.Tensor,
        base_quaternion_wxyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        origins = self.scene.env_origins
        self.scene.reset()
        target_base_position = origins + base_position_object
        target_base_quaternion = base_quaternion_wxyz
        self._root_position, self._root_quaternion = self._base_to_root(
            target_base_position, target_base_quaternion
        )
        robot_state = self.robot.data.root_state_w.clone()
        robot_state[:, :3] = self._root_position
        robot_state[:, 3:7] = self._root_quaternion
        robot_state[:, 7:13] = 0.0
        self.robot.write_root_state_to_sim(robot_state)
        self.robot.write_joint_state_to_sim(
            torch.zeros_like(self.robot.data.joint_pos),
            torch.zeros_like(self.robot.data.joint_vel),
        )

        object_start = origins.clone()
        spawn_quaternion = torch.tensor(
            self.record.spawn_quaternion_wxyz, device=self.device, dtype=torch.float32
        ).expand(self.env_count, -1)
        object_state = self.object.data.root_state_w.clone()
        object_state[:, :3] = object_start
        object_state[:, 3:7] = spawn_quaternion
        object_state[:, 7:13] = 0.0
        self.object.write_root_state_to_sim(object_state)
        self._hold_root()
        return target_base_position, target_base_quaternion

    def _settle_command(
        self,
        target: torch.Tensor,
        command_index: int,
        required_mask: torch.Tensor | None = None,
    ) -> SettlingResult:
        cfg = self.config.settling
        required = (
            torch.ones(self.env_count, dtype=torch.bool, device=self.device)
            if required_mask is None
            else required_mask.to(device=self.device, dtype=torch.bool)
        )
        if required.shape != (self.env_count,) or not bool(torch.any(required)):
            raise ValueError("required_mask must select at least one environment")
        previous = self.object.data.root_pos_w.clone()
        previous_quaternion = self.object.data.root_quat_w.clone()
        previous_joint_position = self.robot.data.joint_pos.clone()
        stable = torch.zeros(self.env_count, dtype=torch.long, device=self.device)
        finished = torch.zeros(self.env_count, dtype=torch.bool, device=self.device)
        environment_steps = torch.full(
            (self.env_count,), cfg.max_steps, dtype=torch.int32, device=self.device
        )
        peak_contact_count = torch.zeros(
            self.env_count, device=self.device, dtype=torch.float32
        )
        captured_position = previous.clone()
        captured_orientation = previous_quaternion.clone()
        captured_joint_position = previous_joint_position.clone()
        captured_linear_velocity = torch.zeros_like(previous)
        captured_angular_velocity = torch.zeros_like(previous)
        captured_contact_count = torch.zeros_like(peak_contact_count)
        captured_actuator_effort = torch.zeros_like(peak_contact_count)
        started = perf_counter()
        for step in range(cfg.max_steps):
            peak_contact_count = torch.maximum(
                peak_contact_count, self._control_step(target)
            )
            position = self.object.data.root_pos_w.clone()
            orientation = self.object.data.root_quat_w.clone()
            joint_position = self.robot.data.joint_pos.clone()
            position_delta = torch.linalg.vector_norm(position - previous, dim=-1)
            delta_quaternion = quat_mul(orientation, quat_inv(previous_quaternion))
            delta_quaternion = torch.where(
                delta_quaternion[:, :1] < 0.0,
                -delta_quaternion,
                delta_quaternion,
            )
            sin_half_angle = torch.linalg.vector_norm(delta_quaternion[:, 1:], dim=-1)
            rotation_delta = 2.0 * torch.atan2(
                sin_half_angle, delta_quaternion[:, 0].clamp_min(0.0)
            )
            angular_scale = torch.where(
                sin_half_angle > 1e-8,
                rotation_delta / sin_half_angle.clamp_min(1e-8),
                torch.full_like(sin_half_angle, 2.0),
            )
            joint_delta = torch.linalg.vector_norm(
                joint_position - previous_joint_position, dim=-1
            )
            control_dt = SIM_DT * CONTROL_DECIMATION
            finite_linear_velocity = (position - previous) / control_dt
            finite_angular_velocity = (
                delta_quaternion[:, 1:] * angular_scale[:, None] / control_dt
            )
            object_settled = (
                (position_delta <= cfg.position_delta_m)
                & (position_delta <= cfg.linear_velocity_m_s * control_dt)
                & (rotation_delta <= cfg.angular_velocity_rad_s * control_dt)
            )
            joints_settled = (
                joint_delta <= cfg.joint_velocity_rad_s * control_dt
            )
            settled = object_settled & joints_settled
            if step + 1 >= cfg.min_steps:
                stable = torch.where(settled, stable + 1, torch.zeros_like(stable))
                newly_finished = required & ~finished & (
                    stable >= cfg.consecutive_steps
                )
                captured_position[newly_finished] = position[newly_finished]
                captured_orientation[newly_finished] = orientation[newly_finished]
                captured_joint_position[newly_finished] = joint_position[
                    newly_finished
                ]
                captured_linear_velocity[newly_finished] = finite_linear_velocity[
                    newly_finished
                ]
                captured_angular_velocity[newly_finished] = finite_angular_velocity[
                    newly_finished
                ]
                captured_contact_count[newly_finished] = peak_contact_count[
                    newly_finished
                ]
                captured_actuator_effort[newly_finished] = torch.abs(
                    self.robot.data.applied_torque[newly_finished]
                ).amax(dim=-1)
                environment_steps[newly_finished] = step + 1
                finished |= newly_finished
                # Physics and tensor reads are asynchronous on CUDA.  A host-side
                # truth conversion synchronizes the complete pipeline, so poll in
                # small blocks instead of forcing a GPU round-trip every step.
                check_interval = min(5, cfg.consecutive_steps)
                if (step + 1) % check_interval == 0 and bool(
                    torch.all(finished | ~required)
                ):
                    result = SettlingResult(
                        step + 1,
                        environment_steps,
                        captured_position,
                        captured_orientation,
                        captured_joint_position,
                        captured_linear_velocity,
                        captured_angular_velocity,
                        captured_contact_count,
                        captured_actuator_effort,
                        finished,
                    )
                    self._restore_settled_states(result, required)
                    return result
            if (step + 1) % 25 == 0 and perf_counter() - started >= 5.0:
                print(
                    f"[SRNO] {self.record.object_id}: command {command_index:02d}/32 "
                    f"still settling at control step {step + 1}/{cfg.max_steps} "
                    f"({perf_counter() - started:.1f}s)",
                    flush=True,
                )
            previous = position
            previous_quaternion = orientation
            previous_joint_position = joint_position
        failed_mask = required & ~finished
        failed = torch.nonzero(failed_mask).flatten().tolist()
        max_position_delta = float(position_delta[failed_mask].amax().item())
        max_linear_speed = float((position_delta[failed_mask] / control_dt).amax().item())
        max_angular_speed = float((rotation_delta[failed_mask] / control_dt).amax().item())
        max_joint_speed = float((joint_delta[failed_mask] / control_dt).amax().item())
        print(
            f"[SRNO] {self.record.object_id}: command {command_index:02d} rejected "
            f"non-settling envs={failed[:16]} after {cfg.max_steps} control steps; "
            f"max_position_delta={max_position_delta:.6g} m, "
            f"max_linear_speed={max_linear_speed:.6g} m/s, "
            f"max_angular_speed={max_angular_speed:.6g} rad/s, "
            f"max_joint_speed={max_joint_speed:.6g} rad/s",
            flush=True,
        )
        result = SettlingResult(
            cfg.max_steps,
            environment_steps,
            captured_position,
            captured_orientation,
            captured_joint_position,
            captured_linear_velocity,
            captured_angular_velocity,
            captured_contact_count,
            captured_actuator_effort,
            finished,
        )
        self._restore_settled_states(result, required & finished)
        return result

    def _restore_settled_states(
        self, result: SettlingResult, mask: torch.Tensor
    ) -> None:
        """Restore each environment to the instant it independently settled."""

        object_state = self.object.data.root_state_w.clone()
        object_state[mask, :3] = result.object_position[mask]
        object_state[mask, 3:7] = result.object_quaternion_wxyz[mask]
        object_state[mask, 7:13] = 0.0
        self.object.write_root_state_to_sim(object_state)
        joint_position = self.robot.data.joint_pos.clone()
        joint_velocity = self.robot.data.joint_vel.clone()
        joint_position[mask] = result.joint_position[mask]
        joint_velocity[mask] = 0.0
        self.robot.write_joint_state_to_sim(joint_position, joint_velocity)
        self._hold_root()

    def _actual_aperture(
        self, joint_position: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Express the measured six-joint state in the same metric aperture used
        # by the schedule-indexed surface asset.  Runtime USD and source URDF use
        # different finger-link origins, so measuring their raw separation would
        # introduce a centimetre-scale inconsistency between state and geometry.
        denominator = torch.sum(self.close_joint_target.square()).clamp_min(1e-8)
        measured_joint_position = (
            self.robot.data.joint_pos
            if joint_position is None
            else joint_position
        )
        closure_fraction = (
            torch.sum(measured_joint_position * self.close_joint_target, dim=-1)
            / denominator
        ).clamp(0.0, 1.0)
        schedule_position = closure_fraction * float(len(self.commanded_aperture) - 1)
        lower = torch.floor(schedule_position).long()
        upper = (lower + 1).clamp(max=len(self.commanded_aperture) - 1)
        alpha = schedule_position - lower.to(schedule_position.dtype)
        return (
            self.commanded_aperture.index_select(0, lower) * (1.0 - alpha)
            + self.commanded_aperture.index_select(0, upper) * alpha
        )

    def _record_state(
        self,
        base_position: torch.Tensor,
        base_quaternion: torch.Tensor,
        settling: SettlingResult,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        position, quaternion = subtract_frame_transforms(
            base_position,
            base_quaternion,
            settling.object_position,
            settling.object_quaternion_wxyz,
        )
        quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-8)
        return position, quaternion, self._actual_aperture(settling.joint_position)

    def _contact_count(self) -> torch.Tensor:
        counts = torch.zeros(self.env_count, device=self.device, dtype=torch.float32)
        for name in ("left_contact", "right_contact"):
            # This collector has no shelf, gravity, or enabled robot
            # self-collisions, so every external force on a contact pad comes
            # from the object.  ``force_matrix_w`` can be allocated yet remain
            # zero when Isaac's filtered-pair expression is not resolved for a
            # cloned environment; ``net_forces_w`` is the reliable signal here.
            forces = self.scene[name].data.net_forces_w
            if forces is None:
                forces = self.scene[name].data.force_matrix_w
            magnitude = torch.linalg.vector_norm(forces, dim=-1)
            while magnitude.ndim > 1:
                magnitude = magnitude.amax(dim=-1)
            counts += (magnitude > 1e-3).to(counts.dtype)
        return counts

    def collect(
        self,
        seeds: PoseSeeds,
        *,
        desired_count: int | None = None,
    ) -> CollectedTrajectories:
        candidate_count = len(seeds.position_m)
        count = candidate_count if desired_count is None else desired_count
        if count <= 0 or count > candidate_count:
            raise ValueError("desired_count must be in [1, number of candidate seeds]")
        states = 33
        accepted_position: list[np.ndarray] = []
        accepted_quaternion: list[np.ndarray] = []
        accepted_aperture: list[np.ndarray] = []
        accepted_source_index: list[np.ndarray] = []
        accepted_diagnostics: dict[str, list[np.ndarray]] = {
            "contact_count": [],
            "actuator_effort": [],
            "linear_velocity": [],
            "angular_velocity": [],
            "settling_substeps": [],
        }
        accepted_count = 0
        rejected_count = 0
        start = 0
        while start < candidate_count and accepted_count < count:
            # The first batch is normally full.  Replacement batches contain
            # only the number of trajectories still missing, so one rejected
            # seed never causes another hundred unnecessary rollouts.
            remaining = count - accepted_count
            live = min(self.env_count, candidate_count - start, remaining)
            stop = start + live
            chosen = np.arange(start, stop)
            if live < self.env_count:
                chosen = np.pad(chosen, (0, self.env_count - live), mode="wrap")
            seed_position = torch.from_numpy(seeds.position_m[chosen]).to(self.device)
            seed_quaternion = torch.from_numpy(seeds.quaternion_wxyz[chosen]).to(self.device)
            batch_started = perf_counter()
            print(
                f"[SRNO] {self.record.object_id}: candidate batch "
                f"{start + 1}-{stop}/{candidate_count}: "
                "placing zero-velocity object and open gripper",
                flush=True,
            )
            position = np.empty((live, states, 3), dtype=np.float32)
            quaternion = np.empty((live, states, 4), dtype=np.float32)
            aperture = np.empty((live, states), dtype=np.float32)
            diagnostics = {
                "contact_count": np.empty((live, 32), dtype=np.float32),
                "actuator_effort": np.empty((live, 32), dtype=np.float32),
                "linear_velocity": np.empty((live, 32, 3), dtype=np.float32),
                "angular_velocity": np.empty((live, 32, 3), dtype=np.float32),
                "settling_substeps": np.empty((live, 32), dtype=np.int32),
            }
            valid = torch.zeros(self.env_count, dtype=torch.bool, device=self.device)
            valid[:live] = True
            base_position, base_quaternion = self._reset_batch(seed_position, seed_quaternion)
            print(
                f"[SRNO] {self.record.object_id}: placement complete in "
                f"{perf_counter() - batch_started:.1f}s",
                flush=True,
            )
            initial_settling = self._settle_command(
                self.open_joint_target, 0, required_mask=valid
            )
            initially_rejected = valid & ~initial_settling.settled_mask
            valid &= initial_settling.settled_mask
            if bool(torch.any(initially_rejected)):
                local_indices = (
                    torch.nonzero(initially_rejected[:live]).flatten().cpu().numpy()
                )
                rejected_sources = seeds.source_index[start:stop][
                    local_indices
                ].tolist()
                print(
                    f"[SRNO] {self.record.object_id}: rejecting source poses "
                    f"{rejected_sources} at initial settling",
                    flush=True,
                )
            print(
                f"[SRNO] {self.record.object_id}: initial state settled in "
                f"{initial_settling.steps} control steps",
                flush=True,
            )
            state_position, state_quaternion, state_aperture = self._record_state(
                base_position, base_quaternion, initial_settling
            )
            position[:, 0] = state_position[:live].cpu().numpy()
            quaternion[:, 0] = state_quaternion[:live, (1, 2, 3, 0)].cpu().numpy()
            aperture[:, 0] = torch.maximum(
                state_aperture[:live], self.commanded_aperture[0]
            ).cpu().numpy()
            previous_quaternion = state_quaternion
            for command_index in range(1, states):
                if not bool(torch.any(valid)):
                    break
                command_started = perf_counter()
                fraction = float(command_index) / 32.0
                target = self.close_joint_target * fraction
                settling = self._settle_command(
                    target, command_index, required_mask=valid
                )
                previously_valid = valid.clone()
                valid &= settling.settled_mask
                newly_rejected = previously_valid & ~valid
                if bool(torch.any(newly_rejected)):
                    local_indices = torch.nonzero(newly_rejected[:live]).flatten().cpu().numpy()
                    rejected_sources = seeds.source_index[start:stop][local_indices].tolist()
                    print(
                        f"[SRNO] {self.record.object_id}: rejecting source poses "
                        f"{rejected_sources} at command {command_index:02d}",
                        flush=True,
                    )
                state_position, state_quaternion, measured_aperture = self._record_state(
                    base_position, base_quaternion, settling
                )
                flip = torch.sum(previous_quaternion * state_quaternion, dim=-1) < 0.0
                state_quaternion = torch.where(flip[:, None], -state_quaternion, state_quaternion)
                previous_quaternion = state_quaternion
                state_index = command_index
                position[:, state_index] = state_position[:live].cpu().numpy()
                quaternion[:, state_index] = state_quaternion[:live, (1, 2, 3, 0)].cpu().numpy()
                lower = self.commanded_aperture[state_index]
                previous = torch.from_numpy(aperture[:, state_index - 1]).to(self.device)
                canonical_aperture = torch.maximum(
                    lower,
                    torch.minimum(previous, measured_aperture[:live]),
                )
                aperture[:, state_index] = canonical_aperture.cpu().numpy()
                diagnostic_index = command_index - 1
                diagnostics["contact_count"][:, diagnostic_index] = (
                    settling.contact_count[:live].cpu().numpy()
                )
                diagnostics["actuator_effort"][:, diagnostic_index] = (
                    settling.actuator_effort[:live].cpu().numpy()
                )
                diagnostics["linear_velocity"][:, diagnostic_index] = (
                    settling.linear_velocity[:live].cpu().numpy()
                )
                diagnostics["angular_velocity"][:, diagnostic_index] = (
                    settling.angular_velocity[:live].cpu().numpy()
                )
                diagnostics["settling_substeps"][:, diagnostic_index] = (
                    settling.environment_steps[:live].cpu().numpy()
                    * CONTROL_DECIMATION
                )
                print(
                    f"[SRNO] {self.record.object_id}: command {command_index:02d}/32 "
                    f"settled in {settling.steps} control steps "
                    f"({perf_counter() - command_started:.1f}s)",
                    flush=True,
                )
            accepted_local = torch.nonzero(valid[:live]).flatten().cpu().numpy()
            remaining = count - accepted_count
            accepted_local = accepted_local[:remaining]
            batch_accepted = len(accepted_local)
            batch_rejected = live - int(valid[:live].sum().item())
            rejected_count += batch_rejected
            if batch_accepted:
                accepted_position.append(position[accepted_local].copy())
                accepted_quaternion.append(quaternion[accepted_local].copy())
                accepted_aperture.append(aperture[accepted_local].copy())
                accepted_source_index.append(
                    seeds.source_index[start:stop][accepted_local].copy()
                )
                for name in accepted_diagnostics:
                    accepted_diagnostics[name].append(
                        diagnostics[name][accepted_local].copy()
                    )
                accepted_count += batch_accepted
            print(
                f"[SRNO] {self.record.object_id}: batch accepted {batch_accepted}, "
                f"rejected {batch_rejected}; total {accepted_count}/{count}",
                flush=True,
            )
            start = stop

        if accepted_count < count:
            raise RuntimeError(
                f"only {accepted_count}/{count} trajectories settled after exhausting "
                f"{candidate_count} unique candidate poses ({rejected_count} rejected)"
            )
        return CollectedTrajectories(
            np.concatenate(accepted_position, axis=0),
            np.concatenate(accepted_quaternion, axis=0),
            np.concatenate(accepted_aperture, axis=0),
            {
                name: np.concatenate(chunks, axis=0)
                for name, chunks in accepted_diagnostics.items()
            },
            np.concatenate(accepted_source_index, axis=0),
        )
