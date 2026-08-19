"""Fail-fast runtime contract for the six implicit gripper drives."""

from __future__ import annotations

from typing import Any

import numpy as np

from srno.sim.config import RelaxationConfig
from srno.sim.isaac_scene import (
    GRIPPER_DRIVE_TYPE,
    GRIPPER_EFFORT_LIMIT,
    GRIPPER_STIFFNESS,
    GRIPPER_TARGET_TYPE,
)


def _values(value: Any) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    return np.asarray(array[0] if array.ndim == 2 else array, dtype=np.float64)


def _joint_indices(actuator: Any) -> list[int]:
    indices = actuator.joint_indices
    if isinstance(indices, slice):
        return list(range(actuator.num_joints))[indices]
    return [int(value) for value in indices]


def _assert_vector(name: str, actual: np.ndarray, expected: float) -> None:
    if actual.shape != (6,) or not np.allclose(
        actual, expected, rtol=0.0, atol=1e-6
    ):
        raise RuntimeError(
            f"runtime actuator {name} mismatch: actual={actual.tolist()}, "
            f"expected={[expected] * 6}"
        )


def verify_runtime_actuator(
    robot: Any,
    *,
    expected_joint_names: tuple[str, ...],
    relaxation: RelaxationConfig,
    position_target: Any | None = None,
) -> dict[str, Any]:
    """Read IsaacLab, PhysX and USD drive state and enforce the dataset law."""

    if len(expected_joint_names) != 6:
        raise ValueError("the SRNO actuator contract requires six joint names")
    runtime_names = tuple(map(str, robot.data.joint_names))
    if runtime_names != expected_joint_names:
        raise RuntimeError(
            "runtime gripper joint order mismatch: "
            f"actual={runtime_names}, expected={expected_joint_names}"
        )
    if set(robot.actuators) != {"quasistatic_gripper"}:
        raise RuntimeError(
            f"unexpected actuator groups: {sorted(robot.actuators)}"
        )
    actuator = robot.actuators["quasistatic_gripper"]
    if not actuator.is_implicit_model:
        raise RuntimeError("quasistatic gripper actuator must be implicit")
    if tuple(map(str, actuator.joint_names)) != expected_joint_names:
        raise RuntimeError("actuator joint order differs from runtime articulation")
    indices = _joint_indices(actuator)
    if indices != list(range(6)):
        raise RuntimeError(f"unexpected actuator joint indices: {indices}")

    isaaclab = {
        "stiffness": _values(actuator.stiffness),
        "damping": _values(actuator.damping),
        "effort_limit": _values(actuator.effort_limit_sim),
        "velocity_limit": _values(actuator.velocity_limit_sim),
    }
    view = robot.root_physx_view
    physx = {
        "stiffness": _values(view.get_dof_stiffnesses())[[*indices]],
        "damping": _values(view.get_dof_dampings())[[*indices]],
        "effort_limit": _values(view.get_dof_max_forces())[[*indices]],
        "velocity_limit": _values(view.get_dof_max_velocities())[[*indices]],
    }
    expected = {
        "stiffness": GRIPPER_STIFFNESS,
        "damping": relaxation.gripper_damping,
        "effort_limit": GRIPPER_EFFORT_LIMIT,
        "velocity_limit": relaxation.gripper_velocity_limit_rad_s,
    }
    for source_name, source in (("IsaacLab", isaaclab), ("PhysX", physx)):
        for property_name, expected_value in expected.items():
            try:
                _assert_vector(
                    f"{source_name}.{property_name}",
                    source[property_name],
                    expected_value,
                )
            except RuntimeError as error:
                raise RuntimeError(str(error)) from error

    target_readback: dict[str, Any] | None = None
    if position_target is not None:
        import torch

        expected_target = torch.as_tensor(
            position_target,
            dtype=robot.data.joint_pos.dtype,
            device=robot.device,
        )
        if expected_target.shape != (6,):
            raise ValueError("position_target must contain the six ordered joints")
        commanded = expected_target.unsqueeze(0).expand(robot.num_instances, -1)
        robot.set_joint_position_target(commanded)
        # This is the same write path used by QuasistaticCollector.  For an
        # implicit actuator it transfers joint_pos_target into the PhysX
        # articulation's position-drive target tensor.
        robot.write_data_to_sim()
        isaaclab_position = _values(robot.data.joint_pos_target)
        physx_position = _values(view.get_dof_position_targets())[[*indices]]
        physx_velocity = _values(view.get_dof_velocity_targets())[[*indices]]
        expected_numpy = expected_target.detach().float().cpu().numpy().astype(
            np.float64
        )
        for source_name, actual in (
            ("IsaacLab.position_target", isaaclab_position),
            ("PhysX.position_target", physx_position),
        ):
            if actual.shape != (6,) or not np.allclose(
                actual, expected_numpy, rtol=0.0, atol=1e-6
            ):
                raise RuntimeError(
                    f"runtime actuator {source_name} mismatch: "
                    f"actual={actual.tolist()}, expected={expected_numpy.tolist()}"
                )
        if physx_velocity.shape != (6,) or not np.allclose(
            physx_velocity, 0.0, rtol=0.0, atol=1e-7
        ):
            raise RuntimeError(
                "runtime actuator PhysX.velocity_target must be zero for the "
                f"position PD law, actual={physx_velocity.tolist()}"
            )
        target_readback = {
            "target_type": GRIPPER_TARGET_TYPE,
            "commanded_position": expected_numpy.tolist(),
            "isaaclab_position": isaaclab_position.tolist(),
            "physx_position": physx_position.tolist(),
            "physx_velocity": physx_velocity.tolist(),
        }

    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    usd_drives: list[dict[str, Any]] = []
    dof_paths = list(view.dof_paths[0])
    for joint_name, joint_index in zip(expected_joint_names, indices, strict=True):
        path = str(dof_paths[joint_index])
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"runtime joint prim not found: {path}")
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            raise RuntimeError(f"angular DriveAPI missing on revolute joint {path}")
        drive_type = str(drive.GetTypeAttr().Get())
        if drive_type != GRIPPER_DRIVE_TYPE:
            raise RuntimeError(
                f"USD drive type mismatch for {joint_name}: "
                f"actual={drive_type!r}, expected={GRIPPER_DRIVE_TYPE!r}"
            )
        usd_drives.append(
            {
                "joint_name": joint_name,
                "path": path,
                "drive_type": drive_type,
                "target_type": GRIPPER_TARGET_TYPE,
                # These authored attributes are retained for provenance.  The
                # definitive gains/limits above are read from the live PhysX
                # articulation because IsaacLab may override authored USD.
                "authored_stiffness": drive.GetStiffnessAttr().Get(),
                "authored_damping": drive.GetDampingAttr().Get(),
                "authored_max_force": drive.GetMaxForceAttr().Get(),
                "authored_target_position": drive.GetTargetPositionAttr().Get(),
                "authored_target_velocity": drive.GetTargetVelocityAttr().Get(),
            }
        )

    return {
        "joint_names": list(expected_joint_names),
        "model": type(actuator).__name__,
        "implicit": True,
        "drive_type": GRIPPER_DRIVE_TYPE,
        "target_type": GRIPPER_TARGET_TYPE,
        "expected": expected,
        "isaaclab": {name: value.tolist() for name, value in isaaclab.items()},
        "physx": {name: value.tolist() for name, value in physx.items()},
        "target_readback": target_readback,
        "usd": usd_drives,
        "effort_semantics": (
            "IsaacLab computed/applied torque is an approximate clipped PD "
            "diagnostic for implicit drives, not a measured contact reaction"
        ),
    }
