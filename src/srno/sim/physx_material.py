"""Verified PhysX material contract for simulator dataset collection."""

from __future__ import annotations

import ctypes
import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np

from srno.data.schema import PhysicsMetadata
from srno.sim.config import SimulatorConfig


MATERIAL_PATH = "/World/SRNOContactMaterial"
SUPPORTED_ISAACSIM_VERSION = "5.1.0.0"
_COMBINE_MODE = {0: "average", 1: "min", 2: "multiply", 3: "max"}
_DISABLE_STRONG_FRICTION = 1 << 1


def expected_physics_metadata(config: SimulatorConfig) -> PhysicsMetadata:
    """Return the immutable physics law intended by the collector config."""

    return PhysicsMetadata(
        static_friction=config.material.static_friction,
        dynamic_friction=config.material.dynamic_friction,
        friction_combine_mode=config.material.friction_combine_mode,
        restitution=config.material.restitution,
        restitution_combine_mode=config.material.restitution_combine_mode,
        strong_friction_enabled=config.material.strong_friction_enabled,
        contact_model="rigid",
        contact_stiffness=config.material.contact_stiffness,
        contact_damping=config.material.contact_damping,
        friction_model="patch",
        contact_generation="PCM",
        solver_type="TGS",
        solver_position_iterations=64,
        solver_velocity_iterations=16,
        simulator="Isaac Sim / PhysX",
        simulator_version=SUPPORTED_ISAACSIM_VERSION,
    )


class PhysxMaterialAudit:
    """Capture and verify the actual PxMaterial created for SRNO contacts."""

    def __init__(self, expected: PhysicsMetadata) -> None:
        expected.validate()
        self.expected = expected
        self._records: list[tuple[int, int]] = []
        self._interface: Any = None
        self._subscription: int | None = None

    def start(self) -> None:
        import omni.physx

        if self._subscription is not None:
            raise RuntimeError("PhysX material audit is already active")

        def on_created(path: int, object_id: int, physx_type: int) -> None:
            if int(physx_type) == 2:  # omni::physx::ePTMaterial
                self._records.append((int(path), int(object_id)))

        self._interface = omni.physx.get_physx_interface()
        self._subscription = self._interface.subscribe_object_changed_notifications(
            object_creation_fn=on_created,
            stop_callback_when_sim_stopped=False,
        )

    def force_load(self) -> None:
        if self._interface is None:
            raise RuntimeError("PhysX material audit was not started")
        self._interface.force_load_physics_from_usd()

    def close(self) -> None:
        if self._interface is not None and self._subscription is not None:
            self._interface.unsubscribe_object_change_notifications(self._subscription)
        self._subscription = None
        self._interface = None

    def verify(self, app: Any) -> PhysicsMetadata:
        from pxr import PhysicsSchemaTools

        try:
            for _ in range(2):
                app.update()
        finally:
            self.close()

        decoded = [
            (str(PhysicsSchemaTools.intToSdfPath(path)), object_id)
            for path, object_id in self._records
        ]
        matches = sorted({object_id for path, object_id in decoded if path == MATERIAL_PATH})
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one live PxMaterial at {MATERIAL_PATH}, found {decoded}"
            )
        actual = self._read_material(matches[0])
        self._assert_expected(actual)
        print(
            "[SRNO] verified live PxMaterial: "
            f"mu_s={actual['static_friction']:.6g}, "
            f"mu_d={actual['dynamic_friction']:.6g}, "
            f"combine={actual['friction_combine_mode']}, "
            f"strong_friction={actual['strong_friction_enabled']}",
            flush=True,
        )
        return self.expected

    @staticmethod
    def _read_material(object_id: int) -> dict[str, Any]:
        installed = importlib.metadata.version("isaacsim")
        if installed != SUPPORTED_ISAACSIM_VERSION:
            raise RuntimeError(
                "PxMaterial probe ABI mismatch: "
                f"Isaac Sim {installed}, expected {SUPPORTED_ISAACSIM_VERSION}"
            )
        library_path = Path(__file__).with_name("native") / "libsrno_physx_material_probe.so"
        if not library_path.is_file():
            raise RuntimeError(f"required PxMaterial runtime probe is missing: {library_path}")
        library = ctypes.CDLL(str(library_path))
        reader = library.srno_read_physx_material
        reader.argtypes = (
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint16),
        )
        reader.restype = ctypes.c_int
        static = ctypes.c_float()
        dynamic = ctypes.c_float()
        combine = ctypes.c_int32()
        restitution = ctypes.c_float()
        damping = ctypes.c_float()
        flags = ctypes.c_uint16()
        status = reader(
            object_id,
            ctypes.byref(static),
            ctypes.byref(dynamic),
            ctypes.byref(combine),
            ctypes.byref(restitution),
            ctypes.byref(damping),
            ctypes.byref(flags),
        )
        if status != 1 or combine.value not in _COMBINE_MODE:
            raise RuntimeError(
                f"cannot read live PxMaterial {object_id}: status={status}, "
                f"combine={combine.value}"
            )
        return {
            "static_friction": float(static.value),
            "dynamic_friction": float(dynamic.value),
            "friction_combine_mode": _COMBINE_MODE[combine.value],
            "restitution": float(restitution.value),
            "contact_damping": float(damping.value),
            "strong_friction_enabled": not bool(flags.value & _DISABLE_STRONG_FRICTION),
        }

    def _assert_expected(self, actual: dict[str, Any]) -> None:
        expected = self.expected
        numeric = {
            "static_friction": expected.static_friction,
            "dynamic_friction": expected.dynamic_friction,
            "restitution": expected.restitution,
            "contact_damping": expected.contact_damping,
        }
        errors = [
            f"{name}: actual={actual[name]!r}, intended={value!r}"
            for name, value in numeric.items()
            if not np.isclose(actual[name], value, rtol=0.0, atol=1e-6)
        ]
        for name, value in {
            "friction_combine_mode": expected.friction_combine_mode,
            "strong_friction_enabled": expected.strong_friction_enabled,
        }.items():
            if actual[name] != value:
                errors.append(f"{name}: actual={actual[name]!r}, intended={value!r}")
        if errors:
            raise RuntimeError("live PxMaterial violates dataset physics contract: " + "; ".join(errors))
