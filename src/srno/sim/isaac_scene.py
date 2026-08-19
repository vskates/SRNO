"""Validation-derived, zero-gravity Isaac Lab scene construction.

Import this module only after :class:`isaaclab.app.AppLauncher` is running.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg

from srno.sim.assets import ObjectAssetRecord, SimulatorAssetCatalog
from srno.sim.config import ContactMaterialConfig, RelaxationConfig


SIM_DT = 1.0 / 120.0
CONTROL_DECIMATION = 2
GRIPPER_DRIVE_TYPE = "force"
GRIPPER_TARGET_TYPE = "position"
GRIPPER_STIFFNESS = 14.0
GRIPPER_EFFORT_LIMIT = 480.0
CONTACT_LINKS = (
    "astribot_gripper_right_Link_L11",
    "astribot_gripper_right_Link_R11",
)


def _rigid_material_cfg(material: ContactMaterialConfig) -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=material.static_friction,
        dynamic_friction=material.dynamic_friction,
        restitution=material.restitution,
        friction_combine_mode=material.friction_combine_mode,
        restitution_combine_mode=material.restitution_combine_mode,
        compliant_contact_stiffness=material.contact_stiffness,
        compliant_contact_damping=material.contact_damping,
    )


def make_simulation_cfg(
    device: str, material: ContactMaterialConfig | None = None
) -> sim_utils.SimulationCfg:
    material = ContactMaterialConfig() if material is None else material
    cfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        render_interval=CONTROL_DECIMATION,
        gravity=(0.0, 0.0, 0.0),
        device=device,
        physics_material=_rigid_material_cfg(material),
    )
    cfg.physx.bounce_threshold_velocity = 0.01
    cfg.physx.gpu_found_lost_aggregate_pairs_capacity = 4 * 1024 * 1024
    cfg.physx.gpu_total_aggregate_pairs_capacity = 128 * 1024
    cfg.physx.friction_correlation_distance = 0.00625
    cfg.physx.enable_ccd = True
    cfg.physx.gpu_max_rigid_contact_count = 1_048_576
    cfg.physx.gpu_max_rigid_patch_count = 262_144
    return cfg


def _gripper_cfg(
    catalog: SimulatorAssetCatalog, relaxation: RelaxationConfig
) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(catalog.gripper.runtime_usd),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
                solver_position_iteration_count=64,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos=dict(catalog.gripper.open_joint_position_rad),
        ),
        actuators={
            # The validation actuator computes the same six target positions
            # through an explicit mimic spring.  That controller chatters at its
            # velocity limit and has no equilibrium in a zero-gravity free-body
            # scene.  The dataset collector therefore sends the exact consistent
            # six-joint schedule to PhysX's implicit PD solve instead.
            "quasistatic_gripper": ImplicitActuatorCfg(
                joint_names_expr=list(catalog.gripper.close_joint_position_rad),
                effort_limit=GRIPPER_EFFORT_LIMIT,
                effort_limit_sim=GRIPPER_EFFORT_LIMIT,
                velocity_limit=relaxation.gripper_velocity_limit_rad_s,
                velocity_limit_sim=relaxation.gripper_velocity_limit_rad_s,
                stiffness=GRIPPER_STIFFNESS,
                damping=relaxation.gripper_damping,
            )
        },
    )


def _object_cfg(
    record: ObjectAssetRecord, relaxation: RelaxationConfig
) -> RigidObjectCfg:
    spawn_options = {}
    if record.spawn_scale is not None:
        # Supplying even (1, 1, 1) makes Isaac overwrite the default prim's
        # authored xformOp:scale.  Validation leaves this option unset for the
        # vendored grocery assets, whose root scale performs normalization.
        spawn_options["scale"] = record.spawn_scale
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(record.usd_path),
            **spawn_options,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=10.0,
                linear_damping=relaxation.object_linear_damping_s_inv,
                angular_damping=relaxation.object_angular_damping_s_inv,
                max_contact_impulse=float("inf"),
                max_angular_velocity=10.0,
                max_linear_velocity=10.0,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=16,
                stabilization_threshold=0.1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=record.spawn_quaternion_wxyz,
        ),
    )


def make_scene_cfg(
    catalog: SimulatorAssetCatalog,
    record: ObjectAssetRecord,
    *,
    num_envs: int,
    relaxation: RelaxationConfig,
) -> InteractiveSceneCfg:
    """Build a homogeneous scene for one catalog object."""

    cfg = InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=0.8,
        lazy_sensor_update=True,
        replicate_physics=True,
        filter_collisions=True,
    )
    cfg.robot = _gripper_cfg(catalog, relaxation)
    cfg.object = _object_cfg(record, relaxation)
    # The USD default prim is grafted onto ``Robot`` by UsdFileCfg, leaving
    # the inner articulation scope as the only additional path component.
    link_root = "{ENV_REGEX_NS}/Robot/astribot_gripper"
    cfg.left_contact = ContactSensorCfg(
        prim_path=f"{link_root}/{CONTACT_LINKS[0]}",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    cfg.right_contact = ContactSensorCfg(
        prim_path=f"{link_root}/{CONTACT_LINKS[1]}",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    cfg.dome_light = AssetBaseCfg(
        prim_path="/World/Domelight",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=300.0),
    )
    return cfg


def apply_contact_materials(
    scene: object, material: ContactMaterialConfig | None = None
) -> None:
    """Bind validation's high-friction material to object and contact pads."""

    import omni.usd
    from pxr import Sdf, Usd, UsdPhysics

    material = ContactMaterialConfig() if material is None else material
    stage = omni.usd.get_context().get_stage()
    material_path = "/World/SRNOContactMaterial"
    runtime_cfg = _rigid_material_cfg(material)
    runtime_cfg.func(material_path, runtime_cfg)

    def bind_and_verify(prim: object) -> None:
        """Author and verify a direct physics-purpose material binding."""

        sim_utils.bind_physics_material(
            prim.GetPath(), material_path, stage=stage, stronger_than_descendants=True
        )
        relationship = prim.GetRelationship("material:binding:physics")
        targets = relationship.GetTargets() if relationship.IsValid() else []
        if targets != [Sdf.Path(material_path)]:
            raise RuntimeError(
                f"failed to bind {material_path} to collision prim {prim.GetPath()}; "
                f"authored targets={list(map(str, targets))}"
            )

    bound_paths: list[str] = []
    for environment_path in scene.env_prim_paths:
        object_prim = stage.GetPrimAtPath(f"{environment_path}/Object")
        if not object_prim.IsValid():
            raise RuntimeError(f"spawned object prim not found: {object_prim.GetPath()}")
        for prim in Usd.PrimRange(object_prim):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            bind_and_verify(prim)
            bound_paths.append(str(prim.GetPath()))
        for link_name in CONTACT_LINKS:
            link = stage.GetPrimAtPath(
                f"{environment_path}/Robot/astribot_gripper/{link_name}"
            )
            if not link.IsValid():
                raise RuntimeError(f"contact link not found in runtime gripper: {link.GetPath()}")
            for prim in Usd.PrimRange(link):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                bind_and_verify(prim)
                bound_paths.append(str(prim.GetPath()))
    if not bound_paths:
        raise RuntimeError("no collision prims received the SRNO contact material")
