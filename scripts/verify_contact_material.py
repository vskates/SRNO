#!/usr/bin/env python3
"""Launch one environment and verify the live SRNO PxMaterial contract."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import traceback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--object")
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    from srno.sim.config import SimulatorConfig
    from srno.sim.memory_guard import MemoryWatchdog

    config = SimulatorConfig.load(args.config)
    watchdog = MemoryWatchdog(config.memory_limit_gib, config.memory_check_interval_s)
    watchdog.start()
    app = AppLauncher({"headless": True, "device": config.device}).app
    try:
        import omni.usd
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext

        from srno.sim.assets import SimulatorAssetCatalog
        from srno.sim.isaac_scene import (
            apply_contact_materials,
            make_scene_cfg,
            make_simulation_cfg,
        )
        from srno.sim.physx_material import PhysxMaterialAudit, expected_physics_metadata

        catalog = SimulatorAssetCatalog.load(config.catalog)
        object_id = args.object or catalog.object_ids[0]
        record = catalog.object(object_id)
        omni.usd.get_context().new_stage()
        for _ in range(10):
            app.update()
        simulation = SimulationContext(make_simulation_cfg(config.device, config.material))
        scene = None
        audit = PhysxMaterialAudit(expected_physics_metadata(config))
        try:
            audit.start()
            scene = InteractiveScene(
                make_scene_cfg(catalog, record, num_envs=1, relaxation=config.relaxation)
            )
            apply_contact_materials(scene, config.material)
            audit.force_load()
            simulation.reset()
            physics = audit.verify(app)
            print(json.dumps(physics.to_dict(), indent=2, sort_keys=True), flush=True)
        finally:
            audit.close()
            del scene
            simulation.clear_all_callbacks()
            simulation.clear_instance()
            del simulation
            gc.collect()
            omni.usd.get_context().close_stage()
            for _ in range(5):
                app.update()
    except BaseException:
        traceback.print_exc()
        os._exit(1)
    else:
        app.close()
        watchdog.stop()


if __name__ == "__main__":
    main()
