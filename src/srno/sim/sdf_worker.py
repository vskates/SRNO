"""Isolated PhysX-cooked SDF worker."""

from __future__ import annotations

import argparse
import os
import sys
import traceback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--object", action="append", dest="objects", required=True)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher
    from srno.sim import SimulatorAssetCatalog
    from srno.sim.config import SimulatorConfig
    from srno.sim.memory_guard import MemoryWatchdog
    from srno.sim.usd_geometry import load_or_generate_sdf

    config = SimulatorConfig.load(args.config)
    catalog = SimulatorAssetCatalog.load(config.catalog)
    with MemoryWatchdog(config.memory_limit_gib, config.memory_check_interval_s):
        launcher = AppLauncher({"headless": True, "device": config.device})
        app = launcher.app
        try:
            for object_id in args.objects:
                record = catalog.object(object_id)
                cache = config.output_dir / ".cache" / "sdf" / f"{record.object_id}.npz"
                dense = load_or_generate_sdf(
                    record.usd_path,
                    cache,
                    config.sdf,
                    distance_backend="warp",
                    device=config.device,
                )
                print(
                    f"[SRNO SDF] {record.object_id}: shape={dense.values.shape}, "
                    f"voxel_max={float(dense.voxel_size_xyz.max()):.6f} m, "
                    f"geometry={dense.representation}, sha256={dense.geometry_sha256}",
                    flush=True,
                )
        except BaseException:
            traceback.print_exc()
            sys.stderr.flush()
            # Kit can terminate the interpreter with status 0 from close()
            # while an exception is unwinding, masking a failed SDF build.
            os._exit(1)
        else:
            app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
