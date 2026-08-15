"""One-object SDF subprocess; exits to release mesh/proximity memory before Isaac starts."""

from __future__ import annotations

import argparse

from srno.sim.pxr_bootstrap import ensure_pxr_available


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--object", required=True)
    args = parser.parse_args()
    ensure_pxr_available(reexec=True)

    from srno.sim import SimulatorAssetCatalog
    from srno.sim.config import SimulatorConfig
    from srno.sim.memory_guard import MemoryWatchdog
    from srno.sim.usd_geometry import load_or_generate_sdf

    config = SimulatorConfig.load(args.config)
    catalog = SimulatorAssetCatalog.load(config.catalog)
    record = catalog.object(args.object)
    cache = config.output_dir / ".cache" / "sdf" / f"{record.object_id}.npz"
    with MemoryWatchdog(config.memory_limit_gib, config.memory_check_interval_s):
        dense = load_or_generate_sdf(record.usd_path, cache, config.sdf)
    print(
        f"[SRNO SDF] {record.object_id}: shape={dense.values.shape}, "
        f"voxel_max={float(dense.voxel_size_xyz.max()):.6f} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
