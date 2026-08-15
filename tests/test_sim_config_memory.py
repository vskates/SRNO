from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from srno.sim.config import SimulatorConfig
from srno.sim.memory_guard import MemorySnapshot, MemoryWatchdog, read_system_used_bytes
from srno.cli import build_parser
from srno.sim.runner import _check_existing_shard


def test_simulator_config_is_minimal_and_zero_gravity_mode_has_memory_guard() -> None:
    config = SimulatorConfig.load("configs/simulator.toml")
    assert config.headless is True
    assert config.num_envs == 100
    assert config.trajectories_per_object == 100
    assert config.trajectory.command_steps == 32
    assert config.memory_limit_gib == 14.0
    assert config.memory_check_interval_s == 0.25
    assert config.relaxation.object_linear_damping_s_inv > 0.0
    assert config.relaxation.object_angular_damping_s_inv > 0.0
    assert config.relaxation.gripper_velocity_limit_rad_s == 0.1
    assert config.catalog.name == "catalog.json"


def test_memory_watchdog_triggers_on_system_or_process_tree_limit() -> None:
    violations: list[tuple[MemorySnapshot, int]] = []
    snapshot = MemorySnapshot(system_used_bytes=15 * 1024**3, process_tree_rss_bytes=1)
    watchdog = MemoryWatchdog(
        limit_gib=14.0,
        sampler=lambda _: snapshot,
        terminator=lambda current, limit: violations.append((current, limit)),
    )
    assert watchdog.check_now() == snapshot
    assert violations == [(snapshot, 14 * 1024**3)]


def test_linux_memory_reader_uses_memavailable(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       16000000 kB\nMemFree: 100 kB\nMemAvailable: 2000000 kB\n",
        encoding="utf-8",
    )
    assert read_system_used_bytes(meminfo) == 14_000_000 * 1024


def test_resume_rejects_smoke_shard_with_wrong_trajectory_count(tmp_path: Path) -> None:
    shard = tmp_path / "smoke.h5"
    with h5py.File(shard, "w") as handle:
        group = handle.create_group("objects/000000")
        group.attrs["object_id"] = "object-a"
        group.create_dataset("position", data=np.zeros((1, 33, 3), dtype=np.float32))

    _check_existing_shard(shard, "object-a", 1)
    with pytest.raises(ValueError, match="pass --overwrite"):
        _check_existing_shard(shard, "object-a", 2_000)


def test_collect_cli_supports_explicit_resume_mode() -> None:
    args = build_parser().parse_args(
        ["sim", "collect", "--config", "configs/simulator.toml", "--resume"]
    )
    assert args.resume is True
    assert args.overwrite is None
