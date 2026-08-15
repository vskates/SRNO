"""Low-overhead RAM watchdog for Isaac Sim entry points."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Callable


MEMORY_LIMIT_EXIT_CODE = 86


@dataclass(frozen=True)
class MemorySnapshot:
    system_used_bytes: int
    process_tree_rss_bytes: int

    @property
    def maximum_bytes(self) -> int:
        return max(self.system_used_bytes, self.process_tree_rss_bytes)


def read_system_used_bytes(meminfo_path: str | Path = "/proc/meminfo") -> int:
    fields: dict[str, int] = {}
    for line in Path(meminfo_path).read_text(encoding="utf-8").splitlines():
        name, separator, remainder = line.partition(":")
        if separator and name in {"MemTotal", "MemAvailable"}:
            fields[name] = int(remainder.strip().split()[0]) * 1024
    if set(fields) != {"MemTotal", "MemAvailable"}:
        raise RuntimeError("/proc/meminfo does not contain MemTotal and MemAvailable")
    return max(0, fields["MemTotal"] - fields["MemAvailable"])


def read_process_tree_rss_bytes(root_pid: int | None = None) -> int:
    """Sum resident memory for a process and all of its descendants."""

    root = os.getpid() if root_pid is None else int(root_pid)
    processes: dict[int, tuple[int, int]] = {}
    for status_path in Path("/proc").glob("[0-9]*/status"):
        try:
            pid = int(status_path.parent.name)
            parent = -1
            rss_kib = 0
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("PPid:"):
                    parent = int(line.split()[1])
                elif line.startswith("VmRSS:"):
                    rss_kib = int(line.split()[1])
            processes[pid] = (parent, rss_kib * 1024)
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    descendants = {root}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _) in processes.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(processes.get(pid, (-1, 0))[1] for pid in descendants)


def read_memory_snapshot(root_pid: int | None = None) -> MemorySnapshot:
    return MemorySnapshot(
        system_used_bytes=read_system_used_bytes(),
        process_tree_rss_bytes=read_process_tree_rss_bytes(root_pid),
    )


class MemoryWatchdog:
    """Terminate Isaac if host RAM or collector-tree RSS crosses a limit.

    The production monitor is a separate Python process.  A thread in the
    collector is not sufficient here: long Kit/native calls can keep the GIL
    long enough for Linux's OOM killer to win the race.  Injected samplers and
    terminators retain the lightweight thread path used by unit tests.
    """

    def __init__(
        self,
        limit_gib: float = 12.0,
        interval_s: float = 0.25,
        *,
        root_pid: int | None = None,
        sampler: Callable[[int | None], MemorySnapshot] = read_memory_snapshot,
        terminator: Callable[[MemorySnapshot, int], None] | None = None,
    ) -> None:
        if limit_gib <= 0.0:
            raise ValueError("memory limit must be positive")
        if interval_s <= 0.0:
            raise ValueError("memory check interval must be positive")
        self.limit_bytes = int(limit_gib * 1024**3)
        self.interval_s = float(interval_s)
        self.root_pid = os.getpid() if root_pid is None else root_pid
        self._sampler = sampler
        self._terminator = terminator or self._terminate_process
        self._external = sampler is read_memory_snapshot and terminator is None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None

    def check_now(self) -> MemorySnapshot:
        snapshot = self._sampler(self.root_pid)
        if snapshot.maximum_bytes > self.limit_bytes:
            self._terminator(snapshot, self.limit_bytes)
        return snapshot

    def start(self) -> "MemoryWatchdog":
        if self._thread is not None or self._process is not None:
            raise RuntimeError("memory watchdog is already running")
        self.check_now()
        if self._external:
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "srno.sim.memory_guard",
                    "--monitor-pid",
                    str(self.root_pid),
                    "--limit-bytes",
                    str(self.limit_bytes),
                    "--interval-s",
                    str(self.interval_s),
                ],
                start_new_session=True,
            )
            return self
        self._thread = threading.Thread(
            target=self._monitor,
            name="srno-memory-watchdog",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2.0 * self.interval_s))
            self._thread = None
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=max(1.0, 2.0 * self.interval_s))
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            self._process = None

    def __enter__(self) -> "MemoryWatchdog":
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()

    def _monitor(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.check_now()
            except Exception as error:
                print(f"[SRNO memory watchdog] monitoring error: {error}", file=sys.stderr, flush=True)

    @staticmethod
    def _terminate_process(snapshot: MemorySnapshot, limit_bytes: int) -> None:
        gib = 1024**3
        print(
            "[SRNO memory watchdog] RAM limit exceeded: "
            f"system_used={snapshot.system_used_bytes / gib:.2f} GiB, "
            f"process_tree_rss={snapshot.process_tree_rss_bytes / gib:.2f} GiB, "
            f"limit={limit_bytes / gib:.2f} GiB. Terminating collector.",
            file=sys.stderr,
            flush=True,
        )
        os._exit(MEMORY_LIMIT_EXIT_CODE)


def monitor_parent(root_pid: int, limit_bytes: int, interval_s: float) -> int:
    """External monitor entry point; never depends on the collector's GIL."""

    while True:
        try:
            os.kill(root_pid, 0)
        except ProcessLookupError:
            return 0
        except PermissionError:
            return 1

        snapshot = read_memory_snapshot(root_pid)
        if snapshot.maximum_bytes > limit_bytes:
            gib = 1024**3
            print(
                "[SRNO memory watchdog] RAM limit exceeded: "
                f"system_used={snapshot.system_used_bytes / gib:.2f} GiB, "
                f"process_tree_rss={snapshot.process_tree_rss_bytes / gib:.2f} GiB, "
                f"limit={limit_bytes / gib:.2f} GiB. Killing collector.",
                file=sys.stderr,
                flush=True,
            )
            try:
                os.kill(root_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return MEMORY_LIMIT_EXIT_CODE
        time.sleep(interval_s)


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="SRNO external RAM watchdog")
    parser.add_argument("--monitor-pid", type=int, required=True)
    parser.add_argument("--limit-bytes", type=int, required=True)
    parser.add_argument("--interval-s", type=float, required=True)
    args = parser.parse_args()
    if args.monitor_pid <= 0 or args.limit_bytes <= 0 or args.interval_s <= 0.0:
        parser.error("monitor PID, memory limit and interval must be positive")
    return monitor_parent(args.monitor_pid, args.limit_bytes, args.interval_s)


if __name__ == "__main__":
    raise SystemExit(_main())
