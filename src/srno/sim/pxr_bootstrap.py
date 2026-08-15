"""Expose Isaac Sim's bundled USD Python bindings without launching Kit."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys


BOOTSTRAP_FLAG = "SRNO_PXR_BOOTSTRAPPED"


def build_pxr_environment() -> dict[str, str]:
    import isaacsim

    environment = dict(os.environ)
    isaacsim_root = Path(isaacsim.__file__).resolve().parent
    candidates = sorted((isaacsim_root / "extscache").glob("omni.usd.libs-*"))
    if not candidates:
        raise ModuleNotFoundError("Isaac Sim's omni.usd.libs extension was not found")
    usd_root = candidates[0]
    conda_library = Path(sys.executable).resolve().parent.parent / "lib"
    _prepend(environment, "PYTHONPATH", str(usd_root))
    _prepend(environment, "LD_LIBRARY_PATH", str(conda_library), str(usd_root / "bin"))
    return environment


def ensure_pxr_available(*, reexec: bool) -> None:
    try:
        importlib.import_module("pxr")
        return
    except ModuleNotFoundError:
        pass
    environment = build_pxr_environment()
    if reexec and os.environ.get(BOOTSTRAP_FLAG) != "1":
        environment[BOOTSTRAP_FLAG] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], environment)
    raise ModuleNotFoundError("pxr is unavailable; the SDF worker must be allowed to re-exec")


def _prepend(environment: dict[str, str], key: str, *values: str) -> None:
    current = [item for item in environment.get(key, "").split(os.pathsep) if item]
    prefix = [item for item in values if item and item not in current]
    environment[key] = os.pathsep.join((*prefix, *current))
