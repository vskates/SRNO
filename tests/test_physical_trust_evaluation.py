from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.evaluate_physical_one_step_trust import (
    _bootstrap_bias_distance,
    _bootstrap_difference,
)


def test_paired_terminal_bootstrap_preserves_known_difference() -> None:
    object_index = np.repeat(np.arange(3), 8)
    baseline = np.linspace(0.1, 0.3, len(object_index))
    candidate = baseline - 0.02
    result = _bootstrap_difference(candidate, baseline, object_index, replicates=500)
    assert result["mean"] == pytest.approx(-0.02)
    assert result["ci95_lower"] == pytest.approx(-0.02)
    assert result["ci95_upper"] == pytest.approx(-0.02)


def test_bias_distance_bootstrap_uses_paired_rows() -> None:
    object_index = np.repeat(np.arange(3), 8)
    local = np.linspace(-0.1, 0.1, len(object_index))
    old = local + 0.04
    candidate = local + 0.01
    result = _bootstrap_bias_distance(
        candidate, local, old, object_index, replicates=500
    )
    assert result["mean"] == pytest.approx(-0.03)
    assert result["ci95_upper"] < 0.0
