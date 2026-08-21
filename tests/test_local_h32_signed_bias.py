from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.compare_local_h32_signed_bias import (
    _classify_split_vz,
    _tf_feature_vector,
)


def test_tf_feature_vector_uses_all_steps_and_correct_bands() -> None:
    values = np.zeros((2, 32, 6), dtype=np.float64)
    values[:, 31, 2] = 0.008
    features = _tf_feature_vector(values)
    assert features.shape == (2, 30)
    assert np.allclose(features[:, 20], 0.001)
    assert np.allclose(features[:, 26], 0.008)
    assert np.count_nonzero(features) == 4


def _summary(mean: float, lower: float, upper: float) -> dict[str, np.ndarray]:
    values = {
        "mean": np.zeros(30),
        "ci95_lower": np.zeros(30),
        "ci95_upper": np.zeros(30),
    }
    values["mean"][26] = mean
    values["ci95_lower"][26] = lower
    values["ci95_upper"][26] = upper
    return values


def test_vz_classification_separates_local_and_rollout_bias() -> None:
    unbiased = _summary(0.0, -0.001, 0.001)
    positive = _summary(0.004, 0.003, 0.005)
    positive_delta = _summary(0.004, 0.002, 0.006)
    assert (
        _classify_split_vz(unbiased, positive, positive_delta)
        == "bias_introduced_by_rollout"
    )
    already_local = _summary(0.003, 0.002, 0.004)
    assert (
        _classify_split_vz(already_local, positive, unbiased)
        == "bias_already_in_local"
    )


def test_vz_classification_keeps_uncertain_paired_difference_explicit() -> None:
    unbiased = _summary(0.0, -0.001, 0.001)
    positive = _summary(0.004, 0.003, 0.005)
    uncertain_delta = _summary(0.002, -0.001, 0.005)
    assert (
        _classify_split_vz(unbiased, positive, uncertain_delta)
        == "h32_only_but_paired_difference_inconclusive"
    )
