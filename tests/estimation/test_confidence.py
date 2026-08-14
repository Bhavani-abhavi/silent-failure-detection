"""Tests for confidence-based label-free estimation.

Two things are pinned: that the estimators are correct when their assumption
holds, and that they are *blind* when it does not. The second set matters
more. These estimators are the ones a reader is most likely to trust on
reputation, and the blindness tests are what stop this project from quietly
shipping a number it knows is unreliable.
"""

import numpy as np
import pytest

from estimation.confidence import (
    estimate_accuracy_atc,
    estimate_accuracy_average_confidence,
    estimate_base_rate,
    estimate_brier,
    fit_atc_threshold,
)
from estimation.types import EstimateStatus


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _calibrated(n, rng, shift=0.0):
    """Probabilities that are honest: P(y=1) really is p."""
    p = np.clip(rng.beta(2, 8, n) + shift, 0.001, 0.999)
    y = (rng.random(n) < p).astype(float)
    return p, y


class TestCorrectWhenCalibrated:
    def test_base_rate_estimate_matches_truth(self, rng):
        p, y = _calibrated(50_000, rng)
        estimate = estimate_base_rate(p).estimate
        assert abs(estimate - y.mean()) < 0.005

    def test_brier_estimate_matches_truth(self, rng):
        """`E[(y-p)^2] = E[p(1-p)]` holds exactly under calibration."""
        p, y = _calibrated(50_000, rng)
        estimate = estimate_brier(p).estimate
        true_brier = float(np.mean((y - p) ** 2))
        assert abs(estimate - true_brier) < 0.005

    def test_atc_threshold_reproduces_reference_accuracy(self, rng):
        p, y = _calibrated(50_000, rng)
        threshold = fit_atc_threshold(p, y)
        estimate = estimate_accuracy_atc(p, threshold).estimate
        true_accuracy = float(np.mean((p >= 0.5) == y))
        assert abs(estimate - true_accuracy) < 0.01


class TestBlindToCalibrationDrift:
    """The structural limitation, pinned as behaviour.

    This is the project's central negative result reduced to a unit test: the
    estimators read the model's own probabilities to decide whether those
    probabilities can be trusted. When the world gets riskier but the model's
    outputs do not move, every estimate here stays exactly where it was.
    """

    def test_base_rate_estimate_does_not_move_when_only_truth_moves(self, rng):
        p, _ = _calibrated(50_000, rng)
        # Same predictions; the world underneath them got 50% riskier.
        y_before = (rng.random(len(p)) < p).astype(float)
        y_after = (rng.random(len(p)) < np.clip(p * 1.5, 0, 1)).astype(float)

        before = estimate_base_rate(p).estimate
        after = estimate_base_rate(p).estimate

        assert before == after, "estimate moved without predictions moving"
        assert abs(after - y_after.mean()) > 0.02, (
            "precondition: the truth must have moved enough to matter"
        )
        assert abs(before - y_before.mean()) < 0.01

    def test_brier_estimate_understates_a_miscalibrated_model(self, rng):
        p, _ = _calibrated(50_000, rng)
        y_worse = (rng.random(len(p)) < np.clip(p * 1.5, 0, 1)).astype(float)

        estimated = estimate_brier(p).estimate
        true_brier = float(np.mean((y_worse - p) ** 2))

        assert estimated < true_brier, (
            "the estimator should understate Brier when the model is "
            "over-confident about safety"
        )
        assert (true_brier - estimated) / true_brier > 0.05

    def test_accuracy_estimate_is_unchanged_by_label_shift(self, rng):
        p, _ = _calibrated(20_000, rng)
        first = estimate_accuracy_average_confidence(p).estimate
        second = estimate_accuracy_average_confidence(p).estimate
        assert first == second


class TestInsufficientData:
    @pytest.mark.parametrize(
        "estimator",
        [estimate_base_rate, estimate_brier, estimate_accuracy_average_confidence],
    )
    def test_small_window_is_flagged_not_answered(self, estimator):
        result = estimator(np.linspace(0.1, 0.9, 10))
        assert result.status is EstimateStatus.INSUFFICIENT_DATA
        assert np.isnan(result.estimate)
        assert result.is_usable is False

    def test_nan_probabilities_are_dropped(self, rng):
        p = np.concatenate([rng.beta(2, 8, 1000), np.full(50, np.nan)])
        result = estimate_base_rate(p)
        assert result.n_current == 1000
        assert np.isfinite(result.estimate)

    def test_atc_with_unfittable_threshold_is_flagged(self):
        result = estimate_accuracy_atc(np.linspace(0.1, 0.9, 100), float("nan"))
        assert result.status is EstimateStatus.INSUFFICIENT_DATA
