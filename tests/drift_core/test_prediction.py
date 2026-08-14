"""Tests for prediction drift."""

import numpy as np
import pytest

from drift_core.prediction import (
    PREDICTION_FEATURE_NAME,
    detect_prediction_drift,
    mean_score_shift,
)
from drift_core.types import DriftKind, Severity, WindowSpec

WINDOW = WindowSpec(window_id="w1", n_samples=1000)


@pytest.fixture
def rng():
    return np.random.default_rng(11)


class TestPredictionDrift:
    def test_stable_scores_no_drift(self, rng):
        ref = rng.beta(2, 5, 3000)
        cur = rng.beta(2, 5, 3000)
        result = detect_prediction_drift(ref, cur, window=WINDOW)
        assert result.is_drifted is False
        assert result.kind is DriftKind.PREDICTION

    def test_shifted_scores_detected(self, rng):
        ref = rng.beta(2, 5, 3000)
        cur = rng.beta(5, 2, 3000)
        result = detect_prediction_drift(ref, cur, window=WINDOW)
        assert result.is_drifted is True
        assert result.severity is Severity.ALERT

    def test_tagged_as_prediction_not_data(self, rng):
        # This matters for reporting: prediction drift and feature drift must
        # never be aggregated into one "drift count" in a governance report.
        result = detect_prediction_drift(
            rng.beta(2, 5, 500), rng.beta(2, 5, 500), window=WINDOW
        )
        assert result.kind is DriftKind.PREDICTION
        assert result.kind is not DriftKind.DATA
        assert result.feature_name == PREDICTION_FEATURE_NAME

    @pytest.mark.parametrize("method", ["psi", "ks", "wasserstein"])
    def test_all_methods_detect_obvious_shift(self, rng, method):
        ref = rng.beta(2, 8, 2000)
        cur = rng.beta(8, 2, 2000)
        result = detect_prediction_drift(ref, cur, window=WINDOW, method=method)
        assert result.is_drifted is True
        assert result.method == f"prediction_{method}"

    @pytest.mark.parametrize("method", ["psi", "ks", "wasserstein"])
    def test_all_methods_quiet_under_null(self, rng, method):
        ref = rng.beta(2, 8, 2000)
        cur = rng.beta(2, 8, 2000)
        result = detect_prediction_drift(ref, cur, window=WINDOW, method=method)
        assert result.is_drifted is False

    def test_ks_method_populates_pvalue(self, rng):
        result = detect_prediction_drift(
            rng.beta(2, 5, 1000), rng.beta(2, 5, 1000),
            window=WINDOW, method="ks",
        )
        assert result.p_value is not None

    def test_unknown_method_raises(self, rng):
        with pytest.raises(ValueError, match="unknown method"):
            detect_prediction_drift(
                rng.beta(2, 5, 100), rng.beta(2, 5, 100),
                window=WINDOW, method="nonsense",
            )


class TestMeanScoreShift:
    def test_reports_direction_of_movement(self):
        ref = np.full(1000, 0.10)
        cur = np.full(1000, 0.25)
        summary = mean_score_shift(ref, cur)
        assert summary["reference_mean"] == pytest.approx(0.10)
        assert summary["current_mean"] == pytest.approx(0.25)
        assert summary["absolute_change"] == pytest.approx(0.15)
        assert summary["relative_change"] == pytest.approx(1.5)

    def test_positive_rate_at_default_cutoff(self):
        ref = np.array([0.1, 0.2, 0.6, 0.9])
        cur = np.array([0.7, 0.8, 0.9, 0.95])
        summary = mean_score_shift(ref, cur)
        assert summary["reference_positive_rate"] == pytest.approx(0.5)
        assert summary["current_positive_rate"] == pytest.approx(1.0)

    def test_zero_reference_mean_does_not_divide_by_zero(self):
        summary = mean_score_shift(np.zeros(100), np.full(100, 0.3))
        assert np.isnan(summary["relative_change"])
        assert summary["absolute_change"] == pytest.approx(0.3)
