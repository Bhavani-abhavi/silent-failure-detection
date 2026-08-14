"""Tests for estimation-error scoring.

The behaviour that matters here is how a *suppressed* estimate is treated. An
estimator that declines to answer when its assumptions fail is behaving
correctly; if suppression were scored as a large error, the summary would
punish the honest estimator and flatter the reckless one, and the conclusion
drawn from component B would invert.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.scoring import (
    estimator_tracked_degradation,
    score_estimates,
    summarise_estimation_error,
)

WINDOWS = [f"2014-{m:02d}" for m in range(1, 7)]


@pytest.fixture
def truth():
    return pd.DataFrame(
        {
            "base_rate": [0.10, 0.12, 0.14, 0.16, 0.18, 0.20],
            "brier": [0.09, 0.10, 0.11, 0.12, 0.13, 0.14],
            "accuracy": [0.90, 0.88, 0.86, 0.84, 0.82, 0.80],
        },
        index=pd.Index(WINDOWS, name="window_id"),
    )


def _estimates(values, *, metric="base_rate", method="average_confidence",
               status="ok", reason=""):
    return pd.DataFrame(
        {
            "window_id": WINDOWS,
            "metric": metric,
            "method": method,
            "estimate": values,
            "status": status,
            "effective_sample_size": 1000.0,
            "suppression_reason": reason,
        }
    )


class TestScoreEstimates:
    def test_error_is_estimate_minus_truth(self, truth):
        scored = score_estimates(_estimates([0.10] * 6), truth)
        assert scored["error"].iloc[0] == pytest.approx(0.0)
        assert scored["error"].iloc[-1] == pytest.approx(-0.10)
        assert scored["abs_error"].iloc[-1] == pytest.approx(0.10)

    def test_relative_error_uses_the_true_value(self, truth):
        scored = score_estimates(_estimates([0.05] * 6), truth)
        assert scored["relative_error"].iloc[0] == pytest.approx(-0.5)

    def test_unknown_metric_is_skipped_not_crashed(self, truth):
        estimates = _estimates([0.1] * 6, metric="something_else")
        assert score_estimates(estimates, truth).empty

    def test_window_absent_from_truth_is_skipped(self, truth):
        estimates = _estimates([0.1] * 6)
        estimates.loc[0, "window_id"] = "1999-01"
        assert len(score_estimates(estimates, truth)) == 5


class TestSuppressionHandling:
    def test_suppressed_windows_are_excluded_from_error(self, truth):
        estimates = pd.concat([
            _estimates([0.10, 0.12, 0.14] + [np.nan] * 3,
                       method="importance_weighted").iloc[:3],
            _estimates([np.nan] * 6, method="importance_weighted",
                       status="suppressed", reason="ess_collapse").iloc[3:],
        ])
        scored = score_estimates(estimates, truth)
        summary = summarise_estimation_error(scored)
        row = summary.iloc[0]
        assert row["n_windows"] == 6
        assert row["coverage"] == pytest.approx(0.5)
        # Perfect on the windows it answered; the NaNs must not pollute this.
        assert row["mean_abs_error"] == pytest.approx(0.0, abs=1e-9)

    def test_coverage_distinguishes_silent_from_wrong(self, truth):
        silent = score_estimates(
            _estimates([np.nan] * 6, status="suppressed", reason="ess_collapse"),
            truth,
        )
        wrong = score_estimates(_estimates([0.5] * 6), truth)
        silent_summary = summarise_estimation_error(silent).iloc[0]
        wrong_summary = summarise_estimation_error(wrong).iloc[0]

        assert silent_summary["coverage"] == 0.0
        assert wrong_summary["coverage"] == 1.0
        assert wrong_summary["mean_abs_error"] > 0.3


class TestSystematicBias:
    def test_always_same_direction_is_flagged(self, truth):
        """An estimator wrong by the same sign every window is a different
        and more serious defect than one that is noisy, and only the signed
        mean reveals it."""
        scored = score_estimates(_estimates([0.08] * 6), truth)
        summary = summarise_estimation_error(scored).iloc[0]
        assert bool(summary["always_same_direction"])
        assert summary["mean_error"] < 0
        assert summary["mean_error"] == pytest.approx(-summary["mean_abs_error"])

    def test_unbiased_noise_is_not_flagged(self, truth):
        estimates = truth["base_rate"].to_numpy() + np.array(
            [0.01, -0.01, 0.01, -0.01, 0.01, -0.01]
        )
        scored = score_estimates(_estimates(estimates), truth)
        summary = summarise_estimation_error(scored).iloc[0]
        assert not bool(summary["always_same_direction"])
        assert abs(summary["mean_error"]) < summary["mean_abs_error"]


class TestTracking:
    def test_flat_estimator_has_near_zero_range_ratio(self, truth):
        """The signature of structural blindness: the truth moves, the
        estimate does not. Mean error alone would not reveal this — a
        constant sitting in the middle of the true range has a modest
        average error and is useless for monitoring."""
        scored = score_estimates(_estimates([0.15] * 6), truth)
        tracking = estimator_tracked_degradation(
            scored, "base_rate", "average_confidence"
        )
        assert tracking["range_ratio"] == pytest.approx(0.0, abs=1e-9)

    def test_tracking_estimator_has_ratio_near_one(self, truth):
        scored = score_estimates(
            _estimates(truth["base_rate"].to_numpy() + 0.01), truth
        )
        tracking = estimator_tracked_degradation(
            scored, "base_rate", "average_confidence"
        )
        assert tracking["correlation"] == pytest.approx(1.0, abs=1e-6)
        assert tracking["range_ratio"] == pytest.approx(1.0, abs=1e-6)

    def test_too_few_windows_returns_nan_not_a_number(self, truth):
        scored = score_estimates(_estimates([0.1] * 6), truth).head(2)
        tracking = estimator_tracked_degradation(
            scored, "base_rate", "average_confidence"
        )
        assert np.isnan(tracking["correlation"])
