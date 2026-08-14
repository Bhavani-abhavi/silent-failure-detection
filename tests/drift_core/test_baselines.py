"""Tests for the naive baselines.

These also pin the baselines' known blind spots. Those blind spots are the
argument for the sophisticated detectors, so they need to be demonstrated
rather than asserted — and if a blind-spot test ever starts passing, the
corresponding argument for the complex detector has weakened and the
benchmark section of the writeup needs revisiting.
"""

import numpy as np
import pytest

from drift_core.baselines import mean_shift_baseline, missingness_baseline
from drift_core.types import WindowSpec

WINDOW = WindowSpec(window_id="w1", n_samples=1000)


@pytest.fixture
def rng():
    return np.random.default_rng(31)


class TestMeanShiftBaseline:
    def test_no_shift_does_not_fire(self, rng):
        result = mean_shift_baseline(
            rng.normal(0, 1, 5000), rng.normal(0, 1, 1000),
            feature_name="x", window=WINDOW,
        )
        assert result.is_drifted is False

    def test_clear_shift_fires(self, rng):
        result = mean_shift_baseline(
            rng.normal(0, 1, 5000), rng.normal(1, 1, 1000),
            feature_name="x", window=WINDOW,
        )
        assert result.is_drifted is True

    def test_blind_to_variance_only_change(self, rng):
        # Same mean, tripled spread. The baseline cannot see this; PSI can.
        # This is the concrete justification for using distributional tests.
        #
        # Regression guard: an earlier version computed the standard error
        # from the reference std alone, which understates the sampling
        # variability of the current mean once the current variance grows.
        # It therefore fired here — a false alarm that would have been
        # scored as a successful detection in the benchmark. Averaged over
        # seeds below so the assertion tests the calibration rather than
        # one lucky draw.
        fires = 0
        for seed in range(30):
            r = np.random.default_rng(seed)
            result = mean_shift_baseline(
                r.normal(0, 1, 5000), r.normal(0, 3, 5000),
                feature_name="x", window=WINDOW,
            )
            fires += int(result.is_drifted)
        assert fires <= 1, f"variance-only change triggered {fires}/30 false alarms"

    def test_blind_to_bimodal_split_with_same_mean(self, rng):
        # Current window splits into two clusters around the reference mean.
        # Mean is unchanged; the distribution is unrecognizable.
        ref = rng.normal(0, 1, 5000)
        cur = np.concatenate([rng.normal(-4, 0.3, 2500), rng.normal(4, 0.3, 2500)])
        result = mean_shift_baseline(
            ref, cur, feature_name="x", window=WINDOW
        )
        assert result.is_drifted is False

    def test_constant_reference_does_not_divide_by_zero(self):
        result = mean_shift_baseline(
            np.full(1000, 5.0), np.full(500, 9.0),
            feature_name="x", window=WINDOW,
        )
        assert np.isfinite(result.statistic)

    def test_sensitivity_grows_with_current_window_size(self, rng):
        # Standard error shrinks with n, so the same shift is more
        # significant on a larger window. Worth pinning because it means
        # baseline fire rates are not comparable across windows of
        # different sizes — a trap when benchmarking detection latency.
        ref = rng.normal(0, 1, 10000)
        small = mean_shift_baseline(
            ref, rng.normal(0.1, 1, 50), feature_name="x", window=WINDOW
        )
        large = mean_shift_baseline(
            ref, rng.normal(0.1, 1, 5000), feature_name="x", window=WINDOW
        )
        assert large.statistic > small.statistic


class TestMissingnessBaseline:
    def test_stable_missingness_does_not_fire(self, rng):
        ref = np.where(rng.random(1000) < 0.1, np.nan, rng.normal(0, 1, 1000))
        cur = np.where(rng.random(1000) < 0.1, np.nan, rng.normal(0, 1, 1000))
        result = missingness_baseline(
            ref, cur, feature_name="x", window=WINDOW
        )
        assert result.is_drifted is False

    def test_broken_upstream_feed_fires(self, rng):
        # The classic production incident: a join breaks and a feature goes
        # mostly null. Trivial to detect, and worth including in every
        # benchmark precisely because it is trivial.
        ref = rng.normal(0, 1, 1000)
        cur = np.where(rng.random(1000) < 0.8, np.nan, rng.normal(0, 1, 1000))
        result = missingness_baseline(
            ref, cur, feature_name="x", window=WINDOW
        )
        assert result.is_drifted is True
        assert result.extra["current_missing_rate"] > 0.7

    def test_reports_both_rates(self, rng):
        ref = np.concatenate([np.full(100, np.nan), rng.normal(0, 1, 900)])
        cur = np.concatenate([np.full(300, np.nan), rng.normal(0, 1, 700)])
        result = missingness_baseline(
            ref, cur, feature_name="x", window=WINDOW
        )
        assert result.extra["reference_missing_rate"] == pytest.approx(0.1)
        assert result.extra["current_missing_rate"] == pytest.approx(0.3)
        assert result.statistic == pytest.approx(0.2)
