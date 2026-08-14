"""Tests for feature-level drift detectors.

Structure of each test group:
  - a NULL test (identical distributions -> statistic near zero / no alarm),
  - a SIGNAL test (known shift -> statistic exceeds the null),
  - a MONOTONICITY test (bigger shift -> bigger statistic),
  - edge cases that would silently corrupt a production monitor.

Monotonicity matters more than any specific threshold: thresholds get
recalibrated in component 5, but a statistic that doesn't increase with
shift magnitude is simply broken.
"""

import numpy as np
import pytest

from drift_core.types import DriftKind, Severity, WindowSpec
from drift_core.univariate import (
    detect_kl_drift,
    detect_ks_drift,
    detect_psi_drift,
    detect_wasserstein_drift,
    kl_divergence,
    ks_test,
    population_stability_index,
    wasserstein,
)

WINDOW = WindowSpec(window_id="w1", n_samples=1000)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestPSI:
    def test_identical_distributions_give_near_zero(self, rng):
        ref = rng.normal(0, 1, 5000)
        cur = rng.normal(0, 1, 5000)
        psi, _ = population_stability_index(ref, cur)
        assert psi < 0.05

    def test_same_array_gives_zero(self, rng):
        ref = rng.normal(0, 1, 1000)
        psi, _ = population_stability_index(ref, ref)
        assert psi == pytest.approx(0.0, abs=1e-9)

    def test_shifted_distribution_detected(self, rng):
        ref = rng.normal(0, 1, 5000)
        cur = rng.normal(1.0, 1, 5000)
        psi, _ = population_stability_index(ref, cur)
        assert psi > 0.25

    def test_monotonic_in_shift_size(self, rng):
        ref = rng.normal(0, 1, 5000)
        psis = [
            population_stability_index(ref, rng.normal(shift, 1, 5000))[0]
            for shift in (0.0, 0.25, 0.5, 1.0, 2.0)
        ]
        assert psis == sorted(psis)

    def test_variance_change_with_same_mean_detected(self, rng):
        # A pure scale change has zero mean shift — this is the case the
        # naive mean-shift baseline cannot see, so PSI must.
        ref = rng.normal(0, 1, 5000)
        cur = rng.normal(0, 2.5, 5000)
        psi, _ = population_stability_index(ref, cur)
        assert psi > 0.25

    def test_categorical_mode(self):
        ref = np.array(["a"] * 700 + ["b"] * 200 + ["c"] * 100)
        cur = np.array(["a"] * 300 + ["b"] * 300 + ["c"] * 400)
        psi, extra = population_stability_index(ref, cur, categorical=True)
        assert psi > 0.25
        assert extra["categories"] == ["a", "b", "c"]

    def test_unseen_category_does_not_crash(self):
        ref = np.array(["a"] * 500 + ["b"] * 500)
        cur = np.array(["a"] * 400 + ["b"] * 400 + ["z"] * 200)
        psi, extra = population_stability_index(ref, cur, categorical=True)
        assert np.isfinite(psi)
        assert "z" in extra["categories"]

    def test_values_outside_reference_range_are_binned_not_dropped(self, rng):
        # Outer edges are +/-inf; if they weren't, an entirely out-of-range
        # current window would produce an empty histogram and understate drift.
        ref = rng.normal(0, 1, 2000)
        cur = rng.normal(50, 1, 2000)
        psi, extra = population_stability_index(ref, cur)
        assert psi > 1.0
        # Reported proportions are the true observed ones (not epsilon-clipped),
        # so they sum to 1 even when most bins are empty.
        assert sum(extra["current_proportions"]) == pytest.approx(1.0, rel=1e-6)
        assert extra["current_proportions"][-1] == pytest.approx(1.0)

    def test_constant_reference_feature_does_not_crash(self):
        ref = np.full(1000, 3.0)
        cur = np.full(1000, 3.0)
        psi, _ = population_stability_index(ref, cur)
        assert np.isfinite(psi)

    def test_constant_reference_with_moved_current_detected(self):
        ref = np.full(1000, 3.0)
        cur = np.full(1000, 9.0)
        psi, _ = population_stability_index(ref, cur)
        assert psi > 0.25

    def test_nans_are_excluded_not_propagated(self, rng):
        ref = rng.normal(0, 1, 1000)
        cur = np.concatenate([rng.normal(0, 1, 900), np.full(100, np.nan)])
        psi, _ = population_stability_index(ref, cur)
        assert np.isfinite(psi)


class TestKLDivergence:
    def test_identical_distributions_near_zero(self, rng):
        ref = rng.normal(0, 1, 5000)
        cur = rng.normal(0, 1, 5000)
        kl, _ = kl_divergence(ref, cur)
        assert kl < 0.05

    def test_non_negative(self, rng):
        # KL is non-negative by construction; clipping must not break that.
        for _ in range(10):
            ref = rng.normal(0, 1, 1000)
            cur = rng.normal(rng.uniform(-2, 2), rng.uniform(0.5, 2), 1000)
            kl, _ = kl_divergence(ref, cur)
            assert kl >= -1e-9

    def test_asymmetric(self, rng):
        # KL(P||Q) != KL(Q||P) — documented as intentional, so pin it.
        ref = rng.normal(0, 1, 5000)
        cur = rng.normal(0, 3, 5000)
        forward, _ = kl_divergence(ref, cur)
        reverse, _ = kl_divergence(cur, ref)
        assert forward != pytest.approx(reverse, rel=0.05)

    def test_monotonic_in_shift_size(self, rng):
        ref = rng.normal(0, 1, 5000)
        kls = [
            kl_divergence(ref, rng.normal(shift, 1, 5000))[0]
            for shift in (0.0, 0.5, 1.0, 2.0)
        ]
        assert kls == sorted(kls)


class TestKSTest:
    def test_identical_distributions_high_pvalue(self, rng):
        ref = rng.normal(0, 1, 2000)
        cur = rng.normal(0, 1, 2000)
        _, p = ks_test(ref, cur)
        assert p > 0.05

    def test_false_positive_rate_near_alpha_under_null(self, rng):
        # The p-value must be calibrated, not merely "large when identical".
        # Under the null it is uniform, so ~5% of trials fire at alpha=0.05.
        # This is the property that makes multiple-testing correction in
        # component 5 valid; if it fails, corrected thresholds are meaningless.
        fires = sum(
            ks_test(rng.normal(0, 1, 500), rng.normal(0, 1, 500))[1] < 0.05
            for _ in range(200)
        )
        assert 0 <= fires <= 25  # generous band around the expected 10

    def test_shift_detected(self, rng):
        ref = rng.normal(0, 1, 2000)
        cur = rng.normal(0.5, 1, 2000)
        stat, p = ks_test(ref, cur)
        assert p < 0.01
        assert stat > 0.1

    def test_statistic_bounded_in_unit_interval(self, rng):
        stat, _ = ks_test(rng.normal(0, 1, 500), rng.normal(10, 1, 500))
        assert 0.0 <= stat <= 1.0


class TestWasserstein:
    def test_identical_distributions_near_zero(self, rng):
        ref = rng.normal(0, 1, 5000)
        cur = rng.normal(0, 1, 5000)
        stat, _ = wasserstein(ref, cur)
        assert stat < 0.1

    def test_equals_mean_difference_for_pure_translation(self):
        # For a pure shift, EMD is exactly the shift magnitude. Normalizing
        # by reference std must preserve that in std units.
        ref = np.linspace(0, 10, 1000)
        cur = ref + 2.0
        stat, extra = wasserstein(ref, cur, normalize=False)
        assert stat == pytest.approx(2.0, rel=1e-6)
        assert extra["raw_distance"] == pytest.approx(2.0, rel=1e-6)

    def test_normalization_makes_scales_comparable(self, rng):
        # Same shift in std units on two features with wildly different
        # units must produce the same normalized statistic.
        small = rng.normal(0, 1, 5000)
        small_shifted = rng.normal(1, 1, 5000)
        large = small * 1000
        large_shifted = small_shifted * 1000
        stat_small, _ = wasserstein(small, small_shifted)
        stat_large, _ = wasserstein(large, large_shifted)
        assert stat_small == pytest.approx(stat_large, rel=0.02)

    def test_monotonic_in_shift_size(self, rng):
        ref = rng.normal(0, 1, 5000)
        stats = [
            wasserstein(ref, rng.normal(shift, 1, 5000))[0]
            for shift in (0.0, 0.5, 1.0, 2.0)
        ]
        assert stats == sorted(stats)


class TestDetectorWrappers:
    def test_psi_detector_returns_data_kind(self, rng):
        result = detect_psi_drift(
            rng.normal(0, 1, 1000),
            rng.normal(0, 1, 1000),
            feature_name="age",
            window=WINDOW,
        )
        assert result.kind is DriftKind.DATA
        assert result.feature_name == "age"
        assert result.method == "psi"
        assert result.is_drifted is False
        assert result.severity is Severity.NONE

    def test_psi_detector_escalates_severity(self, rng):
        result = detect_psi_drift(
            rng.normal(0, 1, 5000),
            rng.normal(2, 1, 5000),
            feature_name="age",
            window=WINDOW,
        )
        assert result.is_drifted is True
        assert result.severity is Severity.ALERT

    def test_watch_band_between_thresholds(self, rng):
        ref = rng.normal(0, 1, 20000)
        cur = rng.normal(0.35, 1, 20000)
        result = detect_psi_drift(
            ref, cur, feature_name="x", window=WINDOW,
            watch_threshold=0.1, alert_threshold=0.25,
        )
        assert result.severity is Severity.WATCH
        assert result.is_drifted is True

    def test_ks_detector_populates_pvalue(self, rng):
        result = detect_ks_drift(
            rng.normal(0, 1, 1000),
            rng.normal(0, 1, 1000),
            feature_name="x",
            window=WINDOW,
        )
        assert result.p_value is not None
        assert 0.0 <= result.p_value <= 1.0

    def test_wasserstein_detector_records_sample_counts(self, rng):
        result = detect_wasserstein_drift(
            rng.normal(0, 1, 700),
            rng.normal(0, 1, 300),
            feature_name="x",
            window=WINDOW,
        )
        assert result.n_reference == 700
        assert result.n_current == 300

    def test_kl_detector_returns_data_kind(self, rng):
        result = detect_kl_drift(
            rng.normal(0, 1, 1000),
            rng.normal(3, 1, 1000),
            feature_name="x",
            window=WINDOW,
        )
        assert result.kind is DriftKind.DATA
        assert result.is_drifted is True

    def test_all_detectors_agree_on_obvious_drift(self, rng):
        ref = rng.normal(0, 1, 5000)
        cur = rng.normal(3, 1, 5000)
        kwargs = {"feature_name": "x", "window": WINDOW}
        results = [
            detect_psi_drift(ref, cur, **kwargs),
            detect_ks_drift(ref, cur, **kwargs),
            detect_wasserstein_drift(ref, cur, **kwargs),
            detect_kl_drift(ref, cur, **kwargs),
        ]
        assert all(r.is_drifted for r in results)

    def test_all_detectors_agree_on_no_drift(self, rng):
        ref = rng.normal(0, 1, 5000)
        cur = rng.normal(0, 1, 5000)
        kwargs = {"feature_name": "x", "window": WINDOW}
        results = [
            detect_psi_drift(ref, cur, **kwargs),
            detect_ks_drift(ref, cur, **kwargs),
            detect_wasserstein_drift(ref, cur, **kwargs),
            detect_kl_drift(ref, cur, **kwargs),
        ]
        assert not any(r.is_drifted for r in results)
