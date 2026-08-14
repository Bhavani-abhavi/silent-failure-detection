"""Tests for the shared validity / minimum-detectable-effect contract.

These are regression tests for a class of bug that has now appeared twice in
this codebase in unrelated detectors: a detector that cannot see reporting
that all is well. Both instances are reproduced here directly, so the rule
cannot silently regress:

  - `test_ks_all_nan_reference_does_not_report_no_drift` — the original bug,
    found on real Lending Club data where a bureau feature is 100% null
    before 2012.
  - `test_permutation_floor_raises` — the original bug, found when a
    perfectly separable pair of windows (AUC 1.0) returned "no drift".
"""

import numpy as np
import pytest

from drift_core.types import DriftKind, ResultStatus, Severity, WindowSpec
from drift_core.univariate import (
    detect_kl_drift,
    detect_ks_drift,
    detect_psi_drift,
    detect_wasserstein_drift,
)
from drift_core.validity import (
    check_windows,
    coverage,
    insufficient_data_result,
    ks_minimum_detectable_effect,
    n_valid,
    psi_null_expectation,
    require_detectable_alpha,
)

WINDOW = WindowSpec(window_id="w1", n_samples=1000)
DETECTORS = {
    "psi": detect_psi_drift,
    "ks": detect_ks_drift,
    "wasserstein": detect_wasserstein_drift,
    "kl": detect_kl_drift,
}


@pytest.fixture
def rng():
    return np.random.default_rng(5)


class TestCoverageHelpers:
    def test_coverage_counts_only_finite(self):
        values = np.array([1.0, 2.0, np.nan, np.inf, -np.inf, 5.0])
        assert coverage(values) == pytest.approx(3 / 6)

    def test_coverage_of_empty_is_zero(self):
        assert coverage(np.array([])) == 0.0

    def test_n_valid_excludes_nan_and_inf(self):
        assert n_valid(np.array([1.0, np.nan, 3.0, np.inf])) == 2


class TestWindowChecks:
    def test_healthy_windows_pass(self, rng):
        valid, reason = check_windows(rng.normal(0, 1, 500), rng.normal(0, 1, 500))
        assert valid is True
        assert reason == ""

    def test_all_nan_reference_is_rejected(self, rng):
        valid, reason = check_windows(np.full(500, np.nan), rng.normal(0, 1, 500))
        assert valid is False
        assert "reference" in reason

    def test_all_nan_current_names_schema_change(self, rng):
        # The message matters: this is nearly always an upstream pipeline or
        # vendor schema change, and saying so saves the on-call engineer from
        # investigating the model instead of the feed.
        valid, reason = check_windows(rng.normal(0, 1, 500), np.full(500, np.nan))
        assert valid is False
        assert "schema or pipeline change" in reason

    def test_empty_current_is_rejected(self, rng):
        valid, _ = check_windows(rng.normal(0, 1, 500), np.array([]))
        assert valid is False

    def test_small_sample_is_rejected(self, rng):
        valid, reason = check_windows(
            rng.normal(0, 1, 500), rng.normal(0, 1, 10), min_samples=30
        )
        assert valid is False
        assert "below min_samples" in reason

    def test_low_coverage_is_rejected(self, rng):
        sparse = np.where(rng.random(500) < 0.9, np.nan, rng.normal(0, 1, 500))
        valid, reason = check_windows(
            rng.normal(0, 1, 500), sparse, min_coverage=0.5
        )
        assert valid is False
        assert "coverage" in reason


class TestMinimumDetectableEffect:
    def test_ks_mde_shrinks_with_sample_size(self):
        mdes = [ks_minimum_detectable_effect(n, n) for n in (50, 200, 1000, 10000)]
        assert mdes == sorted(mdes, reverse=True)

    def test_ks_mde_exceeds_one_for_tiny_samples(self):
        # KS is bounded by 1, so an MDE above 1 means literally nothing is
        # detectable — not even two disjoint distributions.
        assert ks_minimum_detectable_effect(3, 3) > 1.0

    def test_ks_mde_matches_known_critical_value(self):
        # D_crit = 1.358 * sqrt((n1+n2)/(n1*n2)); for n1=n2=100 that is
        # 1.358 * sqrt(200/10000) = 0.1921.
        assert ks_minimum_detectable_effect(100, 100, 0.05) == pytest.approx(
            0.1921, abs=1e-3
        )

    def test_psi_null_expectation_matches_chi_square_form(self):
        # (bins - 1) * (1/n_ref + 1/n_cur)
        assert psi_null_expectation(1000, 1000, 10) == pytest.approx(9 * 0.002)

    def test_psi_null_expectation_shrinks_with_sample_size(self):
        values = [psi_null_expectation(n, n, 10) for n in (100, 1000, 10000)]
        assert values == sorted(values, reverse=True)

    def test_psi_folklore_threshold_is_noise_at_small_n(self):
        # The headline reason this function exists: on two 200-row windows
        # with 10 bins, the null expectation is 0.09 — so the conventional
        # 0.1 "watch" threshold is essentially the noise floor, and PSI
        # monitoring at that size reports drift that is not there.
        assert psi_null_expectation(200, 200, 10) == pytest.approx(0.09, abs=1e-6)

    def test_psi_detector_flags_no_power_when_threshold_below_noise(self, rng):
        result = detect_psi_drift(
            rng.normal(0, 1, 150),
            rng.normal(0, 1, 150),
            feature_name="x",
            window=WINDOW,
            watch_threshold=0.05,  # below the 0.12 null expectation at n=150
        )
        assert result.status is ResultStatus.NO_POWER
        assert "fires on sampling noise" in result.extra["reason"]

    def test_ks_detector_reports_its_own_mde(self, rng):
        result = detect_ks_drift(
            rng.normal(0, 1, 500), rng.normal(0, 1, 500),
            feature_name="x", window=WINDOW,
        )
        assert result.minimum_detectable_effect == pytest.approx(
            ks_minimum_detectable_effect(500, 500, 0.05)
        )


class TestSilentFailureRegressions:
    """The two original bugs, pinned."""

    def test_ks_all_nan_reference_does_not_report_no_drift(self, rng):
        # BEFORE THE FIX: returned statistic=nan, is_drifted=False,
        # severity=NONE — indistinguishable from a clean result.
        result = detect_ks_drift(
            np.full(1000, np.nan), rng.normal(0, 1, 1000),
            feature_name="vanished_feature", window=WINDOW,
        )
        assert result.status is ResultStatus.INSUFFICIENT_DATA
        assert np.isnan(result.statistic)
        assert result.extra["reason"]

    def test_permutation_floor_raises(self):
        with pytest.raises(ValueError, match="cannot produce a p-value below"):
            require_detectable_alpha(n_permutations=15, alpha=0.05)

    def test_permutation_floor_allows_sufficient_counts(self):
        require_detectable_alpha(n_permutations=99, alpha=0.05)

    @pytest.mark.parametrize("name", sorted(DETECTORS))
    def test_no_detector_crashes_on_all_nan_reference(self, name, rng):
        # Three of the four used to raise (IndexError/ValueError) and the
        # fourth returned a false all-clear. A monitoring sweep over
        # thousands of feature-windows must survive a retired feature.
        result = DETECTORS[name](
            np.full(500, np.nan), rng.normal(0, 1, 500),
            feature_name="x", window=WINDOW,
        )
        assert result.status is ResultStatus.INSUFFICIENT_DATA
        assert result.is_drifted is False

    @pytest.mark.parametrize("name", sorted(DETECTORS))
    def test_no_detector_crashes_on_empty_current(self, name, rng):
        result = DETECTORS[name](
            rng.normal(0, 1, 500), np.array([]),
            feature_name="x", window=WINDOW,
        )
        assert result.status is ResultStatus.INSUFFICIENT_DATA

    @pytest.mark.parametrize("name", sorted(DETECTORS))
    def test_insufficient_data_never_looks_like_a_clean_pass(self, name, rng):
        """The contract callers depend on: is_drifted=False is only
        meaningful when status is OK."""
        result = DETECTORS[name](
            np.full(500, np.nan), rng.normal(0, 1, 500),
            feature_name="x", window=WINDOW,
        )
        clean_pass = result.is_drifted is False and result.status is ResultStatus.OK
        assert not clean_pass


class TestInsufficientDataResult:
    def test_carries_nan_statistic_not_zero(self):
        # Zero would be a plausible-looking "no drift" magnitude. NaN
        # propagates visibly through any downstream arithmetic instead.
        result = insufficient_data_result(
            feature_name="x", method="psi", kind=DriftKind.DATA,
            window=WINDOW, reason="test", n_reference=0, n_current=100,
        )
        assert np.isnan(result.statistic)
        assert result.severity is Severity.NONE
        assert result.status is ResultStatus.INSUFFICIENT_DATA
