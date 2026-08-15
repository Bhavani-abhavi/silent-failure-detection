"""Tests for the serial-correlation-robust significance tools.

These functions overturned a headline claim, so they need to be at least as
well tested as the detectors. The tests that matter most are the ones that
construct a series with a KNOWN answer and check the naive test gets it wrong
while the corrected test gets it right — otherwise there is no evidence the
correction does anything.
"""

import numpy as np
import pytest
from scipy import stats

from backtest.significance import (
    correlation_p_value,
    effective_sample_size,
    fisher_ci,
    hac_slope,
    lag1_autocorrelation,
    moving_block_bootstrap_correlation,
    sign_test,
)


def ar1(n, phi, rng, scale=1.0):
    """AR(1) series: x[t] = phi*x[t-1] + noise."""
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, scale)
    return x


class TestFisherCI:
    def test_interval_brackets_the_estimate(self):
        lo, hi = fisher_ci(0.5, 30)
        assert lo < 0.5 < hi

    def test_interval_narrows_with_n(self):
        narrow = fisher_ci(0.5, 500)
        wide = fisher_ci(0.5, 20)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_reproduces_the_reported_headline_interval(self):
        # The number quoted in the README for the naive reading.
        lo, hi = fisher_ci(-0.3835, 35)
        assert lo == pytest.approx(-0.635, abs=0.005)
        assert hi == pytest.approx(-0.058, abs=0.005)

    def test_degenerate_inputs_return_nan(self):
        assert np.isnan(fisher_ci(0.5, 3)[0])
        assert np.isnan(fisher_ci(1.0, 30)[0])


class TestCorrelationPValue:
    def test_matches_scipy_at_full_n(self):
        rng = np.random.default_rng(0)
        x, y = rng.normal(size=40), rng.normal(size=40)
        r, p = stats.pearsonr(x, y)
        assert correlation_p_value(r, 40) == pytest.approx(p, rel=1e-9)

    def test_reducing_n_raises_the_p_value(self):
        """The whole mechanism of the correction in one assertion."""
        assert correlation_p_value(-0.3835, 13.0) > correlation_p_value(-0.3835, 35)

    def test_reproduces_the_correction(self):
        assert correlation_p_value(-0.3835, 35) == pytest.approx(0.023, abs=0.002)
        assert correlation_p_value(-0.3835, 13.0) == pytest.approx(0.195, abs=0.005)


class TestLag1Autocorrelation:
    def test_white_noise_is_near_zero(self):
        rng = np.random.default_rng(1)
        assert abs(lag1_autocorrelation(rng.normal(size=5000))) < 0.05

    def test_persistent_series_is_positive(self):
        rng = np.random.default_rng(2)
        assert lag1_autocorrelation(ar1(5000, 0.8, rng)) > 0.7

    def test_alternating_series_is_negative(self):
        assert lag1_autocorrelation(np.tile([1.0, -1.0], 50)) < -0.9

    def test_constant_series_returns_nan(self):
        assert np.isnan(lag1_autocorrelation(np.ones(50)))


class TestEffectiveSampleSize:
    def test_white_noise_keeps_almost_all_of_n(self):
        rng = np.random.default_rng(3)
        n_eff, _, _ = effective_sample_size(
            rng.normal(size=200), rng.normal(size=200)
        )
        assert n_eff > 170

    def test_two_persistent_series_collapse(self):
        rng = np.random.default_rng(4)
        n_eff, a1, b1 = effective_sample_size(ar1(200, 0.8, rng), ar1(200, 0.8, rng))
        assert a1 > 0.6 and b1 > 0.6
        assert n_eff < 60, f"n_eff {n_eff} did not collapse for phi=0.8 series"

    def test_never_exceeds_n(self):
        """Anti-persistent series can push the formula above n, which would
        claim more information than was collected."""
        alternating = np.tile([1.0, -1.0], 50) + 1e-9
        n_eff, _, _ = effective_sample_size(alternating, alternating.copy())
        assert n_eff <= 100

    def test_never_below_three(self):
        rng = np.random.default_rng(5)
        n_eff, _, _ = effective_sample_size(ar1(50, 0.99, rng), ar1(50, 0.99, rng))
        assert n_eff >= 3


class TestNaiveTestFailsWhereCorrectedOneDoesNot:
    def test_two_independent_random_walks_fool_the_naive_test(self):
        """The canonical spurious-regression setup.

        Two INDEPENDENT random walks. Any correlation between them is
        spurious by construction. The naive test should reject H0 far more
        than 5% of the time; the corrected test much less.
        """
        rng = np.random.default_rng(7)
        naive_rejections, adjusted_rejections = 0, 0
        trials = 300
        for _ in range(trials):
            x = np.cumsum(rng.normal(size=35))
            y = np.cumsum(rng.normal(size=35))
            r, p_naive = stats.pearsonr(x, y)
            n_eff, _, _ = effective_sample_size(x, y)
            if p_naive < 0.05:
                naive_rejections += 1
            if correlation_p_value(r, n_eff) < 0.05:
                adjusted_rejections += 1

        naive_rate = naive_rejections / trials
        adjusted_rate = adjusted_rejections / trials
        assert naive_rate > 0.30, (
            f"naive false-positive rate {naive_rate:.1%} — the fixture is not "
            f"reproducing the spurious-correlation problem"
        )
        assert adjusted_rate < naive_rate / 2, (
            f"correction barely helped: {adjusted_rate:.1%} vs {naive_rate:.1%}"
        )


class TestHACSlope:
    def test_detects_a_real_trend_under_autocorrelated_noise(self):
        rng = np.random.default_rng(8)
        y = 0.01 * np.arange(60) + ar1(60, 0.6, rng, scale=0.02)
        slope, _, p = hac_slope(y)
        assert slope > 0
        assert p < 0.05

    def test_flat_autocorrelated_series_is_not_a_trend(self):
        rng = np.random.default_rng(9)
        rejections = 0
        for seed in range(200):
            y = ar1(60, 0.6, np.random.default_rng(seed), scale=0.02)
            if hac_slope(y)[2] < 0.05:
                rejections += 1
        # HAC is known to under-correct at short n; require it to be far
        # better than the ~30%+ an uncorrected OLS gives here, not perfect.
        assert rejections / 200 < 0.20

    def test_standard_error_exceeds_ols_under_positive_autocorrelation(self):
        """The correction must actually widen the interval, or it is inert."""
        rng = np.random.default_rng(10)
        y = ar1(80, 0.8, rng)
        _, hac_se, _ = hac_slope(y)
        ols = stats.linregress(np.arange(80, dtype=float), y)
        assert hac_se > ols.stderr

    def test_too_short_returns_nan(self):
        assert np.isnan(hac_slope([1.0, 2.0])[0])


class TestMovingBlockBootstrap:
    def test_ci_brackets_a_strong_real_correlation(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=60)
        y = x + rng.normal(0, 0.3, size=60)
        lo, hi = moving_block_bootstrap_correlation(x, y, n_boot=2000)
        assert lo > 0.5 and hi <= 1.0

    def test_ci_includes_zero_for_independent_random_walks(self):
        rng = np.random.default_rng(12)
        x, y = np.cumsum(rng.normal(size=35)), np.cumsum(rng.normal(size=35))
        lo, hi = moving_block_bootstrap_correlation(x, y, n_boot=2000)
        assert lo < 0 < hi

    def test_short_series_returns_nan(self):
        assert np.isnan(moving_block_bootstrap_correlation([1.0, 2.0], [1.0, 2.0])[0])


class TestSignTest:
    def test_all_same_sign_is_extreme(self):
        below, above, p = sign_test(-np.ones(35))
        assert (below, above) == (35, 0)
        assert p == pytest.approx(5.82e-11, rel=0.05)

    def test_balanced_signs_are_not(self):
        _, _, p = sign_test(np.array([1.0, -1.0] * 10))
        assert p > 0.9

    def test_override_reproduces_the_conservative_discount(self):
        """35/35 discounted to 13 effectively independent observations."""
        _, _, p = sign_test(-np.ones(35), n_override=13)
        assert p == pytest.approx(2.44e-4, rel=0.05)

    def test_override_is_more_conservative_than_full_n(self):
        _, _, strict = sign_test(-np.ones(35), n_override=13)
        _, _, naive = sign_test(-np.ones(35))
        assert strict > naive

    def test_zeros_are_excluded_from_both_counts(self):
        below, above, _ = sign_test(np.array([-1.0, 0.0, 1.0, -1.0]))
        assert (below, above) == (2, 1)

    @pytest.mark.parametrize("n_override", [12.0, 13.0, 13.02, 13.5, 12.7, 25.9])
    def test_non_integer_override_never_yields_p_equal_zero(self, n_override):
        """Regression: a non-integer effective n produced p = 0.

        Scaling the majority count against the unrounded float and taking a
        ceiling pushed successes above the trial count, and `binom.sf(n, n, p)`
        is exactly 0 — a value no finite sign test can produce. It reported
        p = 0 for a 35/35 result whose correct discounted value is 2.4e-4.
        """
        _, _, p = sign_test(-np.ones(35), n_override=n_override)
        assert p > 0.0, f"p = 0 is not attainable by a finite sign test"
        assert p <= 1.0

    def test_discounted_p_is_monotone_in_effective_n(self):
        """More effective observations of the same one-sided result should
        never be less surprising."""
        previous = 1.0
        for n_override in (5, 10, 13, 20, 35):
            _, _, p = sign_test(-np.ones(35), n_override=n_override)
            assert p <= previous
            previous = p
