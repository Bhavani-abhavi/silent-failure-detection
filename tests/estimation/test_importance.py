"""Tests for importance-weighted estimation.

The experiment this module exists to run needs both halves to be trustworthy:
importance weighting must actually work under covariate shift, or its failure
under concept drift proves nothing.
"""

import numpy as np
import pytest

from estimation.importance import (
    effective_sample_size,
    fit_density_ratio,
    importance_weighted_estimates,
)
from estimation.types import EstimateStatus


@pytest.fixture
def rng():
    return np.random.default_rng(3)


def _population(n, rng, mean=0.0, coefficient=1.2):
    """X ~ N(mean, 1); P(y=1|x) depends on x through `coefficient`."""
    x = rng.normal(mean, 1.0, (n, 2))
    logit = coefficient * x[:, 0] - 0.5 * x[:, 1]
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(float)
    return x, p, y


class TestEffectiveSampleSize:
    def test_uniform_weights_give_full_n(self):
        assert effective_sample_size(np.ones(500)) == pytest.approx(500.0)

    def test_one_dominant_weight_collapses_to_one(self):
        w = np.concatenate([[1e6], np.ones(999)])
        assert effective_sample_size(w) < 2.0

    def test_zero_weights_give_zero(self):
        assert effective_sample_size(np.zeros(100)) == 0.0


class TestCorrectUnderCovariateShift:
    """`P(X)` moves, `P(Y|X)` holds — the regime importance weighting is
    actually valid in."""

    def test_reweighting_recovers_the_shifted_base_rate(self, rng):
        ref_x, ref_p, ref_y = _population(20_000, rng, mean=0.0)
        cur_x, _, cur_y = _population(20_000, rng, mean=0.8)  # same P(Y|X)

        weights, diagnostics = fit_density_ratio(ref_x, cur_x, max_samples=8000)
        estimates = importance_weighted_estimates(
            ref_y, ref_p, weights, diagnostics, n_current=len(cur_x)
        )
        base_rate = next(e for e in estimates if e.metric == "base_rate")

        unweighted_error = abs(ref_y.mean() - cur_y.mean())
        weighted_error = abs(base_rate.estimate - cur_y.mean())
        assert base_rate.is_usable
        assert weighted_error < unweighted_error, (
            f"reweighting made it worse: {weighted_error:.4f} vs "
            f"{unweighted_error:.4f} unweighted"
        )
        assert weighted_error < 0.03

    def test_no_shift_leaves_the_estimate_alone(self, rng):
        ref_x, ref_p, ref_y = _population(10_000, rng, mean=0.0)
        cur_x, _, cur_y = _population(10_000, rng, mean=0.0)

        weights, diagnostics = fit_density_ratio(ref_x, cur_x, max_samples=6000)
        estimates = importance_weighted_estimates(
            ref_y, ref_p, weights, diagnostics, n_current=len(cur_x)
        )
        base_rate = next(e for e in estimates if e.metric == "base_rate")
        assert abs(base_rate.estimate - cur_y.mean()) < 0.03


class TestBlindToConceptDrift:
    def test_reweighting_cannot_fix_a_changed_conditional(self, rng):
        """The sharp experiment.

        `P(X)` is identical between windows, so there is nothing for the
        density ratio to correct — weights are ~uniform. Only `P(Y|X)` moved.
        Importance weighting therefore cannot help, and it should be seen not
        to help rather than assumed not to.
        """
        ref_x, ref_p, ref_y = _population(20_000, rng, mean=0.0, coefficient=1.2)
        # Same X distribution, much stronger relationship to y.
        cur_x, _, cur_y = _population(20_000, rng, mean=0.0, coefficient=2.6)

        weights, diagnostics = fit_density_ratio(ref_x, cur_x, max_samples=8000)
        estimates = importance_weighted_estimates(
            ref_y, ref_p, weights, diagnostics, n_current=len(cur_x)
        )
        brier = next(e for e in estimates if e.metric == "brier")

        true_brier = float(np.mean((cur_y - ref_p[: len(cur_y)]) ** 2))
        assert abs(brier.estimate - true_brier) > 0.01, (
            "importance weighting appeared to correct a P(Y|X) change, which "
            "would contradict the impossibility argument"
        )


class TestSuppression:
    def test_no_common_support_suppresses_every_metric(self, rng):
        """Windows six SDs apart. There is nothing to reweight across.

        This test is the reason the ESS check is computed on *unclipped*
        weights and paired with a separability check. The first version
        measured ESS after clipping, which caps precisely the large weights
        ESS exists to detect — it reported a healthy effective sample size
        here and returned a confident base-rate estimate of 0.534. A guardrail
        disabled by its own variance control is the project's recurring bug in
        a third costume.
        """
        ref_x, ref_p, ref_y = _population(4000, rng, mean=0.0)
        cur_x, _, _ = _population(4000, rng, mean=6.0)

        weights, diagnostics = fit_density_ratio(ref_x, cur_x, max_samples=3000)
        assert diagnostics["discriminator_auc"] > 0.95, (
            "precondition: the windows must be near-perfectly separable"
        )
        estimates = importance_weighted_estimates(
            ref_y, ref_p, weights, diagnostics,
            n_current=len(cur_x), min_ess_fraction=0.05,
        )
        assert all(e.status is EstimateStatus.SUPPRESSED for e in estimates)
        assert all(np.isnan(e.estimate) for e in estimates), (
            "a suppressed estimate must not carry a number; it would get "
            "plotted and believed"
        )
        # Both guardrails fire on this window — ESS is checked first, so that
        # is the reason reported. They are redundant here and independent in
        # general, which is the point of keeping both.
        assert all(
            e.detail["suppression_reason"] in {"ess_collapse", "no_common_support"}
            for e in estimates
        )
        assert diagnostics["ess_fraction"] < 0.05

    def test_ess_is_measured_before_clipping(self, rng):
        """Pins the fix directly, so a future 'tidy-up' that moves the ESS
        computation below the clip fails here rather than silently."""
        ref_x, _, _ = _population(4000, rng, mean=0.0)
        cur_x, _, _ = _population(4000, rng, mean=6.0)
        _, diagnostics = fit_density_ratio(
            ref_x, cur_x, max_samples=3000, clip_quantile=0.99
        )
        assert diagnostics["ess_fraction"] < 0.05, (
            f"ESS fraction {diagnostics['ess_fraction']:.3f} looks healthy on "
            f"disjoint windows — it is being measured after clipping"
        )

    def test_suppressed_estimates_still_report_their_ess(self, rng):
        ref_x, ref_p, ref_y = _population(4000, rng, mean=0.0)
        cur_x, _, _ = _population(4000, rng, mean=6.0)
        weights, diagnostics = fit_density_ratio(ref_x, cur_x, max_samples=3000)
        estimates = importance_weighted_estimates(
            ref_y, ref_p, weights, diagnostics, n_current=len(cur_x)
        )
        for estimate in estimates:
            assert estimate.effective_sample_size is not None
            assert estimate.is_usable is False


class TestDensityRatioMechanics:
    def test_predictions_are_out_of_fold(self, rng):
        """In-sample discriminator predictions produce weights that are
        confidently wrong in a way nothing downstream would catch. With no
        real shift, out-of-fold weights should stay near uniform."""
        ref_x, _, _ = _population(6000, rng, mean=0.0)
        cur_x, _, _ = _population(6000, rng, mean=0.0)
        weights, diagnostics = fit_density_ratio(ref_x, cur_x, max_samples=4000)
        used = weights[weights > 0]
        assert diagnostics["ess_fraction"] > 0.5, (
            "weights collapsed on identically distributed windows — the "
            "discriminator is memorising rather than generalising"
        )
        assert 0.5 < float(np.median(used)) < 2.0
        assert diagnostics["discriminator_auc"] < 0.55, (
            "the discriminator separated two identical distributions"
        )

    def test_diagnostics_report_the_clipped_fraction(self, rng):
        ref_x, _, _ = _population(4000, rng, mean=0.0)
        cur_x, _, _ = _population(4000, rng, mean=1.5)
        _, diagnostics = fit_density_ratio(
            ref_x, cur_x, max_samples=3000, clip_quantile=0.95
        )
        assert diagnostics["weight_clipped_fraction"] == pytest.approx(0.05, abs=0.02)
