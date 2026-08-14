"""Tests for the data-vs-concept separation module.

Two of these tests encode the impossibility result documented at the top of
`drift_core/concept.py` rather than testing a feature:

  - test_prediction_shift_alone_is_not_tagged_concept
  - test_importance_weighted_reference_reproduces_current_predictions

They exist to make the claim falsifiable and to fail loudly if someone later
"improves" the module by inferring concept drift from unsupervised prediction
movement. That inference is invalid for a fixed deterministic model and these
tests are the guardrail.
"""

import numpy as np
import pytest

from drift_core.concept import (
    confirm_concept_drift_with_labels,
    effective_sample_size_ratio,
    feature_relationship_drift,
    out_of_support_mass,
)
from drift_core.types import DriftKind, Severity, WindowSpec

WINDOW = WindowSpec(window_id="w1", n_samples=1000)


@pytest.fixture
def rng():
    return np.random.default_rng(23)


class TestImpossibilityGuardrails:
    def test_prediction_shift_alone_is_not_tagged_concept(self, rng):
        """Prediction drift must never be reported as concept drift.

        A fixed model fed shifted-but-well-handled inputs produces shifted
        outputs with P(Y|X) unchanged. Anything claiming otherwise is wrong.
        """
        from drift_core.prediction import detect_prediction_drift

        ref = rng.beta(2, 8, 2000)
        cur = rng.beta(8, 2, 2000)
        result = detect_prediction_drift(ref, cur, window=WINDOW)
        assert result.is_drifted is True
        assert result.kind is DriftKind.PREDICTION
        assert result.kind not in (
            DriftKind.CONCEPT_PROXY,
            DriftKind.CONCEPT_CONFIRMED,
        )

    def test_importance_weighted_reference_reproduces_current_predictions(self):
        """The arithmetic behind the impossibility claim.

        With a deterministic model, reweighting reference predictions by the
        true density ratio recovers the current prediction distribution. So
        "prediction drifted beyond what covariate shift explains" has no
        signal to detect. Pinned as a test so the claim is checkable, not
        just asserted in a docstring.
        """
        rng = np.random.default_rng(0)
        # Model: f(x) = sigmoid(x). Reference x ~ N(0,1), current x ~ N(1,1).
        ref_x = rng.normal(0, 1, 200_000)
        cur_x = rng.normal(1, 1, 200_000)
        f = lambda x: 1 / (1 + np.exp(-x))
        ref_pred, cur_pred = f(ref_x), f(cur_x)

        # True density ratio p_cur(x)/p_ref(x) for these two normals.
        w = np.exp(ref_x - 0.5)

        weighted_mean = np.sum(w * ref_pred) / np.sum(w)
        assert weighted_mean == pytest.approx(np.mean(cur_pred), abs=0.01)


class TestFeatureRelationshipDrift:
    def test_stable_relationships_no_drift(self, rng):
        n = 1500
        def make(seed):
            r = np.random.default_rng(seed)
            a = r.normal(0, 1, n)
            b = 2 * a + r.normal(0, 0.5, n)
            c = r.normal(0, 1, n)
            return np.column_stack([a, b, c])

        result = feature_relationship_drift(
            make(1), make(2), window=WINDOW,
            feature_names=["a", "b", "c"], random_state=0,
        )
        assert result.is_drifted is False
        assert result.kind is DriftKind.CONCEPT_PROXY

    def test_broken_relationship_detected(self, rng):
        n = 1500
        a_ref = rng.normal(0, 1, n)
        b_ref = 2 * a_ref + rng.normal(0, 0.3, n)
        c_ref = rng.normal(0, 1, n)
        ref = np.column_stack([a_ref, b_ref, c_ref])

        # Same marginals for b (mean 0, comparable spread) but the link to a
        # is severed — the dependence structure changed, not the margins.
        a_cur = rng.normal(0, 1, n)
        b_cur = rng.normal(0, float(np.std(b_ref)), n)
        c_cur = rng.normal(0, 1, n)
        cur = np.column_stack([a_cur, b_cur, c_cur])

        result = feature_relationship_drift(
            ref, cur, window=WINDOW,
            feature_names=["a", "b", "c"], random_state=0,
        )
        assert result.is_drifted is True
        assert result.severity is Severity.ALERT

    def test_reports_which_relationship_broke(self, rng):
        n = 1500
        a_ref = rng.normal(0, 1, n)
        b_ref = 2 * a_ref + rng.normal(0, 0.3, n)
        ref = np.column_stack([a_ref, b_ref])
        a_cur = rng.normal(0, 1, n)
        b_cur = rng.normal(0, float(np.std(b_ref)), n)
        cur = np.column_stack([a_cur, b_cur])

        result = feature_relationship_drift(
            ref, cur, window=WINDOW, feature_names=["a", "b"], random_state=0
        )
        per_feature = result.extra["per_feature_relative_error_increase"]
        assert per_feature
        assert max(per_feature.values()) > 0.35

    def test_feature_count_mismatch_raises(self, rng):
        with pytest.raises(ValueError, match="same feature count"):
            feature_relationship_drift(
                rng.normal(0, 1, (100, 3)),
                rng.normal(0, 1, (100, 4)),
                window=WINDOW,
                feature_names=["a", "b", "c"],
            )


class TestOutOfSupportMass:
    def test_same_distribution_near_baseline(self, rng):
        ref = rng.normal(0, 1, (3000, 4))
        cur = rng.normal(0, 1, (3000, 4))
        result = out_of_support_mass(ref, cur, window=WINDOW)
        assert result.is_drifted is False
        assert result.kind is DriftKind.CONCEPT_PROXY

    def test_baseline_subtraction_is_applied(self, rng):
        # ~2% per feature x 4 features means the reference itself sits near
        # 8% outside its own 1-99 box. Without baseline subtraction that
        # alone would trip the 5% watch threshold on a no-drift window.
        ref = rng.normal(0, 1, (3000, 4))
        cur = rng.normal(0, 1, (3000, 4))
        result = out_of_support_mass(ref, cur, window=WINDOW)
        assert result.extra["reference_baseline_rate"] > 0.03
        assert result.statistic < result.extra["raw_current_rate"]

    def test_extrapolation_region_detected(self, rng):
        ref = rng.normal(0, 1, (2000, 3))
        cur = rng.normal(5, 1, (2000, 3))
        result = out_of_support_mass(ref, cur, window=WINDOW)
        assert result.is_drifted is True
        assert result.severity is Severity.ALERT
        assert result.statistic > 0.5


class TestEffectiveSampleSize:
    def test_uniform_weights_give_one(self):
        assert effective_sample_size_ratio(np.ones(1000)) == pytest.approx(1.0)

    def test_degenerate_weights_collapse_toward_zero(self):
        w = np.zeros(1000)
        w[0] = 1.0
        assert effective_sample_size_ratio(w) == pytest.approx(0.001, abs=1e-6)

    def test_moderate_spread_between_zero_and_one(self):
        rng = np.random.default_rng(3)
        ess = effective_sample_size_ratio(rng.lognormal(0, 1, 5000))
        assert 0.0 < ess < 1.0

    def test_monotonic_in_weight_dispersion(self):
        # More dispersed weights => lower ESS. This is the property that makes
        # ESS usable as a suppression guardrail for component 2.
        rng = np.random.default_rng(4)
        ratios = [
            effective_sample_size_ratio(rng.lognormal(0, sigma, 5000))
            for sigma in (0.1, 0.5, 1.0, 2.0)
        ]
        assert ratios == sorted(ratios, reverse=True)

    def test_empty_and_zero_weights_return_zero(self):
        assert effective_sample_size_ratio(np.array([])) == 0.0
        assert effective_sample_size_ratio(np.zeros(100)) == 0.0


class TestLabelConfirmedConceptDrift:
    def test_stable_performance_no_drift(self, rng):
        labels = rng.integers(0, 2, 2000)
        scores = np.where(labels == 1, rng.beta(6, 2, 2000), rng.beta(2, 6, 2000))
        labels2 = rng.integers(0, 2, 2000)
        scores2 = np.where(labels2 == 1, rng.beta(6, 2, 2000), rng.beta(2, 6, 2000))

        result = confirm_concept_drift_with_labels(
            scores, labels, scores2, labels2, window=WINDOW
        )
        assert result.is_drifted is False
        assert result.kind is DriftKind.CONCEPT_CONFIRMED

    def test_degraded_performance_detected(self, rng):
        labels = rng.integers(0, 2, 2000)
        scores = np.where(labels == 1, rng.beta(6, 2, 2000), rng.beta(2, 6, 2000))
        # Current window: scores now nearly independent of the outcome.
        labels2 = rng.integers(0, 2, 2000)
        scores2 = rng.beta(3, 3, 2000)

        result = confirm_concept_drift_with_labels(
            scores, labels, scores2, labels2, window=WINDOW
        )
        assert result.is_drifted is True
        assert result.severity is Severity.ALERT
        assert result.extra["reference_auc"] > result.extra["current_auc"]

    def test_is_the_only_confirmed_kind_produced(self, rng):
        """Nothing unsupervised may emit CONCEPT_CONFIRMED."""
        unsupervised = [
            feature_relationship_drift(
                rng.normal(0, 1, (400, 3)), rng.normal(0, 1, (400, 3)),
                window=WINDOW, feature_names=["a", "b", "c"],
            ),
            out_of_support_mass(
                rng.normal(0, 1, (400, 3)), rng.normal(0, 1, (400, 3)),
                window=WINDOW,
            ),
        ]
        assert all(r.kind is DriftKind.CONCEPT_PROXY for r in unsupervised)
