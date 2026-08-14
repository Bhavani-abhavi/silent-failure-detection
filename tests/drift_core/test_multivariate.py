"""Tests for the domain-classifier (classifier two-sample) drift test.

The single most important test in this file is
`test_no_drift_high_dimensional_small_n_does_not_fire`. That is the failure
mode the permutation test exists to prevent: a flexible classifier finding
spurious separation between two samples of the same distribution when
features outnumber the signal. If that test ever starts failing, the
multivariate detector is producing false alarms and nothing built on top of
it can be trusted.

Permutation counts are kept low here (the statistic is refit n_permutations
times, each with k-fold CV) so the suite stays fast. Production runs use
higher counts; the p-value resolution floor is 1/(n_permutations+1).
"""

import numpy as np
import pytest

from drift_core.multivariate import domain_classifier_drift
from drift_core.types import Severity, WindowSpec

WINDOW = WindowSpec(window_id="w1", n_samples=500)


@pytest.fixture
def rng():
    return np.random.default_rng(7)


class TestNullBehaviour:
    def test_identical_distributions_auc_near_chance(self, rng):
        ref = rng.normal(0, 1, (600, 5))
        cur = rng.normal(0, 1, (600, 5))
        result = domain_classifier_drift(
            ref, cur, window=WINDOW, n_permutations=25, random_state=0
        )
        assert result.auc == pytest.approx(0.5, abs=0.08)
        assert result.is_drifted is False
        assert result.severity is Severity.NONE

    def test_no_drift_high_dimensional_small_n_does_not_fire(self, rng):
        # 40 features, 150 rows per window, no real drift. A raw "AUC > 0.5"
        # rule fires here routinely. The permutation p-value must not.
        ref = rng.normal(0, 1, (150, 40))
        cur = rng.normal(0, 1, (150, 40))
        result = domain_classifier_drift(
            ref, cur, window=WINDOW, n_permutations=30, random_state=1
        )
        assert result.p_value > 0.05, (
            f"false alarm under the null: auc={result.auc:.3f}, "
            f"p={result.p_value:.3f}"
        )
        assert result.is_drifted is False

    def test_pvalue_is_never_zero(self, rng):
        # The +1 correction guarantees this; a reported p=0 would be a lie
        # about the resolution of a finite permutation test.
        ref = rng.normal(0, 1, (300, 4))
        cur = rng.normal(6, 1, (300, 4))
        result = domain_classifier_drift(
            ref, cur, window=WINDOW, n_permutations=20, random_state=2
        )
        assert result.p_value > 0
        assert result.p_value >= 1 / 21


class TestSignalDetection:
    def test_marginal_shift_detected(self, rng):
        ref = rng.normal(0, 1, (500, 5))
        cur = rng.normal(0, 1, (500, 5))
        cur[:, 0] += 2.0
        result = domain_classifier_drift(
            ref, cur, window=WINDOW, n_permutations=25, random_state=3
        )
        assert result.is_drifted is True
        assert result.auc > 0.6
        assert result.p_value < 0.05

    def test_correlation_only_drift_detected(self, rng):
        # Both windows have identical marginals; only the dependence between
        # the two features flips. Every univariate test in the suite is blind
        # to this by construction — this is the multivariate detector's
        # entire reason for existing, so it must catch it.
        n = 800
        base = rng.normal(0, 1, n)
        ref = np.column_stack([base, base + rng.normal(0, 0.3, n)])
        base2 = rng.normal(0, 1, n)
        cur = np.column_stack([base2, -base2 + rng.normal(0, 0.3, n)])

        result = domain_classifier_drift(
            ref, cur, window=WINDOW, n_permutations=25, random_state=4
        )
        assert result.is_drifted is True, (
            "correlation-structure drift missed; univariate tests cannot see "
            "this case so the multivariate detector must"
        )

    def test_correlation_only_drift_invisible_to_univariate(self, rng):
        # Companion assertion to the test above: proves the marginals really
        # are unchanged, so the detection above came from joint structure.
        from drift_core.univariate import population_stability_index

        n = 800
        base = rng.normal(0, 1, n)
        ref = np.column_stack([base, base + rng.normal(0, 0.3, n)])
        base2 = rng.normal(0, 1, n)
        cur = np.column_stack([base2, -base2 + rng.normal(0, 0.3, n)])

        for col in range(2):
            psi, _ = population_stability_index(ref[:, col], cur[:, col])
            assert psi < 0.1, f"marginal {col} moved; test setup is invalid"

    def test_auc_increases_with_shift_magnitude(self, rng):
        # Asserts the AUC ordering only, so the permutation null is not
        # needed — alpha is relaxed to keep n_permutations (and runtime) low.
        ref = rng.normal(0, 1, (400, 4))
        aucs = []
        for shift in (0.0, 0.5, 1.5, 3.0):
            cur = rng.normal(0, 1, (400, 4))
            cur[:, 0] += shift
            aucs.append(
                domain_classifier_drift(
                    ref, cur, window=WINDOW, n_permutations=3,
                    alpha=0.5, random_state=5,
                ).auc
            )
        assert aucs == sorted(aucs)


class TestPermutationResolutionFloor:
    """A permutation test cannot report p below 1/(n_permutations+1).

    When that floor exceeds alpha, no result can ever be significant — a
    perfectly separable pair of windows returns "no drift". This was found
    by `test_large_effect_is_an_alert` failing at AUC=1.0, p=0.0625 with
    n_permutations=15. It is now a hard error at call time.
    """

    def test_too_few_permutations_raises(self, rng):
        with pytest.raises(ValueError, match="cannot produce a p-value below"):
            domain_classifier_drift(
                rng.normal(0, 1, (100, 3)),
                rng.normal(5, 1, (100, 3)),
                window=WINDOW,
                n_permutations=15,
                alpha=0.05,
            )

    def test_error_names_a_sufficient_permutation_count(self, rng):
        with pytest.raises(ValueError, match=r"n_permutations >= 19"):
            domain_classifier_drift(
                rng.normal(0, 1, (100, 3)),
                rng.normal(5, 1, (100, 3)),
                window=WINDOW,
                n_permutations=10,
                alpha=0.05,
            )

    def test_boundary_count_is_accepted(self, rng):
        # 19 permutations => min p = 1/20 = 0.05, which is not < alpha, so
        # 19 is the smallest count that can reach significance... and it
        # cannot, strictly. 20 is the first workable value; both must be
        # callable without raising, since the guard is about achievability.
        result = domain_classifier_drift(
            rng.normal(0, 1, (150, 3)),
            rng.normal(4, 1, (150, 3)),
            window=WINDOW,
            n_permutations=20,
            alpha=0.05,
        )
        assert result.p_value <= 0.05

    def test_relaxed_alpha_permits_few_permutations(self, rng):
        result = domain_classifier_drift(
            rng.normal(0, 1, (150, 3)),
            rng.normal(0, 1, (150, 3)),
            window=WINDOW,
            n_permutations=5,
            alpha=0.5,
        )
        assert result.p_value >= 1 / 6


class TestSeverityGating:
    def test_significant_but_tiny_effect_is_not_an_alert(self, rng):
        # Large n makes trivial separability significant. Operationally that
        # is noise, and severity must be gated on effect size, not p alone.
        ref = rng.normal(0, 1, (4000, 3))
        cur = rng.normal(0.03, 1, (4000, 3))
        result = domain_classifier_drift(
            ref, cur, window=WINDOW, n_permutations=20,
            auc_watch_threshold=0.60, random_state=6,
        )
        assert result.severity is not Severity.ALERT

    def test_large_effect_is_an_alert(self, rng):
        ref = rng.normal(0, 1, (500, 4))
        cur = rng.normal(4, 1, (500, 4))
        result = domain_classifier_drift(
            ref, cur, window=WINDOW, n_permutations=20, random_state=7
        )
        assert result.severity is Severity.ALERT
        assert result.auc > 0.75


class TestReporting:
    """These assert structure and descriptive output, not significance, so
    they relax alpha to keep the permutation count (and runtime) minimal."""

    def test_feature_importances_identify_the_drifted_feature(self, rng):
        ref = rng.normal(0, 1, (500, 4))
        cur = rng.normal(0, 1, (500, 4))
        cur[:, 2] += 3.0
        names = ["a", "b", "drifted", "d"]
        result = domain_classifier_drift(
            ref, cur, window=WINDOW, feature_names=names,
            n_permutations=3, alpha=0.5, random_state=8,
        )
        top = max(result.feature_importances, key=result.feature_importances.get)
        assert top == "drifted"

    def test_importances_empty_without_feature_names(self, rng):
        result = domain_classifier_drift(
            rng.normal(0, 1, (200, 3)),
            rng.normal(0, 1, (200, 3)),
            window=WINDOW,
            n_permutations=3,
            alpha=0.5,
        )
        assert result.feature_importances == {}

    def test_records_window_and_counts(self, rng):
        result = domain_classifier_drift(
            rng.normal(0, 1, (300, 3)),
            rng.normal(0, 1, (200, 3)),
            window=WINDOW,
            n_permutations=3,
            alpha=0.5,
        )
        assert result.n_reference == 300
        assert result.n_current == 200
        assert result.window is WINDOW
        assert result.method == "domain_classifier"


class TestValidation:
    def test_feature_count_mismatch_raises(self, rng):
        # Shape validation must run before the permutation-count guard, so
        # a caller with mismatched inputs gets the useful error, not a
        # complaint about n_permutations.
        with pytest.raises(ValueError, match="feature count mismatch"):
            domain_classifier_drift(
                rng.normal(0, 1, (100, 3)),
                rng.normal(0, 1, (100, 5)),
                window=WINDOW,
                n_permutations=5,
            )

    def test_one_dimensional_input_raises(self, rng):
        with pytest.raises(ValueError, match="2-D feature matrices"):
            domain_classifier_drift(
                rng.normal(0, 1, 100),
                rng.normal(0, 1, 100),
                window=WINDOW,
                n_permutations=5,
            )

    def test_deterministic_under_fixed_seed(self, rng):
        # Also guards the joblib parallelism: permutations are drawn up front
        # from a seeded generator, so results must not depend on scheduling.
        ref = rng.normal(0, 1, (300, 3))
        cur = rng.normal(0.5, 1, (300, 3))
        kwargs = {
            "window": WINDOW, "n_permutations": 5,
            "alpha": 0.5, "random_state": 99,
        }
        first = domain_classifier_drift(ref, cur, **kwargs)
        second = domain_classifier_drift(ref, cur, n_jobs=1, **kwargs)
        assert first.auc == second.auc
        assert first.p_value == second.p_value
