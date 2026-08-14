"""Tests for the monitored model.

The model itself does not need to be good, so nothing here asserts a
performance level. What is tested is the set of properties the rest of the
project's claims depend on: that the model is frozen, that it never saw the
future, and that its metrics fail loudly rather than plausibly.
"""

import numpy as np
import pandas as pd
import pytest

from model.baseline import evaluate, train_baseline

NUMERIC = ["a", "b"]
CATEGORICAL = ["g"]


def _frame(n, rng, shift=0.0, categories=("x", "y")):
    return pd.DataFrame(
        {
            "a": rng.normal(shift, 1, n),
            "b": rng.normal(0, 1, n),
            "g": rng.choice(list(categories), n),
        }
    )


def _labels(frame, rng):
    logit = 0.9 * frame["a"] - 0.4 * frame["b"]
    p = 1 / (1 + np.exp(-logit))
    return (rng.random(len(frame)) < p).astype(float)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def trained(rng):
    train, holdout = _frame(3000, rng), _frame(1500, rng)
    return train_baseline(
        train, holdout, _labels(train, rng), _labels(holdout, rng),
        numeric_features=NUMERIC, categorical_features=CATEGORICAL,
        trained_on="2013H1",
    )


class TestModelIsFrozen:
    def test_predictions_are_deterministic(self, trained, rng):
        """`Y_hat = f(X)` for a fixed `f` is the premise of the project's
        impossibility argument. If prediction were stochastic, prediction
        drift would no longer be a pure function of P(X) and the argument
        would not hold."""
        frame = _frame(500, rng)
        first = trained.predict_proba(frame)
        second = trained.predict_proba(frame)
        np.testing.assert_array_equal(first, second)

    def test_scoring_does_not_change_later_predictions(self, trained, rng):
        """No hidden adaptation: scoring a shifted batch must not alter what
        the model says about the original one."""
        original = _frame(400, rng)
        before = trained.predict_proba(original)
        trained.predict_proba(_frame(4000, rng, shift=3.0))
        np.testing.assert_array_equal(before, trained.predict_proba(original))

    def test_no_online_update_surface_is_exposed(self, trained):
        for forbidden in ("partial_fit", "update", "fit"):
            assert not hasattr(trained, forbidden), (
                f"BaselineModel exposes {forbidden}; the model must stay frozen"
            )


class TestTrainingRefusesUnlabelledRows:
    def test_nan_label_in_training_window_raises(self, rng):
        """An unlabelled training row means the maturity cut was skipped or
        applied in the wrong order. Silently dropping those rows would train
        the model on a survivorship-selected subset — the exact bias the cut
        exists to remove, reintroduced one layer down."""
        train, holdout = _frame(500, rng), _frame(200, rng)
        labels = _labels(train, rng)
        labels[:5] = np.nan
        with pytest.raises(ValueError, match="no label"):
            train_baseline(
                train, holdout, labels, _labels(holdout, rng),
                numeric_features=NUMERIC, categorical_features=CATEGORICAL,
            )


class TestReferenceMetrics:
    def test_reference_metrics_come_from_the_holdout(self, trained):
        assert trained.reference_metrics["n"] == 1500
        assert 0.0 <= trained.reference_metrics["auc"] <= 1.0

    def test_model_learned_something(self, trained):
        # Not a quality bar — just enough to confirm the wiring works, since
        # a model at chance would make every downstream signal meaningless.
        assert trained.reference_metrics["auc"] > 0.6


class TestEvaluateFailsLoudly:
    def test_single_class_returns_nan_auc_not_half(self):
        """0.5 would read as "chance performance" and be indistinguishable in
        any plot from a genuinely broken model. A window with one class is an
        unanswerable question, not a bad answer."""
        metrics = evaluate([1.0] * 50, np.linspace(0, 1, 50))
        assert np.isnan(metrics["auc"])
        assert metrics["base_rate"] == 1.0

    def test_empty_window_returns_nan(self):
        metrics = evaluate([], [])
        assert np.isnan(metrics["auc"])
        assert metrics["n"] == 0.0

    def test_nan_labels_are_excluded_not_imputed(self):
        y = np.array([0.0, 1.0, np.nan, 1.0, 0.0])
        scores = np.array([0.1, 0.9, 0.5, 0.8, 0.2])
        assert evaluate(y, scores)["n"] == 4.0

    def test_reports_calibration_and_base_rate_alongside_auc(self):
        y = np.array([0.0, 1.0] * 50)
        metrics = evaluate(y, np.linspace(0, 1, 100))
        assert set(metrics) >= {"auc", "brier", "base_rate", "mean_predicted"}


class TestUnseenCategories:
    def test_unseen_category_does_not_crash(self, trained, rng):
        """Categorical levels appear and vanish over time in real data. A
        monitor that dies on an unseen level stops monitoring precisely when
        something changed."""
        frame = _frame(300, rng, categories=("x", "y", "brand_new_level"))
        scores = trained.predict_proba(frame)
        assert len(scores) == 300
        assert np.isfinite(scores).all()
