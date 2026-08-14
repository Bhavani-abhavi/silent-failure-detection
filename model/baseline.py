"""The monitored model.

This is deliberately the least interesting module in the repo. It exists to
be monitored, not to be good — a strong model and a weak one degrade the same
way, and tuning it would only make the degradation harder to see.

Two properties matter and both are enforced here:

FIXED AND DETERMINISTIC. Once trained, the model never updates. The project's
central impossibility argument depends on it: `Y_hat = f(X)` for a fixed `f`
means every observable is a function of `P(X)` alone. A model that retrained
on incoming data would break that argument and the honesty claim built on it.
`predict_proba` is pure; there is no `partial_fit`, no drift adaptation, no
threshold that moves.

TRAINED ON A TIME PREFIX. Never a random split. A random split over temporal
data puts 2016 loans in the training set for a model that is then "monitored"
across 2016, and every drift measurement downstream becomes meaningless while
the metrics look excellent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder


@dataclass(frozen=True)
class BaselineModel:
    """A trained, frozen classifier plus the metadata needed to monitor it."""

    estimator: HistGradientBoostingClassifier
    encoder: OrdinalEncoder | None
    numeric_features: list[str]
    categorical_features: list[str]
    trained_on: str
    """Opaque label for the training window, carried so that any report can
    state what the model actually saw."""

    reference_metrics: dict[str, float] = field(default_factory=dict)
    """Performance on a held-out *later* slice of the training era. This is
    the "healthy" baseline every monitored window is compared against. It must
    come from a holdout the model never saw, or the reference is optimistic
    and every later window looks like degradation by comparison."""

    @property
    def feature_names(self) -> list[str]:
        return list(self.numeric_features) + list(self.categorical_features)

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = frame[self.numeric_features].to_numpy(dtype=float)
        if not self.categorical_features or self.encoder is None:
            return numeric
        categorical = self.encoder.transform(
            frame[self.categorical_features].astype(object)
        )
        return np.hstack([numeric, categorical])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """P(default) for each row. Pure — no state is updated."""
        return self.estimator.predict_proba(self._matrix(frame))[:, 1]


def _categorical_mask(n_numeric: int, n_categorical: int) -> np.ndarray:
    mask = np.zeros(n_numeric + n_categorical, dtype=bool)
    mask[n_numeric:] = True
    return mask


def evaluate(y_true, y_score) -> dict[str, float]:
    """Discrimination, calibration, and base rate — reported together.

    AUC alone is the wrong summary for a monitoring project. A model can hold
    its ranking perfectly while its probabilities drift far from reality, and
    a model can look stable on AUC purely because the base rate moved under
    it. Reporting all three makes it possible to say *which* kind of
    degradation happened, which is the difference between a finding and a
    number.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    keep = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[keep], y_score[keep]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        # One class present: AUC is undefined, not 0.5. Returning 0.5 here
        # would read as "chance performance" and be indistinguishable from a
        # genuinely broken model in any downstream plot.
        return {
            "n": float(len(y_true)),
            "auc": float("nan"),
            "brier": float("nan"),
            "base_rate": float(y_true.mean()) if len(y_true) else float("nan"),
            "mean_predicted": float(y_score.mean()) if len(y_score) else float("nan"),
            "accuracy": float("nan"),
            "positive_prediction_rate": (
                float(np.mean(y_score >= 0.5)) if len(y_score) else float("nan")
            ),
        }

    return {
        "n": float(len(y_true)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "base_rate": float(y_true.mean()),
        "mean_predicted": float(y_score.mean()),
        # Reported because the label-free literature (ATC, difference-of-
        # confidences) estimates accuracy specifically. On an imbalanced
        # target it is close to degenerate — a model that never crosses 0.5
        # scores `1 - base_rate` by predicting the majority class every time —
        # so it is here to be compared against those estimators, not to be
        # used as a quality measure.
        "accuracy": float(np.mean((y_score >= 0.5) == (y_true >= 0.5))),
        "positive_prediction_rate": float(np.mean(y_score >= 0.5)),
    }


def train_baseline(
    train_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    train_labels,
    holdout_labels,
    *,
    numeric_features: list[str],
    categorical_features: list[str] | None = None,
    trained_on: str = "train",
    random_state: int = 0,
    max_iter: int = 200,
    learning_rate: float = 0.1,
    max_leaf_nodes: int = 31,
) -> BaselineModel:
    """Fit the frozen model on a time prefix and score it on a later holdout.

    `holdout_frame` must come *after* `train_frame` in time. It is not a
    random validation split: it plays the role of "the model is live and
    nothing has gone wrong yet", so it has to be drawn the same way every
    monitored window will be.

    Gradient boosting is used mainly because it consumes NaN natively. Roughly
    a third of the bureau features have real missingness that varies over
    time, and imputing it would inject a preprocessing artifact into exactly
    the signal the drift detectors are supposed to measure.
    """
    categorical_features = list(categorical_features or [])
    numeric_features = list(numeric_features)

    encoder: OrdinalEncoder | None = None
    if categorical_features:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
        )
        encoder.fit(train_frame[categorical_features].astype(object))

    estimator = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        categorical_features=(
            _categorical_mask(len(numeric_features), len(categorical_features))
            if categorical_features
            else None
        ),
        random_state=random_state,
        early_stopping=False,
    )

    model = BaselineModel(
        estimator=estimator,
        encoder=encoder,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        trained_on=trained_on,
    )

    y_train = np.asarray(train_labels, dtype=float)
    keep = np.isfinite(y_train)
    if not keep.all():
        # Unlabelled rows in the training window mean the maturity cut was not
        # applied, or was applied after splitting. Either way the model would
        # be trained on a survivorship-selected subset.
        raise ValueError(
            f"{(~keep).sum()} of {len(y_train)} training rows have no label. "
            f"Apply the matured-vintage cut before training."
        )

    estimator.fit(model._matrix(train_frame), y_train.astype(int))

    reference_metrics = evaluate(holdout_labels, model.predict_proba(holdout_frame))
    return BaselineModel(
        estimator=estimator,
        encoder=encoder,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        trained_on=trained_on,
        reference_metrics=reference_metrics,
    )
