"""Prediction drift: has the model's output distribution moved?

Uses the same univariate machinery as feature drift, applied to the score
stream. Tagged DriftKind.PREDICTION rather than DATA because it means
something different operationally: input drift may be harmless, but output
drift means the model's decisions themselves have changed composition, which
is what downstream processes (case volumes, approval rates, alert queues)
actually feel.

Prediction drift alone does NOT establish concept drift. A model fed shifted
inputs it still handles correctly will produce shifted outputs. Separating
those cases is what concept.py does.
"""

from __future__ import annotations

import numpy as np

from drift_core.types import DriftKind, DriftResult, Severity, WindowSpec
from drift_core.univariate import (
    ks_test,
    population_stability_index,
    wasserstein,
)

PREDICTION_FEATURE_NAME = "__prediction__"


def detect_prediction_drift(
    reference_scores,
    current_scores,
    *,
    window: WindowSpec,
    method: str = "psi",
    bins: int = 10,
    watch_threshold: float = 0.1,
    alert_threshold: float = 0.25,
    alpha: float = 0.05,
) -> DriftResult:
    """Drift on the model's output scores.

    `method` is one of "psi", "ks", "wasserstein". PSI is the default because
    it is the one most commonly reported to model-risk functions, and its
    bins are fixed from the reference window (so the statistic is comparable
    across monitoring windows rather than being re-derived each time).
    """
    ref = np.asarray(reference_scores, dtype=float)
    cur = np.asarray(current_scores, dtype=float)

    p_value: float | None = None
    extra: dict = {}

    if method == "psi":
        statistic, extra = population_stability_index(ref, cur, bins=bins)
        drifted = statistic >= watch_threshold
        severity = (
            Severity.ALERT
            if statistic >= alert_threshold
            else Severity.WATCH
            if drifted
            else Severity.NONE
        )
        threshold = alert_threshold
    elif method == "ks":
        statistic, p_value = ks_test(ref, cur)
        drifted = p_value < alpha
        severity = Severity.ALERT if drifted else Severity.NONE
        threshold = alpha
    elif method == "wasserstein":
        statistic, extra = wasserstein(ref, cur)
        drifted = statistic >= watch_threshold
        severity = (
            Severity.ALERT
            if statistic >= alert_threshold
            else Severity.WATCH
            if drifted
            else Severity.NONE
        )
        threshold = alert_threshold
    else:
        raise ValueError(
            f"unknown method {method!r}; expected 'psi', 'ks', or 'wasserstein'"
        )

    return DriftResult(
        feature_name=PREDICTION_FEATURE_NAME,
        method=f"prediction_{method}",
        statistic=float(statistic),
        kind=DriftKind.PREDICTION,
        window=window,
        p_value=p_value,
        threshold=threshold,
        is_drifted=drifted,
        severity=severity,
        n_reference=len(ref),
        n_current=len(cur),
        extra=extra,
    )


def mean_score_shift(reference_scores, current_scores) -> dict:
    """Plain summary of how the score distribution moved. Not a test — this
    is the descriptive companion to the statistic above, for report tables
    where "PSI 0.31" means nothing to the reader but "average predicted risk
    rose from 0.12 to 0.19" does."""
    ref = np.asarray(reference_scores, dtype=float)
    cur = np.asarray(current_scores, dtype=float)
    return {
        "reference_mean": float(np.mean(ref)),
        "current_mean": float(np.mean(cur)),
        "absolute_change": float(np.mean(cur) - np.mean(ref)),
        "relative_change": (
            float((np.mean(cur) - np.mean(ref)) / np.mean(ref))
            if abs(np.mean(ref)) > 1e-12
            else float("nan")
        ),
        "reference_positive_rate": float(np.mean(ref >= 0.5)),
        "current_positive_rate": float(np.mean(cur >= 0.5)),
    }
