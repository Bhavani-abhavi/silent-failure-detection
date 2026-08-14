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

from dataclasses import replace

from drift_core.types import DriftKind, DriftResult, WindowSpec
from drift_core.univariate import (
    detect_ks_drift,
    detect_psi_drift,
    detect_wasserstein_drift,
)

PREDICTION_FEATURE_NAME = "__prediction__"

_METHODS = {
    "psi": detect_psi_drift,
    "ks": detect_ks_drift,
    "wasserstein": detect_wasserstein_drift,
}


def detect_prediction_drift(
    reference_scores,
    current_scores,
    *,
    window: WindowSpec,
    method: str = "psi",
    **kwargs,
) -> DriftResult:
    """Drift on the model's output scores.

    `method` is one of "psi", "ks", "wasserstein". PSI is the default because
    it is the one most commonly reported to model-risk functions, and its bins
    are fixed from the reference window, so the statistic stays comparable
    across monitoring windows rather than being re-derived each time.

    This delegates to the univariate detectors rather than reimplementing the
    threshold logic. It previously did reimplement it, and drifted out of
    sync: when the univariate detectors gained the requirement that both a
    significance gate and an effect-size gate be passed before firing, this
    function kept alerting on `p < alpha` alone for KS and on effect size
    alone for PSI and Wasserstein. Prediction drift is the signal closest to
    the headline claim, so it was the worst place to keep a weaker rule.
    """
    if method not in _METHODS:
        raise ValueError(
            f"unknown method {method!r}; expected one of {sorted(_METHODS)}"
        )
    result = _METHODS[method](
        reference_scores,
        current_scores,
        feature_name=PREDICTION_FEATURE_NAME,
        window=window,
        **kwargs,
    )
    # Same statistic, different operational meaning: input drift may be
    # harmless, output drift means the composition of the model's decisions
    # changed, which is what downstream processes actually feel.
    return replace(result, kind=DriftKind.PREDICTION, method=f"prediction_{method}")


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
