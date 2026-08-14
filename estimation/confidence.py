"""Confidence-based label-free performance estimation.

The family of methods that infers "how is the model doing" from the model's
own output distribution, with no labels for the current window. Reference
labels are permitted and used — those are historical and available in any real
deployment. It is the *current* window that has no labels, which is the actual
production constraint.

THE STRUCTURAL LIMITATION, STATED UP FRONT
==========================================

Every estimator here is a functional of the predicted probabilities alone.
That buys the honesty of the method and also caps what it can ever detect:

- If the model's **ranking** degrades, confidence still moves, because the
  confidence distribution changes shape. These methods can see that.
- If the model's **probabilities** become systematically wrong while the
  ranking holds, the estimators inherit the error exactly. They are reading
  the broken instrument to decide whether the instrument is broken.

The second case is not hypothetical on this project's data — it is what
actually happened (Brier +17%, calibration gap 27x, AUC flat). So these
estimators are expected to fail here, and the measurement of *how badly* is
the deliverable. See `backtest/scoring.py` for the scoring and the findings
log for the result.
"""

from __future__ import annotations

import numpy as np

from estimation.types import EstimateStatus, PerformanceEstimate

MIN_SAMPLES = 30


def _clean(probabilities) -> np.ndarray:
    arr = np.asarray(probabilities, dtype=float)
    return arr[np.isfinite(arr)]


def _binary_confidence(probabilities: np.ndarray) -> np.ndarray:
    """Confidence in the predicted class: max(p, 1-p)."""
    return np.maximum(probabilities, 1.0 - probabilities)


def estimate_base_rate(
    probabilities, *, window_id: str = "", threshold: float = 0.5
) -> PerformanceEstimate:
    """Estimated positive rate = mean predicted probability.

    Exact if the model is calibrated, and wrong by exactly the calibration
    error otherwise. On a credit model this is the estimate that matters most
    — it is the portfolio loss rate that pricing and approval cutoffs consume
    — and it is also the one with no defence against calibration drift
    whatsoever.
    """
    del threshold
    probs = _clean(probabilities)
    if len(probs) < MIN_SAMPLES:
        return PerformanceEstimate(
            metric="base_rate", method="average_confidence", window_id=window_id,
            estimate=float("nan"), status=EstimateStatus.INSUFFICIENT_DATA,
            n_current=len(probs),
        )
    return PerformanceEstimate(
        metric="base_rate", method="average_confidence", window_id=window_id,
        estimate=float(np.mean(probs)), n_current=len(probs),
    )


def estimate_brier(probabilities, *, window_id: str = "") -> PerformanceEstimate:
    """Expected Brier score under the model's own beliefs: mean p(1-p).

    If the model is calibrated then `E[(y - p)^2] = E[p(1-p)]`, so this is an
    unbiased label-free estimate of the Brier score. The derivation assumes
    calibration, which means the estimator is exactly blind to miscalibration
    — the one failure mode it would be most valuable to catch. That is not a
    fixable bug in the implementation; it is what the identity says.
    """
    probs = _clean(probabilities)
    if len(probs) < MIN_SAMPLES:
        return PerformanceEstimate(
            metric="brier", method="average_confidence", window_id=window_id,
            estimate=float("nan"), status=EstimateStatus.INSUFFICIENT_DATA,
            n_current=len(probs),
        )
    return PerformanceEstimate(
        metric="brier", method="average_confidence", window_id=window_id,
        estimate=float(np.mean(probs * (1.0 - probs))), n_current=len(probs),
    )


def estimate_accuracy_average_confidence(
    probabilities, *, window_id: str = ""
) -> PerformanceEstimate:
    """Estimated accuracy = mean confidence in the predicted class.

    The simplest member of the family, and the most obviously circular: a
    model that is confidently wrong reports high estimated accuracy.
    """
    probs = _clean(probabilities)
    if len(probs) < MIN_SAMPLES:
        return PerformanceEstimate(
            metric="accuracy", method="average_confidence", window_id=window_id,
            estimate=float("nan"), status=EstimateStatus.INSUFFICIENT_DATA,
            n_current=len(probs),
        )
    return PerformanceEstimate(
        metric="accuracy", method="average_confidence", window_id=window_id,
        estimate=float(np.mean(_binary_confidence(probs))), n_current=len(probs),
    )


def fit_atc_threshold(reference_probabilities, reference_labels) -> float:
    """Average Threshold Confidence (Garg et al., 2022), calibration step.

    Finds the confidence level `t` at which the fraction of reference points
    scoring above `t` equals the model's actual reference accuracy. The
    current window's accuracy is then estimated as the fraction of its points
    above the same `t`.

    Uses reference labels, which is legitimate — the constraint is that the
    *current* window is unlabelled, not that history is. ATC is a genuine
    improvement on average confidence because it does not require the raw
    confidence values to be calibrated, only that the confidence-to-accuracy
    *mapping* stays fixed. That is a weaker assumption. It is still an
    assumption about `P(Y|X)` holding, which is precisely what breaks on this
    data.
    """
    probs = np.asarray(reference_probabilities, dtype=float)
    labels = np.asarray(reference_labels, dtype=float)
    keep = np.isfinite(probs) & np.isfinite(labels)
    probs, labels = probs[keep], labels[keep]
    if len(probs) < MIN_SAMPLES:
        return float("nan")

    predicted = (probs >= 0.5).astype(float)
    accuracy = float(np.mean(predicted == labels))
    confidence = _binary_confidence(probs)
    # The threshold whose exceedance rate matches reference accuracy.
    return float(np.quantile(confidence, 1.0 - accuracy))


def estimate_accuracy_atc(
    probabilities, threshold: float, *, window_id: str = ""
) -> PerformanceEstimate:
    probs = _clean(probabilities)
    if len(probs) < MIN_SAMPLES or not np.isfinite(threshold):
        return PerformanceEstimate(
            metric="accuracy", method="atc", window_id=window_id,
            estimate=float("nan"), status=EstimateStatus.INSUFFICIENT_DATA,
            n_current=len(probs),
        )
    estimate = float(np.mean(_binary_confidence(probs) >= threshold))
    return PerformanceEstimate(
        metric="accuracy", method="atc", window_id=window_id,
        estimate=estimate, n_current=len(probs),
        detail={"threshold": float(threshold)},
    )
