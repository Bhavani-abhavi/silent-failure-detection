"""Importance-weighted performance estimation.

The idea: the reference window has labels, the current window does not. If the
only thing that changed is `P(X)`, then reweighting the labelled reference
points by the density ratio `w(x) = P_cur(x) / P_ref(x)` turns reference
performance into an unbiased estimate of current performance. No current
labels required.

WHAT THIS BUYS, AND WHY IT IS THE SHARPER EXPERIMENT
====================================================

Importance weighting is *correct* under covariate shift — `P(X)` moves,
`P(Y|X)` holds. Confidence-based estimation makes no such correction. So
running both separates two hypotheses that otherwise stay tangled:

- if plain confidence estimation fails and importance weighting fixes it,
  the degradation was covariate shift and the model is fine;
- if importance weighting fails *too*, the failure is in `P(Y|X)` and no
  amount of reweighting the inputs can reach it.

The second outcome is a much stronger claim than "our estimator was
inaccurate", because it rules out the explanation everyone reaches for first.

HOW IT FAILS, WHICH IT WILL
===========================

Density ratios are estimated, not known. When the current window moves into
regions the reference barely covered, a few reference points get enormous
weights and the effective sample size collapses — the estimate is then
computed from a handful of rows while looking like it used all of them. That
is why `effective_sample_size` is returned with every estimate and why the
estimator suppresses itself below `min_ess_fraction` rather than reporting a
confident wrong number.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from estimation.types import EstimateStatus, PerformanceEstimate

MIN_SAMPLES = 30


def effective_sample_size(weights) -> float:
    """Kish effective sample size: `(sum w)^2 / sum(w^2)`.

    Equals n when weights are uniform and collapses toward 1 as a single
    point dominates. This is the number that should be quoted as the sample
    size of a weighted estimate, not the row count.
    """
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w) & (w >= 0)]
    total = w.sum()
    if total <= 0:
        return 0.0
    return float(total**2 / np.sum(w**2))


def fit_density_ratio(
    reference_matrix,
    current_matrix,
    *,
    n_splits: int = 3,
    max_samples: int = 20000,
    clip_quantile: float = 0.99,
    random_state: int = 0,
    categorical_mask=None,
) -> tuple[np.ndarray, dict]:
    """Estimate `P_cur(x)/P_ref(x)` at each reference point.

    A discriminator is trained to tell the two windows apart; its odds convert
    to the density ratio via
    `P_cur(x)/P_ref(x) = odds(current | x) * n_ref / n_cur`.

    Predictions on reference points are **out-of-fold**. An in-sample
    discriminator memorises individual reference rows, drives their predicted
    odds toward zero, and produces weights that are confidently wrong in a way
    that no downstream check would catch — the estimate would just be silently
    dominated by whichever rows the model happened to misfit.

    Returns `(weights, diagnostics)`; weights align with `reference_matrix`.
    """
    rng = np.random.default_rng(random_state)
    ref = np.asarray(reference_matrix, dtype=float)
    cur = np.asarray(current_matrix, dtype=float)

    n_ref_full, n_cur_full = len(ref), len(cur)
    ref_index = np.arange(n_ref_full)
    if n_ref_full > max_samples:
        ref_index = rng.choice(n_ref_full, size=max_samples, replace=False)
    cur_sample = cur
    if n_cur_full > max_samples:
        cur_sample = cur[rng.choice(n_cur_full, size=max_samples, replace=False)]

    ref_sample = ref[ref_index]
    X = np.vstack([ref_sample, cur_sample])
    y = np.concatenate([np.zeros(len(ref_sample)), np.ones(len(cur_sample))])

    oof = np.zeros(len(X))
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    for train_idx, test_idx in splitter.split(X, y):
        clf = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.1, max_leaf_nodes=31,
            categorical_features=categorical_mask,
            random_state=random_state, early_stopping=False,
        )
        clf.fit(X[train_idx], y[train_idx])
        oof[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

    p_ref_points = np.clip(oof[: len(ref_sample)], 1e-6, 1 - 1e-6)
    odds = p_ref_points / (1.0 - p_ref_points)
    raw_ratio = odds * (len(ref_sample) / len(cur_sample))

    # ESS IS MEASURED BEFORE CLIPPING, DELIBERATELY.
    #
    # Clipping caps the largest weights, which is exactly the thing ESS is
    # looking for. Computing ESS after clipping therefore reports a healthy
    # effective sample size for a window with almost no support overlap — the
    # guardrail is disabled by the variance control that was supposed to sit
    # underneath it. Found by a test that expected suppression on windows six
    # standard deviations apart and got a confident estimate instead.
    ess = effective_sample_size(raw_ratio)

    cap = (
        float(np.quantile(raw_ratio, clip_quantile))
        if clip_quantile < 1.0
        else np.inf
    )
    clipped = raw_ratio > cap
    ratio = np.minimum(raw_ratio, cap)

    weights = np.zeros(n_ref_full)
    weights[ref_index] = ratio

    # Separability is the direct test of the assumption importance weighting
    # rests on. Common support is required; an AUC near 1 says a classifier
    # can tell every reference row from every current row, so the density
    # ratio is extrapolation rather than reweighting. This catches support
    # failure even where the weight distribution happens to look benign.
    discriminator_auc = float(roc_auc_score(y, oof))

    diagnostics = {
        "n_reference_used": int(len(ref_sample)),
        "n_current_used": int(len(cur_sample)),
        "effective_sample_size": ess,
        "ess_fraction": float(ess / max(len(ref_sample), 1)),
        "weight_max": float(raw_ratio.max()),
        "weight_clipped_fraction": float(clipped.mean()),
        "discriminator_auc": discriminator_auc,
    }
    return weights, diagnostics


def _weighted_mean(values, weights) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not keep.any():
        return float("nan")
    return float(np.sum(v[keep] * w[keep]) / np.sum(w[keep]))


def importance_weighted_estimates(
    reference_labels,
    reference_probabilities,
    weights,
    diagnostics: dict,
    *,
    window_id: str = "",
    n_current: int = 0,
    min_ess_fraction: float = 0.05,
    max_discriminator_auc: float = 0.95,
) -> list[PerformanceEstimate]:
    """Reweighted reference performance as an estimate of current performance.

    Suppresses every metric on either of two independent failures:

    - **ESS collapse** (`ess_fraction < min_ess_fraction`): the reweighted
      reference is effectively a handful of rows wearing the sample size of
      thousands.
    - **Support failure** (`discriminator_auc > max_discriminator_auc`): the
      two windows are near-perfectly separable, so there is no common support
      to reweight across and the ratio is extrapolating.

    Two checks rather than one because they fail independently — the ESS
    number can look healthy on a window with no overlap once weights are
    clipped, which is how the first version of this passed a test it should
    have failed.
    """
    y = np.asarray(reference_labels, dtype=float)
    p = np.asarray(reference_probabilities, dtype=float)
    w = np.asarray(weights, dtype=float)

    ess_fraction = diagnostics.get("ess_fraction", 0.0)
    ess = diagnostics.get("effective_sample_size", 0.0)
    discriminator_auc = diagnostics.get("discriminator_auc", 0.0)
    suppressed = (
        ess_fraction < min_ess_fraction
        or discriminator_auc > max_discriminator_auc
        or len(y) < MIN_SAMPLES
    )

    def _make(metric: str, value: float) -> PerformanceEstimate:
        return PerformanceEstimate(
            metric=metric,
            method="importance_weighted",
            window_id=window_id,
            estimate=float("nan") if suppressed else value,
            status=(
                EstimateStatus.SUPPRESSED if suppressed else EstimateStatus.OK
            ),
            effective_sample_size=float(ess),
            n_current=n_current,
            detail=dict(
                diagnostics,
                min_ess_fraction=min_ess_fraction,
                max_discriminator_auc=max_discriminator_auc,
                suppression_reason=(
                    ""
                    if not suppressed
                    else "ess_collapse"
                    if ess_fraction < min_ess_fraction
                    else "no_common_support"
                    if discriminator_auc > max_discriminator_auc
                    else "too_few_reference_rows"
                ),
            ),
        )

    correct = ((p >= 0.5).astype(float) == y).astype(float)
    return [
        _make("base_rate", _weighted_mean(y, w)),
        _make("brier", _weighted_mean((y - p) ** 2, w)),
        _make("accuracy", _weighted_mean(correct, w)),
    ]
