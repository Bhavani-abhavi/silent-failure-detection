"""Separating DATA drift from CONCEPT drift.

READ THIS BEFORE USING ANYTHING IN THIS MODULE.
================================================

There is an impossibility result at the heart of this project, and the honest
version of it is:

    For a fixed, deterministic model f, EVERY unsupervised signal you can
    compute is a function of P(X) alone.

Because Y_hat = f(X) with no randomness, the predicted-score distribution is
fully determined by the input distribution. Importance-weighting reference
predictions by w(x) = p_cur(x)/p_ref(x) reproduces the current prediction
distribution exactly (given common support). So "prediction drifted more than
covariate shift explains" is not a detectable event for a fixed model — it is
arithmetically zero. Any tutorial that presents prediction drift as evidence
of concept drift is wrong, and this module does not do that.

Concept drift is a change in P(Y|X). Y appears in that expression. Without
labels, it cannot be measured. Full stop.

What this module therefore provides is NOT a concept-drift detector. It is a
set of unsupervised RISK signals — conditions under which a concept change
would (a) be more likely to have occurred and (b) be more likely to hurt,
none of which observe P(Y|X):

  1. feature_relationship_drift
        Train a surrogate on the reference window to predict feature j from
        the other features. If that relationship degrades on the current
        window, the joint dependence structure among inputs has changed —
        not just the marginals. A shifting data-generating process is the
        most common upstream cause of a shifting P(Y|X), so this is the
        closest genuine unsupervised analogue to "the relationships moved".
        It is correlational evidence about the world, not a measurement of
        the model's error.

  2. out_of_support_mass
        Fraction of the current window falling outside the reference
        region the model was fit on. In that region the model extrapolates
        and its P(Y|X) was never validated. High novelty mass does not mean
        performance dropped; it means you have no evidence it didn't.

  3. effective_sample_size_ratio
        Degeneracy of the importance weights that the label-free performance
        estimator (component 2) depends on. When this collapses, the
        estimator's own assumptions have failed and its output should be
        suppressed rather than reported. This is a guardrail on our own
        method, not a drift signal about the model.

All three return CONCEPT_PROXY results. The only thing that produces a
CONCEPT_CONFIRMED result is `confirm_concept_drift_with_labels`, which runs
after delayed labels arrive and is the ground truth that the proxies get
scored against.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

from drift_core.types import DriftKind, DriftResult, Severity, WindowSpec

RELATIONSHIP_FEATURE_NAME = "__feature_relationships__"
SUPPORT_FEATURE_NAME = "__out_of_support__"
ESS_FEATURE_NAME = "__effective_sample_size__"


def feature_relationship_drift(
    reference,
    current,
    *,
    window: WindowSpec,
    feature_names: list[str],
    target_indices: list[int] | None = None,
    max_targets: int = 10,
    watch_threshold: float = 0.15,
    alert_threshold: float = 0.35,
    random_state: int = 0,
) -> DriftResult:
    """Has the dependence structure among inputs changed?

    For each probed feature j: fit reference-window surrogate g_j predicting
    x_j from x_-j, then compare its normalized error on the reference window
    (out-of-sample via a held-out half) against its error on the current
    window. The statistic is the median relative error increase across probed
    features.

    Normalizing by reference error matters: a feature that is simply noisy
    will have high absolute error in both windows and should not register.
    What registers is error *growing* relative to what that relationship
    used to support.

    Deliberately excluded: this cannot distinguish "relationship changed" from
    "current window entered a region where the relationship was always harder
    to fit". Out-of-support mass is reported separately for exactly that
    reason, and the two should be read together.
    """
    X_ref = np.asarray(reference, dtype=float)
    X_cur = np.asarray(current, dtype=float)
    if X_ref.shape[1] != X_cur.shape[1]:
        raise ValueError("reference and current must have the same feature count")
    n_features = X_ref.shape[1]

    if target_indices is None:
        # Probe the highest-variance features: relationships involving
        # near-constant features carry little information and their relative
        # error blows up numerically.
        variances = np.var(X_ref, axis=0)
        target_indices = list(np.argsort(variances)[::-1][:max_targets])

    rng = np.random.default_rng(random_state)
    split = rng.permutation(len(X_ref))
    half = len(X_ref) // 2
    fit_idx, holdout_idx = split[:half], split[half:]

    per_feature: dict[str, float] = {}
    surrogate = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=random_state,
    )

    for j in target_indices:
        others = [k for k in range(n_features) if k != j]
        if not others:
            continue
        model = clone(surrogate)
        model.fit(X_ref[np.ix_(fit_idx, others)], X_ref[fit_idx, j])

        ref_err = float(
            np.mean(
                np.abs(
                    model.predict(X_ref[np.ix_(holdout_idx, others)])
                    - X_ref[holdout_idx, j]
                )
            )
        )
        cur_err = float(
            np.mean(np.abs(model.predict(X_cur[:, others]) - X_cur[:, j]))
        )
        if ref_err < 1e-12:
            continue
        per_feature[feature_names[j]] = (cur_err - ref_err) / ref_err

    if not per_feature:
        statistic = 0.0
    else:
        statistic = float(np.median(list(per_feature.values())))

    if statistic >= alert_threshold:
        severity, drifted = Severity.ALERT, True
    elif statistic >= watch_threshold:
        severity, drifted = Severity.WATCH, True
    else:
        severity, drifted = Severity.NONE, False

    return DriftResult(
        feature_name=RELATIONSHIP_FEATURE_NAME,
        method="feature_relationship_surrogate",
        statistic=statistic,
        kind=DriftKind.CONCEPT_PROXY,
        window=window,
        threshold=alert_threshold,
        is_drifted=drifted,
        severity=severity,
        n_reference=len(X_ref),
        n_current=len(X_cur),
        extra={"per_feature_relative_error_increase": per_feature},
    )


def out_of_support_mass(
    reference,
    current,
    *,
    window: WindowSpec,
    quantile_range: tuple[float, float] = (0.01, 0.99),
    watch_threshold: float = 0.05,
    alert_threshold: float = 0.15,
) -> DriftResult:
    """Fraction of current rows with at least one feature outside the
    reference window's central range.

    Uses per-feature quantile boxes rather than a fitted density: it is
    transparent enough to defend to a model risk committee, and in high
    dimensions a fitted density estimator would be the less trustworthy of
    the two. The cost is that it ignores joint novelty (a row can be
    in-range on every feature individually while being a combination never
    seen); the domain classifier in multivariate.py is what covers that gap.
    """
    X_ref = np.asarray(reference, dtype=float)
    X_cur = np.asarray(current, dtype=float)
    lo = np.quantile(X_ref, quantile_range[0], axis=0)
    hi = np.quantile(X_ref, quantile_range[1], axis=0)
    outside = (X_cur < lo) | (X_cur > hi)
    row_outside = outside.any(axis=1)
    statistic = float(np.mean(row_outside))

    # Baseline: by construction ~2% of REFERENCE rows fall outside a 1-99%
    # box per feature, so a non-zero rate is expected even with no drift.
    # Report the excess over that baseline, not the raw rate.
    ref_outside = ((X_ref < lo) | (X_ref > hi)).any(axis=1)
    baseline = float(np.mean(ref_outside))
    excess = max(0.0, statistic - baseline)

    if excess >= alert_threshold:
        severity, drifted = Severity.ALERT, True
    elif excess >= watch_threshold:
        severity, drifted = Severity.WATCH, True
    else:
        severity, drifted = Severity.NONE, False

    return DriftResult(
        feature_name=SUPPORT_FEATURE_NAME,
        method="quantile_box_novelty",
        statistic=excess,
        kind=DriftKind.CONCEPT_PROXY,
        window=window,
        threshold=alert_threshold,
        is_drifted=drifted,
        severity=severity,
        n_reference=len(X_ref),
        n_current=len(X_cur),
        extra={"raw_current_rate": statistic, "reference_baseline_rate": baseline},
    )


def effective_sample_size_ratio(weights) -> float:
    """Kish effective sample size of importance weights, as a fraction of n.

    ESS/n = (sum w)^2 / (n * sum w^2). Near 1.0 the weights are benign; near
    0 a handful of reference rows carry all the mass and any importance-
    weighted estimate is effectively computed from those few rows. Component
    2 must refuse to report an estimate below a floor on this value — a
    confidently wrong performance estimate is worse than an absent one.
    """
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w)]
    if len(w) == 0 or np.sum(w) <= 0:
        return 0.0
    return float(np.sum(w) ** 2 / (len(w) * np.sum(w**2)))


def confirm_concept_drift_with_labels(
    reference_scores,
    reference_labels,
    current_scores,
    current_labels,
    *,
    window: WindowSpec,
    watch_threshold: float = 0.03,
    alert_threshold: float = 0.07,
) -> DriftResult:
    """The label-confirmed measurement. Runs only after delayed labels land.

    Statistic is the drop in AUC from reference to current. AUC is used
    rather than accuracy because it is invariant to the shifting base rate
    that accompanies most real drift — an accuracy drop on a window whose
    positive rate moved tells you about prevalence, not about the model.

    Caveat that belongs in every report using this: an AUC drop is consistent
    with concept drift but does not prove it. Pure covariate shift into a
    region where the outcome is inherently harder to predict lowers AUC with
    P(Y|X) completely unchanged. Distinguishing those requires comparing
    against importance-weighted reference performance, which is what the
    validation harness in component 2 does.
    """
    ref_auc = float(roc_auc_score(reference_labels, reference_scores))
    cur_auc = float(roc_auc_score(current_labels, current_scores))
    drop = ref_auc - cur_auc

    if drop >= alert_threshold:
        severity, drifted = Severity.ALERT, True
    elif drop >= watch_threshold:
        severity, drifted = Severity.WATCH, True
    else:
        severity, drifted = Severity.NONE, False

    return DriftResult(
        feature_name="__label_confirmed__",
        method="auc_drop",
        statistic=drop,
        kind=DriftKind.CONCEPT_CONFIRMED,
        window=window,
        threshold=alert_threshold,
        is_drifted=drifted,
        severity=severity,
        n_reference=len(reference_scores),
        n_current=len(current_scores),
        extra={"reference_auc": ref_auc, "current_auc": cur_auc},
    )
