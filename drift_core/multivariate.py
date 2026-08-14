"""Multivariate drift via the domain-classifier (classifier two-sample test).

Idea: train a discriminator to tell reference rows from current rows. If it
can't do better than chance, the joint distributions are indistinguishable to
that model class. If it can, something moved jointly — including correlation
structure that per-feature tests miss entirely.

Why this is not just "AUC > 0.5":
    With enough features and finite samples, a flexible classifier will find
    *some* separating signal between any two samples, even when drawn from
    the same distribution. Held-out AUC controls part of that but is still
    biased upward in high dimensions at small n. So the reported evidence is
    a permutation p-value: shuffle the reference/current labels, refit, and
    build the null distribution of held-out AUC under "no drift". The
    observed AUC is only meaningful relative to that null.

Cost: (n_permutations + 1) x n_splits model fits per window. Measured at
~600 ms per forest fit on a 1000x5 window, that is ~78 s per call at the
default settings — the dominant cost in the whole system by a wide margin.
Parallelism is applied ACROSS permutations (embarrassingly parallel, one
core per permutation) rather than inside each forest, where it measured as
worth almost nothing (598 ms vs 662 ms for n_jobs=-1 vs n_jobs=1). The
smallest achievable p-value is 1/(n_permutations+1), so n_permutations sets
the resolution floor, not just the runtime.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from drift_core.types import (
    DriftKind,
    MultivariateDriftResult,
    Severity,
    WindowSpec,
)


def _default_classifier(random_state: int) -> BaseEstimator:
    """Shallow forest: strong enough to catch interactions, constrained
    enough to limit the finite-sample overfitting described above. Depth is
    capped deliberately — an unconstrained forest inflates the observed AUC
    and the permutation null together, which mostly wastes compute."""
    # n_jobs=1 is deliberate: parallelism is applied across permutations by
    # the caller, so threading inside each forest would only contend for the
    # same cores. 100 trees rather than 200 — for a two-sample test the AUC
    # is stable well before 200, and the fit cost is linear in tree count.
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=20,
        n_jobs=1,
        random_state=random_state,
    )


def _cross_val_auc(
    X: np.ndarray,
    y: np.ndarray,
    classifier: BaseEstimator,
    n_splits: int,
    random_state: int,
) -> float:
    """Out-of-fold AUC over the whole set. Out-of-fold rather than a single
    holdout so small windows still produce a stable estimate."""
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    oof_scores = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in splitter.split(X, y):
        model = clone(classifier)
        model.fit(X[train_idx], y[train_idx])
        oof_scores[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return float(roc_auc_score(y, oof_scores))


def domain_classifier_drift(
    reference,
    current,
    *,
    window: WindowSpec,
    feature_names: list[str] | None = None,
    classifier: BaseEstimator | None = None,
    n_splits: int = 5,
    n_permutations: int = 100,
    alpha: float = 0.05,
    auc_watch_threshold: float = 0.60,
    n_jobs: int = -1,
    random_state: int = 0,
) -> MultivariateDriftResult:
    """Classifier two-sample test between reference and current feature
    matrices.

    Returns AUC (effect size — how separable) alongside a permutation
    p-value (evidence — is that separability more than noise). Both are
    required to call drift: a statistically significant AUC of 0.53 on a
    million rows is real but operationally meaningless, which is why
    `auc_watch_threshold` gates the severity independently of `alpha`.

    `n_jobs` parallelises the permutation loop (default: all cores). Set to
    1 when this is already running inside an outer parallel loop, e.g. a
    backtest sweeping many windows at once — nesting the two oversubscribes
    the CPU and runs slower than either alone.
    """
    X_ref = np.asarray(reference, dtype=float)
    X_cur = np.asarray(current, dtype=float)
    if X_ref.ndim != 2 or X_cur.ndim != 2:
        raise ValueError("reference and current must be 2-D feature matrices")
    if X_ref.shape[1] != X_cur.shape[1]:
        raise ValueError(
            f"feature count mismatch: reference has {X_ref.shape[1]}, "
            f"current has {X_cur.shape[1]}"
        )

    # A permutation test cannot report a p-value below 1/(n_permutations+1).
    # If that floor sits above alpha, NOTHING can ever reach significance —
    # a perfectly separable pair of windows (AUC = 1.0) returns "no drift".
    # That is a silent failure of exactly the kind this project exists to
    # catch, and it fails quietly at the one moment it matters most, so it
    # is a hard error rather than a warning.
    min_achievable_p = 1.0 / (n_permutations + 1)
    if min_achievable_p > alpha:
        raise ValueError(
            f"n_permutations={n_permutations} cannot produce a p-value below "
            f"{min_achievable_p:.4f}, so no result can ever reach alpha="
            f"{alpha}. Use n_permutations >= {int(np.ceil(1 / alpha)) - 1} "
            f"for alpha={alpha}, or raise alpha."
        )

    X = np.vstack([X_ref, X_cur])
    y = np.concatenate([np.zeros(len(X_ref)), np.ones(len(X_cur))])

    if classifier is None:
        classifier = _default_classifier(random_state)

    observed_auc = _cross_val_auc(X, y, classifier, n_splits, random_state)

    # Permutation null: shuffle group labels, refit, recompute. Under the null
    # of no drift, the observed AUC is exchangeable with these.
    #
    # Permutations are drawn up front from a single seeded generator rather
    # than inside the workers, so the null is reproducible regardless of how
    # joblib schedules the jobs.
    rng = np.random.default_rng(random_state)
    permuted_labels = [rng.permutation(y) for _ in range(n_permutations)]
    null_aucs = np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(_cross_val_auc)(
                X, y_perm, classifier, n_splits, random_state + i + 1
            )
            for i, y_perm in enumerate(permuted_labels)
        ),
        dtype=float,
    )

    # +1 in numerator and denominator: the observed value is itself one draw
    # from the null under H0, which keeps the test valid (never reports p=0).
    p_value = float((np.sum(null_aucs >= observed_auc) + 1) / (n_permutations + 1))

    significant = p_value < alpha
    if significant and observed_auc >= 0.75:
        severity = Severity.ALERT
    elif significant and observed_auc >= auc_watch_threshold:
        severity = Severity.WATCH
    else:
        severity = Severity.NONE
    is_drifted = severity is not Severity.NONE

    # Refit once on the full set purely to report which features drove the
    # separation. This is descriptive output for governance reporting, not
    # part of the test statistic.
    importances: dict[str, float] = {}
    final_model = clone(classifier)
    final_model.fit(X, y)
    if hasattr(final_model, "feature_importances_") and feature_names:
        importances = {
            name: float(score)
            for name, score in zip(feature_names, final_model.feature_importances_)
        }

    return MultivariateDriftResult(
        method="domain_classifier",
        auc=observed_auc,
        p_value=p_value,
        window=window,
        n_reference=len(X_ref),
        n_current=len(X_cur),
        is_drifted=is_drifted,
        severity=severity,
        feature_importances=importances,
    )


__all__ = ["domain_classifier_drift", "DriftKind"]
