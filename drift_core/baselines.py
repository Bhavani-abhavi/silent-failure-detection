"""Naive baselines that every detector must be benchmarked against.

The project rule is: if a dumb threshold does comparably well, report that
honestly. That only happens if the dumb threshold is actually implemented and
run on the same windows, so it lives here as a first-class citizen rather
than as an afterthought in a notebook.

If these win, the finding is not "the project failed" — the finding is
"expensive multivariate drift detection bought nothing over a mean shift
alarm on this data", which is a genuinely useful result for anyone deciding
what to deploy.
"""

from __future__ import annotations

import numpy as np

from drift_core.types import DriftKind, DriftResult, Severity, WindowSpec


def mean_shift_baseline(
    reference,
    current,
    *,
    feature_name: str,
    window: WindowSpec,
    n_sigma: float = 3.0,
) -> DriftResult:
    """Fire if the two window means differ by more than n_sigma standard
    errors. This is the "what a competent engineer would write in ten
    minutes" bar.

    Uses a Welch-style two-sample standard error, sqrt(var_ref/n_ref +
    var_cur/n_cur), rather than the reference standard deviation alone. The
    reference-only version is what most quick implementations do and it is
    wrong in a way that matters here: when the current window's variance
    grows, the true sampling variability of its mean grows with it, but a
    reference-only SE does not — so the baseline fires on pure variance
    change via a spurious mean difference. That would be a false alarm
    credited as a detection, which would flatter our own detectors in the
    benchmark. A strawman baseline is worse than no baseline.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    n_ref, n_cur = max(len(ref), 1), max(len(cur), 1)
    se = float(np.sqrt(np.var(ref, ddof=1) / n_ref + np.var(cur, ddof=1) / n_cur))
    if se < 1e-12:
        statistic = 0.0
    else:
        statistic = float(abs(np.mean(cur) - np.mean(ref)) / se)

    drifted = statistic >= n_sigma
    return DriftResult(
        feature_name=feature_name,
        method="baseline_mean_shift",
        statistic=statistic,
        kind=DriftKind.DATA,
        window=window,
        threshold=n_sigma,
        is_drifted=drifted,
        severity=Severity.ALERT if drifted else Severity.NONE,
        n_reference=len(ref),
        n_current=len(cur),
    )


def missingness_baseline(
    reference,
    current,
    *,
    feature_name: str,
    window: WindowSpec,
    absolute_threshold: float = 0.05,
) -> DriftResult:
    """Fire if the missing-value rate moved by more than a fixed amount.

    Included because in practice this trivial check catches a large share of
    real production incidents (an upstream join breaking, a vendor feed
    changing schema) that elaborate distributional tests also catch but more
    slowly and with more false alarms. Any honest benchmark has to include it.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref_missing = float(np.mean(np.isnan(ref)))
    cur_missing = float(np.mean(np.isnan(cur)))
    statistic = abs(cur_missing - ref_missing)
    drifted = statistic >= absolute_threshold
    return DriftResult(
        feature_name=feature_name,
        method="baseline_missingness",
        statistic=statistic,
        kind=DriftKind.DATA,
        window=window,
        threshold=absolute_threshold,
        is_drifted=drifted,
        severity=Severity.ALERT if drifted else Severity.NONE,
        n_reference=len(ref),
        n_current=len(cur),
        extra={
            "reference_missing_rate": ref_missing,
            "current_missing_rate": cur_missing,
        },
    )
