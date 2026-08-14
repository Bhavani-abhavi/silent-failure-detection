"""Validity and minimum-detectable-effect gates shared by every detector.

THE RULE THIS MODULE ENFORCES
=============================

Every detector must be able to answer, at call time, "what is the smallest
effect this configuration could possibly detect?" — and must never quietly
report "no drift" when the answer is "none".

This exists because the same bug has now been found twice in this codebase,
in two unrelated detectors:

  1. The permutation test had a resolution floor: with n_permutations=15 the
     smallest achievable p-value (0.0625) exceeded alpha=0.05, so perfectly
     separable windows (AUC = 1.0) returned "no drift".
  2. The KS test on an all-NaN reference window returned
     `statistic=nan, is_drifted=False` — a valid-looking result asserting
     that a feature which had entirely vanished was fine.

Both are silent failures inside a silent-failure detector, and both fail at
the moment they matter most. A third variant will appear in components 2-6
unless the rule is enforced centrally, so detectors call these helpers rather
than each re-deriving the check.

TWO KINDS OF PROBLEM, TWO DIFFERENT RESPONSES
=============================================

*Configuration* errors are knowable before looking at any data — a
permutation count too small for the requested alpha, a threshold that cannot
be exceeded. These RAISE. They are programmer errors and there is no
sensible result to return.

*Data* insufficiency is discovered per feature, per window — a feature that
is 100% missing this quarter, a window with nine rows. These return a result
carrying `ResultStatus.INSUFFICIENT_DATA`. A monitoring run sweeping
thousands of feature-window pairs must not die because one feature was
retired upstream, but the result must never be mistaken for "checked, fine".

`is_drifted=False` is therefore not sufficient on its own to conclude
anything. Callers must check `status is ResultStatus.OK` first, and reports
must render non-OK results as a distinct category — not folded into the
"no drift detected" count.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from drift_core.types import DriftKind, DriftResult, ResultStatus, Severity, WindowSpec

# KS critical-value coefficients: D_crit ~= c(alpha) * sqrt((n1+n2)/(n1*n2)).
_KS_COEFFICIENTS = {0.10: 1.224, 0.05: 1.358, 0.025: 1.480, 0.01: 1.628}


def coverage(values) -> float:
    """Fraction of entries that are present and finite."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(arr)))


def n_valid(values) -> int:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0
    return int(np.sum(np.isfinite(arr)))


def check_windows(
    reference,
    current,
    *,
    min_samples: int = 30,
    min_coverage: float = 0.0,
) -> tuple[bool, str]:
    """Are both windows usable at all?

    Returns (is_valid, reason). `min_samples` defaults to 30 — below that,
    every test in this package has so little power that a "no drift" verdict
    carries no information.
    """
    n_ref, n_cur = n_valid(reference), n_valid(current)

    if n_ref == 0 and n_cur == 0:
        return False, "feature is entirely missing in both windows"
    if n_ref == 0:
        return False, (
            "feature is entirely missing in the reference window — cannot "
            "establish a baseline to compare against"
        )
    if n_cur == 0:
        return False, (
            "feature is entirely missing in the current window — this is a "
            "schema or pipeline change, not a distributional shift"
        )
    if n_ref < min_samples:
        return False, f"reference has {n_ref} valid values, below min_samples={min_samples}"
    if n_cur < min_samples:
        return False, f"current has {n_cur} valid values, below min_samples={min_samples}"

    if min_coverage > 0:
        cov_ref, cov_cur = coverage(reference), coverage(current)
        if cov_ref < min_coverage:
            return False, f"reference coverage {cov_ref:.1%} below {min_coverage:.1%}"
        if cov_cur < min_coverage:
            return False, f"current coverage {cov_cur:.1%} below {min_coverage:.1%}"

    return True, ""


def ks_minimum_detectable_effect(n_ref: int, n_cur: int, alpha: float = 0.05) -> float:
    """Smallest KS statistic that could reach significance at `alpha`.

    D_crit = c(alpha) * sqrt((n_ref + n_cur) / (n_ref * n_cur)). Since the KS
    statistic is bounded above by 1, a D_crit at or above 1 means no
    difference whatsoever — not even two entirely disjoint distributions —
    could be called significant.
    """
    if n_ref <= 0 or n_cur <= 0:
        return float("inf")
    coefficient = _KS_COEFFICIENTS.get(alpha)
    if coefficient is None:
        # Asymptotic form for alphas not tabulated above.
        coefficient = float(np.sqrt(-0.5 * np.log(alpha / 2)))
    return float(coefficient * np.sqrt((n_ref + n_cur) / (n_ref * n_cur)))


def psi_null_expectation(n_ref: int, n_cur: int, bins: int) -> float:
    """Expected PSI under the null hypothesis of no drift.

    PSI is asymptotically (1/n_ref + 1/n_cur) * chi-square(bins - 1), so its
    expectation under "nothing changed" is (bins - 1) * (1/n_ref + 1/n_cur).

    This is the number that makes the folklore thresholds checkable. The
    usual "PSI > 0.25 means substantial shift" rule assumes large samples: on
    two windows of 200 rows with 10 bins the null expectation is already
    0.09, so a 0.1 "watch" threshold is triggered by pure sampling noise
    roughly half the time. Reported in every PSI result so the threshold can
    be judged against it rather than taken on faith.
    """
    if n_ref <= 0 or n_cur <= 0:
        return float("inf")
    return float((bins - 1) * (1.0 / n_ref + 1.0 / n_cur))


def psi_p_value(psi: float, n_ref: int, n_cur: int, bins: int) -> float:
    """Probability of a PSI this large under "nothing changed".

    PSI is asymptotically `(1/n_ref + 1/n_cur) * chi2(bins - 1)`, so dividing
    by the scale factor gives a chi-square variate directly. `bins` must be the
    *realized* bin count, not the requested one — quantile edges collapse on
    ties, and using the requested count would overstate the degrees of freedom
    and make the test anti-conservative on exactly the lumpy, heavily-tied
    features where it is most likely to be wrong.
    """
    scale = 1.0 / n_ref + 1.0 / n_cur
    df = bins - 1
    if df < 1 or scale <= 0 or not np.isfinite(psi):
        return float("nan")
    return float(stats.chi2.sf(psi / scale, df=df))


def kl_p_value(kl: float, n_ref: int, n_cur: int, bins: int) -> float:
    """Same asymptotic, halved.

    PSI is exactly the symmetrized KL — `sum((c-r)(log c - log r))` expands to
    `KL(c||r) + KL(r||c)` — and under the null the two halves have the same
    expectation. So `2 * KL` inherits PSI's null distribution. Verified against
    simulation rather than left as algebra in
    `test_univariate.py::TestAsymptoticNullCalibration`.
    """
    return psi_p_value(2.0 * kl, n_ref, n_cur, bins)


def permutation_p_value(
    reference,
    current,
    statistic_fn,
    *,
    n_permutations: int = 199,
    max_samples: int = 5000,
    random_state: int | None = None,
) -> tuple[float, float, int]:
    """Permutation null for a statistic with no usable asymptotic form.

    Returns `(p_value, observed_at_capped_n, n_per_side)`.

    THE CAP IS LORE, NOT AN OPTIMISATION
    ------------------------------------
    Both windows are subsampled to `max_samples` per side, and the observed
    statistic is recomputed on that same subsample. Building the null at a
    reduced n while comparing it against a statistic computed at full n would
    be a serious error in the *anti*-conservative direction for any statistic
    that shrinks with n — the full-n observed value would sit deep inside a
    null built from noisier small-n draws, and the test would essentially
    never reject no matter how large the true shift.

    The price is honest and one-directional: at capped n the test has less
    power than the data could support, so it under-fires rather than
    over-fires. Callers get the full-n effect size separately, which is the
    number that should be doing the work anyway.
    """
    rng = np.random.default_rng(random_state)

    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]

    n_per_side = int(min(max_samples, len(ref), len(cur)))
    if n_per_side < 2:
        return float("nan"), float("nan"), n_per_side

    ref_s = rng.choice(ref, size=n_per_side, replace=False)
    cur_s = rng.choice(cur, size=n_per_side, replace=False)
    observed = float(statistic_fn(ref_s, cur_s))

    pool = np.concatenate([ref_s, cur_s])
    at_least_as_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(pool)
        if float(statistic_fn(pool[:n_per_side], pool[n_per_side:])) >= observed:
            at_least_as_extreme += 1

    # +1 in both terms: the observed arrangement is itself one of the
    # permutations, and omitting it makes p=0 reachable, which is a claim no
    # finite permutation test can support.
    p_value = (at_least_as_extreme + 1) / (n_permutations + 1)
    return float(p_value), observed, n_per_side


def insufficient_data_result(
    *,
    feature_name: str,
    method: str,
    kind: DriftKind,
    window: WindowSpec,
    reason: str,
    n_reference: int,
    n_current: int,
) -> DriftResult:
    """The result a detector returns when it cannot evaluate.

    `statistic` is NaN and `is_drifted` is False, but `status` is
    INSUFFICIENT_DATA — so any caller that checks status cannot mistake this
    for a clean bill of health, and any caller that does not check status
    gets a NaN rather than a plausible number.
    """
    return DriftResult(
        feature_name=feature_name,
        method=method,
        statistic=float("nan"),
        kind=kind,
        window=window,
        is_drifted=False,
        severity=Severity.NONE,
        status=ResultStatus.INSUFFICIENT_DATA,
        n_reference=n_reference,
        n_current=n_current,
        extra={"reason": reason},
    )


def require_detectable_alpha(n_permutations: int, alpha: float) -> None:
    """Configuration guard for permutation-based tests.

    Raises rather than returning a status: this is knowable without data, so
    reaching it means the caller configured a test that can never fire.
    """
    min_achievable_p = 1.0 / (n_permutations + 1)
    if min_achievable_p > alpha:
        raise ValueError(
            f"n_permutations={n_permutations} cannot produce a p-value below "
            f"{min_achievable_p:.4f}, so no result can ever reach alpha={alpha}. "
            f"Use n_permutations >= {int(np.ceil(1 / alpha)) - 1} for alpha={alpha}, "
            f"or raise alpha."
        )
