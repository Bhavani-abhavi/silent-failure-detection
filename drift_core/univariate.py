"""Feature-level drift detectors: PSI, KL divergence, KS test, Wasserstein
distance.

These test only P(X) for a single feature — DriftKind.DATA. None of them
touch labels or predictions. Everything here is domain-agnostic: callers
pass plain arrays and a feature name; nothing here knows what the feature
means.

EVERY DETECTOR HERE REQUIRES TWO GATES TO FIRE
==============================================

`is_drifted` is true only when the shift is BOTH statistically distinguishable
from sampling noise (`p_value < alpha`) AND large enough to matter
(`statistic >= min_effect_size`). Neither alone is usable, and this codebase
has measurements for both failure modes:

- **Significance alone.** On real Lending Club windows (n ~ 40,000) the KS test
  alerted on **81-94%** of feature-quarters while being correctly calibrated at
  4.3% against a synthetic null. It is not broken; it is answering "is there
  *any* difference", and on real credit data the answer is always yes. Multiple-
  testing correction does not help — the family-wise error rate is not the
  problem, the null being tested is.
- **Effect size alone.** The conventional PSI 0.1/0.25 thresholds are a
  sample-size artifact. At n=250 per window the 0.1 threshold sits on the noise
  floor and false-alarms 14.6% of the time; at n=20,000 it sits 111x above the
  floor, so the detector is effectively switched off while appearing to be on.

So `min_effect_size` defaults remain provisional and should be calibrated per
deployment against `minimum_detectable_effect`, which every result reports.
What is *not* provisional is the requirement that both gates exist.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from drift_core.types import (
    DriftKind,
    DriftResult,
    ResultStatus,
    Severity,
    WindowSpec,
)
from drift_core.validity import (
    check_windows,
    insufficient_data_result,
    kl_p_value,
    ks_minimum_detectable_effect,
    n_valid,
    permutation_p_value,
    psi_null_expectation,
    psi_p_value,
    require_detectable_alpha,
)


def _as_float_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return arr


def _quantile_bin_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    """Bin edges from reference quantiles, with open outer edges so current
    values outside the reference range still land in a bin instead of being
    silently dropped (dropping them would understate drift, not overstate it)."""
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        # Degenerate reference (constant feature). Needs FOUR edges, not two:
        # a single (-inf, inf) bin would swallow every current value and
        # report PSI = 0 no matter how far the feature moved. Three bins
        # (below / at the constant / above) keep a moved current window
        # detectable, which is the whole point of monitoring a feature that
        # is supposed to be constant.
        value = float(reference[0])
        eps = max(abs(value), 1.0) * 1e-9
        return np.array([-np.inf, value - eps, value + eps, np.inf])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bin_proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    if total == 0:
        return np.zeros(len(counts))
    return counts / total


def _categorical_proportions(
    values: np.ndarray, categories: np.ndarray
) -> np.ndarray:
    values = np.asarray(values)
    counts = np.array([(values == c).sum() for c in categories], dtype=float)
    total = counts.sum()
    if total == 0:
        return np.zeros(len(counts))
    return counts / total


def population_stability_index(
    reference,
    current,
    *,
    bins: int = 10,
    epsilon: float = 1e-4,
    categorical: bool = False,
) -> tuple[float, dict]:
    """PSI = sum((cur_pct - ref_pct) * ln(cur_pct / ref_pct)) over bins fixed
    from the reference distribution. Standard read: <0.1 negligible,
    0.1-0.25 moderate, >0.25 substantial shift. Symmetric-ish but not a true
    distance metric (doesn't satisfy triangle inequality)."""
    if categorical:
        ref_arr = np.asarray(reference)
        cur_arr = np.asarray(current)
        categories = np.unique(np.concatenate([ref_arr, cur_arr]))
        ref_prop = _categorical_proportions(ref_arr, categories)
        cur_prop = _categorical_proportions(cur_arr, categories)
        extra = {"categories": categories.tolist()}
    else:
        ref_arr = _as_float_array(reference)
        cur_arr = _as_float_array(current)
        edges = _quantile_bin_edges(ref_arr, bins)
        ref_prop = _bin_proportions(ref_arr, edges)
        cur_prop = _bin_proportions(cur_arr, edges)
        extra = {"bin_edges": edges.tolist()}

    # Report the true observed proportions; clip only for the log. Storing
    # the clipped values would make reported proportions sum to more than 1
    # whenever a bin is empty, which is wrong in any report a human reads.
    extra["reference_proportions"] = ref_prop.tolist()
    extra["current_proportions"] = cur_prop.tolist()

    ref_safe = np.clip(ref_prop, epsilon, None)
    cur_safe = np.clip(cur_prop, epsilon, None)
    psi = float(np.sum((cur_safe - ref_safe) * np.log(cur_safe / ref_safe)))
    return psi, extra


def kl_divergence(
    reference,
    current,
    *,
    bins: int = 10,
    epsilon: float = 1e-4,
    categorical: bool = False,
) -> tuple[float, dict]:
    """KL(current || reference): expected extra "surprise" (in nats) of
    modeling the current window with a distribution fit to the reference
    window. Asymmetric by design — this is "how wrong would reference-based
    assumptions be about what we're seeing now", which is the direction that
    matters for a monitor built on a fixed reference."""
    if categorical:
        ref_arr = np.asarray(reference)
        cur_arr = np.asarray(current)
        categories = np.unique(np.concatenate([ref_arr, cur_arr]))
        ref_prop = _categorical_proportions(ref_arr, categories)
        cur_prop = _categorical_proportions(cur_arr, categories)
        extra = {"categories": categories.tolist()}
    else:
        ref_arr = _as_float_array(reference)
        cur_arr = _as_float_array(current)
        edges = _quantile_bin_edges(ref_arr, bins)
        ref_prop = _bin_proportions(ref_arr, edges)
        cur_prop = _bin_proportions(cur_arr, edges)
        extra = {"bin_edges": edges.tolist()}

    extra["reference_proportions"] = ref_prop.tolist()
    extra["current_proportions"] = cur_prop.tolist()

    ref_safe = np.clip(ref_prop, epsilon, None)
    cur_safe = np.clip(cur_prop, epsilon, None)
    kl = float(np.sum(cur_safe * np.log(cur_safe / ref_safe)))
    return kl, extra


def ks_test(reference, current) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test. Continuous features only. Returns
    (statistic, p_value); p_value is exact/asymptotic per scipy's choice and
    is what alerting.py should feed into multiple-testing correction rather
    than comparing raw statistics across features."""
    ref_arr = _as_float_array(reference)
    cur_arr = _as_float_array(current)
    result = stats.ks_2samp(ref_arr, cur_arr)
    return float(result.statistic), float(result.pvalue)


def wasserstein(reference, current, *, normalize: bool = True) -> tuple[float, dict]:
    """Earth-mover's distance between reference and current. Normalized by
    reference std by default so magnitudes are roughly comparable across
    features on different scales — without normalization a feature measured
    in the thousands will always dominate one measured in fractions."""
    ref_arr = _as_float_array(reference)
    cur_arr = _as_float_array(current)
    raw = float(stats.wasserstein_distance(ref_arr, cur_arr))
    ref_std = float(np.std(ref_arr))
    if normalize and ref_std > 1e-12:
        stat = raw / ref_std
    else:
        stat = raw
    return stat, {"raw_distance": raw, "reference_std": ref_std}


def _gated_severity(
    statistic: float,
    p_value: float,
    *,
    alpha: float,
    min_effect_size: float,
    alert_threshold: float,
) -> tuple[Severity, bool]:
    """Both gates, or nothing.

    A statistically significant shift below `min_effect_size` is real and
    operationally meaningless — it must not page anyone. A large-looking shift
    that a null model reproduces routinely at this sample size is noise. Only
    the intersection is worth acting on.
    """
    significant = np.isfinite(p_value) and p_value < alpha
    if not significant or statistic < min_effect_size:
        return Severity.NONE, False
    if statistic >= alert_threshold:
        return Severity.ALERT, True
    return Severity.WATCH, True


def detect_psi_drift(
    reference,
    current,
    *,
    feature_name: str,
    window: WindowSpec,
    bins: int = 10,
    categorical: bool = False,
    alpha: float = 0.05,
    min_effect_size: float = 0.1,
    alert_threshold: float = 0.25,
    min_samples: int = 30,
) -> DriftResult:
    n_ref, n_cur = n_valid(reference), n_valid(current)
    valid, reason = check_windows(reference, current, min_samples=min_samples)
    if not valid:
        return insufficient_data_result(
            feature_name=feature_name, method="psi", kind=DriftKind.DATA,
            window=window, reason=reason, n_reference=n_ref, n_current=n_cur,
        )

    psi, extra = population_stability_index(
        reference, current, bins=bins, categorical=categorical
    )
    # Realized, not requested: quantile edges collapse on ties, and a heavily
    # tied feature can end up with far fewer bins than asked for.
    realized_bins = len(extra["reference_proportions"])
    p_value = psi_p_value(psi, n_ref, n_cur, realized_bins)
    severity, drifted = _gated_severity(
        psi, p_value, alpha=alpha,
        min_effect_size=min_effect_size, alert_threshold=alert_threshold,
    )

    # The PSI a pair of identical distributions would produce at these sample
    # sizes. If min_effect_size sits below it, the effect gate is not gating —
    # it admits pure sampling noise and leaves alpha doing all the work.
    noise_floor = psi_null_expectation(n_ref, n_cur, realized_bins)
    extra["null_expectation"] = noise_floor
    extra["realized_bins"] = realized_bins
    status = (
        ResultStatus.NO_POWER if min_effect_size < noise_floor else ResultStatus.OK
    )
    if status is ResultStatus.NO_POWER:
        extra["reason"] = (
            f"min_effect_size={min_effect_size} is below the null expectation "
            f"{noise_floor:.4f} at n_ref={n_ref}, n_cur={n_cur}; this "
            f"configuration fires on sampling noise"
        )

    return DriftResult(
        feature_name=feature_name,
        method="psi",
        statistic=psi,
        kind=DriftKind.DATA,
        window=window,
        p_value=p_value,
        threshold=alert_threshold,
        is_drifted=drifted,
        severity=severity,
        status=status,
        minimum_detectable_effect=noise_floor,
        n_reference=n_ref,
        n_current=n_cur,
        extra=extra,
    )


def detect_ks_drift(
    reference,
    current,
    *,
    feature_name: str,
    window: WindowSpec,
    alpha: float = 0.05,
    min_effect_size: float = 0.05,
    alert_threshold: float = 0.1,
    min_samples: int = 30,
) -> DriftResult:
    """Two-sample KS with an effect-size gate.

    `min_effect_size` is in units of the KS statistic — the maximum vertical
    gap between the two empirical CDFs, so 0.05 means "the distributions
    disagree about where at least 5% of the mass sits". It defaults to a real
    value rather than 0 because on this project's data KS at
    significance-only alerted on 81-94% of feature-quarters. That behaviour is
    correct and useless; the gate is what makes the detector answer a question
    somebody would act on.
    """
    n_ref, n_cur = n_valid(reference), n_valid(current)

    # Without this gate an all-NaN reference window returned
    # statistic=nan, is_drifted=False — a confident "no drift" verdict on a
    # feature that had entirely disappeared. Found on real Lending Club data.
    valid, reason = check_windows(reference, current, min_samples=min_samples)
    if not valid:
        return insufficient_data_result(
            feature_name=feature_name, method="ks", kind=DriftKind.DATA,
            window=window, reason=reason, n_reference=n_ref, n_current=n_cur,
        )

    statistic, p_value = ks_test(reference, current)
    severity, drifted = _gated_severity(
        statistic, p_value, alpha=alpha,
        min_effect_size=min_effect_size, alert_threshold=alert_threshold,
    )

    # KS is bounded by 1, so a critical value at or above 1 means even two
    # completely disjoint distributions could not reach significance. The
    # effect gate raises that floor: the detector cannot fire below
    # min_effect_size either, whatever the p-value says.
    mde = max(ks_minimum_detectable_effect(n_ref, n_cur, alpha), min_effect_size)
    status = ResultStatus.NO_POWER if mde >= 1.0 else ResultStatus.OK

    return DriftResult(
        feature_name=feature_name,
        method="ks",
        statistic=statistic,
        kind=DriftKind.DATA,
        window=window,
        p_value=p_value,
        threshold=alpha,
        is_drifted=drifted,
        severity=severity,
        status=status,
        minimum_detectable_effect=mde,
        n_reference=n_ref,
        n_current=n_cur,
        extra=(
            {}
            if status is ResultStatus.OK
            else {
                "reason": (
                    f"effective KS floor {mde:.3f} >= 1.0 at n_ref={n_ref}, "
                    f"n_cur={n_cur}, min_effect_size={min_effect_size}; no "
                    f"difference is detectable at alpha={alpha}"
                )
            }
        ),
    )


def detect_wasserstein_drift(
    reference,
    current,
    *,
    feature_name: str,
    window: WindowSpec,
    alpha: float = 0.05,
    min_effect_size: float = 0.1,
    alert_threshold: float = 0.25,
    normalize: bool = True,
    min_samples: int = 30,
    n_permutations: int = 199,
    max_permutation_samples: int = 5000,
    random_state: int | None = 0,
) -> DriftResult:
    """Earth-mover distance with a permutation null.

    Wasserstein has no usable closed-form two-sample null, so unlike PSI and
    KL this one pays for its p-value. The null is built at
    `max_permutation_samples` per side (see `permutation_p_value` for why the
    observed statistic is recomputed at the same n), while the reported
    `statistic` is the full-n effect size — the number the gate should be
    tuned against.
    """
    n_ref, n_cur = n_valid(reference), n_valid(current)
    valid, reason = check_windows(reference, current, min_samples=min_samples)
    if not valid:
        return insufficient_data_result(
            feature_name=feature_name, method="wasserstein", kind=DriftKind.DATA,
            window=window, reason=reason, n_reference=n_ref, n_current=n_cur,
        )

    # Knowable without touching the data, so it raises rather than returning a
    # quiet no-drift result. This is the permutation-floor bug's home.
    require_detectable_alpha(n_permutations, alpha)

    statistic, extra = wasserstein(reference, current, normalize=normalize)

    ref_std = extra["reference_std"]
    scale = ref_std if (normalize and ref_std > 1e-12) else 1.0

    def _statistic(a, b):
        return stats.wasserstein_distance(a, b) / scale

    p_value, observed_capped, n_per_side = permutation_p_value(
        reference, current, _statistic,
        n_permutations=n_permutations,
        max_samples=max_permutation_samples,
        random_state=random_state,
    )
    extra["permutation_n_per_side"] = n_per_side
    extra["statistic_at_permutation_n"] = observed_capped
    extra["n_permutations"] = n_permutations

    severity, drifted = _gated_severity(
        statistic, p_value, alpha=alpha,
        min_effect_size=min_effect_size, alert_threshold=alert_threshold,
    )
    return DriftResult(
        feature_name=feature_name,
        method="wasserstein",
        statistic=statistic,
        kind=DriftKind.DATA,
        window=window,
        p_value=p_value,
        threshold=alert_threshold,
        is_drifted=drifted,
        severity=severity,
        minimum_detectable_effect=min_effect_size,
        n_reference=n_ref,
        n_current=n_cur,
        extra=extra,
    )


def detect_kl_drift(
    reference,
    current,
    *,
    feature_name: str,
    window: WindowSpec,
    bins: int = 10,
    categorical: bool = False,
    alpha: float = 0.05,
    min_effect_size: float = 0.05,
    alert_threshold: float = 0.125,
    min_samples: int = 30,
) -> DriftResult:
    """KL(current || reference) with the halved PSI asymptotic as its null.

    Effect-size defaults are half PSI's, because KL is one of the two halves
    PSI sums — carrying PSI's 0.1/0.25 across unchanged would silently make
    this detector twice as hard to trip.
    """
    n_ref, n_cur = n_valid(reference), n_valid(current)
    valid, reason = check_windows(reference, current, min_samples=min_samples)
    if not valid:
        return insufficient_data_result(
            feature_name=feature_name, method="kl_divergence", kind=DriftKind.DATA,
            window=window, reason=reason, n_reference=n_ref, n_current=n_cur,
        )

    kl, extra = kl_divergence(reference, current, bins=bins, categorical=categorical)
    realized_bins = len(extra["reference_proportions"])
    p_value = kl_p_value(kl, n_ref, n_cur, realized_bins)
    severity, drifted = _gated_severity(
        kl, p_value, alpha=alpha,
        min_effect_size=min_effect_size, alert_threshold=alert_threshold,
    )
    noise_floor = psi_null_expectation(n_ref, n_cur, realized_bins) / 2.0
    extra["null_expectation"] = noise_floor
    extra["realized_bins"] = realized_bins
    status = (
        ResultStatus.NO_POWER if min_effect_size < noise_floor else ResultStatus.OK
    )
    if status is ResultStatus.NO_POWER:
        extra["reason"] = (
            f"min_effect_size={min_effect_size} is below the null expectation "
            f"{noise_floor:.4f} at n_ref={n_ref}, n_cur={n_cur}"
        )
    return DriftResult(
        feature_name=feature_name,
        method="kl_divergence",
        statistic=kl,
        kind=DriftKind.DATA,
        window=window,
        p_value=p_value,
        threshold=alert_threshold,
        is_drifted=drifted,
        severity=severity,
        status=status,
        minimum_detectable_effect=noise_floor,
        n_reference=n_ref,
        n_current=n_cur,
        extra=extra,
    )
