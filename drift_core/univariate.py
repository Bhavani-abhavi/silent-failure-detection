"""Feature-level drift detectors: PSI, KL divergence, KS test, Wasserstein
distance.

These test only P(X) for a single feature — DriftKind.DATA. None of them
touch labels or predictions. Everything here is domain-agnostic: callers
pass plain arrays and a feature name; nothing here knows what the feature
means.

Thresholds set below (`watch_threshold`/`alert_threshold`) are heuristic
defaults, not calibrated ones. Component 5 (alerting) is responsible for
picking thresholds that hold a target false-positive rate on real stable
periods, with multiple-testing correction across features and windows —
these defaults exist so component 1 is independently testable and usable,
not as the final word on severity.
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
    ks_minimum_detectable_effect,
    n_valid,
    psi_null_expectation,
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


def _severity(value: float, watch: float, alert: float) -> tuple[Severity, bool]:
    if value >= alert:
        return Severity.ALERT, True
    if value >= watch:
        return Severity.WATCH, True
    return Severity.NONE, False


def detect_psi_drift(
    reference,
    current,
    *,
    feature_name: str,
    window: WindowSpec,
    bins: int = 10,
    categorical: bool = False,
    watch_threshold: float = 0.1,
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
    severity, drifted = _severity(psi, watch_threshold, alert_threshold)

    # The PSI a pair of identical distributions would produce at these sample
    # sizes. If the watch threshold sits below it, the detector is measuring
    # sampling noise and the severity below is not trustworthy.
    noise_floor = psi_null_expectation(n_ref, n_cur, bins)
    extra["null_expectation"] = noise_floor
    status = (
        ResultStatus.NO_POWER if watch_threshold < noise_floor else ResultStatus.OK
    )
    if status is ResultStatus.NO_POWER:
        extra["reason"] = (
            f"watch_threshold={watch_threshold} is below the null expectation "
            f"{noise_floor:.4f} at n_ref={n_ref}, n_cur={n_cur}; this "
            f"configuration fires on sampling noise"
        )

    return DriftResult(
        feature_name=feature_name,
        method="psi",
        statistic=psi,
        kind=DriftKind.DATA,
        window=window,
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
    min_samples: int = 30,
) -> DriftResult:
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
    drifted = p_value < alpha
    severity = Severity.ALERT if drifted else Severity.NONE

    # KS is bounded by 1, so a critical value at or above 1 means even two
    # completely disjoint distributions could not reach significance.
    mde = ks_minimum_detectable_effect(n_ref, n_cur, alpha)
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
                    f"KS critical value {mde:.3f} >= 1.0 at n_ref={n_ref}, "
                    f"n_cur={n_cur}; no difference is detectable at alpha={alpha}"
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
    watch_threshold: float = 0.1,
    alert_threshold: float = 0.25,
    normalize: bool = True,
    min_samples: int = 30,
) -> DriftResult:
    n_ref, n_cur = n_valid(reference), n_valid(current)
    valid, reason = check_windows(reference, current, min_samples=min_samples)
    if not valid:
        return insufficient_data_result(
            feature_name=feature_name, method="wasserstein", kind=DriftKind.DATA,
            window=window, reason=reason, n_reference=n_ref, n_current=n_cur,
        )

    statistic, extra = wasserstein(reference, current, normalize=normalize)
    severity, drifted = _severity(statistic, watch_threshold, alert_threshold)
    return DriftResult(
        feature_name=feature_name,
        method="wasserstein",
        statistic=statistic,
        kind=DriftKind.DATA,
        window=window,
        threshold=alert_threshold,
        is_drifted=drifted,
        severity=severity,
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
    watch_threshold: float = 0.1,
    alert_threshold: float = 0.25,
    min_samples: int = 30,
) -> DriftResult:
    n_ref, n_cur = n_valid(reference), n_valid(current)
    valid, reason = check_windows(reference, current, min_samples=min_samples)
    if not valid:
        return insufficient_data_result(
            feature_name=feature_name, method="kl_divergence", kind=DriftKind.DATA,
            window=window, reason=reason, n_reference=n_ref, n_current=n_cur,
        )

    kl, extra = kl_divergence(reference, current, bins=bins, categorical=categorical)
    severity, drifted = _severity(kl, watch_threshold, alert_threshold)
    noise_floor = psi_null_expectation(n_ref, n_cur, bins)
    extra["null_expectation"] = noise_floor
    return DriftResult(
        feature_name=feature_name,
        method="kl_divergence",
        statistic=kl,
        kind=DriftKind.DATA,
        window=window,
        threshold=alert_threshold,
        is_drifted=drifted,
        severity=severity,
        minimum_detectable_effect=noise_floor,
        n_reference=n_ref,
        n_current=n_cur,
        extra=extra,
    )
