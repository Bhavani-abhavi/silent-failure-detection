"""Is the anti-correlation between estimate and truth real, or a trend artifact?

    .venv/Scripts/python.exe scripts/verify_correlation.py

Reads reports/backtest/estimation_error.csv. Recomputes nothing expensive.

THE CLAIM UNDER TEST, AND HOW IT DIED

The first write-up of component B reported that the confidence-based label-free
estimator was *anti-correlated* with the truth it estimates (r = -0.38 over 35
windows, naive p = 0.023) — "not blind but actively misleading".

The 35 observations are consecutive months and both series are strongly
autocorrelated. A Pearson test assumes independent observations; correlating
two serially dependent trending series is the oldest way to manufacture a
significant result from nothing. Corrected, the effective sample size is 13 and
p is 0.195.

Four tests are run and all four reported, because a finding that survives only
the most permissive one is not a finding:

1. naive Pearson with a Fisher-z 95% CI     (the optimistic reading)
2. the same at a serial-correlation-adjusted effective n
3. correlation of FIRST DIFFERENCES          (trend removed)
4. moving-block bootstrap CI                 (serial structure preserved)

Then the claims that replaced it: a sign test for systematic bias, and
Newey-West HAC trend slopes for whether the estimate moved at all while the
truth did.

The statistics live in `backtest/significance.py` and are unit-tested there,
including a fixture of independent random walks that shows the naive test
false-rejecting >30% of the time where the correction cuts it by more than
half. Untested code has no business overturning a headline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from backtest.significance import (
    correlation_p_value,
    effective_sample_size,
    fisher_ci,
    hac_slope,
    moving_block_bootstrap_correlation,
    sign_test,
)

OUT = Path("reports/backtest")
PAIRS = [
    ("base_rate", "average_confidence"),
    ("brier", "average_confidence"),
    ("accuracy", "average_confidence"),
    ("accuracy", "atc"),
]


def verify(label: str, estimate: np.ndarray, truth: np.ndarray) -> dict:
    n = len(estimate)
    r, p_naive = stats.pearsonr(estimate, truth)
    lo, hi = fisher_ci(r, n)
    rho, p_spearman = stats.spearmanr(estimate, truth)

    n_eff, a1, b1 = effective_sample_size(estimate, truth)
    p_adjusted = correlation_p_value(r, n_eff)
    lo_adj, hi_adj = fisher_ci(r, int(round(n_eff)))

    d_est, d_truth = np.diff(estimate), np.diff(truth)
    r_diff, p_diff = stats.pearsonr(d_est, d_truth)
    n_eff_diff, _, _ = effective_sample_size(d_est, d_truth)
    p_diff_adjusted = correlation_p_value(r_diff, n_eff_diff)

    boot_lo, boot_hi = moving_block_bootstrap_correlation(estimate, truth)

    print(f"\n{'=' * 78}\n{label}   (n = {n} windows)\n{'=' * 78}")
    print(f"  1. naive Pearson      r = {r:+.4f}   p = {p_naive:.4f}"
          f"   95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"     Spearman           rho = {rho:+.4f} p = {p_spearman:.4f}")
    print(f"  2. lag-1 autocorr     estimate {a1:+.3f}, truth {b1:+.3f}"
          f"  ->  effective n {n_eff:.1f} of {n}")
    print(f"     adjusted           p = {p_adjusted:.4f}"
          f"   95% CI [{lo_adj:+.3f}, {hi_adj:+.3f}]")
    print(f"  3. first differences  r = {r_diff:+.4f}   p = {p_diff:.2e}"
          f"   (adjusted p = {p_diff_adjusted:.2e}, n_eff {n_eff_diff:.1f})")
    print(f"  4. block bootstrap    95% CI [{boot_lo:+.3f}, {boot_hi:+.3f}]")

    survives = (
        p_naive < 0.05 and p_adjusted < 0.05 and boot_hi < 0
    )
    print(f"\n  LEVEL CORRELATION SURVIVES: {'YES' if survives else 'NO'}")
    if not survives:
        reasons = []
        if p_naive >= 0.05:
            reasons.append("naive test not significant")
        if p_adjusted >= 0.05:
            reasons.append(f"not significant at effective n = {n_eff:.1f}")
        if boot_hi >= 0:
            reasons.append("block-bootstrap CI includes zero")
        if np.sign(r_diff) != np.sign(r):
            reasons.append("sign flips once the trend is differenced out")
        print(f"    {'; '.join(reasons)}")

    return {
        "series": label, "n": n, "r": r, "p_naive": p_naive,
        "ci_low": lo, "ci_high": hi,
        "spearman_rho": rho, "spearman_p": p_spearman,
        "lag1_estimate": a1, "lag1_truth": b1, "n_effective": n_eff,
        "p_adjusted": p_adjusted,
        "ci_adj_low": lo_adj, "ci_adj_high": hi_adj,
        "r_first_diff": r_diff, "p_first_diff_adjusted": p_diff_adjusted,
        "boot_ci_low": boot_lo, "boot_ci_high": boot_hi,
        "level_correlation_survives": survives,
    }


def replacement_claims(
    label: str, estimate: np.ndarray, truth: np.ndarray, error: np.ndarray
) -> dict:
    """The claims that did survive: systematic bias, and a widening gap."""
    n_below, n_above, p_sign = sign_test(error)
    n_eff, _, _ = effective_sample_size(estimate, truth)
    _, _, p_sign_discounted = sign_test(error, n_override=n_eff)

    print(f"\n  BIAS   below/above truth: {n_below}/{n_above} of {len(error)}"
          f"   mean error {error.mean():+.5f}")
    print(f"         sign test p = {p_sign:.3g}"
          f"   (discounted to n_eff={n_eff:.0f}: p = {p_sign_discounted:.3g})")

    print("  TREND  (Newey-West HAC, 4 lags)")
    out = {"series": label, "n_below": n_below, "n_above": n_above,
           "mean_error": float(error.mean()), "sign_p": p_sign,
           "sign_p_discounted": p_sign_discounted}
    for name, series in (("estimate", estimate), ("truth", truth),
                         ("gap (truth-est)", truth - estimate)):
        slope, se, p = hac_slope(series)
        low, high = slope - 1.96 * se, slope + 1.96 * se
        verdict = "excludes 0" if (low > 0 or high < 0) else "INCLUDES 0"
        print(f"         {name:<16}{slope:+.6f}/window  p = {p:.4f}"
              f"  95% CI [{low:+.6f}, {high:+.6f}]  {verdict}")
        key = name.split()[0]
        out[f"{key}_slope"] = slope
        out[f"{key}_slope_p"] = p
    return out


def main() -> None:
    scored = pd.read_csv(OUT / "estimation_error.csv")
    level_rows, replacement_rows = [], []

    for metric, method in PAIRS:
        subset = scored[
            (scored["metric"] == metric)
            & (scored["method"] == method)
            & (scored["status"] == "ok")
        ].sort_values("window_id")
        if len(subset) < 5:
            print(f"\nskipping {metric}/{method}: only {len(subset)} windows")
            continue
        label = f"{metric}  -  {method}"
        estimate = subset["estimate"].to_numpy()
        truth = subset["true_value"].to_numpy()
        level_rows.append(verify(label, estimate, truth))
        replacement_rows.append(
            replacement_claims(label, estimate, truth, subset["error"].to_numpy())
        )

    pd.DataFrame(level_rows).to_csv(OUT / "correlation_significance.csv", index=False)
    pd.DataFrame(replacement_rows).to_csv(OUT / "bias_and_trend.csv", index=False)
    print(f"\nwrote {OUT / 'correlation_significance.csv'}"
          f" and {OUT / 'bias_and_trend.csv'}")


if __name__ == "__main__":
    main()
