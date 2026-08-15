"""Significance testing for statistics computed over a sequence of windows.

Everything in this project's backtest produces one number per monthly window,
so every inferential claim made about those numbers is a claim about a short,
serially correlated time series. The standard tests assume independent
observations and are badly anti-conservative here.

This module exists because that mistake was actually made: a correlation of
r = −0.38 between a label-free estimate and the truth was reported as a
finding on the strength of a naive Pearson p-value of 0.023. Corrected for
serial dependence the effective sample size was 13 of 35 and p became 0.195.
See the correction entry in `docs/findings_log.md`.

TWO TESTS THAT LOOK RIGOROUS AND ARE WRONG HERE
===============================================

- **Block-bootstrapping a trend slope.** Resampling blocks and regressing them
  against a fixed time index scrambles the ordering, destroying any trend by
  construction. It returns "CI includes zero" for everything, including series
  with obvious trends. Use `hac_slope` instead. (Block bootstrap *is* valid for
  a correlation between two series, where paired blocks preserve the joint
  structure — hence `moving_block_bootstrap_correlation` below.)

- **A t-test on first differences, as a trend test.** The mean first difference
  reduces to `(last - first) / (n - 1)`, so it is an endpoint comparison
  carrying month-to-month variance. It returns p = 0.63 on a series that
  `hac_slope` resolves at p = 0.0002.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Fisher z-transform confidence interval for a correlation."""
    if n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return float("nan"), float("nan")
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    crit = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


def correlation_p_value(r: float, n: float) -> float:
    """Two-sided p for H0: rho = 0, given r and an (effective) sample size.

    `n` is a float so a serial-correlation-adjusted effective sample size can
    be passed straight in — which is the entire point of this module.
    """
    if n <= 2 or not np.isfinite(r) or abs(r) >= 1:
        return float("nan")
    t = r * np.sqrt((n - 2) / (1 - r**2))
    return float(2 * stats.t.sf(abs(t), df=n - 2))


def lag1_autocorrelation(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    centred = x - x.mean()
    denominator = float(np.sum(centred**2))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(centred[:-1] * centred[1:]) / denominator)


def effective_sample_size(x, y) -> tuple[float, float, float]:
    """Bartlett/Quenouille adjustment for correlating two autocorrelated series.

    `n_eff = n * (1 - r1*r2) / (1 + r1*r2)`, where r1 and r2 are the lag-1
    autocorrelations. When both series are persistent, n_eff collapses far
    below n; that collapse is what a naive test silently ignores.

    Returns `(n_eff, lag1_x, lag1_y)`. Clipped to [3, n]: two anti-persistent
    series can push the formula above n, which would be a claim to more
    information than was collected.
    """
    x = np.asarray(x, dtype=float)
    a1, b1 = lag1_autocorrelation(x), lag1_autocorrelation(y)
    n = len(x)
    product = a1 * b1
    if not np.isfinite(product) or product <= -1:
        return float(n), a1, b1
    return float(np.clip(n * (1 - product) / (1 + product), 3, n)), a1, b1


def hac_slope(y, lags: int = 4) -> tuple[float, float, float]:
    """OLS slope against a time index, with Newey-West standard errors.

    Returns `(slope, standard_error, p_value)`. The HAC correction is what
    makes the trend claim survivable where the correlation claim did not.
    """
    y = np.asarray(y, dtype=float)
    m = len(y)
    if m < 3:
        return float("nan"), float("nan"), float("nan")
    t = np.arange(m, dtype=float)
    X = np.column_stack([np.ones(m), t])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    u = resid[:, None] * X
    S = u.T @ u
    for lag in range(1, min(lags, m - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        G = u[lag:].T @ u[:-lag]
        S += weight * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    variance = float(cov[1, 1])
    if not np.isfinite(variance) or variance <= 0:
        return float(beta[1]), float("nan"), float("nan")
    se = float(np.sqrt(variance))
    return float(beta[1]), se, float(2 * stats.norm.sf(abs(beta[1] / se)))


def moving_block_bootstrap_correlation(
    x, y, *, block: int | None = None, n_boot: int = 10000, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI for a correlation, preserving short-range serial structure.

    Blocks of consecutive *paired* (x, y) observations are resampled with
    replacement, so local dependence survives into each replicate. An i.i.d.
    bootstrap would destroy exactly the structure that inflates the naive test.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 4:
        return float("nan"), float("nan")
    if block is None:
        block = max(2, int(round(n ** (1 / 3))))
    block = min(block, n)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))

    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        index = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        xb, yb = x[index], y[index]
        if np.std(xb) < 1e-12 or np.std(yb) < 1e-12:
            continue
        out.append(np.corrcoef(xb, yb)[0, 1])
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def sign_test(values, *, n_override: float | None = None) -> tuple[int, int, float]:
    """Two-sided sign test on the signs of `values`.

    Returns `(n_below_zero, n_above_zero, p_value)`.

    `n_override` allows the test to be re-run at a conservative effective
    sample size. A sign test assumes independent signs, and serially
    correlated errors produce runs; quoting p = 5.8e-11 for "low in 35/35
    windows" overstates it, while the same statement discounted to 13
    effectively independent observations still gives p = 2.4e-4.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n_below = int(np.sum(arr < 0))
    n_above = int(np.sum(arr > 0))
    total = n_below + n_above
    if total < 1:
        return n_below, n_above, float("nan")

    # Round the sample size FIRST, then scale the observed majority into it and
    # clamp. Scaling against the unrounded float and then taking a ceiling can
    # push the success count above the trial count, and `binom.sf(n, n, p)` is
    # exactly 0 — a p-value no finite test can produce. That is how this
    # reported "p = 0" for a 35-of-35 result whose true discounted value is
    # 2.4e-4, which would have been the fourth instance of this project's
    # recurring bug had it reached the write-up.
    n_trials = total if n_override is None else max(1, int(round(n_override)))
    observed = max(n_below, n_above)
    successes = observed if n_override is None else observed / total * n_trials
    successes = min(int(round(successes)), n_trials)

    p = float(2 * stats.binom.sf(successes - 1, n_trials, 0.5))
    return n_below, n_above, min(p, 1.0)
