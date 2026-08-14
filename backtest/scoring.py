"""Score label-free estimates against the truth that arrived later.

This is component B's deliverable. The estimators in `estimation/` are built
to be plausible, not to be right; the number that matters is how far off they
were once matured labels made the answer knowable.

Suppressed estimates are counted separately and never as errors. An estimator
that declines to answer when its assumptions fail is behaving correctly, and
folding those windows into a mean absolute error would punish exactly the
behaviour worth having. Coverage — the share of windows answered at all — is
reported next to accuracy so that "always right, rarely willing to speak" and
"always willing, often wrong" are distinguishable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRUTH_COLUMN = {
    "base_rate": "base_rate",
    "brier": "brier",
    "accuracy": "accuracy",
}


def score_estimates(estimates: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Per-window estimation error, long form.

    Returns one row per (window, metric, method) with the estimate, the true
    value, and signed / absolute / relative error.
    """
    rows = []
    for _, row in estimates.iterrows():
        metric = row["metric"]
        column = TRUTH_COLUMN.get(metric)
        if column is None or row["window_id"] not in truth.index:
            continue
        true_value = float(truth.loc[row["window_id"], column])
        estimate = float(row["estimate"])
        error = estimate - true_value
        rows.append(
            {
                "window_id": row["window_id"],
                "metric": metric,
                "method": row["method"],
                "estimate": estimate,
                "true_value": true_value,
                "error": error,
                "abs_error": abs(error),
                "relative_error": (
                    error / true_value if abs(true_value) > 1e-12 else np.nan
                ),
                "status": row["status"],
                "suppression_reason": row.get("suppression_reason", ""),
            }
        )
    return pd.DataFrame(rows)


def summarise_estimation_error(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (metric, method): coverage, bias, error magnitude.

    `mean_error` is kept alongside `mean_abs_error` deliberately. A signed mean
    near zero with a large absolute mean is an estimator that is noisy but
    unbiased; a signed mean equal to the absolute mean is one that is wrong in
    the same direction every single window, which is the more serious defect
    and the one that would be invisible if only magnitude were reported.
    """
    if scored.empty:
        return pd.DataFrame()

    answered = scored[scored["status"] == "ok"]
    rows = []
    for (metric, method), group in scored.groupby(["metric", "method"]):
        usable = answered[
            (answered["metric"] == metric) & (answered["method"] == method)
        ]
        rows.append(
            {
                "metric": metric,
                "method": method,
                "n_windows": len(group),
                "coverage": len(usable) / len(group) if len(group) else np.nan,
                "mean_error": usable["error"].mean(),
                "mean_abs_error": usable["abs_error"].mean(),
                "max_abs_error": usable["abs_error"].max(),
                "mean_relative_error": usable["relative_error"].mean(),
                "mean_abs_relative_error": usable["relative_error"].abs().mean(),
                "always_same_direction": (
                    bool((usable["error"] > 0).all() or (usable["error"] < 0).all())
                    if len(usable)
                    else False
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["metric", "method"]).reset_index(drop=True)


def estimator_tracked_degradation(
    scored: pd.DataFrame, metric: str, method: str
) -> dict:
    """Did the estimate move *with* the truth, or stay flat while it moved?

    The headline question for component B. An estimator can have a modest
    average error and still be useless for monitoring if it is a constant —
    what a monitor needs is for the estimate to fall when performance falls.
    Correlation across windows answers that; mean error does not.
    """
    subset = scored[
        (scored["metric"] == metric)
        & (scored["method"] == method)
        & (scored["status"] == "ok")
    ].sort_values("window_id")

    if len(subset) < 3:
        return {"metric": metric, "method": method, "n": len(subset),
                "correlation": np.nan, "estimate_range": np.nan,
                "true_range": np.nan, "range_ratio": np.nan}

    estimate_range = float(subset["estimate"].max() - subset["estimate"].min())
    true_range = float(subset["true_value"].max() - subset["true_value"].min())
    return {
        "metric": metric,
        "method": method,
        "n": len(subset),
        "correlation": float(subset["estimate"].corr(subset["true_value"])),
        "estimate_range": estimate_range,
        "true_range": true_range,
        # <<1 means the estimate barely moved while the truth did — the
        # signature of an estimator that is structurally blind rather than
        # merely imprecise.
        "range_ratio": estimate_range / true_range if true_range > 1e-12 else np.nan,
    }
