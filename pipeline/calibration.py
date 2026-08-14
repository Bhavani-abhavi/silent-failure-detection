"""Threshold calibration: what false-positive rate do these thresholds buy?

Two independent estimates, because neither alone is trustworthy.

INTRINSIC FPR (synthetic null)
    Split a single window at random into two halves. There is no drift by
    construction — the two halves are draws from the same distribution, the
    same month, the same population. Any alert is therefore a false positive,
    definitionally. This measures the detector's own noise floor at a given
    sample size, and it needs no assumption that any real period was stable.

REALISTIC FPR (real candidate windows)
    Run the detector across periods nominated as stable by evidence EXOGENOUS
    to the detector — macro conditions and documented company events. Alerts
    here mix genuine mild drift with false positives and cannot separate
    them, so this is an upper bound on FPR, not a measurement of it.

WHY THE CANDIDATE WINDOWS MUST BE CHOSEN BEFORE LOOKING AT DRIFT OUTPUT
    Picking the period where the detector reports least drift, then measuring
    the false-positive rate there, is circular: it selects for low readings
    and then reports low readings as a property of the thresholds. The
    resulting FPR is guaranteed optimistic and means nothing. Candidate
    windows are therefore declared in the domain adapter from external
    evidence, and ALL of them are reported — the spread across them is the
    actual result, because it quantifies how regime-dependent the choice is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from drift_core import ResultStatus
from drift_core.types import WindowSpec
from pipeline.monitor import DETECTORS


def synthetic_null_fpr(
    frame: pd.DataFrame,
    feature_names: list[str],
    *,
    n_splits: int = 50,
    methods: list[str] | None = None,
    sample_size: int | None = None,
    random_state: int = 0,
    min_samples: int = 30,
    **detector_kwargs,
) -> pd.DataFrame:
    """Split one window in half at random, repeatedly, and count alerts.

    Every alert is a false positive by construction. `sample_size` caps each
    half so the FPR can be measured as a function of window size — the
    thresholds are sample-size dependent, and a rate measured on 50,000 rows
    says nothing about behaviour on 500.
    """
    methods = methods or list(DETECTORS)
    rng = np.random.default_rng(random_state)
    rows: list[dict] = []

    for split_index in range(n_splits):
        shuffled = rng.permutation(len(frame))
        half = len(frame) // 2
        left_idx, right_idx = shuffled[:half], shuffled[half : 2 * half]
        if sample_size is not None:
            left_idx = left_idx[:sample_size]
            right_idx = right_idx[:sample_size]

        left = frame.iloc[left_idx]
        right = frame.iloc[right_idx]
        window = WindowSpec(window_id=f"null_{split_index}", n_samples=len(right))

        for feature in feature_names:
            reference_values = left[feature].to_numpy(dtype=float)
            current_values = right[feature].to_numpy(dtype=float)
            for method in methods:
                result = DETECTORS[method](
                    reference_values,
                    current_values,
                    feature_name=feature,
                    window=window,
                    min_samples=min_samples,
                    **detector_kwargs.get(method, {}),
                )
                rows.append(
                    {
                        "split": split_index,
                        "feature": feature,
                        "method": result.method,
                        "statistic": result.statistic,
                        "is_drifted": result.is_drifted,
                        "status": result.status.value,
                        "mde": result.minimum_detectable_effect,
                        "n_per_half": len(left_idx),
                    }
                )

    return pd.DataFrame(rows)


def summarise_fpr(results: pd.DataFrame, *, by: list[str] | None = None) -> pd.DataFrame:
    """False-positive rate over evaluable tests only."""
    by = by or ["method"]
    evaluable = results[results.status == ResultStatus.OK.value]
    summary = evaluable.groupby(by).agg(
        n_tests=("is_drifted", "size"),
        false_positive_rate=("is_drifted", "mean"),
        median_statistic=("statistic", "median"),
        p95_statistic=("statistic", lambda s: s.quantile(0.95)),
    )
    excluded = results[results.status != ResultStatus.OK.value]
    if len(excluded):
        counts = excluded.groupby(by).size().rename("n_not_evaluable")
        summary = summary.join(counts, how="left")
        summary["n_not_evaluable"] = summary["n_not_evaluable"].fillna(0).astype(int)
    else:
        summary["n_not_evaluable"] = 0
    return summary.round(4)


def per_feature_fpr(results: pd.DataFrame) -> pd.DataFrame:
    """FPR broken out by feature.

    Aggregate FPR hides the shape of the problem: a 6% average can be one
    pathological feature firing on every split while the rest stay silent.
    Which case it is determines whether the fix is a threshold change or
    dropping a feature from monitoring.
    """
    return summarise_fpr(results, by=["method", "feature"]).sort_values(
        "false_positive_rate", ascending=False
    )
