"""Run the drift core across a WindowedPanel and collect tidy results.

Orchestration only. Every statistical decision lives in `drift_core`; this
module chooses which detectors to run over which features and flattens the
results into a frame for reporting.
"""

from __future__ import annotations

import pandas as pd

from drift_core import (
    ResultStatus,
    detect_ks_drift,
    detect_psi_drift,
    detect_wasserstein_drift,
)
from drift_core.types import DriftResult, WindowSpec
from pipeline.windowing import WindowedPanel

DETECTORS = {
    "psi": detect_psi_drift,
    "ks": detect_ks_drift,
    "wasserstein": detect_wasserstein_drift,
}


def _to_row(result: DriftResult) -> dict:
    return {
        "window_id": result.window.window_id,
        "feature": result.feature_name,
        "method": result.method,
        "statistic": result.statistic,
        "p_value": result.p_value,
        "is_drifted": result.is_drifted,
        "severity": result.severity.value,
        "status": result.status.value,
        "mde": result.minimum_detectable_effect,
        "n_reference": result.n_reference,
        "n_current": result.n_current,
        "reason": result.extra.get("reason", ""),
    }


def run_univariate_sweep(
    panel: WindowedPanel,
    *,
    methods: list[str] | None = None,
    min_samples: int = 30,
    **detector_kwargs,
) -> pd.DataFrame:
    """Every (feature, window, method) combination, as a tidy frame.

    Returns results for non-OK statuses too rather than dropping them. A
    feature that could not be evaluated is a reportable event — dropping it
    would recreate, at the reporting layer, exactly the silent failure the
    validity gates were added to prevent.
    """
    methods = methods or list(DETECTORS)
    rows: list[dict] = []

    for window_id, frame in panel.windows:
        window = WindowSpec(window_id=window_id, n_samples=len(frame))
        for feature in panel.feature_names:
            reference_values = panel.reference[feature].to_numpy(dtype=float)
            current_values = frame[feature].to_numpy(dtype=float)
            for method in methods:
                result = DETECTORS[method](
                    reference_values,
                    current_values,
                    feature_name=feature,
                    window=window,
                    min_samples=min_samples,
                    **detector_kwargs.get(method, {}),
                )
                rows.append(_to_row(result))

    return pd.DataFrame(rows)


def alert_rate(results: pd.DataFrame, *, method: str | None = None) -> float:
    """Share of EVALUABLE tests that fired.

    Non-OK statuses are excluded from the denominator. Including them would
    understate the alert rate by padding it with tests that never ran — a
    feature that is 100% missing is not evidence of stability.
    """
    frame = results if method is None else results[results.method == method]
    evaluable = frame[frame.status == ResultStatus.OK.value]
    if len(evaluable) == 0:
        return float("nan")
    return float(evaluable.is_drifted.mean())


def status_breakdown(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["method", "status"])
        .size()
        .unstack(fill_value=0)
        .assign(total=lambda f: f.sum(axis=1))
    )
