"""MLflow experiment tracking for backtest runs.

The CSVs under `reports/backtest/` are the evidence for the README and stay
in git. MLflow adds what a folder of CSVs cannot: every run of the backtest
kept side by side with the exact parameters that produced it, so a change to
the onset rule, the reference window or the model can be compared against the
previous run instead of overwriting it.

What gets logged per run:
  - params:  every frozen setting of the run (era, windows, onset rule, ...)
  - metrics: per-window series logged with `step` = window ordinal, so the
             MLflow UI plots them as curves over deployment time; plus
             scalar summaries (latency per signal, mean estimation error)
  - artifacts: the report CSVs, and a window-id <-> step lookup table

mlflow is an optional extra (`pip install -e ".[tracking]"`); nothing else in
the project imports this module.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

try:
    import mlflow
except ImportError:  # pragma: no cover - exercised only without the extra
    mlflow = None  # type: ignore[assignment]

EXPERIMENT = "silent-failure-detection"


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def log_backtest_run(
    *,
    params: Mapping[str, object],
    window_metrics: pd.DataFrame,
    summary_metrics: Mapping[str, float] | None = None,
    artifacts: Iterable[Path] = (),
    run_name: str | None = None,
    experiment: str = EXPERIMENT,
    tracking_uri: str | None = None,
) -> str:
    """Log one backtest run and return its MLflow run id.

    `window_metrics` is indexed by window id in deployment order; every numeric
    column becomes a metric series. Non-finite values (a window where AUC is
    undefined, say) are skipped rather than logged as NaN, because MLflow
    plots NaN as zero and a fake zero is worse than a gap.
    """
    if mlflow is None:
        raise ImportError("mlflow is not installed. Install the optional extra: pip install -e \".[tracking]\"")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    numeric = window_metrics.select_dtypes("number")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        for step, (window_id, row) in enumerate(numeric.iterrows()):
            for name, value in row.items():
                if _finite(value):
                    mlflow.log_metric(str(name), float(value), step=step)
        for name, value in (summary_metrics or {}).items():
            if _finite(value):
                mlflow.log_metric(str(name), float(value))

        with tempfile.TemporaryDirectory() as tmp:
            lookup = Path(tmp) / "window_steps.csv"
            pd.DataFrame({"step": range(len(numeric)), "window_id": list(numeric.index)}).to_csv(
                lookup, index=False
            )
            mlflow.log_artifact(str(lookup))
        for path in artifacts:
            if Path(path).exists():
                mlflow.log_artifact(str(path), artifact_path="reports")
        return run.info.run_id
