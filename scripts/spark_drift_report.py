"""Per-window PSI for every feature over the full Lending Club panel, with Spark.

    .venv/Scripts/python.exe scripts/spark_drift_report.py

Same windows and reference as `run_backtest.py` (reference = the 2013 H2
holdout, monthly monitoring windows from 2014-01). Writes
`reports/spark/psi_by_window.csv`, then recomputes every value with the NumPy
implementation on the same data and prints the largest disagreement, so the
Spark numbers are never trusted on their own.

Needs data/raw/loan.csv (see README) and the 'spark' extra.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from domains.finance import lending_club as lc
from drift_core.univariate import population_stability_index
from pipeline.spark_drift import spark_session, windowed_psi
from pipeline.windowing import split_time_windows

OUT = Path("reports/spark")
ERA = "2013+"
REFERENCE_START = "2013-07-01"
REFERENCE_END = "2014-01-01"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    frame = lc.load(era=ERA)
    frame = lc.matured_vintages(frame)
    frame = frame[frame[lc.TIME_COLUMN] >= pd.Timestamp(REFERENCE_START)].reset_index(drop=True)
    numeric = list(lc.numeric_features(ERA))
    categorical = list(lc.CATEGORICAL_FEATURES)
    features = numeric + categorical
    print(f"rows: {len(frame):,}  features: {len(features)}")

    spark = spark_session("sfd-psi-report")
    sdf = spark.createDataFrame(frame[[lc.TIME_COLUMN, *features]])
    started = time.perf_counter()
    psi = windowed_psi(
        sdf, time_column=lc.TIME_COLUMN, reference_start=REFERENCE_START,
        reference_end=REFERENCE_END, features=features, categorical=categorical, freq="M",
    )
    print(f"spark: {len(psi):,} (window, feature) PSI values in {time.perf_counter() - started:.1f}s")
    psi.to_csv(OUT / "psi_by_window.csv", index=False)
    spark.stop()

    # Cross-check against the NumPy reference implementation on the same data.
    panel = split_time_windows(
        frame, time_column=lc.TIME_COLUMN, freq="M", reference_start=REFERENCE_START,
        reference_end=REFERENCE_END, feature_names=features,
    )
    expected = {
        (window_id, feature): population_stability_index(
            panel.reference[feature].to_numpy(), window[feature].to_numpy(),
            categorical=feature in categorical,
        )[0]
        for window_id, window in panel.windows
        for feature in features
    }
    diffs = np.array([abs(row.psi - expected[(row.window_id, row.feature)]) for row in psi.itertuples()])
    print(f"max |spark - numpy| = {diffs.max():.2e} over {len(diffs):,} values")


if __name__ == "__main__":
    main()
