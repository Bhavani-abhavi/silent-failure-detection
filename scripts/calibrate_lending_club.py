"""Threshold calibration on Lending Club.

Produces two numbers:
  1. INTRINSIC FPR   — synthetic null, no drift by construction
  2. REALISTIC FPR   — across all exogenously-nominated candidate windows

Run:  ./.venv/Scripts/python.exe scripts/calibrate_lending_club.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from domains.finance import lending_club as lc
from domains.finance.stable_periods import candidate_stable_periods
from pipeline.calibration import per_feature_fpr, summarise_fpr, synthetic_null_fpr

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

ERA = "2013+"
OUT_DIR = Path("reports/calibration")


def main(parts: set[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features = lc.numeric_features(ERA)
    print(f"Loading Lending Club (era {ERA}, {len(features)} numeric features)...")
    frame = lc.load(era=ERA)
    frame = frame[frame[lc.TIME_COLUMN] >= "2013-01-01"]
    print(f"  {len(frame):,} rows, {frame[lc.TIME_COLUMN].min().date()} "
          f"-> {frame[lc.TIME_COLUMN].max().date()}\n")

    # ---------------------------------------------------------------- 1
    # INTRINSIC FPR — synthetic null.
    # Sweep sample size: thresholds are sample-size dependent, so a single
    # FPR number without a stated n is not interpretable.
    # ----------------------------------------------------------------
    if "intrinsic" not in parts:
        print("(skipping part 1)\n")
        _realistic(frame, features)
        return

    print("=" * 78)
    print("1. INTRINSIC FPR  (synthetic null: one window split in half at random)")
    print("=" * 78)

    pool = frame[frame[lc.TIME_COLUMN].between("2014-01-01", "2014-12-31")]
    print(f"pool: 2014 vintage, {len(pool):,} rows\n")

    intrinsic_frames = []
    for n_per_half in (250, 1000, 5000, 20000):
        results = synthetic_null_fpr(
            pool, features, n_splits=40, sample_size=n_per_half, random_state=0
        )
        summary = summarise_fpr(results).assign(n_per_half=n_per_half)
        intrinsic_frames.append(summary.reset_index())
        print(f"--- n = {n_per_half:,} per half ---")
        print(summary[["n_tests", "false_positive_rate", "median_statistic",
                       "p95_statistic"]].to_string())
        print()

    intrinsic = pd.concat(intrinsic_frames, ignore_index=True)
    intrinsic.to_csv(OUT_DIR / "intrinsic_fpr.csv", index=False)

    print("--- worst features at n=20,000 (synthetic null) ---")
    worst = synthetic_null_fpr(
        pool, features, n_splits=40, sample_size=20000, random_state=0
    )
    print(per_feature_fpr(worst).head(12).to_string())
    per_feature_fpr(worst).to_csv(OUT_DIR / "intrinsic_fpr_by_feature.csv")
    print()

    _realistic(frame, features)


def _realistic(frame: pd.DataFrame, features: list[str]) -> None:
    # ---------------------------------------------------------------- 2
    # REALISTIC FPR — every exogenously-nominated candidate window.
    # ----------------------------------------------------------------
    print("=" * 78)
    print("2. REALISTIC FPR  (all exogenously-nominated candidate windows)")
    print("=" * 78)

    rows = []
    for period in candidate_stable_periods():
        subset = frame[
            frame[lc.TIME_COLUMN].between(period.start, period.end, inclusive="left")
        ]
        if len(subset) < 1000:
            print(f"{period.name}: only {len(subset)} rows, skipped")
            continue

        # Quarterly windows inside the candidate period, first quarter as
        # reference. Any alert in a period with no exogenous shock is treated
        # as a false positive -- an UPPER BOUND, since genuine mild drift is
        # not separable from noise here.
        panel = lc.build_windows(
            subset,
            reference_end=(
                pd.Timestamp(period.start) + pd.DateOffset(months=3)
            ).strftime("%Y-%m-%d"),
            reference_start=period.start,
            era=ERA,
            freq="Q",
            min_rows=200,
        )
        from pipeline.monitor import run_univariate_sweep

        results = run_univariate_sweep(panel, min_samples=30)
        summary = summarise_fpr(results.rename(columns={"is_drifted": "is_drifted"}))
        for method, row in summary.iterrows():
            rows.append(
                {
                    "period": period.name,
                    "method": method,
                    "n_windows": len(panel.windows),
                    "n_reference_rows": len(panel.reference),
                    "n_tests": int(row.n_tests),
                    "alert_rate": row.false_positive_rate,
                    "n_not_evaluable": int(row.n_not_evaluable),
                }
            )
        print(f"{period.name}: {len(panel.windows)} windows, "
              f"ref n={len(panel.reference):,}")

    realistic = pd.DataFrame(rows)
    realistic.to_csv(OUT_DIR / "realistic_fpr.csv", index=False)
    print()
    print(realistic.pivot(index="period", columns="method",
                          values="alert_rate").round(3).to_string())
    print()
    print("--- spread across candidate windows (the actual result) ---")
    spread = realistic.groupby("method").alert_rate.agg(["min", "max", "median"])
    spread["range"] = spread["max"] - spread["min"]
    print(spread.round(3).to_string())
    spread.to_csv(OUT_DIR / "fpr_spread.csv")


if __name__ == "__main__":
    import sys

    requested = set(sys.argv[1:]) or {"intrinsic", "realistic"}
    main(requested)
