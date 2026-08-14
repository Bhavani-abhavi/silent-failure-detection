"""Apply the matured-vintage cut to Lending Club and report what survives.

Run: .venv/Scripts/python.exe scripts/report_maturity_cut.py

Writes reports/maturity/ and prints the summary. The point of this script is
to make the cost of the cut explicit and quotable — how many rows and how
many monitoring windows were paid for an unbiased ground truth.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from domains.finance import lending_club as lc

OUT = Path("reports/maturity")
ERA = "2013+"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    frame = lc.load(era=ERA)
    snapshot = lc.snapshot_date(frame)
    cutoff = lc.maturity_cutoff(frame)

    print(f"loaded            : {len(frame):,} rows, era {ERA}")
    print(f"issue_d range     : {frame[lc.TIME_COLUMN].min().date()}"
          f" .. {frame[lc.TIME_COLUMN].max().date()}")
    print(f"snapshot (inferred): {snapshot.date()}")
    print(f"horizon           : {lc.DEFAULT_HORIZON_MONTHS} months"
          f" + {lc.CHARGEOFF_BOOKING_LAG_MONTHS} months booking lag")
    print(f"maturity cutoff   : issue_d <= {cutoff.date()}")

    report = lc.horizon_maturity_report(frame)
    report.to_csv(OUT / "vintage_comparison.csv")
    print("\n=== PER-VINTAGE: snapshot label vs 24-month horizon label ===")
    print(report.to_string())

    matured = lc.matured_vintages(frame)
    label = lc.default_label_within_horizon(matured)

    kept_share = len(matured) / len(frame)
    quarters = matured[lc.TIME_COLUMN].dt.to_period("Q")

    print("\n=== WHAT REMAINS AFTER THE CUT ===")
    print(f"rows kept         : {len(matured):,} of {len(frame):,} ({kept_share:.1%})")
    print(f"rows discarded    : {len(frame) - len(matured):,}"
          f" (vintages after {cutoff.date()})")
    print(f"date range kept   : {matured[lc.TIME_COLUMN].min().date()}"
          f" .. {matured[lc.TIME_COLUMN].max().date()}")
    print(f"monitoring quarters: {quarters.nunique()}")
    print(f"labelled share    : {label.notna().mean():.4%}  (must be 100%)")
    print(f"default rate      : {label.mean():.4%}")
    print(f"positives         : {int(label.sum()):,}")

    per_quarter = pd.DataFrame(
        {"quarter": quarters.astype(str), "default": label}
    ).groupby("quarter").agg(n_loans=("default", "size"),
                             default_rate=("default", "mean"))
    per_quarter.to_csv(OUT / "quarterly_default_rate.csv")
    print("\n=== QUARTERLY DEFAULT RATE (the series a monitor must track) ===")
    print(per_quarter.round(4).to_string())

    # The residual the horizon rule does not fix, quantified rather than assumed.
    late_residual = int(report["still_late_in_horizon"].sum())
    print(f"\nresidual censoring: {late_residual:,} loans"
          f" ({late_residual / max(len(matured), 1):.3%} of kept rows) stopped"
          f" paying inside the horizon but are still `Late`, not yet charged"
          f" off, and are labelled 0.")


if __name__ == "__main__":
    main()
