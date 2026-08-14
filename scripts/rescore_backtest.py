"""Re-score a completed backtest under different onset definitions.

    .venv/Scripts/python.exe scripts/rescore_backtest.py

Reads the CSVs written by `run_backtest.py`; recomputes nothing expensive.
This exists because the choice of ground-truth metric is contestable, and the
right response to a contestable choice is to show the answer under all of
them rather than defend one.

THE THREE CANDIDATE DEFINITIONS OF "TRUE PERFORMANCE DROP"

- **Brier** — the standard proper scoring rule. Its weakness on this data is
  that it conflates two things: `Brier = calibration error + irreducible
  uncertainty`, and the irreducible part rises mechanically as the base rate
  moves toward 0.5. Some of the measured Brier increase is the problem
  getting genuinely harder, not the model getting worse.
- **Calibration gap** (`mean_predicted - base_rate`) — isolates the model
  being wrong about the *level* of risk, independent of how hard the problem
  became. On a credit model this is the quantity that maps to money: pricing
  and approval cutoffs consume the probability.
- **AUC** — discrimination. Reported for completeness and because its null
  result is itself a finding.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.degradation import find_onset, reference_variability
from backtest.latency import score_signal

OUT = Path("reports/backtest")

SIGNALS = [
    "psi_fired", "ks_fired", "wasserstein_fired", "kl_fired",
    "prediction_fired", "multivariate_fired",
]

ONSET_METRICS = [
    ("brier", "increase"),
    ("calibration_gap", "decrease"),
    ("abs_calibration_gap", "increase"),
    ("auc", "decrease"),
]


def main() -> None:
    truth = pd.read_csv(OUT / "truth.csv").set_index("window_id")
    signals = pd.read_csv(OUT / "signals.csv").set_index("window_id")
    healthy = pd.read_csv(OUT / "reference_monthly.csv")

    healthy = healthy.assign(
        calibration_gap=healthy["mean_predicted"] - healthy["base_rate"],
    )
    healthy["abs_calibration_gap"] = healthy["calibration_gap"].abs()

    window_ids = list(truth.index)
    all_rows = []

    for metric, direction in ONSET_METRICS:
        if metric not in truth.columns or metric not in healthy.columns:
            print(f"skipping {metric}: not present in both tables")
            continue

        ref_mean, ref_sd = reference_variability(healthy[metric])
        print("\n" + "=" * 78)
        print(f"ONSET METRIC: {metric}  (worse = {direction})")
        print(f"  healthy reference: mean {ref_mean:+.5f}  sd {ref_sd:.5f}"
              f"  ({len(healthy)} months)")

        onset = find_onset(
            window_ids, truth[metric].to_numpy(), metric=metric,
            reference_mean=ref_mean, reference_sd=ref_sd,
            n_sd=3.0, persistence=2, direction=direction,
        )
        print(f"  threshold        : {onset.threshold:+.5f}")
        print(f"  {onset.describe()}")

        if not onset.occurred:
            print("  -> no latency computable for this metric")
            all_rows.append({"onset_metric": metric, "onset_window": None})
            continue

        print(f"\n  {'signal':<16}{'latency':>9}{'first fire':>13}"
              f"{'pre-onset rate':>16}{'total rate':>12}")
        print("  " + "-" * 66)
        for signal in SIGNALS:
            if signal not in signals.columns:
                continue
            scored = score_signal(
                signal.replace("_fired", ""), window_ids,
                signals[signal].to_numpy(dtype=bool),
                onset_index=onset.onset_index, onset_window=onset.onset_window,
                persistence=2,
            )
            latency = (
                f"{scored.latency_windows:+d}"
                if scored.latency_windows is not None
                else "—"
            )
            print(f"  {scored.signal:<16}{latency:>9}"
                  f"{str(scored.first_fire_window):>13}"
                  f"{scored.pre_onset_alert_rate:>15.1%}"
                  f"{scored.total_alert_rate:>12.1%}")
            all_rows.append(
                {
                    "onset_metric": metric,
                    "onset_window": onset.onset_window,
                    "onset_threshold": onset.threshold,
                    "signal": scored.signal,
                    "first_fire_window": scored.first_fire_window,
                    "latency_windows": scored.latency_windows,
                    "pre_onset_alert_rate": scored.pre_onset_alert_rate,
                    "total_alert_rate": scored.total_alert_rate,
                }
            )

    pd.DataFrame(all_rows).to_csv(OUT / "latency_by_onset_metric.csv", index=False)
    print(f"\nwrote {OUT / 'latency_by_onset_metric.csv'}")


if __name__ == "__main__":
    main()
