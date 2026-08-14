"""Train the baseline model and measure its TRUE performance over time.

This is the ground truth the headline latency number is measured against, and
it is the project's main risk: if the model does not measurably degrade, there
is no "true performance drop" for an unsupervised signal to anticipate, and
the headline result cannot exist in the form the project assumes.

Run before building anything on top of it:
    .venv/Scripts/python.exe scripts/measure_true_performance.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from domains.finance import lending_club as lc
from model.baseline import evaluate, train_baseline

OUT = Path("reports/truth")
ERA = "2013+"

# Schema era 2013+ starts here; earlier rows have the bureau block 100% null.
ERA_START = "2013-01-01"
TRAIN_END = "2013-07-01"      # train on 2013 H1
HOLDOUT_END = "2014-01-01"    # reference holdout: 2013 H2, never trained on
FREQ = "M"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    frame = lc.load(era=ERA)
    frame = lc.matured_vintages(frame)
    frame = frame[frame[lc.TIME_COLUMN] >= pd.Timestamp(ERA_START)]
    frame = frame.reset_index(drop=True)
    labels = lc.default_label_within_horizon(frame)

    numeric = lc.numeric_features(ERA)
    categorical = list(lc.CATEGORICAL_FEATURES)

    time = frame[lc.TIME_COLUMN]
    is_train = time < pd.Timestamp(TRAIN_END)
    is_holdout = (time >= pd.Timestamp(TRAIN_END)) & (time < pd.Timestamp(HOLDOUT_END))
    is_monitor = time >= pd.Timestamp(HOLDOUT_END)

    print(f"era {ERA}, matured only: {len(frame):,} rows"
          f"  {time.min().date()} .. {time.max().date()}")
    print(f"  train    2013-01 .. 2013-06 : {int(is_train.sum()):,}")
    print(f"  holdout  2013-07 .. 2013-12 : {int(is_holdout.sum()):,}")
    print(f"  monitor  2014-01 ..         : {int(is_monitor.sum()):,}")
    print(f"  features : {len(numeric)} numeric + {len(categorical)} categorical")

    model = train_baseline(
        frame[is_train], frame[is_holdout],
        labels[is_train], labels[is_holdout],
        numeric_features=numeric, categorical_features=categorical,
        trained_on="2013-01..2013-06",
    )
    ref = model.reference_metrics
    print(f"\nreference (2013 H2 holdout): AUC {ref['auc']:.4f}"
          f"  Brier {ref['brier']:.4f}  base rate {ref['base_rate']:.4f}"
          f"  mean pred {ref['mean_predicted']:.4f}")

    monitor = frame[is_monitor]
    monitor_labels = labels[is_monitor]
    scores = model.predict_proba(monitor)

    rows = []
    period = monitor[lc.TIME_COLUMN].dt.to_period(FREQ)
    for window_id, index in monitor.groupby(period).groups.items():
        metrics = evaluate(monitor_labels.loc[index], scores[monitor.index.get_indexer(index)])
        metrics["window_id"] = str(window_id)
        metrics["auc_delta"] = metrics["auc"] - ref["auc"]
        rows.append(metrics)

    truth = pd.DataFrame(rows).set_index("window_id")
    truth = truth[["n", "auc", "auc_delta", "brier", "base_rate", "mean_predicted"]]
    truth.to_csv(OUT / f"true_performance_{FREQ}.csv")

    print(f"\n=== TRUE PERFORMANCE PER WINDOW (freq={FREQ}) ===")
    print(truth.round(4).to_string())

    print("\n=== SUMMARY ===")
    print(f"windows            : {len(truth)}")
    print(f"reference AUC      : {ref['auc']:.4f}")
    print(f"final-window AUC   : {truth['auc'].iloc[-1]:.4f}")
    print(f"worst-window AUC   : {truth['auc'].min():.4f}"
          f"  ({truth['auc'].idxmin()})")
    print(f"total AUC drift    : {truth['auc'].iloc[-1] - ref['auc']:+.4f}")
    print(f"base rate drift    : {truth['base_rate'].iloc[-1] - ref['base_rate']:+.4f}")
    print(f"Brier drift        : {truth['brier'].iloc[-1] - ref['brier']:+.4f}")

    first_half = truth["auc"].iloc[: len(truth) // 2].mean()
    second_half = truth["auc"].iloc[len(truth) // 2 :].mean()
    print(f"mean AUC 1st half  : {first_half:.4f}")
    print(f"mean AUC 2nd half  : {second_half:.4f}")
    print(f"half-over-half     : {second_half - first_half:+.4f}")


if __name__ == "__main__":
    main()
