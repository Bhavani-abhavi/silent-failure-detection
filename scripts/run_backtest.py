"""Produce the headline: detection latency per unsupervised signal.

    .venv/Scripts/python.exe scripts/run_backtest.py

Design decisions worth stating, because they constrain what the number means:

REFERENCE = THE HOLDOUT, NOT THE TRAINING WINDOW. Drift, the ATC threshold,
and the importance weights are all anchored on 2013 H2, which the model never
trained on. Anchoring them on the training window would make the reference
predictions in-sample: reference Brier would be optimistic, the ATC threshold
would be fitted to memorised rows, and every monitored window would look
degraded by comparison to a baseline that never existed in production.

ONSET IS CALIBRATED ON HEALTHY MONTHS ONLY, and its two free parameters
(`n_sd`, `persistence`) are swept rather than chosen. A latency that survives
only one threshold setting is not a finding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.degradation import find_onset, reference_variability
from backtest.latency import false_positive_rate, score_signal
from backtest.runner import WindowData, run_backtest
from backtest.scoring import (
    estimator_tracked_degradation,
    score_estimates,
    summarise_estimation_error,
)
from domains.finance import lending_club as lc
from model.baseline import evaluate, train_baseline

OUT = Path("reports/backtest")
ERA = "2013+"
ERA_START = "2013-01-01"
TRAIN_END = "2013-07-01"
REFERENCE_END = "2014-01-01"
FREQ = "M"

SIGNALS = [
    "psi_fired", "ks_fired", "wasserstein_fired", "kl_fired",
    "prediction_fired", "multivariate_fired",
]

# Declared before any signal is computed.
ONSET_METRIC = "brier"
ONSET_DIRECTION = "increase"
ONSET_N_SD = 3.0
ONSET_PERSISTENCE = 2


def _monthly(frame, labels, predictions, time_column):
    windows = []
    for period, index in frame.groupby(frame[time_column].dt.to_period(FREQ)).groups.items():
        rows = frame.loc[index]
        positions = frame.index.get_indexer(index)
        windows.append((str(period), rows, labels.loc[index], predictions[positions]))
    return windows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    frame = lc.load(era=ERA)
    frame = lc.matured_vintages(frame)
    frame = frame[frame[lc.TIME_COLUMN] >= pd.Timestamp(ERA_START)].reset_index(drop=True)
    labels = lc.default_label_within_horizon(frame)
    numeric = lc.numeric_features(ERA)
    categorical = list(lc.CATEGORICAL_FEATURES)

    time = frame[lc.TIME_COLUMN]
    is_train = time < pd.Timestamp(TRAIN_END)
    is_reference = (time >= pd.Timestamp(TRAIN_END)) & (time < pd.Timestamp(REFERENCE_END))
    is_monitor = time >= pd.Timestamp(REFERENCE_END)

    print(f"train     : {int(is_train.sum()):,}")
    print(f"reference : {int(is_reference.sum()):,}  (holdout, never trained on)")
    print(f"monitor   : {int(is_monitor.sum()):,}")

    model = train_baseline(
        frame[is_train], frame[is_reference], labels[is_train], labels[is_reference],
        numeric_features=numeric, categorical_features=categorical,
        trained_on="2013-01..2013-06",
    )
    print(f"reference metrics: {model.reference_metrics}")

    reference_frame = frame[is_reference].copy()
    reference_labels = labels[is_reference]
    reference_predictions = model.predict_proba(reference_frame)

    # --- reference variability, from healthy monthly sub-windows only ------
    healthy = []
    for period, index in reference_frame.groupby(
        reference_frame[lc.TIME_COLUMN].dt.to_period(FREQ)
    ).groups.items():
        positions = reference_frame.index.get_indexer(index)
        healthy.append(
            evaluate(reference_labels.loc[index], reference_predictions[positions])
        )
    healthy = pd.DataFrame(healthy)
    healthy.to_csv(OUT / "reference_monthly.csv", index=False)
    print(f"\nhealthy reference months ({len(healthy)}):")
    print(healthy[["n", "auc", "brier", "base_rate", "mean_predicted"]].round(4).to_string())

    # --- monitoring windows ------------------------------------------------
    monitor_frame = frame[is_monitor].copy()
    monitor_labels = labels[is_monitor]
    monitor_predictions = model.predict_proba(monitor_frame)

    reference_window = WindowData(
        window_id="2013H2", features=reference_frame,
        predictions=reference_predictions,
        labels=reference_labels.to_numpy(dtype=float),
    )
    windows = [
        WindowData(
            window_id=window_id, features=rows,
            predictions=predictions, labels=window_labels.to_numpy(dtype=float),
        )
        for window_id, rows, window_labels, predictions in _monthly(
            monitor_frame, monitor_labels, monitor_predictions, lc.TIME_COLUMN
        )
    ]
    print(f"\nmonitoring {len(windows)} windows\n")

    result = run_backtest(
        reference_window, windows,
        feature_names=numeric,
        reference_metrics=model.reference_metrics,
    )
    result.signals.to_csv(OUT / "signals.csv")
    result.truth.to_csv(OUT / "truth.csv")
    result.estimates.to_csv(OUT / "estimates.csv", index=False)
    result.feature_results.to_csv(OUT / "feature_results.csv", index=False)

    window_ids = list(result.truth.index)

    # --- B: estimation error ----------------------------------------------
    scored = score_estimates(result.estimates, result.truth)
    scored.to_csv(OUT / "estimation_error.csv", index=False)
    summary = summarise_estimation_error(scored)
    summary.to_csv(OUT / "estimation_error_summary.csv", index=False)

    print("=" * 78)
    print("COMPONENT B — LABEL-FREE ESTIMATION ERROR vs MATURED LABELS")
    print("=" * 78)
    print(summary.round(4).to_string(index=False))

    print("\n--- did the estimate MOVE with the truth? ---")
    tracking = pd.DataFrame([
        estimator_tracked_degradation(scored, metric, method)
        for metric in ("base_rate", "brier", "accuracy")
        for method in ("average_confidence", "atc", "importance_weighted")
    ])
    tracking = tracking.dropna(subset=["correlation"], how="all")
    tracking.to_csv(OUT / "estimator_tracking.csv", index=False)
    print(tracking.round(4).to_string(index=False))

    # --- C: onset + latency ------------------------------------------------
    ref_mean, ref_sd = reference_variability(healthy[ONSET_METRIC])
    onset = find_onset(
        window_ids, result.truth[ONSET_METRIC].to_numpy(),
        metric=ONSET_METRIC, reference_mean=ref_mean, reference_sd=ref_sd,
        n_sd=ONSET_N_SD, persistence=ONSET_PERSISTENCE, direction=ONSET_DIRECTION,
    )

    print("\n" + "=" * 78)
    print("COMPONENT C — DETECTION LATENCY")
    print("=" * 78)
    print(f"onset rule : {ONSET_METRIC} {ONSET_DIRECTION}, "
          f"{ONSET_N_SD} SD over {ONSET_PERSISTENCE} consecutive windows")
    print(f"reference  : mean {ref_mean:.5f}, sd {ref_sd:.5f} "
          f"({len(healthy)} healthy months)")
    print(f"threshold  : {onset.threshold:.5f}")
    print(onset.describe())

    latencies = [
        score_signal(
            signal.replace("_fired", ""), window_ids,
            result.signals[signal].to_numpy(dtype=bool),
            onset_index=onset.onset_index, onset_window=onset.onset_window,
            persistence=ONSET_PERSISTENCE,
        )
        for signal in SIGNALS if signal in result.signals.columns
    ]
    print("\nsignal                       latency   first fire   pre-onset alert rate")
    print("-" * 78)
    for latency in latencies:
        print(latency.describe())

    pd.DataFrame([vars(latency) for latency in latencies]).to_csv(
        OUT / "latency.csv", index=False
    )

    # --- sensitivity: does the headline survive the free parameters? -------
    print("\n--- onset sensitivity (latency of the best-leading signal) ---")
    rows = []
    for n_sd in (2.0, 2.5, 3.0, 4.0):
        for persistence in (1, 2, 3):
            candidate = find_onset(
                window_ids, result.truth[ONSET_METRIC].to_numpy(),
                metric=ONSET_METRIC, reference_mean=ref_mean, reference_sd=ref_sd,
                n_sd=n_sd, persistence=persistence, direction=ONSET_DIRECTION,
            )
            row = {"n_sd": n_sd, "persistence": persistence,
                   "onset_window": candidate.onset_window}
            for signal in SIGNALS:
                if signal not in result.signals.columns:
                    continue
                scored_signal = score_signal(
                    signal.replace("_fired", ""), window_ids,
                    result.signals[signal].to_numpy(dtype=bool),
                    onset_index=candidate.onset_index,
                    onset_window=candidate.onset_window,
                    persistence=persistence,
                )
                row[signal.replace("_fired", "")] = scored_signal.latency_windows
            rows.append(row)
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(OUT / "onset_sensitivity.csv", index=False)
    print(sensitivity.to_string(index=False))

    # --- false-positive rate on a genuine null -----------------------------
    print("\n--- FPR on a synthetic null (random splits of the healthy period) ---")
    fpr = _null_alert_rates(reference_frame, numeric, n_splits=20)
    fpr.to_csv(OUT / "null_fpr.csv", index=False)
    print(fpr.round(4).to_string(index=False))

    print("\n--- alert rate on real pre-onset windows (NOT an FPR) ---")
    for signal in SIGNALS:
        if signal not in result.signals.columns:
            continue
        flags = result.signals[signal].to_numpy(dtype=bool)
        pre = flags[: onset.onset_index] if onset.onset_index else flags
        print(f"  {signal.replace('_fired', ''):<16}"
              f" {false_positive_rate(pre):6.1%}  ({len(pre)} windows)")


def _null_alert_rates(reference_frame, feature_names, *, n_splits=20):
    """Alert rate when both windows are drawn from the same healthy period.

    A true null: any alert here is a false positive by construction. This is
    the clean FPR. The alert rate on real pre-onset windows is a different
    quantity and is reported separately — those windows are the run-up to a
    genuine event, so counting every alert there as a false alarm would
    penalise exactly the early warning the project is looking for.
    """
    from drift_core.types import WindowSpec
    from drift_core.univariate import (
        detect_kl_drift, detect_ks_drift, detect_psi_drift, detect_wasserstein_drift,
    )

    detectors = {
        "psi": detect_psi_drift, "ks": detect_ks_drift,
        "wasserstein": detect_wasserstein_drift, "kl": detect_kl_drift,
    }
    rng = np.random.default_rng(0)
    counts = {method: [0, 0] for method in detectors}

    for split in range(n_splits):
        order = rng.permutation(len(reference_frame))
        half = len(order) // 2
        left = reference_frame.iloc[order[:half]]
        right = reference_frame.iloc[order[half:]]
        spec = WindowSpec(window_id=f"null{split}", n_samples=half)
        for method, detector in detectors.items():
            for feature in feature_names:
                result = detector(
                    left[feature], right[feature],
                    feature_name=feature, window=spec,
                )
                if result.status.value != "ok":
                    continue
                counts[method][1] += 1
                # `is_drifted`, matching the fire condition used for the
                # monitoring signals. Measuring FPR on a stricter rule than
                # the one that actually fires would understate it.
                if result.is_drifted:
                    counts[method][0] += 1

    return pd.DataFrame([
        {
            "method": method,
            "alerts": alerts,
            "evaluable": total,
            "false_positive_rate": alerts / total if total else np.nan,
        }
        for method, (alerts, total) in counts.items()
    ])


if __name__ == "__main__":
    main()
