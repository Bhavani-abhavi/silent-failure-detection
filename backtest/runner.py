"""Roll the frozen model forward across windows and record everything.

Order of operations matters and is enforced by structure: for each window the
unsupervised signals and the label-free estimates are computed first, from
features and predictions only, and the matured labels are touched afterwards
purely to record the truth. Nothing in the signal path can see a label.

The output is three aligned tables — signals, estimates, truth — indexed by
window. Scoring them against each other is `latency.py` and `scoring.py`'s
job, deliberately downstream of collection, so that the expensive sweep can be
run once and re-scored under different onset rules without recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from drift_core.multivariate import domain_classifier_drift
from drift_core.prediction import detect_prediction_drift
from drift_core.types import ResultStatus, Severity, WindowSpec
from drift_core.univariate import (
    detect_kl_drift,
    detect_ks_drift,
    detect_psi_drift,
    detect_wasserstein_drift,
)
from estimation.confidence import (
    estimate_accuracy_atc,
    estimate_accuracy_average_confidence,
    estimate_base_rate,
    estimate_brier,
    fit_atc_threshold,
)
from estimation.importance import fit_density_ratio, importance_weighted_estimates

UNIVARIATE_DETECTORS = {
    "psi": detect_psi_drift,
    "ks": detect_ks_drift,
    "wasserstein": detect_wasserstein_drift,
    "kl": detect_kl_drift,
}


@dataclass(frozen=True)
class WindowData:
    """One monitoring window. Labels are carried but must not be read by the
    signal path — they exist so the truth can be recorded in the same pass."""

    window_id: str
    features: pd.DataFrame
    predictions: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class BacktestResult:
    signals: pd.DataFrame
    """Per window: one boolean `<method>_fired` column per signal, plus the
    continuous statistic behind it."""

    estimates: pd.DataFrame
    truth: pd.DataFrame
    reference_metrics: dict
    feature_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    """Long-form per-feature detector output, kept so a firing signal can be
    traced to the features responsible."""


def _feature_matrix(frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    return frame[feature_names].to_numpy(dtype=float)


def _univariate_signals(
    reference: WindowData,
    window: WindowData,
    feature_names: list[str],
    spec: WindowSpec,
    detector_kwargs: dict,
) -> tuple[dict, list[dict]]:
    """Run every univariate detector over every feature; summarise per method.

    A window-level signal fires when at least one feature is `is_drifted` —
    the detector's own verdict, which already requires passing both the
    significance gate and the effect-size gate. The stricter ALERT tier is
    recorded separately rather than used as the fire condition: it is a
    severity band for triage, and using it here would mean a signal only
    counts once the drift is severe, which systematically shortens every
    measured lead time.

    The *share* of features drifting is recorded alongside the boolean,
    because "one feature moved" and "two thirds of the book moved" are
    different events that a boolean cannot distinguish.
    """
    summary: dict = {}
    rows: list[dict] = []

    for method, detector in UNIVARIATE_DETECTORS.items():
        kwargs = detector_kwargs.get(method, {})
        drifted, alerts, evaluable, statistics = 0, 0, 0, []
        for feature in feature_names:
            result = detector(
                reference.features[feature],
                window.features[feature],
                feature_name=feature,
                window=spec,
                **kwargs,
            )
            rows.append(
                {
                    "window_id": window.window_id,
                    "method": method,
                    "feature": feature,
                    "statistic": result.statistic,
                    "p_value": result.p_value,
                    "severity": result.severity.value,
                    "status": result.status.value,
                    "minimum_detectable_effect": result.minimum_detectable_effect,
                }
            )
            # Non-OK results are excluded from the denominator, not counted as
            # passes. A retired feature is not evidence of stability.
            if result.status is not ResultStatus.OK:
                continue
            evaluable += 1
            statistics.append(result.statistic)
            if result.is_drifted:
                drifted += 1
            if result.severity is Severity.ALERT:
                alerts += 1

        summary[f"{method}_drift_count"] = drifted
        summary[f"{method}_alert_count"] = alerts
        summary[f"{method}_evaluable"] = evaluable
        summary[f"{method}_drift_share"] = drifted / evaluable if evaluable else np.nan
        summary[f"{method}_alert_share"] = alerts / evaluable if evaluable else np.nan
        summary[f"{method}_max_statistic"] = max(statistics) if statistics else np.nan
        summary[f"{method}_fired"] = drifted > 0

    return summary, rows


def run_backtest(
    reference: WindowData,
    windows: list[WindowData],
    *,
    feature_names: list[str],
    reference_metrics: dict,
    detector_kwargs: dict | None = None,
    include_multivariate: bool = True,
    include_importance_weighting: bool = True,
    multivariate_permutations: int = 49,
    multivariate_max_samples: int = 4000,
    multivariate_splits: int = 3,
    multivariate_trees: int = 50,
    multivariate_auc_threshold: float = 0.60,
    random_state: int = 0,
    verbose: bool = True,
) -> BacktestResult:
    """Sweep every window, collecting signals, estimates, and truth.

    THE MULTIVARIATE DEFAULTS ARE A MEASURED COMPROMISE, NOT A GUESS.

    The permutation null costs `n_splits x (n_permutations + 1)` forest fits
    per window. At the detector's own defaults (20,000 per side, 5 folds, 99
    permutations, 100 trees) one window took over ten minutes here, so the
    35-window sweep would have run for most of a day. Timings drove these
    values down to roughly a minute per window.

    What is given up is power, and only power: subsampling to 4,000 per side
    makes the test *less* able to detect small joint shifts, so it under-fires
    rather than over-fires. `n_permutations=49` puts the p-value floor at
    0.02, still below alpha=0.05 — `require_detectable_alpha` raises if that
    ever stops being true. The effect sizes this project is looking for are
    multi-year population shifts, comfortably above the resolution left.
    """
    detector_kwargs = detector_kwargs or {}
    rng = np.random.default_rng(random_state)

    reference_matrix = _feature_matrix(reference.features, feature_names)
    atc_threshold = fit_atc_threshold(reference.predictions, reference.labels)

    signal_rows, estimate_rows, truth_rows, feature_rows = [], [], [], []

    for position, window in enumerate(windows):
        spec = WindowSpec(window_id=window.window_id, n_samples=len(window.features))
        if verbose:
            print(
                f"[{position + 1:>2}/{len(windows)}] {window.window_id}"
                f"  n={len(window.features):,}",
                flush=True,
            )

        # ---- unsupervised signals: features and predictions only ----------
        summary, rows = _univariate_signals(
            reference, window, feature_names, spec, detector_kwargs
        )
        feature_rows.extend(rows)

        prediction_result = detect_prediction_drift(
            reference.predictions, window.predictions, window=spec, method="psi"
        )
        summary["prediction_psi"] = prediction_result.statistic
        summary["prediction_psi_p"] = prediction_result.p_value
        summary["prediction_fired"] = prediction_result.is_drifted

        summary["mean_prediction"] = float(np.mean(window.predictions))

        if include_multivariate:
            current_matrix = _feature_matrix(window.features, feature_names)
            ref_sub = reference_matrix
            cur_sub = current_matrix
            if len(ref_sub) > multivariate_max_samples:
                ref_sub = ref_sub[
                    rng.choice(len(ref_sub), multivariate_max_samples, replace=False)
                ]
            if len(cur_sub) > multivariate_max_samples:
                cur_sub = cur_sub[
                    rng.choice(len(cur_sub), multivariate_max_samples, replace=False)
                ]
            mv = domain_classifier_drift(
                ref_sub, cur_sub, window=spec, feature_names=feature_names,
                classifier=RandomForestClassifier(
                    n_estimators=multivariate_trees, max_depth=6,
                    min_samples_leaf=20, n_jobs=1, random_state=random_state,
                ),
                n_splits=multivariate_splits,
                n_permutations=multivariate_permutations,
                auc_watch_threshold=multivariate_auc_threshold,
                random_state=random_state,
            )
            summary["multivariate_auc"] = mv.auc
            summary["multivariate_p"] = mv.p_value
            summary["multivariate_fired"] = mv.is_drifted

        summary["window_id"] = window.window_id
        signal_rows.append(summary)

        # ---- label-free estimates: predictions only ------------------------
        window_estimates = [
            estimate_base_rate(window.predictions, window_id=window.window_id),
            estimate_brier(window.predictions, window_id=window.window_id),
            estimate_accuracy_average_confidence(
                window.predictions, window_id=window.window_id
            ),
            estimate_accuracy_atc(
                window.predictions, atc_threshold, window_id=window.window_id
            ),
        ]

        if include_importance_weighting:
            current_matrix = _feature_matrix(window.features, feature_names)
            weights, diagnostics = fit_density_ratio(
                reference_matrix, current_matrix, random_state=random_state
            )
            window_estimates.extend(
                importance_weighted_estimates(
                    reference.labels, reference.predictions, weights, diagnostics,
                    window_id=window.window_id, n_current=len(window.features),
                )
            )

        for estimate in window_estimates:
            estimate_rows.append(
                {
                    "window_id": estimate.window_id,
                    "metric": estimate.metric,
                    "method": estimate.method,
                    "estimate": estimate.estimate,
                    "status": estimate.status.value,
                    "effective_sample_size": estimate.effective_sample_size,
                    "suppression_reason": estimate.detail.get("suppression_reason", ""),
                    "discriminator_auc": estimate.detail.get("discriminator_auc"),
                }
            )

        # ---- truth: labels touched only here, after everything above -------
        from model.baseline import evaluate  # local import keeps the boundary visible

        metrics = evaluate(window.labels, window.predictions)
        metrics["window_id"] = window.window_id
        metrics["calibration_gap"] = metrics["mean_predicted"] - metrics["base_rate"]
        metrics["abs_calibration_gap"] = abs(metrics["calibration_gap"])
        truth_rows.append(metrics)

    return BacktestResult(
        signals=pd.DataFrame(signal_rows).set_index("window_id"),
        estimates=pd.DataFrame(estimate_rows),
        truth=pd.DataFrame(truth_rows).set_index("window_id"),
        reference_metrics=dict(reference_metrics),
        feature_results=pd.DataFrame(feature_rows),
    )
