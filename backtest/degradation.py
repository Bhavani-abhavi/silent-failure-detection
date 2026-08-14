"""When did true performance actually drop?

The headline number is a gap between two events, so it is only as defensible
as the later one. This module defines that event, and the definition is the
part most open to being gamed — a threshold chosen after seeing the signals
would let the latency be whatever the author wanted.

THE RULE, AND WHY IT IS THIS ONE
================================

Onset is the first window that begins a run of `persistence` consecutive
windows breaching a threshold set from the *reference period's own
variability*: `reference_mean + n_sd * reference_sd`.

Three properties this buys:

- **Calibrated to the model, not to a convention.** "Brier rose above 0.11" is
  arbitrary. "Brier left the range it occupied while nothing was wrong" is
  not, and it transfers to any model and metric without retuning.
- **Computed from healthy windows only.** The reference statistics come from
  sub-windows of the holdout period, before any monitored window is scored.
  Nothing about the monitoring period can influence where the threshold lands.
- **Persistence kills single-window noise.** One bad month is a fluctuation;
  a run is a regime. Without this, onset lands on whichever window happened to
  have the noisiest draw, and the latency measured against it is noise too.

`n_sd` and `persistence` are the two free parameters. Both are declared before
the signals are computed, and `backtest/runner.py` reports the headline under
a sweep of them rather than a single favourable setting — an onset rule that
only works at one threshold is not a finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DegradationOnset:
    """The label-confirmed event that unsupervised signals are scored against."""

    metric: str
    direction: str
    """"increase" for metrics where worse is higher (Brier, calibration gap),
    "decrease" for metrics where worse is lower (AUC)."""

    threshold: float
    reference_mean: float
    reference_sd: float
    n_sd: float
    persistence: int

    onset_window: str | None = None
    onset_index: int | None = None
    breaching_windows: list[str] = field(default_factory=list)

    @property
    def occurred(self) -> bool:
        return self.onset_index is not None

    def describe(self) -> str:
        if not self.occurred:
            return (
                f"{self.metric}: no sustained breach of {self.threshold:.4f} "
                f"({self.reference_mean:.4f} {'+' if self.direction == 'increase' else '-'} "
                f"{self.n_sd}x{self.reference_sd:.4f}) — no degradation to detect"
            )
        return (
            f"{self.metric}: onset at {self.onset_window} (index "
            f"{self.onset_index}), threshold {self.threshold:.4f}, "
            f"{len(self.breaching_windows)} windows breaching"
        )


def reference_variability(values) -> tuple[float, float]:
    """Mean and standard deviation of a metric across healthy sub-windows.

    The sub-windows must be the same size and cadence as the monitoring
    windows. A reference SD computed on six months of pooled data would be far
    too small — it would measure the variability of a large sample, not the
    variability of a monthly window, and every monitoring window would breach
    it immediately.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def find_onset(
    window_ids: list[str],
    values,
    *,
    metric: str,
    reference_mean: float,
    reference_sd: float,
    n_sd: float = 3.0,
    persistence: int = 2,
    direction: str = "increase",
) -> DegradationOnset:
    """First window beginning a sustained breach of the reference band."""
    if direction not in {"increase", "decrease"}:
        raise ValueError(f"direction must be 'increase' or 'decrease', got {direction!r}")
    if persistence < 1:
        raise ValueError(f"persistence must be >= 1, got {persistence}")

    if not np.isfinite(reference_sd) or reference_sd <= 0:
        raise ValueError(
            f"reference_sd is {reference_sd}; a threshold cannot be calibrated "
            f"without a positive reference variability. Supply at least two "
            f"healthy sub-windows of the same cadence as the monitoring windows."
        )

    threshold = (
        reference_mean + n_sd * reference_sd
        if direction == "increase"
        else reference_mean - n_sd * reference_sd
    )

    arr = np.asarray(values, dtype=float)
    breached = (
        (arr > threshold) if direction == "increase" else (arr < threshold)
    )
    # A window that could not be evaluated is not a breach, and must not be
    # allowed to break a run either — treating NaN as "fine" would let a run
    # of unevaluable windows silently satisfy the persistence requirement.
    breached = breached & np.isfinite(arr)

    onset_index = None
    for i in range(len(arr) - persistence + 1):
        if breached[i : i + persistence].all():
            onset_index = i
            break

    return DegradationOnset(
        metric=metric,
        direction=direction,
        threshold=float(threshold),
        reference_mean=float(reference_mean),
        reference_sd=float(reference_sd),
        n_sd=float(n_sd),
        persistence=int(persistence),
        onset_window=window_ids[onset_index] if onset_index is not None else None,
        onset_index=onset_index,
        breaching_windows=[
            window_ids[i] for i in range(len(arr)) if breached[i]
        ],
    )
