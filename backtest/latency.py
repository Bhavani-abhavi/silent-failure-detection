"""Detection latency, and the false-positive rate that makes it meaningful.

Latency alone is trivially gameable: a detector that fires on every window has
maximum lead time on every event and is worthless. The two numbers only mean
something together, and this module refuses to produce one without the other.

WHAT "LATENCY" MEANS HERE
=========================

`latency = onset_index - first_sustained_fire_index`, in windows.

- **positive** — the signal fired *before* the label-confirmed drop. This is
  the useful case and the number the project exists to report.
- **zero** — fired in the same window. No warning, but no lag either.
- **negative** — fired after. The signal is a lagging indicator, which is
  worth reporting plainly rather than hiding: labels would have arrived first.
- **None** — never fired, or nothing to detect. Distinct from zero, and
  conflating them would let a detector that never fires look punctual.

THE AMBIGUITY THAT CANNOT BE MEASURED AWAY
==========================================

A signal firing 8 windows before onset is either an excellent early warning or
an alarm that was already ringing for unrelated reasons. Nothing in the
retrospective data distinguishes them, and the honest response is to report
both readings rather than pick the flattering one:

- `latency_windows` — the optimistic reading (it fired first, so it led).
- `pre_onset_alert_rate` — what share of pre-onset windows it was firing on.
  A signal firing on 90% of pre-onset windows did not predict the onset; it
  was on the whole time and happened to be on when the onset arrived.

A long latency paired with a high pre-onset alert rate is a false alarm that
got lucky. `backtest/runner.py` reports them side by side for this reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignalLatency:
    signal: str
    first_fire_window: str | None
    first_fire_index: int | None
    onset_window: str | None
    onset_index: int | None
    latency_windows: int | None
    pre_onset_alert_rate: float
    """Share of strictly-pre-onset windows on which this signal fired. The
    discount factor on `latency_windows`."""

    total_alert_rate: float
    n_windows: int
    persistence: int

    @property
    def led_the_drop(self) -> bool:
        return self.latency_windows is not None and self.latency_windows > 0

    def describe(self) -> str:
        if self.latency_windows is None:
            reason = (
                "never fired" if self.first_fire_index is None else "no onset to score"
            )
            return f"{self.signal:<28} —      ({reason})"
        return (
            f"{self.signal:<28} {self.latency_windows:+3d} windows"
            f"   first fire {self.first_fire_window}"
            f"   pre-onset alert rate {self.pre_onset_alert_rate:5.1%}"
        )


def first_sustained_fire(
    fired, *, persistence: int = 2
) -> int | None:
    """Index of the first window beginning a run of `persistence` fires.

    The same persistence rule the onset definition uses. Applying a run
    requirement to the ground-truth event but not to the signal would hand the
    signal free lead time — it could fire on a single noisy window years early
    and be credited with the detection.
    """
    flags = np.asarray(fired, dtype=bool)
    if persistence < 1:
        raise ValueError(f"persistence must be >= 1, got {persistence}")
    for i in range(len(flags) - persistence + 1):
        if flags[i : i + persistence].all():
            return i
    return None


def score_signal(
    signal: str,
    window_ids: list[str],
    fired,
    *,
    onset_index: int | None,
    onset_window: str | None,
    persistence: int = 2,
) -> SignalLatency:
    """Latency and alert rates for one signal against one onset event."""
    flags = np.asarray(fired, dtype=bool)
    if len(flags) != len(window_ids):
        raise ValueError(
            f"{signal}: {len(flags)} fire flags for {len(window_ids)} windows"
        )

    fire_index = first_sustained_fire(flags, persistence=persistence)

    latency = None
    if fire_index is not None and onset_index is not None:
        latency = int(onset_index - fire_index)

    pre_onset = flags[:onset_index] if onset_index else np.array([], dtype=bool)

    return SignalLatency(
        signal=signal,
        first_fire_window=window_ids[fire_index] if fire_index is not None else None,
        first_fire_index=fire_index,
        onset_window=onset_window,
        onset_index=onset_index,
        latency_windows=latency,
        pre_onset_alert_rate=float(np.mean(pre_onset)) if len(pre_onset) else 0.0,
        total_alert_rate=float(np.mean(flags)) if len(flags) else 0.0,
        n_windows=len(flags),
        persistence=persistence,
    )


def false_positive_rate(fired) -> float:
    """Alert rate over windows known to contain no degradation.

    Meaningful only when every window passed in is genuinely stable. On this
    project the clean version comes from splitting the healthy reference
    period against itself — the pre-onset monitoring windows are *not* a
    stable period, they are the run-up to a real event, and quoting an alert
    rate over them as an FPR would count every genuine early warning as a
    false alarm.
    """
    flags = np.asarray(fired, dtype=bool)
    if len(flags) == 0:
        return float("nan")
    return float(np.mean(flags))
