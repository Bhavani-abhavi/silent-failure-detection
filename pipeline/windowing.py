"""Generic time-windowing shared by all domain adapters.

This lives in `pipeline`, not `drift_core`. The core takes plain arrays and an
opaque `window_id`; it has no concept of calendar time and must not acquire
one. Turning a timestamped table into ordered windows is orchestration, and
every domain needs the same version of it — so it is shared here rather than
duplicated per domain.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


_PERIOD_ALIASES = {"QS": "Q", "MS": "M", "YS": "Y", "AS": "Y", "A": "Y"}


@dataclass(frozen=True)
class WindowedPanel:
    """A reference window plus ordered monitoring windows.

    This is the single contract every domain adapter must satisfy. If a
    domain cannot express its data this way, that is worth knowing — it
    would be the first real evidence that the "one core, three domains"
    claim does not hold.
    """

    reference: pd.DataFrame
    windows: list[tuple[str, pd.DataFrame]]
    feature_names: list[str]
    reference_id: str

    def __post_init__(self) -> None:
        missing = set(self.feature_names) - set(self.reference.columns)
        if missing:
            raise ValueError(f"reference is missing declared features: {sorted(missing)}")
        for window_id, frame in self.windows:
            gap = set(self.feature_names) - set(frame.columns)
            if gap:
                raise ValueError(
                    f"window {window_id} is missing declared features: {sorted(gap)}"
                )

    def describe(self) -> pd.DataFrame:
        rows = [
            {
                "window_id": self.reference_id,
                "role": "reference",
                "n_rows": len(self.reference),
            }
        ]
        rows += [
            {"window_id": wid, "role": "monitor", "n_rows": len(frame)}
            for wid, frame in self.windows
        ]
        return pd.DataFrame(rows)


def split_time_windows(
    frame: pd.DataFrame,
    *,
    time_column: str,
    freq: str = "Q",
    reference_end: str | pd.Timestamp,
    feature_names: list[str],
    reference_start: str | pd.Timestamp | None = None,
    min_rows: int = 200,
) -> WindowedPanel:
    """Split a timestamped frame into a reference period and later windows.

    Time-based only — there is no random-split option, deliberately. A random
    split of temporal data leaks future information into the reference set and
    makes every drift measurement downstream meaningless.

    `min_rows` drops windows too small to test. Lending Club's early years
    have months with a few dozen loans; a KS test on 40 rows has almost no
    power and would contribute noise to the false-positive rate rather than
    signal.
    """
    if time_column not in frame.columns:
        raise ValueError(f"time_column {time_column!r} not in frame")

    # Period aliases are not offset aliases: pandas accepts "QS"/"MS" as
    # resample frequencies but rejects them for to_period, which wants the
    # anchorless "Q"/"M". Callers reasonably pass either.
    freq = _PERIOD_ALIASES.get(freq.upper(), freq)

    ordered = frame.sort_values(time_column)
    reference_end = pd.Timestamp(reference_end)

    in_reference = ordered[time_column] < reference_end
    if reference_start is not None:
        in_reference &= ordered[time_column] >= pd.Timestamp(reference_start)

    reference = ordered[in_reference]
    if len(reference) < min_rows:
        raise ValueError(
            f"reference period has {len(reference)} rows, below min_rows={min_rows}"
        )

    later = ordered[ordered[time_column] >= reference_end]
    windows: list[tuple[str, pd.DataFrame]] = []
    for period, group in later.groupby(later[time_column].dt.to_period(freq)):
        if len(group) >= min_rows:
            windows.append((str(period), group))

    ref_start_label = (
        pd.Timestamp(reference_start).date() if reference_start is not None
        else reference[time_column].min().date()
    )
    return WindowedPanel(
        reference=reference,
        windows=windows,
        feature_names=feature_names,
        reference_id=f"ref[{ref_start_label}..{reference_end.date()})",
    )
