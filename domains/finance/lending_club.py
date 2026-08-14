"""Lending Club domain adapter.

Owns everything Lending Club specific: column names, the target definition,
leakage exclusions, and the schema eras. `drift_core` sees none of it — it
receives arrays and opaque feature-name strings.

Three hazards in this dataset that any honest use has to handle:

1. LEAKAGE. Of the 145 columns, a large block is populated *after*
   origination — payments received, recoveries, hardship plans, settlement
   terms, last payment date. Several of them (`recoveries`,
   `debt_settlement_flag`) are near-perfect proxies for the target. They are
   excluded by explicit allowlist, not by blocklist: a blocklist silently
   admits any new leaky column, an allowlist fails closed.

2. SCHEMA ERAS. Feature availability changes twice. Roughly 16 bureau fields
   are 100% null before 2012 and populated from 2013; four more
   (`open_acc_6m`, `il_util`, `all_util`, `inq_last_12m`) are null before
   2015 and populated from 2016. A feature switching from absent to present
   is a vendor schema change, not drift, and monitoring across that boundary
   measures the vendor rather than the model.

3. LABEL MATURITY. 919,695 of 2.26M loans are still `Current`, concentrated
   in recent vintages. Restricting to resolved loans therefore selects, in
   late vintages, for loans that resolved *early* — early payoff or early
   default — which is a survivorship bias, not a sample. `default_label`
   exposes this rather than hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.windowing import WindowedPanel, split_time_windows

# --------------------------------------------------------------------------
# Feature allowlists, by the era in which the field becomes available.
# --------------------------------------------------------------------------

ALWAYS_AVAILABLE: list[str] = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
]
"""Populated from the 2007 vintage onward. The only set usable for a
reference window that starts before 2012."""

AVAILABLE_FROM_2013: list[str] = [
    "tot_coll_amt",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "acc_open_past_24mths",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "mo_sin_old_rev_tl_op",
    "mort_acc",
    "num_actv_bc_tl",
    "num_sats",
    "pct_tl_nvr_dlq",
    "percent_bc_gt_75",
    "tot_hi_cred_lim",
    "total_bal_ex_mort",
    "total_bc_limit",
]

AVAILABLE_FROM_2016: list[str] = [
    "open_acc_6m",
    "il_util",
    "all_util",
    "inq_last_12m",
]

CATEGORICAL_FEATURES: list[str] = [
    "term",
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
]

TIME_COLUMN = "issue_d"
STATUS_COLUMN = "loan_status"

RESOLVED_BAD = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}
RESOLVED_GOOD = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}
UNRESOLVED = {
    "Current",
    "Late (31-120 days)",
    "Late (16-30 days)",
    "In Grace Period",
}
"""The label-delay problem, present in the raw data. These loans have no
outcome yet. In production this is the majority of your live population;
here it is 41% of the dataset."""


@dataclass(frozen=True)
class SchemaEra:
    name: str
    start: str
    features: list[str]


def schema_eras() -> list[SchemaEra]:
    """Feature sets that are internally consistent over a date range.

    Monitoring must stay inside one era, or compare eras knowing that any
    drift found at the boundary is a schema change.
    """
    return [
        SchemaEra("2007+", "2007-06-01", list(ALWAYS_AVAILABLE)),
        SchemaEra("2013+", "2013-01-01", ALWAYS_AVAILABLE + AVAILABLE_FROM_2013),
        SchemaEra(
            "2016+",
            "2016-01-01",
            ALWAYS_AVAILABLE + AVAILABLE_FROM_2013 + AVAILABLE_FROM_2016,
        ),
    ]


def numeric_features(era: str = "2013+") -> list[str]:
    for candidate in schema_eras():
        if candidate.name == era:
            return list(candidate.features)
    raise ValueError(
        f"unknown era {era!r}; expected one of "
        f"{[e.name for e in schema_eras()]}"
    )


def load(
    path: str | Path = "data/raw/loan.csv",
    *,
    era: str = "2013+",
    include_categoricals: bool = True,
    nrows: int | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """Load Lending Club with an allowlist of origination-time columns only.

    Reads just the needed columns — the full 145-column frame is ~1.2 GB on
    disk and materially larger in memory. Parses in ~48 s, so the cleaned
    frame is cached as parquet: calibration sweeps are re-run often and
    paying the CSV parse each time discourages re-running them.
    """
    features = numeric_features(era)
    cache_path = (
        Path("data/processed") / f"lending_club_{era.replace('+', 'plus')}.parquet"
    )
    if cache and nrows is None and cache_path.exists():
        return pd.read_parquet(cache_path)

    columns = features + [TIME_COLUMN, STATUS_COLUMN]
    if include_categoricals:
        columns += CATEGORICAL_FEATURES

    frame = pd.read_csv(path, usecols=columns, low_memory=False, nrows=nrows)
    frame[TIME_COLUMN] = pd.to_datetime(
        frame[TIME_COLUMN], format="%b-%Y", errors="coerce"
    )
    frame = frame.dropna(subset=[TIME_COLUMN])

    # int_rate and revol_util arrive as strings with a percent sign in some
    # vintages and as floats in others.
    for column in ("int_rate", "revol_util"):
        if column in frame.columns and frame[column].dtype == object:
            frame[column] = pd.to_numeric(
                frame[column].astype(str).str.rstrip("%").str.strip(),
                errors="coerce",
            )

    if "term" in frame.columns and frame["term"].dtype == object:
        frame["term"] = pd.to_numeric(
            frame["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        )

    frame = frame.sort_values(TIME_COLUMN).reset_index(drop=True)

    if cache and nrows is None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path, index=False)

    return frame


def default_label(frame: pd.DataFrame) -> pd.Series:
    """Binary default label; NaN where the outcome has not arrived.

    NaN rather than 0 for unresolved loans. Treating "no outcome yet" as
    "did not default" is the single most common way this dataset is misused,
    and it biases default rates downward by an amount that grows with how
    recent the vintage is — which would look exactly like a favourable trend.
    """
    status = frame[STATUS_COLUMN]
    label = pd.Series(np.nan, index=frame.index, dtype=float)
    label[status.isin(RESOLVED_BAD)] = 1.0
    label[status.isin(RESOLVED_GOOD)] = 0.0
    return label


def label_maturity_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Resolved-label share by vintage year.

    Exists to make the survivorship bias visible before anyone computes an
    AUC on a recent vintage. A 2018 vintage with 30% resolved labels is not
    a 30% sample of that vintage — it is the subset that resolved fastest.
    """
    label = default_label(frame)
    out = pd.DataFrame(
        {
            "year": frame[TIME_COLUMN].dt.year,
            "resolved": label.notna(),
            "default": label,
        }
    )
    grouped = out.groupby("year").agg(
        n_loans=("resolved", "size"),
        resolved_share=("resolved", "mean"),
        default_rate_among_resolved=("default", "mean"),
    )
    return grouped.round(4)


def build_windows(
    frame: pd.DataFrame,
    *,
    reference_end: str,
    reference_start: str | None = None,
    era: str = "2013+",
    freq: str = "Q",
    min_rows: int = 200,
) -> WindowedPanel:
    """Reference period plus later monitoring windows, time-ordered."""
    return split_time_windows(
        frame,
        time_column=TIME_COLUMN,
        freq=freq,
        reference_start=reference_start,
        reference_end=reference_end,
        feature_names=numeric_features(era),
        min_rows=min_rows,
    )
