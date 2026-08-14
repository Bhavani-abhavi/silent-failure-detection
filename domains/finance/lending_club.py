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
   exposes this rather than hiding it, and
   `default_label_within_horizon` is the version anything downstream should
   actually use.
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

# --------------------------------------------------------------------------
# Label construction. These columns are post-origination and are loaded for
# one purpose only: to build and *time* the outcome. They must never appear
# in a feature list — `tests/domains/test_lending_club.py` pins that.
# --------------------------------------------------------------------------

LAST_PAYMENT_COLUMN = "last_pymnt_d"
LAST_CREDIT_PULL_COLUMN = "last_credit_pull_d"

LABEL_COLUMNS: list[str] = [
    STATUS_COLUMN,
    LAST_PAYMENT_COLUMN,
    LAST_CREDIT_PULL_COLUMN,
]

DEFAULT_HORIZON_MONTHS = 24
"""Outcome window: did the loan stop paying within 24 months of origination?

Chosen against measurement, not convention. Among charged-off loans the
median gap from origination to last payment is 14 months, and only 42.9%
stop paying inside 12 months — so a 12-month horizon would define away most
of the risk it claims to measure. 24 months captures 80.5%. Going further
(full 36-month term) captures ~100% but forces dropping every 60-month loan
and costs 70% of the rows, leaving too few quarters to measure a latency in.
"""

CHARGEOFF_BOOKING_LAG_MONTHS = 4
"""Lending Club books a charge-off at roughly 120+ days delinquent.

A loan that stops paying in month 24 therefore does not *show* as charged
off until about month 28. Without this buffer the most recent kept vintage
would be systematically under-labelled — its late defaults not yet visible —
which is the same survivorship bias the horizon exists to remove, just
pushed to the boundary instead of the tail.
"""

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
    # The cache name carries a schema version. When the loaded column set
    # changes, a stale parquet from a previous version would load fine and
    # then fail somewhere far away with a missing column — bump the version
    # instead of relying on anyone remembering to clear data/processed.
    cache_path = (
        Path("data/processed")
        / f"lending_club_{era.replace('+', 'plus')}_v2.parquet"
    )
    if cache and nrows is None and cache_path.exists():
        return pd.read_parquet(cache_path)

    columns = features + [TIME_COLUMN] + LABEL_COLUMNS
    if include_categoricals:
        columns += CATEGORICAL_FEATURES

    frame = pd.read_csv(path, usecols=columns, low_memory=False, nrows=nrows)
    for column in (TIME_COLUMN, LAST_PAYMENT_COLUMN, LAST_CREDIT_PULL_COLUMN):
        frame[column] = pd.to_datetime(
            frame[column], format="%b-%Y", errors="coerce"
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


def snapshot_date(frame: pd.DataFrame) -> pd.Timestamp:
    """The date this extract was pulled, inferred from the data itself.

    Hardcoding it would be a silent time bomb: re-run against a later
    Lending Club extract and every maturity calculation would quietly be
    computed against the wrong "now" while still producing numbers.
    """
    observed = [
        frame[column].max()
        for column in (LAST_CREDIT_PULL_COLUMN, LAST_PAYMENT_COLUMN)
        if column in frame.columns
    ]
    observed = [value for value in observed if pd.notna(value)]
    if not observed:
        raise ValueError(
            f"cannot infer snapshot date: none of {LABEL_COLUMNS} are present "
            f"and populated. Load with the label columns included."
        )
    return max(observed)


def _months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Whole calendar months from `start` to `end`.

    Both columns are month-precision in the source (`Dec-2015`), so a day-
    based difference would invent precision the data does not have.
    """
    return (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)


def maturity_cutoff(
    frame: pd.DataFrame | None = None,
    *,
    horizon_months: int = DEFAULT_HORIZON_MONTHS,
    booking_lag_months: int = CHARGEOFF_BOOKING_LAG_MONTHS,
    snapshot: str | pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Latest origination date whose `horizon_months` outcome is fully observable."""
    if snapshot is not None:
        snapshot_ts = pd.Timestamp(snapshot)
    elif frame is not None:
        snapshot_ts = snapshot_date(frame)
    else:
        raise ValueError("pass either `frame` or an explicit `snapshot`")
    return snapshot_ts - pd.DateOffset(months=horizon_months + booking_lag_months)


def default_label_within_horizon(
    frame: pd.DataFrame,
    *,
    horizon_months: int = DEFAULT_HORIZON_MONTHS,
    booking_lag_months: int = CHARGEOFF_BOOKING_LAG_MONTHS,
    snapshot: str | pd.Timestamp | None = None,
) -> pd.Series:
    """Did this loan stop paying within `horizon_months` of origination?

    1.0 = yes, 0.0 = no, NaN = the vintage is too recent to know yet.

    WHY THIS AND NOT `default_label`
    --------------------------------
    `default_label` asks "has this loan defaulted by the snapshot date",
    which is a question whose answer depends on how long the loan has been
    observed. Older vintages have had years to fail; 2018 has had months. Any
    performance metric computed that way is partly a measurement of vintage
    age, so a genuine decline can present as an improvement — the 2018
    vintage reads 14.7% against 2016's 24.3% purely because 90% of it has not
    resolved.

    A fixed horizon asks the same question of every vintage, so the answer is
    comparable across them. It also *recovers* the 919k `Current` loans as
    real negatives rather than discarding them: a loan that would have
    defaulted inside the horizon would already be charged off, so one that is
    still current has genuinely survived. Within the observable range there is
    no maturity bias left to correct — not less of it, none.

    The cost is stated plainly: this measures 24-month default, not lifetime
    default, and 19.5% of eventual charge-offs happen after month 24 and are
    labelled 0 here.

    KNOWN RESIDUAL
    --------------
    A loan that stopped paying inside the horizon but is still `Late` rather
    than `Charged Off` at snapshot is labelled 0. `horizon_maturity_report`
    counts these so the size of the compromise is visible rather than
    assumed; the booking-lag buffer exists to keep it small.
    """
    cutoff = maturity_cutoff(
        frame,
        horizon_months=horizon_months,
        booking_lag_months=booking_lag_months,
        snapshot=snapshot,
    )

    observable = frame[TIME_COLUMN] <= cutoff
    stopped_paying = _months_between(frame[TIME_COLUMN], frame[LAST_PAYMENT_COLUMN])
    bad_outcome = frame[STATUS_COLUMN].isin(RESOLVED_BAD)

    label = pd.Series(np.nan, index=frame.index, dtype=float)
    label[observable] = 0.0
    label[observable & bad_outcome & (stopped_paying <= horizon_months)] = 1.0
    # Charged off with no payment ever recorded: defaulted at month zero, which
    # is inside every horizon. Left as NaN it would silently become a negative.
    label[observable & bad_outcome & frame[LAST_PAYMENT_COLUMN].isna()] = 1.0
    return label


def matured_vintages(
    frame: pd.DataFrame,
    *,
    horizon_months: int = DEFAULT_HORIZON_MONTHS,
    booking_lag_months: int = CHARGEOFF_BOOKING_LAG_MONTHS,
    snapshot: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Rows whose horizon outcome is fully observable. The tail is cut, not weighted.

    Reweighting or modelling the censored tail was considered and rejected: it
    would put a modelling assumption underneath the ground truth that the
    headline latency number is measured against, and a contested denominator
    makes the whole result arguable.
    """
    cutoff = maturity_cutoff(
        frame,
        horizon_months=horizon_months,
        booking_lag_months=booking_lag_months,
        snapshot=snapshot,
    )
    return frame[frame[TIME_COLUMN] <= cutoff].copy()


def horizon_maturity_report(
    frame: pd.DataFrame,
    *,
    horizon_months: int = DEFAULT_HORIZON_MONTHS,
    booking_lag_months: int = CHARGEOFF_BOOKING_LAG_MONTHS,
    snapshot: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Per-vintage comparison of the snapshot label against the horizon label.

    Prints the bias next to its correction, including the residual censoring
    the horizon rule does not fix (`still_late_in_horizon`).
    """
    horizon = default_label_within_horizon(
        frame,
        horizon_months=horizon_months,
        booking_lag_months=booking_lag_months,
        snapshot=snapshot,
    )
    snapshot_label = default_label(frame)
    stopped_paying = _months_between(frame[TIME_COLUMN], frame[LAST_PAYMENT_COLUMN])
    still_late = (
        frame[STATUS_COLUMN].isin(UNRESOLVED)
        & (frame[STATUS_COLUMN] != "Current")
        & (stopped_paying <= horizon_months)
        & horizon.notna()
    )

    out = pd.DataFrame(
        {
            "year": frame[TIME_COLUMN].dt.year,
            "snapshot_label": snapshot_label,
            "horizon_label": horizon,
            "still_late": still_late,
        }
    )
    grouped = out.groupby("year").agg(
        n_loans=("snapshot_label", "size"),
        snapshot_resolved_share=("snapshot_label", lambda s: s.notna().mean()),
        snapshot_default_rate=("snapshot_label", "mean"),
        horizon_labelled_share=("horizon_label", lambda s: s.notna().mean()),
        horizon_default_rate=("horizon_label", "mean"),
        still_late_in_horizon=("still_late", "sum"),
    )
    return grouped.round(4)


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
