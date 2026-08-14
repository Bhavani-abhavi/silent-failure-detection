"""Tests for the Lending Club adapter.

The tests that matter most here are the leakage and label-maturity ones.
A drift-monitoring project that trains on leaked outcome columns produces
beautiful, meaningless results, and the failure is invisible — the model
just looks unusually good.
"""

import numpy as np
import pandas as pd
import pytest

from domains.finance import lending_club as lc
from domains.finance.stable_periods import (
    STRUCTURAL_BREAKS,
    candidate_stable_periods,
)

POST_ORIGINATION_COLUMNS = [
    "out_prncp", "total_pymnt", "total_rec_prncp", "total_rec_int",
    "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
    "next_pymnt_d", "last_credit_pull_d", "debt_settlement_flag",
    "settlement_amount", "hardship_flag", "hardship_amount", "loan_status",
]


class TestLeakageExclusion:
    @pytest.mark.parametrize("era", ["2007+", "2013+", "2016+"])
    def test_no_post_origination_column_is_a_feature(self, era):
        features = set(lc.numeric_features(era))
        leaked = features & set(POST_ORIGINATION_COLUMNS)
        assert not leaked, f"post-origination columns used as features: {leaked}"

    def test_recoveries_is_not_a_feature(self):
        # `recoveries` is money recovered after charge-off. It is a
        # near-perfect proxy for the target and its presence would make any
        # model look excellent while learning nothing.
        for era in ("2007+", "2013+", "2016+"):
            assert "recoveries" not in lc.numeric_features(era)

    def test_feature_lists_are_allowlists_not_blocklists(self):
        # An allowlist fails closed: a new leaky column added by the vendor
        # is excluded by default. A blocklist would silently admit it.
        for era in ("2007+", "2013+", "2016+"):
            for feature in lc.numeric_features(era):
                assert feature in (
                    lc.ALWAYS_AVAILABLE
                    + lc.AVAILABLE_FROM_2013
                    + lc.AVAILABLE_FROM_2016
                )


class TestSchemaEras:
    def test_eras_are_nested_and_grow(self):
        sets = [set(lc.numeric_features(e)) for e in ("2007+", "2013+", "2016+")]
        assert sets[0] < sets[1] < sets[2]

    def test_always_available_features_are_in_every_era(self):
        for era in ("2007+", "2013+", "2016+"):
            assert set(lc.ALWAYS_AVAILABLE) <= set(lc.numeric_features(era))

    def test_unknown_era_raises(self):
        with pytest.raises(ValueError, match="unknown era"):
            lc.numeric_features("2020+")

    def test_no_feature_appears_in_two_era_lists(self):
        combined = (
            lc.ALWAYS_AVAILABLE + lc.AVAILABLE_FROM_2013 + lc.AVAILABLE_FROM_2016
        )
        assert len(combined) == len(set(combined))


class TestDefaultLabel:
    def _frame(self, statuses):
        return pd.DataFrame(
            {
                lc.STATUS_COLUMN: statuses,
                lc.TIME_COLUMN: pd.to_datetime(["2015-01-01"] * len(statuses)),
            }
        )

    def test_unresolved_loans_are_nan_not_zero(self):
        """The single most consequential line in this adapter.

        Treating `Current` as "did not default" biases the default rate
        downward by an amount that grows with how recent the vintage is —
        which would look exactly like the portfolio improving over time.
        """
        frame = self._frame(list(lc.UNRESOLVED))
        label = lc.default_label(frame)
        assert label.isna().all()
        assert (label == 0).sum() == 0

    def test_charged_off_is_one(self):
        label = lc.default_label(self._frame(["Charged Off"]))
        assert label.iloc[0] == 1.0

    def test_fully_paid_is_zero(self):
        label = lc.default_label(self._frame(["Fully Paid"]))
        assert label.iloc[0] == 0.0

    def test_policy_flagged_statuses_are_classified(self):
        frame = self._frame(
            [
                "Does not meet the credit policy. Status:Charged Off",
                "Does not meet the credit policy. Status:Fully Paid",
            ]
        )
        label = lc.default_label(frame)
        assert label.tolist() == [1.0, 0.0]

    def test_resolved_and_unresolved_sets_are_disjoint(self):
        assert not (lc.RESOLVED_BAD & lc.RESOLVED_GOOD)
        assert not (lc.RESOLVED_BAD & lc.UNRESOLVED)
        assert not (lc.RESOLVED_GOOD & lc.UNRESOLVED)

    def test_maturity_report_exposes_declining_resolution(self):
        frame = pd.DataFrame(
            {
                lc.STATUS_COLUMN: ["Fully Paid"] * 50 + ["Current"] * 50,
                lc.TIME_COLUMN: pd.to_datetime(
                    ["2014-01-01"] * 50 + ["2018-01-01"] * 50
                ),
            }
        )
        report = lc.label_maturity_report(frame)
        assert report.loc[2014, "resolved_share"] == 1.0
        assert report.loc[2018, "resolved_share"] == 0.0


class TestHorizonLabel:
    """The fixed-horizon label is the ground truth the headline latency
    number is measured against. If it is wrong, the headline is wrong and
    nothing downstream can reveal it."""

    SNAPSHOT = "2019-03-01"

    def _frame(self, rows):
        """rows: (issue_d, status, last_pymnt_d)"""
        return pd.DataFrame(
            {
                lc.TIME_COLUMN: pd.to_datetime([r[0] for r in rows]),
                lc.STATUS_COLUMN: [r[1] for r in rows],
                lc.LAST_PAYMENT_COLUMN: pd.to_datetime([r[2] for r in rows]),
                lc.LAST_CREDIT_PULL_COLUMN: pd.to_datetime(
                    [self.SNAPSHOT] * len(rows)
                ),
            }
        )

    def _label(self, rows, **kwargs):
        return lc.default_label_within_horizon(
            self._frame(rows), snapshot=self.SNAPSHOT, **kwargs
        )

    def test_cutoff_subtracts_horizon_and_booking_lag(self):
        cutoff = lc.maturity_cutoff(snapshot=self.SNAPSHOT, horizon_months=24)
        # 2019-03 minus (24 + 4) months.
        assert cutoff == pd.Timestamp("2016-11-01")

    def test_snapshot_is_inferred_from_data_not_hardcoded(self):
        frame = self._frame([("2014-01-01", "Fully Paid", "2015-06-01")])
        assert lc.snapshot_date(frame) == pd.Timestamp(self.SNAPSHOT)

    def test_snapshot_inference_fails_loudly_without_label_columns(self):
        bare = pd.DataFrame({lc.TIME_COLUMN: pd.to_datetime(["2014-01-01"])})
        with pytest.raises(ValueError, match="cannot infer snapshot"):
            lc.snapshot_date(bare)

    def test_vintage_too_recent_is_nan(self):
        # Issued after the cutoff: its 24-month outcome has not happened yet.
        label = self._label([("2018-06-01", "Current", "2019-02-01")])
        assert label.isna().all()

    def test_current_loan_in_matured_vintage_is_a_real_negative(self):
        """The whole point of the horizon rule.

        Under the snapshot label this loan is unusable (NaN). It is in fact a
        perfectly good negative: it was issued in 2014 and had it defaulted
        within 24 months it would read `Charged Off` now, not `Current`.
        Recovering these is what removes the survivorship bias rather than
        merely trimming it.
        """
        label = self._label([("2014-01-01", "Current", "2019-02-01")])
        assert label.iloc[0] == 0.0

    def test_default_inside_horizon_is_positive(self):
        label = self._label([("2014-01-01", "Charged Off", "2014-09-01")])
        assert label.iloc[0] == 1.0

    def test_default_after_horizon_is_a_negative(self):
        """A loan that paid for 30 months and then failed did NOT default
        within 24 months. Labelling it 1 would reintroduce exactly the
        vintage-age dependence the horizon exists to remove — older vintages
        have had longer to accumulate these."""
        label = self._label([("2014-01-01", "Charged Off", "2016-07-01")])
        assert label.iloc[0] == 0.0

    def test_boundary_month_is_inside_the_horizon(self):
        label = self._label([("2014-01-01", "Charged Off", "2016-01-01")])
        assert label.iloc[0] == 1.0

    def test_charged_off_with_no_payment_ever_is_positive(self):
        # Defaulted at month zero. Left to the month arithmetic this is NaT,
        # the comparison is False, and it would silently become a negative.
        label = self._label([("2014-01-01", "Charged Off", None)])
        assert label.iloc[0] == 1.0

    def test_fully_paid_is_negative(self):
        label = self._label([("2014-01-01", "Fully Paid", "2015-03-01")])
        assert label.iloc[0] == 0.0

    def test_matured_vintages_are_fully_labelled(self):
        """No NaN may survive the cut — an unlabelled row inside the matured
        frame would be silently dropped by downstream metrics and reintroduce
        selection."""
        rows = [
            ("2014-01-01", "Current", "2019-02-01"),
            ("2014-06-01", "Charged Off", "2015-01-01"),
            ("2015-01-01", "Fully Paid", "2016-01-01"),
            ("2018-06-01", "Current", "2019-02-01"),
        ]
        frame = self._frame(rows)
        matured = lc.matured_vintages(frame, snapshot=self.SNAPSHOT)
        label = lc.default_label_within_horizon(matured, snapshot=self.SNAPSHOT)
        assert len(matured) == 3
        assert label.notna().all()

    def test_horizon_label_is_flat_where_snapshot_label_trends(self):
        """Regression test for the bias itself, not just its handling.

        Two vintages with identical true 24-month behaviour, differing only
        in how much of the later one has resolved. The snapshot label must
        show a spurious difference and the horizon label must not.
        """
        early = [("2013-01-01", "Charged Off", "2013-09-01")] * 10
        early += [("2013-01-01", "Fully Paid", "2015-06-01")] * 90
        # Same 10% 24-month default rate, but 60 of the good loans are still
        # running rather than paid off — invisible to the snapshot label.
        late = [("2016-01-01", "Charged Off", "2016-09-01")] * 10
        late += [("2016-01-01", "Fully Paid", "2018-06-01")] * 30
        late += [("2016-01-01", "Current", "2019-02-01")] * 60

        frame = self._frame(early + late)
        report = lc.horizon_maturity_report(frame, snapshot=self.SNAPSHOT)

        assert report.loc[2013, "snapshot_default_rate"] == 0.10
        assert report.loc[2016, "snapshot_default_rate"] == 0.25  # the artifact
        assert report.loc[2013, "horizon_default_rate"] == 0.10
        assert report.loc[2016, "horizon_default_rate"] == 0.10  # corrected

    def test_horizon_is_configurable_and_changes_the_answer(self):
        rows = [("2014-01-01", "Charged Off", "2015-06-01")]  # month 17
        assert self._label(rows, horizon_months=24).iloc[0] == 1.0
        assert self._label(rows, horizon_months=12).iloc[0] == 0.0


class TestLabelColumnsAreNotFeatures:
    def test_label_columns_never_appear_as_features(self):
        """`last_pymnt_d` dates the outcome, so it encodes the outcome. It is
        loaded deliberately and must stay on the label side of the wall."""
        for era in ("2007+", "2013+", "2016+"):
            overlap = set(lc.numeric_features(era)) & set(lc.LABEL_COLUMNS)
            assert not overlap, f"label columns leaked into features: {overlap}"

    def test_label_columns_are_not_categorical_features_either(self):
        assert not set(lc.CATEGORICAL_FEATURES) & set(lc.LABEL_COLUMNS)


class TestCandidateStablePeriods:
    def test_all_candidates_start_within_one_schema_era(self):
        # Candidates spanning a schema boundary would measure the vendor's
        # column additions as drift and corrupt the false-positive rate.
        for period in candidate_stable_periods():
            assert pd.Timestamp(period.start) >= pd.Timestamp("2013-01-01")

    def test_candidates_are_well_formed(self):
        for period in candidate_stable_periods():
            assert pd.Timestamp(period.start) < pd.Timestamp(period.end)
            assert period.rationale
            assert period.known_confounders, (
                f"{period.name} declares no confounders; every real window has "
                f"some, and an empty field means they were not considered"
            )

    def test_scandal_break_is_documented(self):
        assert "2016-05" in STRUCTURAL_BREAKS
        assert "resigned" in STRUCTURAL_BREAKS["2016-05"]

    def test_no_candidate_window_contains_the_2016_break(self):
        break_date = pd.Timestamp("2016-05-01")
        for period in candidate_stable_periods():
            start, end = pd.Timestamp(period.start), pd.Timestamp(period.end)
            assert not (start <= break_date < end), (
                f"{period.name} spans the May 2016 structural break"
            )

    def test_covid_break_recorded_even_though_out_of_range(self):
        # The dataset ends 2018-12, but any extension to later vintages hits
        # this first, so it is recorded rather than omitted.
        assert "2020-03" in STRUCTURAL_BREAKS
