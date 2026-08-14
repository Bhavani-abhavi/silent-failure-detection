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
