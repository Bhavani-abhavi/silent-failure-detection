"""Tests for the onset definition.

The headline is a gap between two events. This module defines the later one,
so a bug here moves the headline number directly and nothing downstream would
contradict it.
"""

import numpy as np
import pytest

from backtest.degradation import DegradationOnset, find_onset, reference_variability

WINDOWS = [f"2014-{m:02d}" for m in range(1, 13)]


class TestReferenceVariability:
    def test_mean_and_sd_of_healthy_windows(self):
        mean, sd = reference_variability([0.10, 0.11, 0.09, 0.10, 0.11])
        assert mean == pytest.approx(0.102, abs=1e-3)
        assert sd > 0

    def test_single_window_cannot_give_a_variability(self):
        mean, sd = reference_variability([0.1])
        assert np.isnan(sd)

    def test_nan_windows_are_dropped(self):
        mean, _ = reference_variability([0.1, np.nan, 0.2])
        assert mean == pytest.approx(0.15)


class TestOnsetRequiresCalibration:
    def test_zero_reference_sd_raises(self):
        """Without a positive reference SD the threshold is just the mean, so
        the first window above average would count as degradation. Failing
        loudly beats producing a latency number from a meaningless onset."""
        with pytest.raises(ValueError, match="reference_sd"):
            find_onset(
                WINDOWS, np.full(12, 0.1), metric="brier",
                reference_mean=0.1, reference_sd=0.0,
            )

    def test_nan_reference_sd_raises(self):
        with pytest.raises(ValueError, match="reference_sd"):
            find_onset(
                WINDOWS, np.full(12, 0.1), metric="brier",
                reference_mean=0.1, reference_sd=float("nan"),
            )


class TestPersistence:
    def test_single_spike_does_not_count_as_onset(self):
        """One bad month is a fluctuation. Without a run requirement the
        onset lands on the noisiest window and the latency measured against
        it is noise as well."""
        values = np.full(12, 0.10)
        values[4] = 0.20  # one spike, then back to normal
        onset = find_onset(
            WINDOWS, values, metric="brier",
            reference_mean=0.10, reference_sd=0.01, n_sd=3.0, persistence=2,
        )
        assert not onset.occurred
        assert onset.onset_index is None

    def test_sustained_breach_is_onset_at_its_first_window(self):
        values = np.full(12, 0.10)
        values[6:] = 0.20
        onset = find_onset(
            WINDOWS, values, metric="brier",
            reference_mean=0.10, reference_sd=0.01, n_sd=3.0, persistence=2,
        )
        assert onset.occurred
        assert onset.onset_index == 6
        assert onset.onset_window == "2014-07"

    def test_persistence_one_accepts_a_single_window(self):
        values = np.full(12, 0.10)
        values[4] = 0.20
        onset = find_onset(
            WINDOWS, values, metric="brier",
            reference_mean=0.10, reference_sd=0.01, persistence=1,
        )
        assert onset.onset_index == 4


class TestDirection:
    def test_decrease_direction_catches_falling_auc(self):
        values = np.full(12, 0.70)
        values[8:] = 0.60
        onset = find_onset(
            WINDOWS, values, metric="auc",
            reference_mean=0.70, reference_sd=0.01,
            n_sd=3.0, persistence=2, direction="decrease",
        )
        assert onset.onset_index == 8

    def test_rising_auc_is_not_degradation(self):
        values = np.linspace(0.70, 0.80, 12)
        onset = find_onset(
            WINDOWS, values, metric="auc",
            reference_mean=0.70, reference_sd=0.01, direction="decrease",
        )
        assert not onset.occurred

    def test_unknown_direction_raises(self):
        with pytest.raises(ValueError, match="direction"):
            find_onset(
                WINDOWS, np.full(12, 0.1), metric="brier",
                reference_mean=0.1, reference_sd=0.01, direction="sideways",
            )


class TestUnevaluableWindows:
    def test_nan_windows_do_not_satisfy_persistence(self):
        """A run of unevaluable windows must not be able to complete a
        breach. `NaN > threshold` is False in numpy, but the explicit finite
        check is what stops a future refactor to `~(arr <= t)` from treating
        every NaN as a breach."""
        values = np.full(12, 0.10)
        values[5] = 0.20
        values[6] = np.nan
        values[7] = 0.20
        onset = find_onset(
            WINDOWS, values, metric="brier",
            reference_mean=0.10, reference_sd=0.01, n_sd=3.0, persistence=2,
        )
        assert not onset.occurred

    def test_all_nan_series_reports_no_onset(self):
        onset = find_onset(
            WINDOWS, np.full(12, np.nan), metric="brier",
            reference_mean=0.10, reference_sd=0.01,
        )
        assert not onset.occurred


class TestThresholdIsCalibratedNotConventional:
    def test_threshold_scales_with_reference_variability(self):
        quiet = find_onset(
            WINDOWS, np.full(12, 0.1), metric="brier",
            reference_mean=0.10, reference_sd=0.005, n_sd=3.0,
        )
        noisy = find_onset(
            WINDOWS, np.full(12, 0.1), metric="brier",
            reference_mean=0.10, reference_sd=0.02, n_sd=3.0,
        )
        assert noisy.threshold > quiet.threshold

    def test_describe_states_the_absence_of_an_event(self):
        onset = find_onset(
            WINDOWS, np.full(12, 0.1), metric="brier",
            reference_mean=0.10, reference_sd=0.01,
        )
        assert "no degradation to detect" in onset.describe()
        assert isinstance(onset, DegradationOnset)
