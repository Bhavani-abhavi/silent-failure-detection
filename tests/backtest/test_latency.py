"""Tests for latency scoring.

The property these protect is that a detector cannot win by firing
constantly. Latency alone rewards exactly that, so every test here checks the
guard as well as the number.
"""

import numpy as np
import pytest

from backtest.latency import (
    false_positive_rate,
    first_sustained_fire,
    score_signal,
)

WINDOWS = [f"w{i:02d}" for i in range(12)]


class TestFirstSustainedFire:
    def test_run_of_required_length_is_found(self):
        fired = [False] * 4 + [True, True] + [False] * 6
        assert first_sustained_fire(fired, persistence=2) == 4

    def test_isolated_fire_does_not_count(self):
        """Without this, a signal that blips once years early is credited
        with the detection and reports an enormous lead time."""
        fired = [False] * 3 + [True] + [False] * 8
        assert first_sustained_fire(fired, persistence=2) is None

    def test_never_firing_returns_none(self):
        assert first_sustained_fire([False] * 12, persistence=2) is None

    def test_always_firing_returns_zero(self):
        assert first_sustained_fire([True] * 12, persistence=2) == 0

    def test_persistence_must_be_positive(self):
        with pytest.raises(ValueError, match="persistence"):
            first_sustained_fire([True] * 5, persistence=0)


class TestLatencySign:
    def _score(self, fired, onset_index):
        return score_signal(
            "sig", WINDOWS, fired,
            onset_index=onset_index,
            onset_window=WINDOWS[onset_index] if onset_index is not None else None,
            persistence=2,
        )

    def test_firing_before_onset_gives_positive_latency(self):
        fired = [False] * 3 + [True] * 9
        result = self._score(fired, onset_index=8)
        assert result.first_fire_index == 3
        assert result.latency_windows == 5
        assert result.led_the_drop is True

    def test_firing_at_onset_gives_zero(self):
        fired = [False] * 8 + [True] * 4
        result = self._score(fired, onset_index=8)
        assert result.latency_windows == 0
        assert result.led_the_drop is False

    def test_firing_after_onset_gives_negative_latency(self):
        """A lagging indicator is reported as lagging rather than dropped.
        Labels would have arrived first, which is worth stating."""
        fired = [False] * 10 + [True] * 2
        result = self._score(fired, onset_index=6)
        assert result.latency_windows == -4
        assert result.led_the_drop is False

    def test_never_firing_is_none_not_zero(self):
        """Conflating "never fired" with "fired exactly on time" would let a
        dead detector look punctual."""
        result = self._score([False] * 12, onset_index=6)
        assert result.latency_windows is None
        assert result.first_fire_index is None
        assert result.led_the_drop is False

    def test_no_onset_gives_no_latency(self):
        result = self._score([True] * 12, onset_index=None)
        assert result.latency_windows is None


class TestAlwaysOnDetectorIsExposed:
    def test_maximum_latency_comes_with_maximum_pre_onset_alert_rate(self):
        """The central guard. A detector firing on every window reports the
        best possible latency; the pre-onset alert rate is what reveals that
        it predicted nothing."""
        result = score_signal(
            "always_on", WINDOWS, [True] * 12,
            onset_index=8, onset_window="w08", persistence=2,
        )
        assert result.latency_windows == 8  # looks excellent
        assert result.pre_onset_alert_rate == 1.0  # and is meaningless
        assert result.total_alert_rate == 1.0

    def test_selective_detector_has_low_pre_onset_rate(self):
        fired = [False] * 6 + [True] * 6
        result = score_signal(
            "selective", WINDOWS, fired,
            onset_index=8, onset_window="w08", persistence=2,
        )
        assert result.latency_windows == 2
        assert result.pre_onset_alert_rate == pytest.approx(2 / 8)

    def test_describe_surfaces_both_numbers(self):
        result = score_signal(
            "always_on", WINDOWS, [True] * 12,
            onset_index=8, onset_window="w08", persistence=2,
        )
        text = result.describe()
        assert "+8" in text and "100.0%" in text


class TestInputValidation:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="fire flags"):
            score_signal(
                "sig", WINDOWS, [True] * 5,
                onset_index=3, onset_window="w03",
            )


class TestFalsePositiveRate:
    def test_rate_over_stable_windows(self):
        assert false_positive_rate([False, False, True, False]) == pytest.approx(0.25)

    def test_empty_is_nan_not_zero(self):
        assert np.isnan(false_positive_rate([]))
