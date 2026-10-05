"""Parity tests: Spark windowed PSI must equal the NumPy reference implementation.

The Spark path is only useful if it is the same statistic. Every edge case the
NumPy path handles deliberately is exercised here on synthetic data with known
structure: a drifting numeric feature, NaNs, a constant feature (degenerate
bin edges), a categorical feature that gains a new category, out-of-range
values beyond the reference min/max, and a window below `min_rows`.

Skipped when pyspark or a Java runtime is unavailable.
"""


import numpy as np
import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")

from drift_core.univariate import population_stability_index  # noqa: E402
from pipeline.spark_drift import spark_session, windowed_psi  # noqa: E402
from pipeline.windowing import split_time_windows  # noqa: E402

TIME = "issue_d"
REFERENCE_END = "2014-01-01"
NUMERIC = ["x", "y", "const"]
CATEGORICAL = ["grade"]
FEATURES = NUMERIC + CATEGORICAL


@pytest.fixture(scope="module")
def spark():
    try:
        session = spark_session("sfd-tests", cores="2")
    except Exception as exc:  # no JVM available
        pytest.skip(f"Spark could not start: {exc}")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    months = pd.date_range("2013-07-01", "2014-06-01", freq="MS")
    frames = []
    for i, month in enumerate(months):
        n = 60 if month == pd.Timestamp("2014-03-01") else 1500  # one window below min_rows
        drift = max(0, i - 6) * 0.25  # flat through the reference, then shifts
        x = rng.normal(drift, 1 + 0.05 * i, n)
        x[:5] = 25.0 + i  # values beyond the reference range must still be counted
        y = rng.exponential(1 + drift, n)
        y[rng.random(n) < 0.1] = np.nan  # NaNs are dropped per feature, not per row
        grades = ["A", "B", "C"] + (["D"] if month >= pd.Timestamp("2014-04-01") else [])
        frames.append(pd.DataFrame({
            TIME: month + pd.to_timedelta(rng.integers(0, 27, n), unit="D"),
            "x": x,
            "y": y,
            "const": np.full(n, 3.0) if i < 9 else np.full(n, 4.0),
            "grade": rng.choice(grades, n),
        }))
    return pd.concat(frames, ignore_index=True)


def _numpy_expected(frame: pd.DataFrame) -> pd.DataFrame:
    panel = split_time_windows(
        frame, time_column=TIME, freq="M", reference_end=REFERENCE_END,
        feature_names=FEATURES, min_rows=200,
    )
    rows = []
    for window_id, window in panel.windows:
        for feature in FEATURES:
            psi, extra = population_stability_index(
                panel.reference[feature].to_numpy(), window[feature].to_numpy(),
                bins=10, categorical=feature in CATEGORICAL,
            )
            rows.append({"window_id": window_id, "feature": feature, "psi": psi,
                         "n_rows": len(window), "realized_bins": len(extra["reference_proportions"])})
    return pd.DataFrame(rows).sort_values(["window_id", "feature"], ignore_index=True)


@pytest.fixture(scope="module")
def both(spark, panel):
    expected = _numpy_expected(panel)
    got = windowed_psi(
        spark.createDataFrame(panel), time_column=TIME, reference_end=REFERENCE_END,
        features=FEATURES, categorical=CATEGORICAL, freq="M", min_rows=200,
    )
    return expected, got


def test_same_windows_and_features(both):
    expected, got = both
    assert list(zip(got.window_id, got.feature)) == list(zip(expected.window_id, expected.feature))


def test_small_window_is_dropped(both):
    _, got = both
    assert "2014-03" not in set(got.window_id)


def test_psi_matches_numpy_to_1e12(both):
    expected, got = both
    np.testing.assert_allclose(got.psi.to_numpy(), expected.psi.to_numpy(), rtol=0, atol=1e-12)


def test_bin_counts_and_window_sizes_match(both):
    expected, got = both
    assert got.realized_bins.tolist() == expected.realized_bins.tolist()
    assert got.n_rows.tolist() == expected.n_rows.tolist()


def test_drift_is_detected_and_grows(both):
    _, got = both
    x = got[got.feature == "x"].set_index("window_id").psi
    assert x.iloc[0] < 0.1  # first monitoring month is still close to the reference
    assert x.iloc[-1] > 0.25  # by the end the shift is substantial
    assert x.iloc[-1] > x.iloc[1]


def test_constant_feature_that_moves_is_detected(both):
    _, got = both
    const = got[got.feature == "const"].set_index("window_id").psi
    assert const.loc["2014-01"] == pytest.approx(0.0, abs=1e-12)  # still 3.0
    assert const.loc["2014-04"] > 1.0  # moved to 4.0: degenerate edges must still catch it


def test_quarterly_window_ids_match_pandas_periods(spark, panel):
    got = windowed_psi(
        spark.createDataFrame(panel), time_column=TIME, reference_end=REFERENCE_END,
        features=["x"], freq="Q", min_rows=200,
    )
    later = panel[panel[TIME] >= pd.Timestamp(REFERENCE_END)]
    expected = sorted(str(p) for p in later[TIME].dt.to_period("Q").unique())
    assert sorted(got.window_id.unique()) == expected


def test_rejects_unknown_categorical(spark, panel):
    with pytest.raises(ValueError, match="categorical features not in features"):
        windowed_psi(spark.createDataFrame(panel), time_column=TIME, reference_end=REFERENCE_END,
                     features=["x"], categorical=["grade"])
