"""Population stability index per monitoring window, computed with PySpark.

The NumPy path (`drift_core.univariate.population_stability_index`) loads every
window into memory. That is fine for one model on one laptop and is still the
reference implementation. This module is the same computation shaped for a
cluster: fit on the reference, score everything else in one distributed pass.

FIT ON THE REFERENCE, ON THE DRIVER. Bin edges come from the reference window
only, and they are computed with the very same `_quantile_bin_edges` the NumPy
path uses. Spark's `approxQuantile` interpolates differently from
`np.quantile` (it returns observed values, never a value between two rows), so
using it would make the two paths disagree on every feature and the parity test
below would be measuring the quantile algorithm, not the drift code. The
reference is one period; collecting a single column of it is cheap.

SCORE THE MONITORING WINDOWS IN SPARK. Every later row is bucketed against
those fixed edges and counted per (window, feature, bin) in one groupBy. This
is the part that grows with the data, and it never leaves the executors as
rows; only the count table comes back.

Semantics match the NumPy path exactly, including the parts that are easy to
get subtly wrong:
  - NaN / null values are dropped before binning, per feature.
  - Outer edges are open (-inf, +inf), so out-of-range values still count.
  - Bins are [a, b) except the last, which is closed, as in `np.histogram`.
  - Proportions are clipped at `epsilon` only inside the log term.
  - Categorical features use the union of reference and window categories.
  - Windows below `min_rows` are dropped, as in `split_time_windows`.

`tests/pipeline/test_spark_drift.py` checks all of this against the NumPy
implementation to 1e-12.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from drift_core.univariate import _quantile_bin_edges

try:  # pyspark is an optional extra: `pip install -e ".[spark]"`
    from pyspark.ml.feature import Bucketizer
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import functions as F
except ImportError:  # pragma: no cover - exercised only without the extra
    SparkDataFrame = None  # type: ignore[assignment,misc]


_FREQ_ALIASES = {"MS": "M", "QS": "Q"}


def _require_spark() -> None:
    if SparkDataFrame is None:
        raise ImportError(
            "pyspark is not installed. Install the optional extra: pip install -e \".[spark]\""
        )


def _window_id_column(time_column: str, freq: str):
    """Window labels that match `str(pd.Period)` for the same frequency, so
    Spark output joins cleanly onto the NumPy path's window ids."""
    freq = _FREQ_ALIASES.get(freq.upper(), freq.upper())
    ts = F.col(time_column)
    if freq == "M":
        return F.date_format(ts, "yyyy-MM")
    if freq == "Q":
        return F.concat(F.year(ts).cast("string"), F.lit("Q"), F.quarter(ts).cast("string"))
    raise ValueError(f"freq must be 'M' or 'Q', got {freq!r}")


def _psi(ref_prop: np.ndarray, cur_prop: np.ndarray, epsilon: float) -> float:
    ref_safe = np.clip(ref_prop, epsilon, None)
    cur_safe = np.clip(cur_prop, epsilon, None)
    return float(np.sum((cur_safe - ref_safe) * np.log(cur_safe / ref_safe)))


def _proportions(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return counts / total if total else np.zeros(len(counts))


def windowed_psi(
    frame: "SparkDataFrame",
    *,
    time_column: str,
    reference_end: str | pd.Timestamp,
    features: Sequence[str],
    categorical: Iterable[str] = (),
    reference_start: str | pd.Timestamp | None = None,
    freq: str = "M",
    bins: int = 10,
    epsilon: float = 1e-4,
    min_rows: int = 200,
) -> pd.DataFrame:
    """PSI for every (monitoring window, feature) pair.

    Returns a pandas frame with one row per pair:
    window_id, feature, psi, n_rows, realized_bins.
    `n_rows` is the window size before per-feature NaN removal, matching what
    `split_time_windows` uses for its `min_rows` cut.
    """
    _require_spark()
    categorical = set(categorical)
    unknown = categorical - set(features)
    if unknown:
        raise ValueError(f"categorical features not in features: {sorted(unknown)}")

    reference_end = pd.Timestamp(reference_end)
    ref_filter = F.col(time_column) < F.lit(reference_end)
    if reference_start is not None:
        ref_filter &= F.col(time_column) >= F.lit(pd.Timestamp(reference_start))

    reference = frame.where(ref_filter)
    n_reference = reference.count()
    if n_reference < min_rows:
        raise ValueError(f"reference period has {n_reference} rows, below min_rows={min_rows}")

    monitor = (
        frame.where(F.col(time_column) >= F.lit(reference_end))
        .withColumn("_window_id", _window_id_column(time_column, freq))
    )
    sizes = {r["_window_id"]: r["count"] for r in monitor.groupBy("_window_id").count().collect()}
    kept = sorted(w for w, n in sizes.items() if n >= min_rows)
    monitor = monitor.where(F.col("_window_id").isin(kept))

    rows: list[dict] = []
    for feature in features:
        if feature in categorical:
            rows += _categorical_feature_psi(reference, monitor, feature, kept, sizes, epsilon)
        else:
            rows += _numeric_feature_psi(reference, monitor, feature, kept, sizes, bins, epsilon)

    out = pd.DataFrame(rows, columns=["window_id", "feature", "psi", "n_rows", "realized_bins"])
    return out.sort_values(["window_id", "feature"], ignore_index=True)


def _numeric_feature_psi(reference, monitor, feature, kept, sizes, bins, epsilon) -> list[dict]:
    ref_values = np.array(
        [r[0] for r in reference.select(F.col(feature).cast("double")).collect()], dtype=float
    )
    ref_values = ref_values[~np.isnan(ref_values)]
    edges = _quantile_bin_edges(ref_values, bins)
    n_bins = len(edges) - 1
    ref_counts, _ = np.histogram(ref_values, bins=edges)
    ref_prop = _proportions(ref_counts.astype(float))

    # Bucketizer: [a, b) except the last bucket, which includes its upper
    # edge -- the same convention as np.histogram. With +/-inf outer edges no
    # finite value is out of range; NaN/null rows are skipped, matching the
    # NaN drop in the NumPy path.
    bucketizer = Bucketizer(
        splits=[float(e) for e in edges],
        inputCol="_value",
        outputCol="_bin",
        handleInvalid="skip",
    )
    values = monitor.select("_window_id", F.col(feature).cast("double").alias("_value")).where(
        F.col("_value").isNotNull() & ~F.isnan("_value")
    )
    counts = (
        bucketizer.transform(values)
        .groupBy("_window_id", "_bin")
        .count()
        .toPandas()
    )

    rows = []
    for window_id in kept:
        cur_counts = np.zeros(n_bins)
        sub = counts[counts["_window_id"] == window_id]
        cur_counts[sub["_bin"].astype(int).to_numpy()] = sub["count"].to_numpy()
        rows.append({
            "window_id": window_id,
            "feature": feature,
            "psi": _psi(ref_prop, _proportions(cur_counts), epsilon),
            "n_rows": int(sizes[window_id]),
            "realized_bins": n_bins,
        })
    return rows


def _categorical_feature_psi(reference, monitor, feature, kept, sizes, epsilon) -> list[dict]:
    ref_counts = {
        r[feature]: r["count"]
        for r in reference.where(F.col(feature).isNotNull()).groupBy(feature).count().collect()
    }
    cur = (
        monitor.where(F.col(feature).isNotNull())
        .groupBy("_window_id", feature)
        .count()
        .toPandas()
    )
    rows = []
    for window_id in kept:
        sub = cur[cur["_window_id"] == window_id]
        cur_counts = dict(zip(sub[feature], sub["count"]))
        categories = sorted(set(ref_counts) | set(cur_counts), key=lambda c: (str(type(c)), c))
        ref_prop = _proportions(np.array([ref_counts.get(c, 0) for c in categories], dtype=float))
        cur_prop = _proportions(np.array([cur_counts.get(c, 0) for c in categories], dtype=float))
        rows.append({
            "window_id": window_id,
            "feature": feature,
            "psi": _psi(ref_prop, cur_prop, epsilon),
            "n_rows": int(sizes[window_id]),
            "realized_bins": len(categories),
        })
    return rows


def spark_session(app_name: str = "silent-failure-detection", *, cores: str = "*"):
    """A local SparkSession with settings that keep small test runs fast.
    On a cluster, build the session from the cluster's own config instead."""
    _require_spark()
    import os
    import sys

    from pyspark.sql import SparkSession

    # Executors must run the same interpreter as the driver; on Windows the
    # default `python` on PATH is often a different install, and the worker
    # then never connects back.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return (
        SparkSession.builder.master(f"local[{cores}]")
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        # pandas <-> Spark conversion through Arrow on the JVM side, without
        # pickling rows through Python workers.
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


__all__ = ["windowed_psi", "spark_session"]
