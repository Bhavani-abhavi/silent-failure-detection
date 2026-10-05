"""MLflow tracking: a logged run must round-trip exactly what the backtest produced."""

import numpy as np
import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")

from reports.tracking import log_backtest_run  # noqa: E402


@pytest.fixture
def window_metrics():
    ids = ["2014-01", "2014-02", "2014-03", "2014-04"]
    return pd.DataFrame(
        {"brier": [0.11, 0.12, 0.13, 0.15], "auc": [0.67, 0.67, np.nan, 0.66], "label": list("abcd")},
        index=pd.Index(ids, name="window_id"),
    )


def test_run_round_trips_params_series_and_summary(tmp_path, monkeypatch, window_metrics):
    monkeypatch.chdir(tmp_path)  # artifacts land under ./mlartifacts; keep them out of the repo
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    report = tmp_path / "latency.csv"
    report.write_text("signal,latency\npsi,3\n")

    run_id = log_backtest_run(
        params={"freq": "M", "onset_n_sd": 3.0},
        window_metrics=window_metrics,
        summary_metrics={"latency_psi": 3, "undefined": float("nan")},
        artifacts=[report, tmp_path / "missing.csv"],
        run_name="test",
        experiment="sfd-test",
        tracking_uri=uri,
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)
    assert run.data.params == {"freq": "M", "onset_n_sd": "3.0"}
    assert run.data.metrics["latency_psi"] == 3
    assert "undefined" not in run.data.metrics  # NaN summaries are skipped, not logged as 0

    brier = client.get_metric_history(run_id, "brier")
    assert [(m.step, m.value) for m in sorted(brier, key=lambda m: m.step)] == [
        (0, 0.11), (1, 0.12), (2, 0.13), (3, 0.15)
    ]
    auc_steps = sorted(m.step for m in client.get_metric_history(run_id, "auc"))
    assert auc_steps == [0, 1, 3]  # the NaN window is a gap, not a zero
    assert "label" not in run.data.metrics  # non-numeric columns are not metrics

    artifacts = {a.path for a in client.list_artifacts(run_id)}
    assert "window_steps.csv" in artifacts
    assert {a.path for a in client.list_artifacts(run_id, "reports")} == {"reports/latency.csv"}
