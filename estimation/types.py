"""Result types for label-free performance estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EstimateStatus(str, Enum):
    OK = "ok"

    SUPPRESSED = "suppressed"
    """The estimator declined to answer.

    An estimator that reports a confident number outside the conditions it is
    valid under is worse than one that reports nothing, because the number
    gets plotted and believed. Suppression is a result, not an error.
    """

    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class PerformanceEstimate:
    """One label-free estimate of one metric, for one window.

    Deliberately carries no true value. Scoring an estimate against the truth
    is `backtest`'s job, and keeping the truth out of this dataclass makes it
    structurally impossible for an estimator to peek at what it is being
    graded on.
    """

    metric: str
    """What is being estimated: "brier", "accuracy", "base_rate"."""

    method: str
    """How: "average_confidence", "atc", "importance_weighted"."""

    window_id: str
    estimate: float
    status: EstimateStatus = EstimateStatus.OK
    effective_sample_size: float | None = None
    """For weighted estimators: the number of reference points actually
    contributing after reweighting. Collapse here is the main way importance
    weighting fails, and it fails quietly."""

    n_current: int = 0
    detail: dict = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        return self.status is EstimateStatus.OK
