"""Shared result types for the drift core.

Everything in `drift_core` communicates through these dataclasses. They carry
no domain knowledge (no column semantics, no "readmission" or "default"
concepts) — only feature names as opaque strings, arrays, and statistics.
This is what lets the same functions run unchanged on clinical, financial,
and commercial data: domain packages are responsible for handing drift_core
plain arrays/frames with feature names attached, and for interpreting the
results back into domain language afterward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DriftKind(str, Enum):
    """What moved. Kept explicit because conflating these is the most common
    mistake in drift monitoring writeups."""

    DATA = "data"
    """P(X) changed: the input distribution moved. Detectable unsupervised,
    with no labels required, at the time it happens."""

    PREDICTION = "prediction"
    """P(Y_hat) changed: the model's output distribution moved. Detectable
    unsupervised. A leading indicator of concept drift, not proof of it —
    a model can shift its outputs on genuinely shifted-but-still-well-handled
    inputs."""

    CONCEPT_PROXY = "concept_proxy"
    """An unsupervised proxy for a change in P(Y|X), built from prediction
    drift + calibration-adjacent signals conditional on regions where the
    input distribution has NOT moved. This is suggestive, not confirmatory —
    see CONCEPT_CONFIRMED."""

    CONCEPT_CONFIRMED = "concept_confirmed"
    """A label-confirmed change in P(Y|X), computable only once delayed
    labels arrive for the current window. This is the ground truth that
    CONCEPT_PROXY is trying to anticipate; comparing the two is how we
    report detection latency and estimation error."""


class Severity(str, Enum):
    NONE = "none"
    WATCH = "watch"
    ALERT = "alert"


class ResultStatus(str, Enum):
    """Whether the detector could actually evaluate this feature/window.

    Severity answers "how bad is the drift". Status answers the prior
    question, "was a verdict possible at all". They are separate fields
    because collapsing them is precisely how a detector ends up reporting
    Severity.NONE for a feature it could not see — see drift_core/validity.py
    for the two occasions that happened here.
    """

    OK = "ok"
    INSUFFICIENT_DATA = "insufficient_data"
    """Not enough present, finite observations to compute the statistic."""

    NO_POWER = "no_power"
    """Computable, but the configuration cannot detect any effect — the
    minimum detectable effect exceeds the maximum possible effect."""


@dataclass(frozen=True)
class WindowSpec:
    """Identifies a time window being compared against the reference. Purely
    an index/label — drift_core never interprets window_id as a calendar
    date; that mapping lives in the domain package."""

    window_id: str
    n_samples: int


@dataclass(frozen=True)
class DriftResult:
    """Result of a single drift test on a single feature (or on the
    prediction stream, using feature_name="__prediction__")."""

    feature_name: str
    method: str
    statistic: float
    kind: DriftKind
    window: WindowSpec
    p_value: float | None = None
    threshold: float | None = None
    is_drifted: bool = False
    severity: Severity = Severity.NONE
    status: ResultStatus = ResultStatus.OK
    """Check this BEFORE reading is_drifted. `is_drifted=False` on a
    non-OK status means "could not evaluate", not "no drift"."""

    minimum_detectable_effect: float | None = None
    """The smallest effect this configuration could have detected, in the
    units of `statistic`. Required by the project rule that every detector
    state its own sensitivity at call time rather than leaving the reader to
    assume a null result means the world was quiet."""

    n_reference: int = 0
    n_current: int = 0
    extra: dict = field(default_factory=dict)
    """Method-specific detail (e.g. bin edges for PSI) for report rendering.
    Never load logic-critical values from here — anything a caller needs to
    branch on belongs in a typed field above."""


@dataclass(frozen=True)
class MultivariateDriftResult:
    """Result of the domain-classifier drift test over a feature set."""

    method: str
    auc: float
    p_value: float
    """From a permutation-test null (see multivariate.py), not from a
    fixed AUC-vs-0.5 threshold. Raw AUC alone is not evidence of drift at
    finite sample sizes."""
    window: WindowSpec
    n_reference: int
    n_current: int
    is_drifted: bool = False
    severity: Severity = Severity.NONE
    status: ResultStatus = ResultStatus.OK
    minimum_detectable_p: float | None = None
    """1/(n_permutations+1) — the resolution floor of the permutation null."""

    feature_importances: dict[str, float] = field(default_factory=dict)
    """Which features the discriminator relied on — the multivariate
    equivalent of "which feature drifted", used for governance reporting."""
