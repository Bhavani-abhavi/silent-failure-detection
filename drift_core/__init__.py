"""Domain-agnostic drift detection core.

Hard rule enforced by the import-linter contract in pyproject.toml:
`drift_core` imports nothing from `domains`, `pipeline`, `dashboard`, or
`reports`. If a change to this package is needed to support a new domain,
that is a design failure and belongs in the findings log.
"""

from drift_core.concept import (
    confirm_concept_drift_with_labels,
    effective_sample_size_ratio,
    feature_relationship_drift,
    out_of_support_mass,
)
from drift_core.multivariate import domain_classifier_drift
from drift_core.prediction import detect_prediction_drift, mean_score_shift
from drift_core.types import (
    DriftKind,
    DriftResult,
    MultivariateDriftResult,
    Severity,
    WindowSpec,
)
from drift_core.univariate import (
    detect_kl_drift,
    detect_ks_drift,
    detect_psi_drift,
    detect_wasserstein_drift,
    kl_divergence,
    ks_test,
    population_stability_index,
    wasserstein,
)

__all__ = [
    "DriftKind",
    "DriftResult",
    "MultivariateDriftResult",
    "Severity",
    "WindowSpec",
    "population_stability_index",
    "kl_divergence",
    "ks_test",
    "wasserstein",
    "detect_psi_drift",
    "detect_kl_drift",
    "detect_ks_drift",
    "detect_wasserstein_drift",
    "domain_classifier_drift",
    "detect_prediction_drift",
    "mean_score_shift",
    "feature_relationship_drift",
    "out_of_support_mass",
    "effective_sample_size_ratio",
    "confirm_concept_drift_with_labels",
]
