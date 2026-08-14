"""Candidate stable periods for Lending Club, declared from EXOGENOUS evidence.

METHODOLOGICAL COMMITMENT
=========================
Every window below was chosen from macroeconomic conditions and documented
Lending Club corporate events — never from drift-detector output. Choosing
the period where the detector reports least drift and then measuring the
false-positive rate there is circular: it selects for low readings and
reports them back as a property of the thresholds. The resulting number
would be guaranteed optimistic and would mean nothing.

Consequently ALL candidates are reported, including whichever ones look
worst. The spread across them is the finding, because it measures how much
threshold calibration depends on which regime you happened to calibrate in.

PROVENANCE AND ITS LIMITS
=========================
The dates below are analyst-declared from public record. The macro claims
(federal funds target changes, unemployment trend) are stated from published
history and are auditable, but this module deliberately does NOT hardcode
macro time series as data. Before any of this appears in a report, the rate
and unemployment series should be pulled from FRED (DFF/FEDFUNDS and UNRATE)
and the regime boundaries re-derived from the actual series rather than from
these annotations. Recorded as an open item in docs/findings_log.md.

Nothing here is claimed to be a *truly* stable period. The honest position is
that these are periods with no identifiable exogenous shock, which is a much
weaker statement. Alerts inside them mix genuine mild drift with false
positives and the two cannot be separated — so real-window FPR is an UPPER
BOUND. The synthetic null in pipeline/calibration.py is the estimate that
does not depend on any of this being right.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidatePeriod:
    name: str
    start: str
    end: str
    rationale: str
    known_confounders: str


STRUCTURAL_BREAKS: dict[str, str] = {
    "2008-10": (
        "SEC registration completed after a six-month quiet period; the "
        "platform's investor base and disclosure regime changed."
    ),
    "2009-2010": (
        "Post-crisis underwriting reset. Loans flagged 'Does not meet the "
        "credit policy' come from the earlier regime and are a different "
        "population entirely."
    ),
    "2014-12": "IPO. Capital structure, scrutiny, and growth incentives all change.",
    "2016-05": (
        "CEO Renaud Laplanche resigned following an internal review that found "
        "loans sold to an institutional investor had application data altered. "
        "Origination volume fell sharply, institutional investors withdrew, and "
        "underwriting was tightened in response. The clearest company-specific "
        "structural break in the dataset."
    ),
    "2015-12": "First federal funds increase since 2006, ending the ZIRP era.",
    "2017-2018": (
        "Sustained tightening cycle — three funds-rate increases in 2017, four "
        "in 2018 — alongside progressive retirement of the riskiest grades."
    ),
    "2020-03": (
        "COVID-19 and the CARES Act forbearance regime. Outside this dataset, "
        "which ends 2018-12, but the single most important break for any "
        "extension of this work to later vintages."
    ),
}


def candidate_stable_periods() -> list[CandidatePeriod]:
    """Periods with no identifiable exogenous shock, in schema era 2013+.

    All start at or after 2013-01 so that feature availability is constant
    within each window — otherwise a schema change would be measured as
    drift and the false-positive rate would be an artefact of the vendor's
    column additions rather than of the thresholds.
    """
    return [
        CandidatePeriod(
            name="A_zirp_recovery_2013H2_2014",
            start="2013-07-01",
            end="2015-01-01",
            rationale=(
                "Federal funds target held at 0-0.25% throughout; unemployment "
                "declining smoothly with no discontinuity; no LC-specific event "
                "until the December 2014 IPO. The most defensible candidate."
            ),
            known_confounders=(
                "Ends at the IPO. Origination volume grew ~75% across the "
                "window, so the applicant population is not constant even "
                "absent a shock."
            ),
        ),
        CandidatePeriod(
            name="B_pre_scandal_2015",
            start="2015-01-01",
            end="2016-04-01",
            rationale=(
                "Between the IPO and the May 2016 leadership crisis. No "
                "company-specific break inside the window."
            ),
            known_confounders=(
                "Contains the December 2015 rate increase, the first in nine "
                "years. Classified as a candidate anyway rather than excluded, "
                "because excluding every window with any macro event would "
                "leave nothing and would itself be a selection decision."
            ),
        ),
        CandidatePeriod(
            name="C_post_scandal_2017",
            start="2017-01-01",
            end="2018-01-01",
            rationale=(
                "After the 2016 disruption had worked through. Volume and "
                "underwriting had re-stabilised at a new level."
            ),
            known_confounders=(
                "Three rate increases during the window, and grade-mix changes "
                "as the riskiest grades were retired. Expected to be the "
                "weakest candidate — included precisely so the reported spread "
                "is not flattered by dropping it."
            ),
        ),
        CandidatePeriod(
            name="D_narrow_2014",
            start="2014-01-01",
            end="2015-01-01",
            rationale=(
                "A single calendar year inside candidate A. Included to "
                "separate the effect of window LENGTH from the effect of "
                "regime — a shorter window means smaller samples and a higher "
                "noise floor, which should move the FPR on its own."
            ),
            known_confounders="Subset of A; not an independent observation.",
        ),
    ]
