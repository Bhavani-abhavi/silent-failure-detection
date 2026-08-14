# Threshold calibration — Lending Club, schema era 2013+

Reproduce: `./.venv/Scripts/python.exe scripts/calibrate_lending_club.py`

Data: 2,164,766 loans, 2013-01 to 2018-12, 28 origination-time numeric
features. Detectors at their default thresholds (PSI watch 0.1 / alert 0.25,
Wasserstein watch 0.1 / alert 0.25 normalised by reference SD, KS alpha 0.05).

---

## Headline

| estimate | KS | PSI | Wasserstein |
|---|---|---|---|
| **Intrinsic FPR** (synthetic null, n=20k) | 4.3% | 0.0% | 0.3% |
| **Realistic alert rate** (real windows) | **81.0% – 93.6%** | 0.9% – 3.6% | 9.5% – 35.7% |

The two columns that matter are the two ends of the Wasserstein range and the
whole KS row. Everything below explains why.

---

## 1. Intrinsic FPR — synthetic null

One window (2014 vintage) split in half at random, 40 times. The halves are
draws from the same distribution in the same year, so **every alert is a false
positive by construction**. No assumption that any real period was stable.

FPR by sample size, 40 splits x 28 features = 1,120 tests per cell:

| rows per half | KS | PSI | Wasserstein |
|---|---|---|---|
| 250 | 2.2% | **14.6%** | **50.5%** |
| 1,000 | 2.9% | 0.0% | 3.8% |
| 5,000 | 3.0% | 0.0% | 0.2% |
| 20,000 | 4.3% | 0.0% | 0.3% |

Median statistic under the null, showing the noise floor collapsing with n:

| rows per half | KS | PSI | Wasserstein |
|---|---|---|---|
| 250 | 0.0640 | 0.0543 | 0.1002 |
| 1,000 | 0.0320 | 0.0138 | 0.0510 |
| 5,000 | 0.0144 | 0.0027 | 0.0238 |
| 20,000 | 0.0072 | 0.0007 | 0.0122 |

### What this shows

**KS is the only well-calibrated detector of the three.** Its FPR sits at
2-4% against a nominal 5%, at every sample size. That is what a correctly
specified hypothesis test should do, and it is the one genuinely reassuring
number in this document.

**PSI's conventional thresholds are not thresholds — they are a sample-size
accident.** The same 0.1 watch threshold produces a 14.6% false-positive rate
at n=250 and *exactly* 0.0% from n=1,000 onward. At n=250 the null expectation
is 9 x (2/250) = 0.072, so the threshold is essentially sitting on the noise
floor. At n=20,000 the null expectation is 0.0009 — the threshold is 111x above
it, and nothing short of a catastrophic shift can trip it. A single fixed
number cannot serve both regimes, and the industry convention of quoting
0.1/0.25 without reference to window size is unsupportable.

**Wasserstein is the worst behaved at small n**: 50.5% FPR at n=250. The
normalised statistic's noise floor (median 0.1002) lands exactly on the 0.1
watch threshold, so it fires on roughly half of all null comparisons.

---

## 2. Realistic alert rate — real candidate windows

Candidate periods were nominated in
[domains/finance/stable_periods.py](../../domains/finance/stable_periods.py)
from macro conditions and documented Lending Club events, **before any drift
output was examined**. All candidates are reported, including the worst.

Reference = first quarter of the period; monitoring windows = subsequent
quarters. Reference sizes 37k-97k rows.

| period | windows | KS | PSI | Wasserstein |
|---|---|---|---|---|
| A — ZIRP recovery, 2013H2-2014 | 5 | 93.6% | 1.4% | **35.7%** |
| B — pre-scandal, 2015-2016Q1 | 4 | 85.7% | 0.9% | 9.8% |
| C — post-scandal, 2017 | 3 | 83.3% | 3.6% | 11.9% |
| D — narrow, 2014 only | 3 | 81.0% | 2.4% | 9.5% |
| **spread (max - min)** | | **12.6 pts** | **2.7 pts** | **26.2 pts** |

These are **upper bounds on FPR, not measurements of it.** Genuine mild drift
exists in every one of these windows and is not separable from noise here.

---

## 3. The two findings

### 3.1 KS is unusable as an alerting rule at production window sizes

KS is well calibrated under the synthetic null (4.3%) and fires on **81-94%**
of tests in periods chosen for having no exogenous shock. Both facts are true
and they are not in conflict.

The synthetic null holds the distribution *exactly* fixed. Real quarterly
credit data never is: the applicant population moves a little every quarter for
a hundred mundane reasons. KS asks "is there **any** difference?" and at
n=40,000 per window it can detect differences far too small to matter. The
answer is essentially always yes.

This is the large-n significance trap, and the fix is the one already applied
to the multivariate detector: **gate on effect size as well as significance.**
`domain_classifier_drift` requires both `p < alpha` and `auc >= auc_watch_threshold`
for exactly this reason. `detect_ks_drift` has no equivalent gate — it alerts
on `p < alpha` alone. That is a design inconsistency inside the drift core and
it should be fixed before component 5.

The subtlety worth stating: the intrinsic/realistic gap is not error. It is the
detector working. Roughly `realistic - intrinsic` = genuine drift, so KS is
telling us ~80% of feature-quarters contain *statistically real* movement.
It is simply not telling us which of that movement matters, and an alerting
system that fires on 85% of checks will be switched off within a week.

### 3.2 No genuinely stable period exists in Lending Club

This was flagged in advance as an acceptable outcome, and it is the outcome.

Even in candidate A — ZIRP throughout, no company-specific event, the most
defensible window available — Wasserstein alerts on 35.7% of tests, the
*highest* of any candidate. That is not a detector failure. 2013H2-2014 was
Lending Club's steepest growth phase, with origination volume up roughly 75%
across the window. The applicant population genuinely moved, a great deal,
with no macro shock and no scandal. **Absence of an exogenous shock does not
imply a stable population**, and the most macro-stable window turned out to be
the least population-stable one.

The practical consequence: **threshold calibration on this data is
regime-dependent, and the dependence is large.** Calibrating Wasserstein on
period B gives 9.8%; the same thresholds on period A give 35.7%. A team that
calibrated in 2015 and deployed in 2013 would have had a 3.6x alert volume
surprise.

---

## 4. What this constrains downstream

- **Component 5 (alerting) cannot use fixed thresholds.** They must be
  sample-size aware. The `psi_null_expectation` and
  `ks_minimum_detectable_effect` helpers in `drift_core/validity.py` give the
  noise floor analytically; thresholds should be expressed as a multiple of it
  rather than as constants.
- **KS needs an effect-size gate** before it is used for alerting, matching
  the multivariate detector's two-condition rule.
- **Report a range, never a single FPR.** Any single figure is a statement
  about one regime.
- **Multiple-testing correction will not fix KS.** At 85% raw alert rate the
  problem is not the family-wise error rate, it is that the null hypothesis
  being tested is not the question anyone cares about.

## 5. Provenance limits

The candidate window boundaries are analyst-declared from public record
(federal funds target changes, the May 2016 leadership resignation, the
December 2014 IPO). They are auditable but not machine-verified. Before
publication the rate and unemployment series should be pulled from FRED
(FEDFUNDS, UNRATE) and the regime boundaries re-derived from the series.
Tracked in [docs/findings_log.md](../../docs/findings_log.md).
