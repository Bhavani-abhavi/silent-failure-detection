# Findings log — what didn't work, and what turned out to be true

Maintained as work happens, not reconstructed afterwards. Entries are append-only;
when something is later disproved, add a new entry rather than editing the old one.

Format: date, what was tried/claimed, what happened, what changed as a result.

---

## 2026-08-13 — Unsupervised concept drift detection is not possible for a fixed model

**Claim tested:** that some unsupervised signal could indicate a change in P(Y|X)
before labels arrive.

**What happened:** it doesn't hold, and the reason is arithmetic rather than
empirical. For a fixed deterministic model f, predictions are Y_hat = f(X), so
every observable quantity is a function of P(X) alone. The natural candidate
signal — "predictions drifted by more than covariate shift explains" — is
identically zero: importance-weighting reference predictions by the true density
ratio reproduces the current prediction distribution exactly, given common
support. Pinned as a test in
`tests/drift_core/test_concept.py::test_importance_weighted_reference_reproduces_current_predictions`
(agreement to ~0.01 on 200k samples).

**What changed:** `drift_core/concept.py` does not claim to detect concept drift.
It returns `CONCEPT_PROXY` risk signals that measure something real but different:

- **feature relationship drift** — the dependence structure among inputs changed
  (upstream process change, a correlational argument about the world)
- **out-of-support mass** — the model is extrapolating into regions where P(Y|X)
  was never validated (absence of evidence, not evidence of degradation)
- **effective sample size** — a guardrail on our own importance-weighted
  estimator, not a signal about the monitored model

`CONCEPT_CONFIRMED` is produced only by `confirm_concept_drift_with_labels`,
which requires delayed labels. Two tests exist specifically to fail if someone
later re-introduces the invalid inference.

**Consequence for the headline result:** detection latency compares an
unsupervised *proxy* firing against a label-confirmed performance drop. The
honest framing is "this proxy anticipated the drop by N windows on this data",
not "we detected concept drift without labels". The former is defensible and
still useful; the latter is false.

---

## 2026-08-13 — Raw domain-classifier AUC is not evidence of drift

**Claim tested:** the common recipe of "train a discriminator on reference vs
current; AUC > 0.5 means drift".

**What happened:** rejected on construction. With enough features relative to
window size, a flexible classifier separates two samples of the *same*
distribution. Held-out AUC reduces but does not eliminate this at small n.

**What changed:** `domain_classifier_drift` reports a permutation-test p-value
(shuffle group labels, refit, build the null distribution of out-of-fold AUC)
alongside the AUC, and gates severity on *both* — significance for evidence,
AUC magnitude for effect size. A significant AUC of 0.53 on 100k rows is real
and operationally meaningless; it must not page anyone.
`test_no_drift_high_dimensional_small_n_does_not_fire` (40 features, 150 rows
per window, no drift) is the regression test for this.

**Cost accepted:** n_permutations × k-fold refits per window. Expensive. If this
becomes the bottleneck on real data, the fallback is a cheaper null (subsampled
permutations, or an analytic approximation) — but not dropping the null entirely.

---

## 2026-08-13 — Detection latency cannot be a live dashboard metric

**What happened:** noted during design, before implementation. Latency is
measured from "unsupervised signal fired" to "performance drop became
measurable", and the second term requires labels that by definition have not
arrived yet at monitoring time.

**What changed:** detection latency belongs to a backtest/experiment module
run over historical windows where both the early signal and the late label are
now in hand. The live monitor reports alerts and proxy severities only. Keeping
these separate avoids the trap of a dashboard implying it knows something it
cannot know.

---

## 2026-08-13 — The permutation test had a silent no-alert floor

**How it surfaced:** `test_large_effect_is_an_alert` failed with AUC = **1.0**
and p = 0.0625. Perfectly separable windows, reported as no drift.

**Cause:** a permutation test cannot produce a p-value below 1/(n_permutations+1).
At n_permutations=15 the floor is 0.0625, which sits above alpha=0.05 — so no
result, at any effect size, could ever reach significance. The detector was
structurally incapable of alerting and said nothing about it.

**Why it matters more than a normal bug:** this is a silent failure in a silent-
failure detector, and it fails at precisely the moment it is needed most (extreme
drift). Nothing in the output distinguished "no drift found" from "cannot detect
drift". A monitoring system with this property is worse than no monitoring,
because it manufactures false confidence.

**What changed:** `domain_classifier_drift` now raises `ValueError` at call time
if 1/(n_permutations+1) > alpha, naming a sufficient permutation count. A hard
error, not a warning — a warning in a scheduled job is a log line nobody reads.
Four regression tests in `TestPermutationResolutionFloor`.

**General lesson to carry into components 2-6:** every detector needs an explicit
answer to "what is the smallest effect this configuration could possibly detect",
and it must fail loudly when the answer is "none". Statistical power is a
configuration property, not a runtime outcome, so it can be checked up front.

---

## 2026-08-13 — Permutation testing was the runtime bottleneck; parallel axis mattered

**What happened:** the multivariate test module ran >10 minutes. Measured cost:
~600 ms per RandomForest fit, x 5 CV folds x (n_permutations+1) = ~78 s per
single detector call at defaults.

**What was tried:**

| change | result |
|---|---|
| `n_jobs=-1` inside the forest | 598 ms vs 662 ms per fit — essentially nothing |
| `HistGradientBoostingClassifier` | 1650 ms/fit — **2.8x slower** than the forest |
| `LogisticRegression` | 14 ms/fit, but cannot see the interaction-only drift the multivariate detector exists to catch |
| parallelise **across permutations**, `n_jobs=1` inside the forest | 42.7 s -> 14.3 s per call (10 cores) |
| 200 -> 100 trees | ~2x, no measurable AUC change on the test cases |

**What changed:** joblib `Parallel` over the permutation loop, forests pinned to
`n_jobs=1`, default `n_estimators` 200 -> 100. Full suite: >600 s -> 112 s.
Permutations are drawn up front from a single seeded generator so results do not
depend on joblib scheduling — pinned by a test comparing `n_jobs=-1` against
`n_jobs=1`.

**Noted for later:** `domain_classifier_drift` exposes `n_jobs`. A backtest that
parallelises across windows must pass `n_jobs=1` to avoid nested oversubscription.

---

## 2026-08-13 — The naive baseline was accidentally a strawman

**How it surfaced:** `test_blind_to_variance_only_change` failed — the mean-shift
baseline fired on a variance-only change it should not be able to see.

**Cause:** the standard error used the reference standard deviation alone. When
the current window's variance grows, the true sampling variability of its mean
grows too, but a reference-only SE does not track that — so an ordinary random
mean difference gets divided by a too-small SE and trips the threshold.

**Why it matters:** it fires as a *false alarm* that a benchmark would score as a
successful detection. That inflates the baseline's apparent recall while wrecking
its precision, and any comparison against it would have been meaningless. A
strawman baseline is worse than no baseline — it produces a favourable number
for our own detectors that does not survive scrutiny.

**What changed:** Welch two-sample standard error,
`sqrt(var_ref/n_ref + var_cur/n_cur)`. The blindness test now averages over 30
seeds so it checks calibration rather than one lucky draw.

---

## 2026-08-13 — Two smaller correctness fixes in PSI binning

- **Constant reference feature returned PSI = 0 no matter what.** The degenerate
  fallback built a single (-inf, inf) bin, which swallowed every current value.
  A feature that is supposed to be constant moving to a different constant is a
  real and serious production event, and it was undetectable. Fixed with a
  three-bin fallback (below / at / above).
- **Reported bin proportions were epsilon-clipped**, so they summed to >1 when
  bins were empty. Harmless to the statistic, wrong in any report a human reads.
  Now the true proportions are reported and clipping applies only inside the log.

---

## 2026-08-14 — Real data broke the drift core in four ways, one of them silently

**What was done:** loaded the full Lending Club dataset (2,260,668 loans,
2007-06 to 2018-12, 145 columns) and ran it through the drift core built against
synthetic data, before building anything on top of it.

**What broke:**

| case | old behaviour | severity |
|---|---|---|
| PSI / KL, all-NaN reference | `IndexError` | crash, but loud |
| Wasserstein, all-NaN reference | `ValueError` | crash, but loud |
| **KS, all-NaN reference** | **`statistic=nan, is_drifted=False`** | **silent false all-clear** |
| PSI, all-NaN *or empty* current | `statistic=6.9, is_drifted=True` | plausible number from nothing |

The KS case is the same bug as the permutation floor wearing different clothes:
a detector that cannot see, reporting that all is well. The PSI case is worse in
one respect — it could not distinguish "feature vanished" from "window has zero
rows", and returned an identical, confident, meaningless 6.9 for both.

**Why real data surfaced this and synthetic data could not:** the synthetic
fixtures always had values. Real feature availability changes over time (see
next entry), so an all-NaN window is not an edge case in production — it is
Tuesday.

**What changed:** new `drift_core/validity.py` holding the shared contract, and
a new `ResultStatus` field on every result. The distinction it encodes:

- **Configuration** errors (knowable without data — permutation floor, a
  threshold that cannot be exceeded) **raise**. There is no sensible result.
- **Data** insufficiency (per feature, per window — feature retired, window too
  small) returns `ResultStatus.INSUFFICIENT_DATA`. A sweep over thousands of
  feature-windows must not die because one feature was retired upstream, but the
  result must never be mistaken for a clean pass.

`is_drifted=False` is now insufficient on its own to conclude anything. Callers
must check `status is OK` first, and `pipeline/monitor.alert_rate` excludes
non-OK results from the denominator — counting an unevaluable feature as
evidence of stability is how this bug would come back at the reporting layer.

---

## 2026-08-14 — Lending Club has three schema eras; naive monitoring measures the vendor

**What happened:** feature availability changes twice, sharply.

| era | features | before that date |
|---|---|---|
| 2007+ | `loan_amnt`, `int_rate`, `dti`, `revol_util`, … (12) | available throughout |
| 2013+ | `tot_cur_bal`, `bc_util`, `mort_acc`, `num_sats`, … (16) | **100% null** before 2012 |
| 2016+ | `open_acc_6m`, `il_util`, `all_util`, `inq_last_12m` (4) | **100% null** before 2015 |

2012 and 2015 are partial-transition years (52% and 95% null respectively).

**Why it matters:** a feature going from absent to fully populated is a vendor
schema change. Every drift detector fires enormously on it, and the alert is
*correct* — the data did change — but it says nothing about the model. Monitoring
across 2013-01 or 2016-01 with a fixed feature list measures Lending Club's data
vendor, not model degradation.

**What changed:** `domains/finance/lending_club.py` declares the eras explicitly
and `numeric_features(era)` returns an internally-consistent set. All calibration
work is confined to a single era. A test asserts every candidate stable window
starts at or after 2013-01.

**Generalisation to the other domains:** MIMIC-IV will have the same problem in a
worse form — ICD-9 to ICD-10 transition (2015-10), changing lab panels, and
changes in what gets charted at all. The era concept is domain-agnostic in shape
but its boundaries are domain knowledge, so it stays in the adapter. Worth
checking whether `SchemaEra` should be promoted to shared infrastructure once a
second domain needs it; one instance is not yet a pattern.

---

## 2026-08-14 — Label maturity is a survivorship trap in the late vintages

**What the data shows** (`label_maturity_report`):

| vintage | n loans | resolved share | default rate among resolved |
|---|---|---|---|
| 2014 | 235,629 | 94.0% | 18.5% |
| 2015 | 421,095 | 88.7% | 20.2% |
| 2016 | 434,407 | 63.2% | 24.3% |
| 2017 | 443,579 | 35.8% | 22.9% |
| 2018 | 495,242 | **9.5%** | **14.7%** |

919,695 loans (41%) are still `Current`.

**The trap:** the 2018 default rate looks like a dramatic improvement. It is not.
Only 9.5% of that vintage has resolved, and on a 36-60 month product the loans
that resolve within months are a biased subset, not a sample. The apparent
improvement is the selection effect.

**What changed:** `default_label` returns **NaN**, not 0, for unresolved loans,
and a test pins it. Encoding `Current` as "did not default" is the most common
misuse of this dataset and it biases default rates downward by an amount that
grows with vintage recency — which reads as a favourable trend.

**Consequence for the headline result:** label-confirmed validation must be
restricted to vintages with enough seasoning, or use a fixed observation horizon
(e.g. default within 12 months of origination) applied uniformly. Using
"whatever has resolved by the snapshot date" would make measured performance a
function of vintage age. This constrains component 2's validation design and is
not yet decided.

---

## 2026-08-14 — KS is unusable for alerting at production window sizes

Full write-up: [reports/calibration/THRESHOLD_CALIBRATION.md](../reports/calibration/THRESHOLD_CALIBRATION.md)

**Two measurements, both on Lending Club schema era 2013+:**

| | KS | PSI | Wasserstein |
|---|---|---|---|
| Intrinsic FPR (synthetic null, n=20k) | 4.3% | 0.0% | 0.3% |
| Realistic alert rate (real windows) | **81.0–93.6%** | 0.9–3.6% | 9.5–35.7% |

**What happened:** KS is *correctly calibrated* — 2-4% FPR against nominal 5%
under a synthetic null at every sample size tested. On real "stable" windows it
fires on 81-94% of tests.

**Why both are true:** the synthetic null holds the distribution exactly fixed.
Real quarterly credit data never is. At n=40,000 per window KS detects
differences far too small to matter operationally. It asks "is there *any*
difference" and the answer on real data is always yes.

**The design inconsistency this exposed:** `domain_classifier_drift` already
requires *both* significance (`p < alpha`) and effect size (`auc >= threshold`)
before alerting — added deliberately because "a significant AUC of 0.53 on 100k
rows is real and operationally meaningless". `detect_ks_drift` has no such gate;
it alerts on `p < alpha` alone. The same reasoning applies and was not applied.
Must be fixed before component 5.

**Not error, and worth stating carefully:** roughly `realistic - intrinsic` is
genuine drift, so KS is correctly reporting that ~80% of feature-quarters
contain statistically real movement. It just cannot say which of it matters.
Multiple-testing correction will not help — the problem is not the family-wise
error rate, it is that the null being tested is not the question anyone cares
about.

---

## 2026-08-14 — PSI's 0.1/0.25 thresholds are a sample-size accident

**What happened:** identical thresholds, synthetic null, varying only n:

| rows per half | PSI FPR | null expectation |
|---|---|---|
| 250 | **14.6%** | 0.072 |
| 1,000 | 0.0% | 0.018 |
| 5,000 | 0.0% | 0.0036 |
| 20,000 | 0.0% | 0.0009 |

At n=250 the 0.1 watch threshold sits essentially on the noise floor. At
n=20,000 it is 111x above it, so nothing short of catastrophe trips it — the
detector is effectively switched off while appearing to be on.

Wasserstein is worse at small n: **50.5% FPR at n=250**, because the normalised
statistic's null median (0.1002) lands exactly on the 0.1 threshold.

**Conclusion:** the industry convention of quoting PSI 0.1/0.25 with no
reference to window size is unsupportable. Component 5 must express thresholds
as a multiple of the analytic noise floor (`psi_null_expectation`,
`ks_minimum_detectable_effect` in `drift_core/validity.py`), not as constants.

---

## 2026-08-14 — No genuinely stable period exists in Lending Club

Flagged in advance as an acceptable outcome. It is the outcome.

Candidate A (2013H2-2014) is the most defensible window by exogenous criteria —
ZIRP throughout, no company event, unemployment declining smoothly. It produced
the **highest** Wasserstein alert rate of any candidate, 35.7% against 9.5-11.9%
elsewhere.

That is not a detector failure. 2013H2-2014 was Lending Club's steepest growth
phase, roughly +75% origination volume across the window. The applicant
population moved a great deal with no macro shock and no scandal.

**The lesson, which generalises beyond this dataset:** absence of an exogenous
shock does not imply a stable population. The most macro-stable window was the
least population-stable one. Any monitoring design that assumes "quiet period =
stable reference" is assuming something that was false here.

**Consequence:** threshold calibration is regime-dependent and the dependence is
large — Wasserstein 9.8% (period B) vs 35.7% (period A) for identical
thresholds, a 3.6x alert-volume swing. Every FPR figure from this project must
be reported as a range with the regime named. Single figures are statements
about one regime and should not be quoted.

---

## 2026-08-14 — The maturity fix is a fixed horizon, not a tail trim, and it *recovers* data

**The decision the previous entry left open.** Four rules were measured against
the raw file rather than argued about:

| rule | issue cutoff | rows kept (era 2013+) | quarters | share of eventual charge-offs captured |
|---|---|---|---|---|
| 12-month horizon | 2018-03 | 1.78M (82%) | 21 | 42.9% |
| **24-month horizon** | **2017-03** | **1.32M (61%)** | **17** | **80.5%** |
| full 36-mo term, 36-mo loans only | 2016-03 | 642k (30%) | 13 | ~100% |
| full maturity, all terms | 2014-03 | 182k (8%) | 5 | 100% |

**The measurement that decided it:** among charged-off loans the median gap from
origination to last payment is **14 months**, and only 42.9% stop paying inside
12 months. The obvious 12-month horizon would have defined away 57% of the risk
it claimed to measure. Full 36-month maturity forces dropping every 60-month
loan — a systematically riskier population, so the model's scope narrows — and
leaves too few windows to measure a latency in.

**The non-obvious part:** a fixed horizon is not the same move as cutting the
recent tail. Cutting the tail keeps only resolved loans and throws the 919,695
`Current` loans away. A fixed horizon *recovers them as valid negatives* — a
loan that would have defaulted inside 24 months would already read
`Charged Off`, so one still `Current` at 28+ months has genuinely survived.
Inside the observable range there is no maturity bias left to correct, not
merely less of it. Labelled share of the kept data is **100.0000%**.

**What it cost, measured** (`scripts/report_maturity_cut.py`):

- 1,285,664 rows kept of 2,260,668 — **56.9%**, discarding 975,004
- issue_d 2007-06 .. **2016-11** (snapshot 2019-03, cutoff = snapshot − 24 − 4)
- 39 quarters total; monitoring is confined to schema era 2013+, so **2013-01
  .. 2016-10** is the usable range
- overall 24-month default rate 12.75%, 163,917 positives

**The bias, before and after.** Same vintages, two label definitions:

| vintage | snapshot-resolved rate | 24-mo horizon rate |
|---|---|---|
| 2013 | 15.60% | 10.66% |
| 2014 | 18.54% | 11.95% |
| 2015 | 20.16% | 13.38% |
| 2016 | 24.27% | 13.65% |
| 2017 | 22.90% | *not observable* |
| 2018 | **14.73%** | *not observable* |

The snapshot column has the wrong sign at the end, exactly as predicted — 2018
reads as the second-best vintage on record when only 9.5% of it has resolved.
The horizon column is monotone and modest.

**The booking-lag buffer, and why it is not decoration.** Lending Club books a
charge-off at ~120+ days delinquent, so a loan that stops paying in month 24
does not *show* as charged off until ~month 28. Without the 4-month buffer the
newest kept vintage would be under-labelled — the same survivorship bias pushed
to the boundary instead of the tail. Measured residual with the buffer in place:
**319 loans, 0.025% of kept rows**, stopped paying inside the horizon but are
still `Late` rather than charged off and are labelled 0. Small enough to state
and move on; it is reported by `horizon_maturity_report` rather than assumed.

**What is honestly given up:** this measures 24-month default, not lifetime
default. 19.5% of eventual charge-offs happen after month 24 and are labelled 0
here. That is a real limitation of the ground truth and every result in this
repo inherits it.

**Two consequences carried forward:**

1. **2016Q4 is a partial quarter** (cutoff falls at 2016-11-01, so only Oct–Nov).
   Its default rate reads 12.31% against 14.35% in 2016Q3, and some of that gap
   is composition rather than signal. Incomplete boundary windows must be
   dropped by the backtest, not analysed.
2. **The target moves, which is the precondition for the whole project.**
   Quarterly 24-month default rate runs 10.39% (2013Q4) → 14.35% (2016Q3), a
   monotone ~38% relative increase. Whether that degrades *model discrimination*
   (AUC) rather than just the base rate is a separate question and is not yet
   answered — a rising base rate alone does not degrade ranking performance.

---

## 2026-08-14 — The model does not degrade on AUC. It degrades on calibration, badly.

This is the load-bearing finding of the project and it invalidated the metric
the headline was originally designed around.

**Setup:** baseline `HistGradientBoostingClassifier`, 28 numeric + 6
categorical origination-time features, trained on 2013-01..2013-06 (53,374
loans), reference holdout 2013-07..2013-12 (81,440, never trained on), then
frozen and scored across 35 monthly windows to 2016-11 (1,054,948 loans).
Labels are the 24-month horizon label throughout.

**Discrimination is flat. There is no AUC degradation to detect.**

| | AUC |
|---|---|
| reference holdout (2013 H2) | 0.6719 |
| final window (2016-11) | 0.6661 |
| worst window (2014-01) | 0.6522 |
| mean, first half of monitoring | 0.6740 |
| mean, second half | **0.6832** |

Second half is *better* than first half by +0.0092. Total drift −0.0059, well
inside window-to-window noise. Had the project defined "true performance drop"
as an AUC decline — the default choice, and the one the original design
assumed — the honest answer would have been "no drop ever occurred, the
headline cannot be computed" and that would have been the entire result.

**Calibration degrades substantially, and in the dangerous direction.**

| | Brier | base rate | mean predicted | gap |
|---|---|---|---|---|
| reference (2013 H2) | 0.0916 | 0.1054 | 0.1035 | −0.0019 |
| 2015-12 | 0.1190 | 0.1434 | 0.0921 | **−0.0513** |
| 2016-05 | 0.1171 | 0.1415 | 0.0952 | −0.0463 |
| 2016-11 | 0.1069 | 0.1256 | 0.0997 | −0.0259 |

Brier rises 0.0916 → 0.1069, **+17% relative**. The calibration gap widens
from −0.0019 to −0.0513, a **27x** growth. At the worst point the model
under-states portfolio default risk by roughly 36% relative.

**This is label-confirmed concept drift, and the direction proves it.** If the
applicant pool had simply become riskier — a pure `P(X)` shift — the model
would have seen the riskier inputs and its mean prediction would have risen
with the base rate. The opposite happened: the base rate rose (0.1054 →
0.1256, +19%) while mean predicted risk *fell* (0.1035 → 0.0997). The
observable covariates say "slightly safer"; the outcomes say "materially
riskier". The same X defaults more than it used to. That is a change in
`P(Y|X)`, and it is confirmable here only because matured labels are in hand —
consistent with the impossibility result, not in tension with it.

**Why this is the better result, not a consolation prize.** A model whose
ranking is intact and whose probabilities have drifted 36% low is the precise
technical meaning of *silent failure*. Every standard monitoring dashboard
watches AUC. This model would have passed AUC monitoring for three years while
systematically under-pricing risk — which for a credit model is the failure
that costs money, since approval cutoffs and pricing consume the probability,
not the rank.

**What changed as a result:**

1. **"True performance drop" is defined on Brier score and the calibration
   gap, not AUC.** Both are reported in every window
   (`model.baseline.evaluate` returns AUC, Brier, base rate, and mean
   prediction together, deliberately) and the AUC null result is reported
   next to the headline rather than buried.
2. **Monthly windows, not quarterly.** All 35 are complete under the maturity
   cut, and latency in months is a more useful number than latency in
   quarters. The partial-quarter artifact noted in the maturity entry affects
   quarterly aggregation only.
3. **Component B's expected failure mode is now specific and predictable.**
   Confidence-based performance estimation reads the model's own probabilities
   to infer how it is doing. Those probabilities are exactly what is broken
   here. A confidence-based estimator should therefore be *structurally blind*
   to this degradation — it should report that everything is fine, right
   through a 36% under-pricing. That is a testable prediction and B is built
   to test it rather than to succeed.

**Corroboration worth noting:** the 2016-07 window is the first sharp break in
several series at once — AUC falls to 0.6647, mean prediction jumps back to
0.1031 from 0.0897. That is two months after Lending Club's CEO resigned
(2016-05, already recorded in `domains/finance/stable_periods.py` as a
structural break, before any of this was measured).

---

## 2026-08-14 — The ESS guardrail was disabled by the clipping meant to protect it

**How it surfaced:** `test_no_common_support_suppresses_every_metric` built two
windows six standard deviations apart — no meaningful support overlap — and
expected the importance-weighted estimator to suppress itself. It returned a
confident base rate of **0.534** instead, with a healthy-looking effective
sample size.

**Cause:** the density-ratio fitter clips weights at the 99th percentile to
control variance, and effective sample size was computed *after* the clip.
Clipping caps exactly the enormous weights that ESS exists to detect. On a
window with almost no overlap the raw ratios span orders of magnitude; after
clipping they are bounded and evenly spread, and Kish ESS reads as healthy.
The guardrail was measuring the output of its own variance control.

**Why it belongs in this log rather than a commit message:** this is the same
failure as the permutation floor and the all-NaN KS result, in a third
costume — a check that cannot see, reporting that all is well. Three
independent instances in one codebase is a pattern, not bad luck. The common
shape: *a safety check computed downstream of a transformation that removes
the evidence it looks for.*

**What changed:**

1. ESS is computed on **unclipped** ratios, before the cap is applied. Pinned
   by `test_ess_is_measured_before_clipping`, which fails if the computation
   is ever moved back below the clip during a tidy-up.
2. A second, independent guardrail: the **discriminator AUC**. Importance
   weighting requires common support, and separability is the direct test of
   it — an AUC near 1 means a classifier can tell every reference row from
   every current row, so the density ratio is extrapolating rather than
   reweighting. On the failing case it read 0.9996, which was already sitting
   in the diagnostics under a misleading name (`discriminator_auc_proxy`,
   which was actually just a mean predicted probability, not an AUC).

Both fire on the test case. They are redundant there and independent in
general, which is the argument for keeping both.

---

## 2026-08-14 — Component B: the label-free estimators did not just fail, they pointed the wrong way

Predicted in the AUC/calibration entry above, before the estimators were run:
"a confidence-based estimator should be structurally blind to this
degradation — it should report that everything is fine." The prediction was
too generous.

**Measured over 35 monthly windows, against matured 24-month labels:**

| metric | method | coverage | mean relative error | worst | same direction every window? |
|---|---|---|---|---|---|
| base rate | average confidence | 35/35 | **−23.4%** | 35.7% | yes |
| Brier | average confidence | 35/35 | **−24.5%** | 35.2% | yes |
| base rate | importance weighted | **4/35** | −19.2% | 30.5% | yes |
| Brier | importance weighted | 4/35 | −18.0% | 28.5% | yes |

**Blind, not inverted.** *(This paragraph was rewritten on 2026-08-14 after the
original "anti-correlated" claim failed verification — see the correction entry
dated below. The numbers here are the ones that survive.)*

The estimate is **flat while the truth moves**:

| | Newey-West HAC slope per window | p | 95% CI |
|---|---|---|---|
| true base rate | **+0.000753** | **0.0002** | [+0.00036, +0.00115] |
| estimate | −0.000208 | 0.099 | [−0.00046, +0.00004] |
| **gap (truth − estimate)** | **+0.000962** | **0.0016** | [+0.00037, +0.00156] |

The truth trends up robustly. The estimate has no statistically detectable
trend at all. The gap between them widens significantly, from +0.0048 in the
first window to a maximum of +0.0512.

And it is biased low in **every single window**: 35/35 below truth, sign test
p = 5.8×10⁻¹¹, and still p = 2.4×10⁻⁴ if the whole series is conservatively
discounted to the 13 effectively independent observations that its serial
correlation implies.

**It does track short-run movement, which makes the failure more specific.**
Correlation of *first differences* — month-over-month change in the estimate
against month-over-month change in the truth — is **r = +0.75** (n = 34,
serial-correlation-adjusted p = 1.2×10⁻⁵; the differenced series are
near-white, lag-1 −0.43 and −0.32, so this test is trustworthy in a way the
level correlation was not).

So the estimator is responsive to month-to-month fluctuation and completely
without purchase on the secular drift. It is not lying about the direction of
change; it simply never moves far enough, in a world that kept moving. For
degradation monitoring that is the useless half of the job to get right.

**Importance weighting did not rescue it, and the way it failed is the useful
part.** This was the sharp experiment: importance weighting is *valid* under
covariate shift, so if it had fixed the estimate, the degradation would have
been a `P(X)` problem and the model would have been fine.

It answered **4 of 35 windows** and suppressed the rest. The discriminator AUC
between reference and current windows ran 0.897 at 2014-01 to **0.9998** by the
end — reference and current become almost perfectly separable, so there is no
common support to reweight across and the density ratio is extrapolation.
**Importance-weighted estimation had a shelf life of four months on this
data.** On the four windows it was willing to answer it was still wrong by
−19%, still in the same direction.

That is the empirical counterpart to the impossibility argument. Correcting
for everything observable about `P(X)` left the error essentially intact,
because the error was in `P(Y|X)`.

**Accuracy estimation is measuring nothing here, and that indicts a whole
literature branch on this kind of target.** ATC and difference-of-confidences
estimate *accuracy*. On a 12% base rate the model crosses the 0.5 threshold on
0.15–0.50% of loans, so accuracy is arithmetically pinned to the majority
class: `corr(accuracy, 1 − base_rate) = 0.9994`. ATC's estimate looks
respectable (+2.5% error, r = 0.05) precisely because it is tracking a
quantity that carries no information about the model. Reporting an accuracy
estimator as "working" on imbalanced credit data would be the most easily
missed error in this project.

**What was NOT done:** none of these estimators were tuned. The first
configuration run is the configuration reported. Tuning against the validation
set is the specific thing component B exists to avoid, and a −23% error found
honestly is worth more than a 5% error found by fitting to the answer.

---

## 2026-08-14 — Component C, THE HEADLINE: no unsupervised signal gave genuine early warning

**The number the project was built to produce, stated plainly: on this data
the answer is zero, and two of the six signals report a positive latency only
because they were already firing.**

**Onset of label-confirmed degradation** (3 SD above the healthy reference
band, sustained 2 windows; reference statistics from the six holdout months
only, computed before any signal was scored):

| onset metric | onset window | index | windows breaching |
|---|---|---|---|
| Brier | 2014-03 | 2 | 33 / 35 |
| calibration gap | 2014-05 | 4 | 31 / 35 |
| AUC | **never** | — | 0 / 35 |

**Latency, against the calibration-gap onset:**

| signal | latency | first fire | pre-onset alert rate | total alert rate |
|---|---|---|---|---|
| KS | +4 | 2014-01 | **100%** | **100%** |
| Wasserstein | +4 | 2014-01 | **100%** | **100%** |
| multivariate | +4 | 2014-01 | **100%** | **100%** |
| PSI | +2 | 2014-03 | 50% | 85.7% |
| KL | +2 | 2014-03 | 50% | 85.7% |
| prediction drift | — | never | 0% | **0%** |

KS, Wasserstein, and the domain classifier fired on the **first monitoring
window and every window after it**. Their "+4 windows of lead time" is not a
detection; it is an always-on detector being credited for the alarm it never
stopped ringing. `backtest/latency.py` reports the pre-onset alert rate
alongside the latency specifically so this cannot be quoted as a success — a
detector that fires on 100% of windows has maximum lead time on every event
and zero information.

PSI and KL are the only signals with a defensible profile, and they achieved
**+2 windows** — with half their pre-onset windows alerting, and with a drift
share of 1 feature out of 28 for most of the period.

**Prediction drift never fired, and this is the finding, not a bug.** Max PSI
on the score distribution across 35 windows was **0.0894**, under the 0.1
effect gate. The model's output distribution barely moved while its true error
grew 27-fold, because the covariates did not move in the direction of risk —
the *relationship* did. Prediction drift is a function of `P(X)`, and the
failure was in `P(Y|X)`. Constraint 1 is not merely a caveat in this project's
README; it is the measured outcome. (Honest caveat: at a 0.05 gate prediction
drift would fire on 11/35 windows, so the "never" is threshold-dependent, not
absolute. It still would not have led the onset.)

**The false-positive rate is 0.0%, which makes the above worse, not better.**
Against a synthetic null — random splits of the healthy reference period, 560
feature-window tests per method — all four univariate detectors alerted **zero
times**. The detectors are correctly calibrated. They fire constantly on real
data because real credit populations genuinely never stop moving, not because
the tests are broken. This is the same finding as the earlier KS entry,
surviving the addition of effect-size gates: the gates cut PSI/KL from
always-on to 85.7% and the first two windows to silent, but left KS and
Wasserstein at 100%.

### The part that is actually useful: the thresholding destroys the signal, not the statistic

Correlation of each **continuous** statistic with the true calibration gap
across the 35 windows:

| statistic | r vs calibration gap | r vs Brier |
|---|---|---|
| **multivariate AUC** | **0.88** | 0.80 |
| Wasserstein drift share | 0.78 | 0.66 |
| KS drift share | 0.75 | 0.67 |
| prediction PSI | 0.74 | 0.49 |
| PSI drift share | 0.41 | 0.43 |

The domain-classifier AUC rises monotonically 0.636 → 0.803 and tracks the
degradation at r = 0.88. As a *binary alert* it is worthless — it is above
threshold from window one. As a *continuous severity index* it is the best
signal in the project.

So the failure is not that unsupervised signals carry no information about
this degradation. They carry a lot. The failure is that converting them to
alerts with a fixed threshold produces either "always on" or "never on", and
neither has a latency worth reporting. **A monitoring design that reports
trend and rate-of-change rather than threshold crossings would have been
usable here; the one that was built, which is the industry-standard one, was
not.**

### Why the latency question was partly unanswerable on this data

The runway was 2–4 windows because degradation began **immediately**: 33 of 35
monitoring windows breach the Brier threshold, starting two months after the
training window ends. There was never a healthy deployment period to detect
across.

That is worth more than the number it prevented. The detection-latency framing
assumes a model that works for a while and then breaks. This model was
degrading from the moment it was deployed, which is what "the world was already
moving when you trained" looks like — and it is probably the common case rather
than the exception.

---

## 2026-08-14 — CORRECTION: "anti-correlated" did not survive verification. I over-claimed.

**What was claimed** in the component B entry above, a few hours earlier: that
the label-free estimator was *anti-correlated* with the truth (r = −0.38 for
base rate, −0.46 for Brier, both with n = 35 windows), and therefore "not
blind but actively misleading — as the model decayed the estimator reported
improvement."

**What happened when it was tested properly:** it fails. The 35 observations
are consecutive months and **both series are strongly autocorrelated** (lag-1
+0.65 for the estimate, +0.71 for the truth). A Pearson test assumes
independent observations; correlating two serially dependent, trending series
is the oldest way to manufacture significance out of nothing.

| test | base rate | Brier |
|---|---|---|
| naive Pearson | r = −0.38, **p = 0.023**, 95% CI [−0.635, −0.058] | r = −0.46, p = 0.006, CI [−0.685, −0.145] |
| effective n after serial-correlation correction | **13.0** of 35 | **12.3** of 35 |
| adjusted p | **0.195** | **0.130** |
| adjusted 95% CI | **[−0.771, +0.212]** | [−0.816, +0.160] |
| moving-block bootstrap 95% CI | **[−0.730, +0.280]** | [−0.778, +0.244] |
| first differences (trend removed) | **r = +0.748** | r = +0.747 |

Three independent ways of saying the same thing: **not significant.** The
adjusted interval spans zero comfortably, the block bootstrap spans zero, and
— most damning — *the sign flips* once the trend is differenced out. The
negative level correlation was two diverging trends, not a relationship.

**Why this one stings.** This project's entire subject is guardrails that
report a clean result while being structurally unable to see the problem. I
built four such guardrails into the detectors, wrote a section of the README
about the pattern, and then committed the identical error in the write-up: a
p-value computed downstream of an assumption that removes the evidence it
would need to be valid. The code was more careful than the prose about it.

It also would have been the single easiest thing for a reviewer to knock down.
Recomputing a correlation from a committed CSV takes about a minute.

**What replaced it** (all serial-correlation-robust, all in
`scripts/verify_correlation.py`, all reproducible from
`reports/backtest/estimation_error.csv`):

1. **Systematic underestimation — robust.** 35/35 windows below truth. Sign
   test p = 5.8×10⁻¹¹; still 2.4×10⁻⁴ discounted to 13 effectively independent
   observations. Mean relative error −23.4%.
2. **The gap widens — robust.** Newey-West HAC slope on (truth − estimate) =
   **+0.000962/window, p = 0.0016**, 95% CI [+0.00037, +0.00156]. Gap grows
   +0.0048 → +0.0512 max.
3. **The truth trends, the estimate does not.** Truth HAC slope +0.000753,
   p = 0.0002. Estimate HAC slope −0.000208, **p = 0.099, CI includes zero.**
   So "the estimator reported improvement" is *also* not supportable — it
   reported approximately nothing, which is a different and more accurate
   accusation.
4. **Short-run tracking is real.** First-difference r = +0.75, adjusted
   p = 1.2×10⁻⁵. The differenced series are near-white (lag-1 −0.43, −0.32),
   so unlike the level correlation this test can be believed.

**The defensible summary, which is the one that was predicted in advance
anyway:** the estimator is *blind*, not inverted. It moves with month-to-month
fluctuation and not at all with the drift, while sitting 23% low in every
window. The original prediction in the AUC/calibration entry — "structurally
blind" — was right. The overclaim was mine, added after seeing a correlation
coefficient and not testing it.

**Method notes, recorded because two plausible tests are wrong here:**

- A *block bootstrap of the trend slope* is invalid. Resampling blocks and
  regressing them against a fixed time index scrambles the ordering and
  destroys any trend by construction; it returns "CI includes zero" for
  everything and looks rigorous doing it. This was tried first and discarded.
  Newey-West HAC standard errors are the right tool.
- A *t-test on first differences* is near-useless for a trend: the mean first
  difference reduces to `(last − first)/(n−1)`, so it is an endpoint
  comparison carrying month-to-month variance. It returns p = 0.63 on the
  truth series that HAC OLS resolves at p = 0.0002.

**Coda — the fix itself shipped the bug once more.** The first version of
`sign_test`, discounting 35/35 to an effective n of 13.02, scaled the majority
count against the unrounded float and took a ceiling. That pushed successes
above the trial count, and `binom.sf(n, n, 0.5)` is exactly **0** — so it
printed `p = 0` for a result whose correct discounted value is 2.4×10⁻⁴. A
p-value of zero is not attainable by any finite test, and it appeared in the
very function written to stop this project from over-claiming. Caught before
the write-up, fixed by rounding the sample size first and clamping, and pinned
by `test_non_integer_override_never_yields_p_equal_zero`. Five instances now.
The pattern is not a bug type; it is a blind spot in how I check things.

---

## Open risks (not yet findings — things expected to break)

- **Importance-weighted performance estimation under regime change.** ATC/DoC
  and importance weighting degrade when current-window support does not overlap
  the reference. A real credit regime shift is exactly that scenario. Expect
  ESS collapse; the estimator should suppress its own output rather than report
  a confident wrong number. Finance domain first specifically to find out early.
- **PSI thresholds (0.1 / 0.25).** Folklore, not calibrated. Component 5 must
  replace them with thresholds that hold a measured false-positive rate on real
  stable periods. Until then, every PSI severity in this repo is provisional.
- **Feature relationship drift confounded with support shift.** The surrogate
  error can grow because the relationship changed *or* because the current
  window entered a harder region. Out-of-support mass is reported alongside for
  this reason, but the two are not yet formally disentangled.
