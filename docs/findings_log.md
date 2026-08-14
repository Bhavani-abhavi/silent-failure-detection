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
