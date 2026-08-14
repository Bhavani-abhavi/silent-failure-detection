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
