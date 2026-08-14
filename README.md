# Silent Failure Detection for Production ML

A credit-default model, frozen after training, deployed across 35 monthly
windows of Lending Club originations (1,054,948 loans, 2014-01 to 2016-11),
monitored with **unsupervised signals only** — then graded against labels that
took two years to arrive.

The question: **how many windows before a measurable performance drop did an
unsupervised signal fire?**

**The answer on this data is zero, and getting to a trustworthy zero was the
work.** Three of six signals report positive lead time only because they never
stopped firing.

---

## Findings

### 1. The model never lost discrimination. It lost calibration, badly.

AUC went 0.6719 → 0.6661 across three years — inside the noise, and the second
half of monitoring scored *higher* than the first (0.6832 vs 0.6740). A
project that defined degradation as "AUC fell" would have concluded nothing
happened and had no result to report.

| | Brier | base rate | mean predicted | gap |
|---|---|---|---|---|
| reference (2013 H2) | 0.0916 | 0.1054 | 0.1035 | −0.0019 |
| 2015-12 | 0.1190 | 0.1434 | 0.0921 | **−0.0513** |
| 2016-11 | 0.1069 | 0.1256 | 0.0997 | −0.0259 |

Brier **+17% relative**. The calibration gap widened **27-fold**. At the worst
point the model under-stated portfolio default risk by ~36% relative.

The direction rules out the easy explanation. If applicants had simply become
riskier, the model would have seen it in the covariates and its mean prediction
would have risen with the base rate. Instead the base rate rose 19% while mean
predicted risk *fell*: `corr(mean predicted risk, true base rate) = −0.38`.
The same inputs default more than they used to — a change in `P(Y|X)`,
confirmable only because matured labels are now in hand.

This is what silent failure means technically. A model with intact ranking and
probabilities 36% low passes AUC monitoring indefinitely while systematically
under-pricing risk — and pricing and approval cutoffs consume the probability,
not the rank.

### 2. The label-free estimators did not just fail. They pointed the wrong way.

Predicted in the findings log *before* they were run: a confidence-based
estimator should be structurally blind to this, because it reads the model's
own probabilities to decide whether those probabilities can be trusted. The
prediction was too generous.

| metric | method | coverage | mean relative error | same direction every window? |
|---|---|---|---|---|
| base rate | average confidence | 35/35 | **−23.4%** | yes |
| Brier | average confidence | 35/35 | **−24.5%** | yes |
| base rate | importance weighted | **4/35** | −19.2% | yes |

Correlation between estimate and truth: **r = −0.38** (base rate), **−0.46**
(Brier). Not flat — *anti-correlated*. As real performance decayed the
estimator reported it improving. A monitoring system built on it would have
drawn a reassuring downward trend line through the worst of the degradation.

**Importance weighting had a shelf life of four months.** This was the sharp
experiment: importance weighting is *valid* under covariate shift, so if it had
fixed the estimate, the degradation would have been a `P(X)` problem and the
model would have been fine. It answered 4 of 35 windows and suppressed the
rest. The reference-vs-current discriminator AUC ran 0.897 → **0.9998**: the
windows become almost perfectly separable, so there is no common support to
reweight across. On the four windows it would answer, it was still wrong by
−19%, still in the same direction.

Correcting for everything observable about `P(X)` left the error intact,
because the error was in `P(Y|X)`.

**Accuracy-based estimation (ATC, difference-of-confidences) is measuring
nothing on this target.** At a 12% base rate the model crosses 0.5 on 0.15–0.50%
of loans, so `corr(accuracy, 1 − base_rate) = 0.9994`. ATC's estimate looks
respectable (+2.5% error) precisely because it tracks a quantity carrying no
information about the model. Reporting it as "working" would have been the most
easily missed error in the project.

Nothing here was tuned. The first configuration run is the one reported.

### 3. The headline: no signal gave genuine early warning

Onset of label-confirmed degradation — 3 SD outside the healthy reference band,
sustained 2 windows, reference statistics computed from the six holdout months
before any signal was scored:

| onset metric | onset window | windows breaching |
|---|---|---|
| Brier | 2014-03 | 33 / 35 |
| calibration gap | 2014-05 | 31 / 35 |
| AUC | **never** | 0 / 35 |

| signal | latency | first fire | pre-onset alert rate | total alert rate |
|---|---|---|---|---|
| KS | +4 | 2014-01 | **100%** | **100%** |
| Wasserstein | +4 | 2014-01 | **100%** | **100%** |
| multivariate | +4 | 2014-01 | **100%** | **100%** |
| PSI | +2 | 2014-03 | 50% | 85.7% |
| KL | +2 | 2014-03 | 50% | 85.7% |
| prediction drift | — | never | 0% | **0%** |

KS, Wasserstein and the domain classifier fired on the first monitoring window
and every window after. Their "+4 windows of lead time" is an always-on
detector being credited for an alarm it never stopped ringing. The pre-onset
alert rate is reported beside every latency for exactly this reason — a
detector firing on 100% of windows has maximum lead time on every event and
zero information.

**Prediction drift never fired, and that is the finding rather than a bug.**
Max PSI on the score distribution was 0.0894, under the 0.1 gate. The output
distribution barely moved while true error grew 27-fold, because the covariates
did not move in the direction of risk — the relationship did. *(At a 0.05 gate
it would fire on 11/35 windows, so "never" is threshold-dependent. It still
would not have led the onset.)*

**False-positive rate: 0.0%** — against a synthetic null (random splits of the
healthy period, 560 feature-window tests per method) all four univariate
detectors alerted zero times. The detectors are correctly calibrated. They fire
constantly on real data because real credit populations never stop moving.

### 4. The thresholding destroys the signal, not the statistic

This is the useful half of the negative result. Correlation of each
**continuous** statistic with the true calibration gap:

| statistic | r vs calibration gap |
|---|---|
| **multivariate AUC** | **0.88** |
| Wasserstein drift share | 0.78 |
| KS drift share | 0.75 |
| prediction PSI | 0.74 |
| PSI drift share | 0.41 |

The domain-classifier AUC rises monotonically 0.636 → 0.803 and tracks
degradation at r = 0.88. As a binary alert it is worthless — above threshold
from window one. As a continuous severity index it is the best signal in the
project.

The unsupervised signals carry substantial information about this degradation.
Converting them to alerts with a fixed threshold produces either "always on" or
"never on", and neither has a latency worth reporting. **A design reporting
trend and rate-of-change would have been usable here. The threshold-crossing
design — the industry-standard one — was not.**

### 5. Why the latency question was partly unanswerable

The runway was 2–4 windows because degradation began *immediately*: 33 of 35
windows breach the Brier threshold, starting two months after the training
window ends. There was never a healthy deployment period to detect across.

That is worth more than the number it prevented. The detection-latency framing
assumes a model that works for a while and then breaks. This one was degrading
from the moment it was deployed — what "the world was already moving when you
trained" looks like, and probably the common case rather than the exception.

---

## What didn't work

The section that took the longest and is worth the most.

**The metric the project was designed around.** Detection latency was to be
measured against an AUC drop. AUC never dropped. The headline had to be
redefined onto Brier and the calibration gap after the measurement came back,
and the AUC null result is reported above rather than quietly dropped.

**The same bug, three times: a guardrail that cannot see.** Each was caught by
a test, in a different module, each reporting "all clear" while structurally
incapable of reporting anything else.

| | symptom | cause |
|---|---|---|
| permutation floor | AUC 1.0 reported as no drift | `1/(n_perm+1) = 0.0625 > alpha` |
| all-NaN KS | `statistic=nan, is_drifted=False` | no validity gate before the test |
| ESS after clipping | confident estimate on disjoint windows | clipping caps the very weights ESS looks for |

The shared shape: **a safety check computed downstream of a transformation that
removes the evidence it looks for.** Three instances is a pattern, so the
response was structural — `drift_core/validity.py` holds the contract, every
result carries a `status` distinct from its severity, and `is_drifted=False` is
never sufficient on its own to conclude anything.

**A strawman baseline that flattered us.** The mean-shift baseline used a
reference-only standard error, so it false-alarmed on variance-only changes — a
false alarm a benchmark would have scored as a successful detection, inflating
its recall and making any comparison meaningless. Fixed to a Welch two-sample
SE.

**Statistical significance, on its own, at production sample sizes.** KS was
correctly calibrated (4.3% FPR against a synthetic null) and alerted on 81–94%
of real feature-quarters. It answers "is there *any* difference", and on real
credit data the answer is always yes. Every detector now requires an effect
size too — which cut PSI/KL to 85.7% and left KS and Wasserstein at 100%.

**PSI's 0.1/0.25 thresholds.** A sample-size artifact: 14.6% FPR at n=250 where
the threshold sits on the noise floor, and 111x above the floor at n=20,000
where the detector is switched off while appearing to be on.

**"Quiet period = stable reference".** The most macro-stable window in the data
(2013H2–2014: ZIRP throughout, no company event, unemployment declining
smoothly) produced the *highest* Wasserstein alert rate of any candidate,
35.7% against 9.5–11.9% elsewhere. It was Lending Club's steepest growth phase,
+75% origination volume. Absence of an exogenous shock does not imply a stable
population.

**The multivariate detector's default settings, on real data.** At the
detector's own defaults one window took over ten minutes; the 35-window sweep
would have run for most of a day. Subsampled to 4,000 per side after timing
measurements. What is given up is power only, so it under-fires rather than
over-fires.

---

## What this project does NOT claim

**It does not detect concept drift without labels.** That is impossible for a
fixed deterministic model, and the repo enforces the claim in code.

For fixed `f`, `Y_hat = f(X)`, so every observable is a function of `P(X)`
alone. Concept drift is a change in `P(Y|X)`. The popular recipe "predictions
drifted more than covariate shift explains" is not merely unreliable, it is
arithmetically empty: importance-weighting reference predictions by the true
density ratio reproduces the current prediction distribution exactly. Pinned as
a test in `tests/drift_core/test_concept.py`.

So unsupervised signals are typed `CONCEPT_PROXY`, never `CONCEPT_CONFIRMED`.
Only `confirm_concept_drift_with_labels` produces the latter.

Finding 2 is the empirical counterpart: correcting for everything observable
about `P(X)` left the estimation error intact. This is not a caveat bolted onto
the project — it is the measured outcome.

Detection latency is therefore **retrospective by construction**. A live
monitor cannot report its own latency, because that needs labels it does not
have.

---

## The ground truth: a fixed 24-month horizon

41% of the dataset is still `Current`. The standard fix — keep only resolved
loans — makes recent vintages look *better*, because loans that resolve early
are a biased subset:

| vintage | snapshot-resolved default rate | 24-month horizon rate |
|---|---|---|
| 2013 | 15.60% | 10.66% |
| 2014 | 18.54% | 11.95% |
| 2015 | 20.16% | 13.38% |
| 2016 | 24.27% | 13.65% |
| 2018 | **14.73%** | not observable |

2018 reads as the second-best vintage on record with 9.5% of it resolved.

A fixed 24-month horizon applied uniformly fixes this and **recovers** data
rather than discarding it: a loan that would have defaulted inside 24 months
would already read `Charged Off`, so one still `Current` at 28+ months is a
valid negative. Labelled share of the kept range: **100.0000%**.

- **1,285,664 rows kept** of 2,260,668 (56.9%); origination cut at 2016-11
- Horizon chosen by measurement: median origination-to-last-payment among
  charged-off loans is **14 months**, so a 12-month horizon would define away
  57% of the risk it claims to measure. 24 months captures 80.5%.
- A 4-month buffer on top covers the charge-off booking lag; without it the
  newest kept vintage is under-labelled and the bias returns at the boundary.
  Measured residual: 319 loans, **0.025%**.
- Honestly given up: this measures 24-month default, not lifetime default.
  19.5% of eventual charge-offs happen later and are labelled 0.

---

## Repo structure

```
drift_core/       Domain-agnostic. Imports nothing below it.
  validity.py       Shared MDE / status contract
  univariate.py     PSI, KL, KS, Wasserstein — each gated on BOTH alpha and
                    min_effect_size
  multivariate.py   Domain classifier + permutation null
  prediction.py     Output-distribution drift (delegates to univariate)
  concept.py        Proxy vs label-confirmed separation
  baselines.py      Naive comparators

domains/finance/  Lending Club: schema eras, leakage allowlist, horizon label
model/            The monitored model. Frozen after training. Boring on purpose.
estimation/       Label-free performance estimation (B)
backtest/         Onset definition, latency, estimation-error scoring (C)
pipeline/         Time windowing
docs/findings_log.md   Maintained as work happened, not reconstructed
```

Two boundaries, each enforced by both `import-linter` and `pytest`:

- **`drift_core` imports nothing domain-specific**, and no domain vocabulary
  appears in any identifier or runtime string (docstrings excluded — the core's
  docstrings legitimately discuss what it refuses to know about).
- **`estimation` cannot import `domains`.** A "label-free" estimator with an
  import path to `default_label_within_horizon` could consume the answer it is
  graded on, and nothing at runtime would reveal it — the estimates would
  simply look excellent.

`backtest/` is separate from `estimation/` for the same reason: it holds the
scoring, the other holds the thing being scored.

## Design decisions worth defending

**Every detector requires two gates**, significance *and* effect size, for the
reasons in "what didn't work". The PSI/KL p-values come from a chi-square
asymptotic (`PSI ~ (1/n_ref + 1/n_cur)·χ²_{bins−1}`; KL is one of PSI's two
halves). That is an argument, not a theorem about the finite-sample statistic,
so it is checked against simulation rather than trusted:

| n per side | PSI | KL | KS | Wasserstein |
|---|---|---|---|---|
| 1,000 | 3.0% | 2.8% | 4.5% | 7.5% |
| 5,000 | 3.2% | 3.2% | 4.5% | — |
| 20,000 | 2.5% | 2.5% | 5.8% | 2.5% |

**Onset thresholds are calibrated to the model's own healthy variability**
(`reference_mean ± n_sd · reference_sd`), not to a convention, and computed
from holdout months before any monitored window is scored. The two free
parameters are swept, not chosen — `reports/backtest/onset_sensitivity.csv`.

**Signal fires use `is_drifted`, not the stricter ALERT tier.** ALERT is a
triage band; using it as the fire condition would mean a signal only counts
once drift is severe, systematically shortening every measured lead time.

**Schema eras are explicit.** Feature availability changes in 2013 and 2016. A
feature going from absent to fully populated is a vendor schema change; every
detector fires enormously and the alert is *correct* while saying nothing about
the model. All work is confined to era 2013+.

**Time-based splits only.** There is no random-split option anywhere.

## Reproducing

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev]"
./.venv/Scripts/python.exe -m pytest tests/ -q          # 273 tests
./.venv/Scripts/lint-imports.exe                        # boundary contracts

# Put loan.csv (Lending Club 2007-2018, ~1.1 GB) in data/raw/, then:
./.venv/Scripts/python.exe scripts/report_maturity_cut.py       # the cut
./.venv/Scripts/python.exe scripts/measure_true_performance.py  # ground truth
./.venv/Scripts/python.exe scripts/run_backtest.py              # ~90 min
./.venv/Scripts/python.exe scripts/rescore_backtest.py          # re-score only
```

Read [docs/findings_log.md](docs/findings_log.md) before extending anything. It
records what was tried and rejected and why, in the order it happened.
