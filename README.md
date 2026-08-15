# Silent Failure Detection for Production ML

A credit-default model, frozen after training on 2013 H1, deployed across 35
monthly windows of Lending Club originations (1,054,948 loans, 2014-01 to
2016-11), monitored with **unsupervised signals only** — then graded against
labels that took two years to arrive.

Ground truth is a fixed 24-month default horizon applied uniformly to every
vintage, so the labels are not a function of how long each loan has been
observed. 1,285,664 rows survive that cut, 100.0000% of them labelled.

---

## 1. Importance weighting corrected for everything observable about P(X). The error remained.

For a fixed deterministic model, `Y_hat = f(X)`, so every unsupervised
observable is a function of `P(X)` alone. Concept drift is a change in
`P(Y|X)`. The standard argument that this makes unsupervised concept-drift
detection impossible is usually left as an argument. Here it is measured.

Importance weighting is *valid* under covariate shift: reweight the labelled
reference window by the density ratio `P_cur(x)/P_ref(x)` and reference
performance becomes an unbiased estimate of current performance, no current
labels needed. So if reweighting had fixed the estimate, the degradation would
have been a `P(X)` problem and the model would have been fine.

It did not. Importance weighting answered **4 of 35 windows** and suppressed
the rest. The reference-vs-current discriminator AUC ran **0.897 → 0.9998** —
the two windows become almost perfectly separable, so there is no common
support to reweight across and the ratio is extrapolation, not correction. On
the four windows it would answer, it was still **−19%** off, still low in every
one.

Correcting for everything observable about `P(X)` left the error essentially
intact, because the error was in `P(Y|X)`. The impossibility result is not a
caveat attached to this project; it is its measured outcome.

## 2. The label-free estimators were blind, not inverted — and the correction matters

**The claim this project originally made here was wrong, and the corrected
version is weaker.** The first write-up reported the estimators as
*anti-correlated* with truth (r = −0.38, n = 35). That does not survive.

Both series are consecutive monthly observations and both are strongly
autocorrelated (lag-1 +0.65 and +0.71). A Pearson test assumes independence:

| | base rate |
|---|---|
| naive Pearson | r = −0.38, p = 0.023, 95% CI [−0.635, −0.058] |
| effective n after serial-correlation correction | **13.0** of 35 |
| adjusted | **p = 0.195**, 95% CI **[−0.771, +0.212]** |
| moving-block bootstrap | 95% CI **[−0.730, +0.280]** |
| first differences (trend removed) | **r = +0.75** — the sign flips |

Not significant, by three independent routes. The negative level correlation
was two diverging trends, not a relationship.

**What does survive**, all serial-correlation-robust:

- **Biased low in 35/35 windows.** Sign test p = 5.8×10⁻¹¹; still 2.4×10⁻⁴
  discounted to the 13 effectively independent observations the autocorrelation
  implies. Mean relative error **−23.4%**.
- **The gap widens.** Newey-West HAC slope on (truth − estimate) =
  **+0.00096/window, p = 0.0016**, 95% CI [+0.00037, +0.00156]. Gap grows from
  +0.0048 to a maximum of +0.0512.
- **The truth trends; the estimate does not.** Truth HAC slope +0.00075,
  p = 0.0002. Estimate HAC slope −0.00021, **p = 0.099, CI spans zero.** So
  "the estimator reported improvement" is also unsupportable — it reported
  approximately nothing.
- **Short-run tracking is real.** First-difference r = +0.75, adjusted
  p = 1.2×10⁻⁵ (differenced series are near-white, so this one can be believed).

The estimator moves with month-to-month fluctuation and not at all with the
drift, while sitting 23% low in every window. That is *blind*, which is what
was predicted before running it. The overclaim was added after seeing a
correlation coefficient and not testing it.

Reproduce: `scripts/verify_correlation.py`.

Nothing here was tuned. The first configuration run is the one reported.

## 3. AUC held steady while calibration collapsed

| | AUC | Brier | base rate | mean predicted | gap |
|---|---|---|---|---|---|
| reference (2013 H2) | 0.6719 | 0.0916 | 0.1054 | 0.1035 | −0.0019 |
| 2015-12 | 0.6863 | 0.1190 | 0.1434 | 0.0921 | **−0.0513** |
| 2016-11 | 0.6661 | 0.1069 | 0.1256 | 0.0997 | −0.0259 |

AUC drifted 0.6719 → 0.6661 across three years, inside the noise, and the
second half of monitoring scored *higher* than the first (0.6832 vs 0.6740).
A project defining degradation as an AUC drop would have concluded nothing
happened and had no result to report.

Meanwhile Brier rose **+17% relative** and the calibration gap widened
**27-fold**. At the worst point the model under-stated portfolio default risk
by ~36% relative.

The direction rules out the easy explanation. If applicants had simply become
riskier, the model would have seen it in the covariates and its mean prediction
would have risen with the base rate. Instead the base rate rose 19% while mean
predicted risk fell slightly. The same inputs default more than they used to.

That is what silent failure means technically: intact ranking, probabilities
36% low, passing AUC monitoring indefinitely — and pricing and approval cutoffs
consume the probability, not the rank.

## 4. One bug pattern, found three times

A safety check computed **downstream of a transformation that removes the
evidence it looks for.** Each instance was caught by a test, in a different
module, and each reported "all clear" while being structurally incapable of
reporting anything else.

| | symptom | the transformation that blinded it |
|---|---|---|
| permutation floor | AUC 1.0 reported as no drift | only 15 permutations, so the smallest achievable p (0.0625) sat above alpha |
| all-NaN KS | `statistic=nan, is_drifted=False` | NaN-dropping ran before any validity check, leaving an empty comparison |
| ESS guardrail | confident estimate on disjoint windows | weight clipping caps exactly the large weights that ESS exists to detect |

The third is the clearest. Effective sample size is the guard against a
weighted estimate secretly resting on a handful of rows. Weights were clipped
at the 99th percentile to control variance, and ESS was computed *after* the
clip — so on two windows six standard deviations apart it read healthy and
returned a confident base rate of 0.534.

**A fourth instance was in the write-up, not the code**: quoting a naive
p-value on two autocorrelated series (finding 2). The code was more careful
about this pattern than the prose was.

**A fifth was in the fix for the fourth.** The sign test written to discount
35/35 down to 13 effectively independent observations scaled the count against
an unrounded float and took a ceiling, pushing successes past the trial count.
`binom.sf(n, n, 0.5)` is exactly zero, so it reported **p = 0** — a value no
finite test can produce — inside the function written specifically to stop this
project over-claiming. Caught before it reached the write-up, pinned by
`test_non_integer_override_never_yields_p_equal_zero`.

The generalisable form: whenever a check consumes a *processed* quantity, ask
what the processing removed. If the processing removes outliers, tails, or
extreme values, and the check exists to detect outliers, tails, or extreme
values, the check is decorative. The response here was structural rather than
three separate fixes — `drift_core/validity.py` holds one shared contract,
every result carries a `status` distinct from its severity, and
`is_drifted=False` is never sufficient on its own to conclude anything.

---

## What didn't work

**No unsupervised signal gave genuine early warning.** That was the headline
number the project was built to produce, and the answer is zero.

| signal | latency | first fire | pre-onset alert rate | total alert rate |
|---|---|---|---|---|
| KS | +4 | 2014-01 | **100%** | **100%** |
| Wasserstein | +4 | 2014-01 | **100%** | **100%** |
| multivariate | +4 | 2014-01 | **100%** | **100%** |
| PSI | +2 | 2014-03 | 50% | 85.7% |
| KL | +2 | 2014-03 | 50% | 85.7% |
| prediction drift | — | never | 0% | **0%** |

Three of six fired on the first monitoring window and every window after.
Their "+4 windows of lead time" is an always-on detector being credited for an
alarm it never stopped ringing. `pre_onset_alert_rate` exists in
`backtest/latency.py` specifically to make that unquotable as a success — a
detector firing on 100% of windows has maximum lead time on every event and
zero information.

**False-positive rate is 0.0%**, which makes this worse rather than better.
Against a synthetic null — random splits of the healthy reference period, 560
feature-window tests per method — all four univariate detectors alerted zero
times. The detectors are correctly calibrated. They fire constantly on real
data because real credit populations never stop moving.

**Accuracy-based estimation (ATC, difference-of-confidences) measures nothing
here.** At a 12% base rate the model crosses 0.5 on 0.15–0.50% of loans, so
`corr(accuracy, 1 − base_rate) = 0.9994`. ATC's estimate looks respectable
(+2.5% error) precisely because it tracks a quantity carrying no information
about the model.

## Honest limits

**There was no healthy deployment period to detect across.** Degradation began
about two months after the training window ended: 33 of 35 monitoring windows
breach the Brier threshold. The runway was 2–4 windows, so the maximum
achievable lead time was 2–4 windows regardless of detector quality. That
partly makes the latency question unanswerable on this data, and it is not a
result the detectors can be blamed for.

It is also probably the common case rather than the exception. The
detection-latency framing assumes a model that works for a while and then
breaks; this one was degrading from deployment.

**Prediction drift's "never fired" is threshold-dependent.** Max PSI on the
score distribution across 35 windows was 0.0894, against a 0.1 effect gate. At
a 0.05 gate it fires on 11/35 windows. It still would not have led the onset.

**The onset definition is contestable**, so it is not defended — latency is
reported under four different ground-truth metrics
(`scripts/rescore_backtest.py`) and the two free parameters of the onset rule
are swept rather than chosen (`reports/backtest/onset_sensitivity.csv`). AUC
never breaches at all, so no latency is computable against it.

**Single dataset, single model class, one training window.** Nothing here
establishes that these results generalise.

## Design conclusion: fixed-threshold alerting is the wrong architecture

The unsupervised signals are not uninformative. Correlation of each
**continuous** statistic with the true calibration gap:

| statistic | r vs calibration gap |
|---|---|
| **multivariate AUC** | **0.88** |
| Wasserstein drift share | 0.78 |
| KS drift share | 0.75 |
| prediction PSI | 0.74 |
| PSI drift share | 0.41 |

The domain-classifier AUC rises monotonically 0.636 → 0.803 and tracks the
degradation at r = 0.88. As a binary alert it is worthless — above threshold
from window one, firing on 100% of windows.

These statistics carry their signal in their **trajectory, not their level**.
Thresholding discards exactly the part that carries information, and produces
either "always on" or "never on". Rate-of-change alerting on the same
statistics would have worked here. That is the design conclusion, and it is the
one thing in this project that would change what I built next.

*(Caveat, given finding 2: these correlations are computed on autocorrelated
series and are descriptive. They are not offered with p-values, and the same
serial-dependence correction that sank r = −0.38 would widen their intervals
substantially. The monotonicity of the multivariate AUC is the more robust
observation.)*

---

## Architecture

```
drift_core/       Domain-agnostic. Imports nothing below it.
  validity.py       Shared MDE / status contract
  univariate.py     PSI, KL, KS, Wasserstein — each gated on BOTH alpha and
                    min_effect_size
  multivariate.py   Domain classifier + permutation null
  prediction.py     Output-distribution drift (delegates to univariate)
  concept.py        Proxy vs label-confirmed separation

domains/finance/  Lending Club: schema eras, leakage allowlist, horizon label
model/            The monitored model. Frozen after training.
estimation/       Label-free performance estimation
backtest/         Onset definition, latency, estimation-error scoring
docs/findings_log.md   Maintained as work happened, including the correction
                       to finding 2
```

Two boundaries, each enforced by `import-linter` **and** `pytest`:

- **`drift_core` imports nothing domain-specific**, and no domain vocabulary
  appears in any identifier or runtime string.
- **`estimation` cannot import `domains`.** A "label-free" estimator with an
  import path to the label function could consume the answer it is graded on,
  and nothing at runtime would reveal it — the estimates would simply look
  excellent.

Every detector requires **both** a significance gate and an effect-size gate.
Significance alone: KS was correctly calibrated (4.3% FPR against a synthetic
null) and alerted on 81–94% of real feature-quarters. Effect size alone: PSI's
conventional 0.1/0.25 thresholds are a sample-size artifact — 14.6% FPR at
n=250, and 111× above the noise floor at n=20,000 where the detector is
switched off while appearing to be on.

PSI and KL p-values come from a chi-square asymptotic
(`PSI ~ (1/n_ref + 1/n_cur)·χ²_{bins−1}`; KL is one of PSI's two halves),
checked against simulation rather than trusted as algebra — measured FPR at
alpha=0.05 is 2.5–3.2% for both across n = 1,000 to 20,000.

Time-based splits only. There is no random-split option anywhere.

## Running it

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev]"
./.venv/Scripts/python.exe -m pytest tests/ -q          # 308 tests
./.venv/Scripts/lint-imports.exe                        # boundary contracts

# Put loan.csv (Lending Club 2007-2018, ~1.1 GB) in data/raw/, then:
./.venv/Scripts/python.exe scripts/report_maturity_cut.py       # the label cut
./.venv/Scripts/python.exe scripts/measure_true_performance.py  # ground truth
./.venv/Scripts/python.exe scripts/run_backtest.py              # ~90 min
./.venv/Scripts/python.exe scripts/rescore_backtest.py          # re-score only
./.venv/Scripts/python.exe scripts/verify_correlation.py        # finding 2
```

Committed CSVs under `reports/` are the evidence behind every number above.

Read [docs/findings_log.md](docs/findings_log.md) before extending anything —
it records what was tried and rejected in the order it happened, including the
entry where the headline correlation claim was walked back.
