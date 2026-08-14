# Silent Failure Detection for Production ML

Label-free degradation monitoring across clinical, financial, and commercial models.

Deployed models don't crash when they break. They get quietly worse while still
reporting high confidence, and nobody finds out until the labels arrive — thirty
days later for a readmission model, months or years for a credit model, never for
plenty of others. This system infers likely degradation from unsupervised signals,
before labels exist, and then scores those inferences against the truth once labels
do arrive.

## The claim this project makes, stated precisely

**It does not detect concept drift without labels. That is not possible, and the
project says so.**

For a fixed deterministic model `f`, predictions are `Y_hat = f(X)`. Every
observable quantity is therefore a function of `P(X)` alone. Concept drift is a
change in `P(Y|X)` — `Y` appears in the expression, and without labels it cannot
be measured. The popular recipe "predictions drifted more than covariate shift
explains, therefore concept drift" is not merely unreliable, it is arithmetically
empty: importance-weighting reference predictions by the true density ratio
reproduces the current prediction distribution exactly. That identity is pinned as
a test in [tests/drift_core/test_concept.py](tests/drift_core/test_concept.py).

What the project actually does:

1. Detect **data drift** (`P(X)` moved) unsupervised, at the time it happens.
2. Detect **prediction drift** (`P(Y_hat)` moved) unsupervised.
3. Compute unsupervised **risk proxies** — conditions under which a concept change
   is more likely to have occurred, or more likely to hurt if it did.
4. Once delayed labels arrive, **confirm** whether performance actually dropped.
5. Report how many windows earlier the unsupervised signal fired — the
   **detection latency** — and how accurate the label-free performance estimate was.

Step 5 is the headline result, and it is retrospective by construction. The live
monitor cannot report its own detection latency, because that requires labels it
does not yet have. Anything claiming otherwise is selling something.

## Repo structure

```
drift_core/          Domain-agnostic. Imports nothing below it. Ever.
  types.py           DriftResult, DriftKind, Severity, WindowSpec
  univariate.py      PSI, KL divergence, KS test, Wasserstein
  multivariate.py    Domain classifier + permutation null
  prediction.py      Output-distribution drift
  concept.py         Data/concept separation; proxies vs label-confirmed
  baselines.py       Naive comparators every detector is benchmarked against

domains/             One adapter per domain. Knows columns, dates, semantics.
  clinical/          MIMIC-IV
  finance/           Lending Club
  commercial/        e-commerce / ad conversion

pipeline/            Windowing, scheduled runs, backtests, detection latency
dashboard/           Streamlit
reports/             Governance output
docs/findings_log.md What didn't work — maintained as work happens
tests/               One test module per core module
```

### The module boundary is the project's central claim

"One drift core, three domains" is only interesting if the core genuinely doesn't
know which domain it's running on. Two mechanisms enforce it:

- an `import-linter` contract in `pyproject.toml` (CI)
- [tests/test_module_boundaries.py](tests/test_module_boundaries.py) (local
  `pytest`), which AST-parses every core module and fails on imports from
  `domains`/`pipeline`/`dashboard`/`reports`, *and* on domain vocabulary appearing
  in any identifier or runtime string — docstrings excluded, since the core's
  docstrings legitimately discuss what it refuses to know about

If a new domain requires a change to `drift_core`, that is a design failure and it
goes in the findings log.

## The drift core interface

Domain adapters hand the core plain arrays plus opaque feature names. The core
hands back typed results. Nothing domain-specific crosses the boundary:

```python
from drift_core import detect_psi_drift, domain_classifier_drift, WindowSpec

window = WindowSpec(window_id="2019-Q3", n_samples=len(current_df))

# Per-feature
result = detect_psi_drift(
    reference_df["some_feature"], current_df["some_feature"],
    feature_name="some_feature", window=window,
)

# Joint — catches correlation-structure drift the univariate tests cannot see
mv = domain_classifier_drift(
    reference_matrix, current_matrix, window=window,
    feature_names=list(reference_df.columns),
)
```

`window_id` is an opaque string. The core never parses it as a date; mapping
windows to calendar time is the domain adapter's job.

## Design decisions worth defending

**Domain-classifier drift reports a permutation p-value, not a raw AUC.** With
enough features relative to window size, a flexible classifier separates two
samples of the *same* distribution. The reported evidence is the observed
out-of-fold AUC against a null built by shuffling group labels and refitting. AUC
gives effect size, the p-value gives evidence, and severity is gated on both — a
significant AUC of 0.53 on 100k rows is real and operationally meaningless.

**Naive baselines are first-class code, not a notebook afterthought.**
`drift_core/baselines.py` holds a mean-shift test and a missingness test. If they
match the sophisticated detectors on real data, that gets reported. The baselines
are also deliberately *fair* — the mean-shift baseline uses a Welch two-sample
standard error, because the reference-only version false-alarms on variance change
and a strawman baseline would flatter our own detectors.

**Thresholds in component 1 are provisional.** The PSI 0.1/0.25 convention is
folklore. Component 5 replaces these with thresholds calibrated to hold a measured
false-positive rate on real stable periods, with multiple-testing correction across
features and windows. Until then every severity in this repo is marked provisional.

## Status

- [x] Component 1 — drift core: univariate, multivariate, prediction, concept
      separation, baselines. 104 tests passing, import-linter contract green.
- [ ] Component 2 — label-free performance estimation *(expected to be the
      hardest; see open risks in the findings log)*
- [ ] Component 3 — subgroup monitoring
- [ ] Component 4 — calibration decay
- [ ] Component 5 — alerting with multiple-testing correction
- [ ] Component 6 — governance reporting

Domain order: finance (Lending Club) first — no credentialing wait, and real
regime changes that stress the label-free estimator where it is weakest. MIMIC-IV
follows once credentialing completes, and serves as the real test of whether the
core is genuinely domain-agnostic.

## Development

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev,dashboard]"
./.venv/Scripts/python.exe -m pytest tests/ -q
```

Read [docs/findings_log.md](docs/findings_log.md) before extending anything. It
records what was tried and rejected, and why — including the impossibility result
above, which constrains what the rest of the project is allowed to claim.
