# Marginal Validity Is Not Enough

**Reliability of Conformal Uncertainty Quantification for Mortgage Delinquency Screening**

M.Sc. Data Science in Business and Economics · Eberhard Karls Universität Tübingen
Author: Moritz Maidl · Examiner: Prof. Dr. Martin Biewen · Submitted 25 August 2026

![Python](https://img.shields.io/badge/python-3.10-blue)
![LightGBM](https://img.shields.io/badge/model-LightGBM-success)
![DuckDB](https://img.shields.io/badge/data-DuckDB%20%2B%20Polars-yellow)
![Rows](https://img.shields.io/badge/panel-2.32B%20loan--months-lightgrey)
![Status](https://img.shields.io/badge/status-submitted-brightgreen)

📄 **[Full thesis (85 pp., PDF)](Master_Thesis_FINAL.pdf)**

---

## The finding in one line

Split conformal prediction hit its **90.30 %** marginal coverage target on the calibration
data and covered the delinquent class at **0.0039 %** — on the very rows the threshold was
drawn from, before any distribution shift.

| Split conformal, α = 0.10 | Marginal | Coverage (y = 1) | Worst FICO×LTV cell | PSI |
|---|---|---|---|---|
| Calibration 2005–06 *(anchor)* | 90.30 % | 0.0039 % | 7.42 % | 0.000 |
| Crisis 2007–12 | 90.51 % | 0.0007 % | 10.31 % | 0.006 |
| Normal 2013–19 | 94.55 % | 0.0002 % | 42.93 % | 0.089 |
| COVID 2020–21 | 94.83 % | 0.0000 % | 58.03 % | 0.124 |
| Rate hike 2022–23 | 96.41 % | 0.0001 % | 65.06 % | 0.199 |

Read the columns against each other. The headline rate *improves* across regimes while the
minority class stays at zero, and the drift monitor a bank would actually run (PSI) ranks the
**least** reliable window as the **most** stable. That inversion is the thesis.

---

## What this project is

An end-to-end empirical audit of conformal prediction (CP) as an uncertainty-quantification
layer for a credit-risk screening system, run on the full **Freddie Mac Single-Family
Loan-Level Dataset** (Release 46, Standard): **2,322,200,609 loan-months** from
**46,648,508 loans**, 1999–2023.

The setting is deliberately hostile to CP's assumptions: severe class imbalance
(1.04–2.53 % positive rate), repeated within-loan observations, 12-month overlapping labels,
cross-sectional macro dependence, and four distinct economic regimes. Exchangeability is
violated on four axes at once, on purpose.

**Four research questions:** (RQ1) through which mechanisms does marginal validity mislead,
and do the failures decompose into distinct kinds? (RQ2) which CP variant repairs which
failure? (RQ3) does standard drift monitoring detect any of it? (RQ4) does the CP layer
change the loans a screen selects?

---

## Pipeline

Ten notebooks, run in order. Each writes a JSON manifest consumed by the next, so the DAG is
explicit and every downstream figure traces to an upstream artefact hash.

| # | Notebook | What it does | Core tech |
|---|---|---|---|
| 01 | `01_data_preparation` | 107 quarterly ZIPs → typed, ZSTD Parquet; repartitioned by reporting period (25.5 GB); row counts reconciled to the published release totals | DuckDB, Parquet |
| 02 | `02_eda` | Schema validation, sentinel/disclosure profiling, delinquency-status inventory, vintage curves, forbearance-field analysis | DuckDB, seaborn |
| 03 | `03_panel_construction` | Loan-month panel: forward-label lookup, lagged features, regime splits, buffer removal, 10 validation checks (leakage screen, duplicate check, fast-default gap, monotonicity) | DuckDB |
| 04 | `04_lgbm_training` | Frozen base scorer: Optuna TPE search → final fit with early stopping | LightGBM, Optuna, SHAP |
| 05a | `05a_score_generation` | Isotonic recalibration + nonconformity scores (`score_scp`, `score_aps`) for every split | scikit-learn, Polars |
| 05b | `05b_cp_calibration` | All conformal thresholds: SCP, Mondrian (4 partitions), APS, ACI/DtACI state init; unit-tested quantile helper | NumPy, Polars |
| 06 | `06_coverage_evaluation` | Two-track evaluation engine — Track A static methods, Track B sequential (ACI/DtACI); year-batched with explicit memory release | Polars, SciPy |
| 07 | `07_coverage_analysis` | Marginal / class-conditional / subgroup / temporal coverage; the PSI–coverage dissociation | pandas, matplotlib |
| 08 | `08_local_reliability_audit` | Pocket-level failure: miscoverage as a supervised target, concentration curves, shallow-tree worst-segment discovery | LightGBM, sklearn trees |
| 09 | `09_operational_translation` | Prediction sets → routing decisions; review-budget curves; matched-budget capture comparison | Polars, matplotlib |
| 10 | `10_logit_robustness_benchmark` | Full parallel pipeline on a penalised logistic scorer; isolates the isotonic map's contribution | scikit-learn |

`README_notebooks.ipynb` holds the notebook-level documentation.

---

## Methods

### Base scorer

| | |
|---|---|
| Model | LightGBM GBDT, 646 trees (early-stopped), `num_leaves` 128, `max_depth` 9, lr 0.054 |
| Imbalance | `scale_pos_weight` = 80.57, computed from the fitted rows only |
| Tuning | Optuna TPE, 80 trials on a deterministic 3 % loan-level sample; `min_child_samples` rescaled to full-panel size |
| Fit | 166.3 M rows (1999–2003), 81.8 M validation rows (2004), AUROC 0.9271 |
| Missingness | Preserved, not imputed — SFLLD sentinels cleaned to nulls + explicit flags; LightGBM learns a default direction |
| Discrimination | AUROC 0.914 (calibration), 0.826–0.883 across the four test regimes |
| Calibration | Isotonic regression (PAVA) fitted on a disjoint 10 % loan-hash holdout; Brier 0.064 → 0.009, AUROC change < 0.001 |
| Benchmark | L2-penalised logistic regression, balanced weights, own isotonic map and own conformal partition — no fitted component shared |

**Features:** 29, in four families — static origination (FICO, LTV, DTI, note rate, term,
vintage, census division), dynamic current-state (UPB, 3-month balance change, 30-DPD
indicator), 12-month delinquency history, and disclosure missingness flags. Eight raw fields
are hard-excluded as servicing-intervention or temporal proxies; a training-window
correlation screen documents the leakage check.

**Target:** `y_t = 1` if the loan reaches 60+ DPD or REO acquisition within `(t, t+12]` —
a fixed-horizon discrete-time event-history design. Loans already 60+ DPD at `t` are
excluded, so the target is *onset* of distress.

### Conformal layer — five methods, one mechanism each

Selection follows a one-mechanism-per-method rule, making the comparison diagnostic rather
than a leaderboard. α = 0.10 throughout, calibrated on 161,421,966 rows.

| Method | Score / mechanism | Targets | Threshold |
|---|---|---|---|
| **SCP** | `s = 1 − p̂_y(x)`, one global quantile | baseline | `q̂ = 0.016224` |
| **Mondrian CP** | separate quantile per group (FICO×LTV, plus FICO, LTV, census variants) | subgroup heterogeneity | 0.0033 → 0.4201 across cells (**128.7×** spread) |
| **APS** | randomised generalised inverse-quantile, cumulative ranked probability mass | class-conditional collapse | `q̂ = 0.899951` |
| **ACI** | online `α_{t+1} = α_t + γ(α − err_t)`, grid γ ∈ {0.005, 0.01, 0.02, 0.05} | temporal drift | monthly, moving |
| **DtACI** | K = 500 experts, exponential aggregation, analytically set η | temporal drift, no γ choice | monthly, moving |

Static thresholds are **frozen on 2005–2006** and applied unchanged to all four test regimes,
so every coverage movement reflects distributional change and not re-fitting.

### Evaluation framework

Coverage is measured at five granularities, because a marginal rate can hold while every
finer one fails:

1. **Marginal** — full test window
2. **Class-conditional** — separately on y = 0 and y = 1
3. **Subgroup** — 10 FICO×LTV cells, summarised by CovGap (max deviation) and worst-group coverage
4. **Temporal** — 12-month rolling, resolving within-regime dynamics a per-regime average smooths away
5. **Pocket-level** — miscoverage `M_i = 1{y_i ∉ C(x_i)}` treated as a supervised target: a GBDT predicts it from the score alone vs. the full feature set (leave-one-regime-out), summarised by a concentration curve and its area, with depth-3 trees surfacing interpretable worst pockets

Plus a monitoring layer (score-PSI over 10 equal-frequency bins, and a label-free
feature-level PSI) and an operational layer: three-way routing (clear / review / escalate),
false auto-clear rate, capture, review precision and lift, and the **equal-review-budget
capture gap** — a bounded-abstention comparison at a matched accepted fraction.

**Inference convention.** At n ≈ 10⁸ with serially correlated, overlapping-label
observations, a binomial interval understates uncertainty and makes every deviation
"significant". Clopper–Pearson intervals are reported for scale, but materiality is decided
by a **±2 pp practical-significance band**, fixed in advance.

---

## Results

**RQ1 — the failure decomposes into two levels.** *Level 1 (intrinsic)*: under a ~1 % positive
rate the split threshold is effectively a quantile of the negative class alone. Mean
nonconformity score is 0.009 for performing loan-months against 0.780 for delinquent ones;
99.996 % of the latter score above `q̂`. This is complete at the calibration anchor, at zero
distributional movement. *Level 2 (distributional)*: regime shifts modulate the deviation on
top, running **upward** here. The two separate by sign and by time-dependence. Subgroup
coverage spans two orders of magnitude, 99.03 % down to 7.42 %. A model predicting miscoverage
from the calibrated score alone reaches AUROC 0.992–0.997 — the full feature set adds ≤ 0.001.

**RQ2 — a division of labour, not a ranking.**

| | Marginal | Coverage (y = 1) | CovGap | Residual failure |
|---|---|---|---|---|
| SCP | 90.5 → 96.4 % | ≈ 0 | 79.7 → 24.9 pp | everything |
| Mondrian | 89.5 → 95.3 % | 0.18–0.98 % | 5.0–7.3 pp | within-cell collapse; a 99.98 %-miscoverage pocket survives |
| APS | 89.0–89.6 % | 17.2–29.6 % | 1.3–3.9 pp | 14.3–17.5 % miscoverage above p̂ = 0.05 |
| DtACI | 90.4 → 96.0 % | ≈ 0 | — | adapts the *target* (0.887), not the coverage |

Only APS reaches materially past its own mechanism. Concentration area is 0.955–0.977 under
SCP/Mondrian against 0.533–0.579 under APS — failure is sharply localised for the first two
and near-diffuse for the third. Every repair is bought with abstention, selection efficiency,
or a lowered target.

**RQ3 — the standard monitor sees almost nothing.** PSI is *identically zero* on the calibration
window, where the failure is already complete: a statistic defined on the size of a movement
cannot register a failure present at zero movement. Across test regimes its ordering is
exactly reversed against every conditional measure. Rolling coverage falls below nominal for
27 consecutive months during the crisis while the window-level PSI reads 0.006. The label-free
feature-level variant reproduces the endpoint dissociation.

**RQ4 — the selection value is exactly zero.** At a matched non-auto-clear budget, SCP and APS
select **exactly** the loans a calibrated-probability threshold selects — Δ = +0.000 pp in
every regime, for both scorers. This is an identity, not a measurement: any clearance rule
that is a fixed lower interval of the score *is* a probability threshold. Mondrian breaks
that monotonicity and captures **3.7–8.7 pp fewer** delinquencies at the same budget — a
conformal construction performing measurably worse than the model beneath it. What screening
does happen runs through *abstention*: SCP covers the delinquent class at ≈ 0 yet routes
47.7–62.1 % of eventual delinquents to review, because capture only requires that a loan was
not auto-cleared.

All results replicate under the logistic benchmark, and under a prepayment-restricted
sensitivity re-run.

---

## Reproducibility

**This repository is a documented record, not a runnable package.** That is a deliberate
statement of scope, and the constraints are worth being explicit about:

- **Data is not redistributable.** The Freddie Mac SFLLD requires a signed user agreement.
  Obtain Release 46 (Standard) directly from Freddie Mac; NB01 reconciles ingested counts to
  the published totals within 0.3 %.
- **Scale.** The panel is 2.32 B rows. NB04 builds a 166 M-row LightGBM `Dataset` (≈ 19 GB
  of encoded features) and was run on a 40-thread high-RAM machine; final training took
  77 minutes, the Optuna search 47 minutes.
- **Environment.** Written for Google Colab with Google Drive as durable storage and local
  NVMe as scratch — paths and `google.colab` imports are hard-coded and would need replacing
  for a local run. Python 3.10.
- **Outputs are committed on purpose.** Since the pipeline cannot be cheaply re-executed,
  every notebook is stored with its executed cells, so the reported numbers are inspectable
  without re-running anything.

**What *is* reproducible** is the sampling and the artefact chain. All subsampling is
deterministic loan-level hashing (CRC32 residue classes, fixed seeds), so every monthly
observation of a loan carries the same role in every notebook, and the isotonic-fit rows
provably never enter conformal calibration. Sampling fractions: 2 % (score drift), 5 %
(local reliability, operational routing, logistic benchmark), 10 % (feature-level PSI),
100 % (model fitting, conformal calibration, coverage evaluation). Each notebook emits a
JSON manifest — `ingest_manifest`, `panel_manifest`, `score_manifest`, `cp_manifest` — that
the next one validates before running.

Core stack: `duckdb` · `polars` · `pandas` · `numpy` · `lightgbm` · `optuna` · `scikit-learn`
· `shap` · `scipy` · `matplotlib` · `seaborn` · `pyarrow`.

---

## Repository contents

```
01_data_preparation.ipynb        ingestion  → typed Parquet
02_eda.ipynb                     validation & exploratory analysis
03_panel_construction.ipynb      loan-month panel + 10 validation checks
04_lgbm_training.ipynb           frozen base scorer
05a_score_generation.ipynb       isotonic recalibration + nonconformity scores
05b_cp_calibration.ipynb         all conformal thresholds
06_coverage_evaluation.ipynb     two-track evaluation engine
07_coverage_analysis.ipynb       coverage analysis + PSI dissociation
08_local_reliability_audit.ipynb pocket-level miscoverage diagnostics
09_operational_translation.ipynb routing, capture, matched-budget comparison
10_logit_robustness_benchmark.ipynb  parallel logistic pipeline
README_notebooks.ipynb           notebook-level documentation
Master_Thesis_FINAL.pdf          full thesis (85 pp.)
```

---

## Scope and limits

Stated plainly, because the thesis states them:

- This is **not** an argument that CP is invalid. The guarantee holds wherever exchangeability
  does; what is studied is how its outputs behave where that condition is deliberately broken.
- **No new conformal method is proposed.** Existing ones are evaluated and characterised.
- Label-conditional (per-class Mondrian) CP is *not* run — a scope choice, flagged as the
  most direct untested repair.
- The four regime labels are *a priori* design hypotheses naming the dominant observable
  shift, not proven causes; four windows cannot estimate a relationship.
- Findings describe one GSE's conforming, fixed-rate book. They show how conformal
  reliability fails under stress, not how it behaves where it holds.

---

## Citation

```bibtex
@mastersthesis{maidl2026marginal,
  author  = {Maidl, Moritz},
  title   = {Marginal Validity Is Not Enough: Reliability of Conformal
             Uncertainty Quantification for Mortgage Delinquency Screening},
  school  = {Eberhard Karls Universit\"at T\"ubingen},
  type    = {Master's thesis},
  year    = {2026},
  month   = {8}
}
```

---

## License

Code released under the MIT License. The thesis text and figures are © 2026 Moritz Maidl.
The Freddie Mac Single-Family Loan-Level Dataset is subject to Freddie Mac's terms of use and
is **not** included in this repository.
