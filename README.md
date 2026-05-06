# Medicare Claims Audit Intelligence Platform

**GPU-accelerated anomaly detection and audit-priority ranking for Medicare Part B claims**

Built by a federal healthcare auditor (HHS/OIG, ~11 years) to demonstrate how domain knowledge of audit-priority billing patterns can be translated into machine learning features that surface unusual billing behavior from public CMS data at population scale. Outputs are public-data decision-support signals — they do not identify fraud, overpayments, or noncompliance.

---

## Project Status

| Notebook | Status | Description |
|----------|--------|-------------|
| 01 — EDA and Domain-Informed Feature Engineering | ✅ Complete (sample) | Builds ~28 provider-service-level audit-priority features from CMS Part B data |
| 02 — Provider-Level Anomaly Detection and Weak Supervision | ✅ Complete (sample) | Aggregates service-level features into 121 provider-level columns; Isolation Forest scoring, weak-supervision audit-priority labels, LEIE cross-reference |
| 03 — Supervised Model Training (XGBoost / LightGBM) | ✅ Complete (sample) | Leakage-controlled supervised models trained on the weak-supervision label |
| 04 — Monte Carlo Audit Capacity & Prioritization Tradeoff Simulation | ⏳ Planned | Audit capacity scenarios, scenario-based recovery potential, and prioritization tradeoffs from full-population scores |

> All ✅ Complete entries above reflect a **development sample run** based on a 500,000-row provider-service slice of the CMS Part B data, not the full production run. See the [Development Sample vs. Full Production Run](#development-sample-vs-full-production-run) section for details.

---

## Problem Statement

Medicare Part B pays ~$250B annually to 1.2M+ providers across 6,000+ HCPCS procedure codes. Audit teams must prioritize limited review capacity against millions of claim lines, and rules-based filters alone miss combined or contextual patterns.

This platform applies gradient-boosted ensemble models and Monte Carlo simulation to rank providers by audit-priority across the entire provider population, using domain-informed review signals rather than manual referral alone. Rankings are a starting point for review, not findings.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  CMS Medicare Physician & Other Practitioners PUF (data.cms.gov)│
│  + Geographic Benchmarks + OIG LEIE Exclusion List              │
└───────────────────────────┬─────────────────────────────────────┘
                            │  download_data.py → load_data.py
┌───────────────────────────▼─────────────────────────────────────┐
│                   FEATURE ENGINEERING                           │
│  Domain-informed audit-priority signals (features.py):          │
│  • Charge-to-allowed ratio → submitted vs allowed ratio         │
│  • Peer specialty z-scores → outlier vs peers                   │
│  • Herfindahl index → service-mix concentration                 │
│  • Services-per-beneficiary → utilization-pattern outliers      │
│  • Place-of-service patterns → unusual facility/office mix      │
│  • Drug revenue share → J-code concentration                    │
│  • Geographic deviation → regional outlier detection            │
└───────────────────────────┬─────────────────────────────────────┘
                            │  features.py
┌───────────────────────────▼─────────────────────────────────────┐
│                      MODEL LAYER                                │
│  XGBoost + LightGBM GPU ensemble (modeling.py)                  │
│  Unsupervised anomaly detection → pseudo-label → supervised     │
│  Monte Carlo audit capacity & prioritization simulation         │
│  (illustrative recovery-potential ranges) (simulation.py)       │
│  RAPIDS cuDF for data pipeline acceleration                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │  evaluation.py
┌───────────────────────────▼─────────────────────────────────────┐
│                      OUTPUT LAYER                               │
│  Provider audit-priority rankings with confidence intervals     │
│  Specialty-level benchmarking visualizations                    │
│  Illustrative recovery-potential ranges (Monte Carlo bands)     │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Tool |
|-----------|------|
| GPU DataFrames | RAPIDS cuDF (falls back to pandas) |
| Gradient Boosting | XGBoost (CUDA), LightGBM (GPU) |
| Uncertainty Quantification | Monte Carlo simulation (NumPy/CuPy) |
| Visualization | Plotly, Matplotlib, Seaborn |
| Environment | WSL2 Ubuntu 24.04, NVIDIA RTX 5080 |
| Data Sources | CMS data.cms.gov public APIs |

## Data Sources

All data is **public, de-identified CMS data** — no PHI/PII, no DUA required:

1. **Medicare Physician & Other Practitioners — by Provider and Service**
   NPI × HCPCS-level utilization, payments, and charges (~10M rows)
2. **Medicare Physician & Other Practitioners — by Provider**
   One row per NPI — aggregate provider summary (~1.2M rows)
3. **Medicare Physician & Other Practitioners — by Geography and Service**
   State/national benchmarks for peer comparison
4. **OIG LEIE — List of Excluded Individuals/Entities**
   Known excluded providers for weak supervision / validation

## Project Structure

```
medicare-claims-audit-intelligence/
├── data/
│   ├── raw/                        # Untouched CMS downloads
│   ├── interim/                    # Intermediate transforms
│   └── processed/                  # Model-ready feature sets (Parquet)
├── notebooks/
│   ├── 01_eda_feature_engineering.py
│   ├── 02_anomaly_detection.py
│   ├── 03_model_training.py
│   └── 04_monte_carlo_simulation.py
├── src/
│   ├── __init__.py
│   ├── config.py                   # Centralized paths, hyperparams, feature lists
│   ├── download_data.py            # CMS data download pipeline
│   ├── load_data.py                # GPU/CPU data loading with dtype optimization
│   ├── features.py                 # Domain-informed audit feature builders
│   ├── modeling.py                 # XGBoost + LightGBM GPU ensemble
│   ├── evaluation.py               # Audit-specific metrics (precision@k, AUCPR)
│   └── simulation.py               # Monte Carlo recovery-potential simulation
├── models/                         # Saved model artifacts
├── reports/                        # Generated plots and analysis
├── outputs/                        # Final deliverables
├── tests/
├── docs/
├── README.md
├── environment.yml
└── .gitignore
```

## Quick Start

```bash
# 1. Create conda environment (includes RAPIDS for GPU)
conda env create -f environment.yml
conda activate medicare-audit

# 2. Download CMS data
#    The static CMS CSV URLs (Provider-Service, Provider, Geography-Service, LEIE)
#    are maintained centrally in src/config.py — update them there if CMS rotates
#    the published file paths. Do not hand-paste CMS portal URLs into commands.
python src/download_data.py

# 3. Run EDA notebook
jupyter lab notebooks/01_eda_feature_engineering.py
```

> The current ✅ Complete notebook results in this README were generated from a
> **500,000-row development sample** of the Provider-Service file. The full
> production run (complete `provider_service_2022.csv`, ~10M rows, ~1.2M
> providers) has **not** been completed yet. See
> [Development Sample vs. Full Production Run](#development-sample-vs-full-production-run).

## Domain Expertise → Feature Engineering

The value proposition isn't the algorithms — the differentiator is knowing *which features to build and why*. Every feature in `src/features.py` maps to an audit-priority pattern auditors commonly use to prioritize where to look first when working from public CMS summary data. Each feature is a public-data decision-support signal — it does not, on its own, indicate wrongdoing.

| Feature | Audit-Priority Pattern | Illustrative Example |
|---------|------------------------|----------------------|
| Charge-to-Allowed Ratio | Unusually high submitted-vs-allowed multiple | DME supplier with submitted charges of $3,000 against a $150 allowed amount |
| Peer Specialty Z-Score | Outlier vs same-specialty peers | High-throughput practice with E&M (99214) volume well above specialty median |
| HHI Concentration | Highly concentrated service mix | Pain practice with ~90% of revenue from facet-joint injections (64493-64495) |
| Services/Beneficiary | Unusual services-per-patient volume | Lab with services-per-encounter well above the typical clinical pattern |
| Facility Ratio | Unusual facility/office POS mix for the specialty | High share of facility-rate (POS 22) lines in an otherwise office-based specialty |
| Drug Revenue Share | High J-code concentration | Oncology practice with most revenue from a single Part B drug code |

## Modeling Approach

The pipeline layers three notebooks so that each stage is auditable on its own and the supervised model never sees the signals that defined its label.

- **Notebook 01 — Domain-Informed Feature Engineering.** Builds approximately **28 provider-service-level** audit-priority features from CMS Part B data (charge-to-allowed ratios, peer specialty z-scores, service-mix concentration, utilization outliers, place-of-service mix, drug revenue share, geographic deviation).
- **Notebook 02 — Anomaly Scores and Weak-Supervision Label.** Aggregates the service-level features into **121 provider-level columns** (in the current development sample run), runs Isolation Forest at provider scale, derives a composite review signal, and emits `weak_label_high_audit_priority` — a weak-supervision label that flags providers worth a follow-up review under this framework. Also performs an end-to-end LEIE cross-reference for validation.
- **Notebook 03 — Leakage-Controlled Supervised Models.** Trains Logistic Regression, XGBoost, and LightGBM against the weak-supervision label, with the anomaly-stage signals excluded from the feature set. Selects the best model by PR-AUC and scores the full sampled provider population for downstream simulation.

## Current Results

> ⚠️ **Development sample run based on 500,000 provider-service rows.**
> The metrics below were generated from a 500,000-row sample of the
> Provider-Service file used to develop and validate the pipeline end-to-end.
> They are **not** the final full Medicare Part B production-run figures.
> The full production run should be executed against the complete
> `provider_service_2022.csv` and is expected to produce a substantially
> larger provider-level population (on the order of ~1M+ providers, vs. the
> 58,866 surfaced by the development sample). All counts, prevalence rates,
> PR-AUC, precision@k, and lift values will change once rerun on the full file.

### Notebook 02 — Anomaly Detection and Weak Labels (development sample run)

| Metric | Value (development sample) |
|--------|----------------------------|
| Provider-service rows in sample | 500,000 |
| Provider-level rows (after aggregation) | 58,866 |
| Provider-level columns (final save) | 121 |
| Isolation Forest anomalies | 2,944 providers (5.00%) |
| Weak-label positives (`weak_label_high_audit_priority`) | 1,631 providers (2.77%) |
| LEIE cross-reference | Ran end-to-end |

> **Limitation noted in Notebook 02:** Population-level anomaly detection can reflect provider-type structure (specialty mix, billing norms). Future work should evaluate provider-type-specific or specialty-specific anomaly detection so that "unusual" is judged against true peer behavior rather than the overall provider population.

### Notebook 03 — Supervised Model Training (development sample run)

Dataset (development sample): 58,866 providers × 121 columns. Features used for modeling: 108. Target: `weak_label_high_audit_priority` (prevalence 2.77% in the sample).

| Model | PR-AUC | ROC AUC | Precision@100 | Precision@500 | Lift@100 |
|-------|--------|---------|---------------|---------------|----------|
| Logistic Regression | 0.8527 | 0.9956 | 0.8900 | 0.6240 | 32.14× |
| XGBoost | 0.9660 | 0.9988 | 1.0000 | 0.6420 | 36.12× |
| **LightGBM (best)** | **0.9759** | **0.9992** | **1.0000** | **0.6480** | **36.12×** |

Best model selected by Average Precision / PR-AUC: **LightGBM**, persisted to `models/best_model_2022.joblib` with companion metadata. These figures characterize the development sample run only and should be regenerated against the full production file before any external interpretation.

## Development Sample vs. Full Production Run

This repository currently reflects a **development sample run**, not a full Medicare Part B production run.

- **What the current metrics show.** The figures in [Current Results](#current-results) demonstrate that the end-to-end pipeline (download → feature engineering → provider aggregation → anomaly detection → weak-supervision label → leakage-controlled supervised models → full-population scoring) runs cleanly against a manageable dev-sized input. They are a **pipeline validation**, not a portfolio-grade headline number.
- **How they were generated.** All ✅ Complete notebook outputs were produced from a **500,000-row sample parquet** of the Provider-Service file used during development to keep iteration cycles short on a single workstation (WSL2 Ubuntu, RTX 5080).
- **What the full production run requires.** The final portfolio metrics should be regenerated from the **complete CMS Provider-Service file** (`provider_service_2022.csv`, ~10M rows) using the same notebooks. The full provider-level population is expected to be on the order of ~1M+ providers rather than 58,866, and every count, prevalence rate, PR-AUC, precision@k, and lift value should be expected to shift accordingly.
- **What to do before publication.** Before publication or recruiter review, rerun Notebooks 01 → 02 → 03 against the full file, regenerate the artifacts listed under [Generated Outputs](#generated-outputs), and update the [Current Results](#current-results) tables with the production figures.

## Leakage Controls

To keep the supervised models from trivially recovering the weak-supervision label, the following columns are excluded from the modeling feature set in Notebook 03:

- `weak_label_high_audit_priority` — the target itself
- `iso_anomaly_flag`, `iso_anomaly_score` — Isolation Forest outputs that participate in label construction
- `extreme_composite_risk`, `extreme_peer_deviation` — composite review signals that drive the weak label
- `in_leie` — LEIE cross-reference flag (held out for validation, not training)
- `Rndrng_NPI` — provider identifier
- `provider_type`, `provider_state`, `provider_entity_type` — categorical identifiers held out to keep the model from memorizing slices
- All `feat_composite_risk*` columns — composite review signals upstream of the label

This keeps the reported PR-AUC, precision@k, and lift figures interpretable as "how well the model reconstructs the weak-supervision audit-priority framework from upstream features," not "how well it memorizes its own label."

## Important Limitation

- The supervised model predicts a **weak-supervision audit-priority label**, not confirmed audit outcomes.
- It does **not** predict confirmed fraud, overpayments, noncompliance, or audit findings.
- The high PR-AUC, precision@k, and lift values reflect how well the model learned the weak-supervision framework defined in Notebook 02. They are **not** measures of confirmed real-world audit outcomes.
- Outputs are intended as **decision support** for prioritizing follow-up review of public CMS billing patterns. Any provider surfaced by the model would still require independent review under the appropriate audit process before any conclusion is drawn.

## Generated Outputs

These artifacts are produced locally by the notebooks. They are not necessarily committed — generated data, model, and report files are local outputs and may be selectively included later.

**Processed data**
- `data/processed/features_provider_service_2022.parquet`
- `data/processed/provider_features_labeled_2022.parquet`
- `data/processed/model_predictions_2022.parquet`
- `data/processed/model_predictions_full_population_2022.parquet`

**Models**
- `models/best_model_2022.joblib`
- `models/best_model_2022_metadata.json`

**Reports / figures**
- `reports/isolation_forest_anomaly_scores.png`
- `reports/provider_type_anomaly_rates.png`
- `reports/notebook03_precision_recall_curve.png`
- `reports/notebook03_xgboost_feature_importance.png`
- `reports/notebook03_lightgbm_feature_importance.png`

## Next Phase

**Notebook 04 — Monte Carlo Audit Capacity & Prioritization Tradeoff Simulation.** Use the full-population model scores from `model_predictions_full_population_2022.parquet` to simulate:

- Audit review capacity scenarios (top-K provider review under realistic staffing)
- Scenario-based recovery potential under varying assumptions
- Uncertainty bands via Monte Carlo resampling (illustrative recovery-potential ranges)
- Prioritization tradeoffs (depth vs. breadth, specialty mix, geographic coverage)

The deliverable is a set of illustrative recovery-potential ranges and prioritization curves — decision support for "where would limited review capacity go furthest under these assumptions," not a forecast of confirmed recoveries. Notebook 04 should ideally be run on full-production scores; the current development sample results are sufficient for wiring the simulation but not for portfolio-grade scenario figures.
