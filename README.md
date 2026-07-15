# Medicare Claims Audit Intelligence Platform

A public-data healthcare audit analytics project that uses CMS Medicare Part B data, OIG LEIE exclusion records, domain-informed feature engineering, anomaly detection, weak-supervision modeling, supervised ML ranking, and Monte Carlo simulation to prioritize providers for follow-up review. Built by a federal healthcare auditor (HHS/OIG, ~11 years) to show how domain knowledge of audit-priority billing patterns translates into machine learning features that surface unusual billing behavior from public CMS data at population scale.

> **Important:** This project produces audit-priority review signals from public summary data. It does not determine fraud, overpayments, noncompliance, medical necessity, documentation sufficiency, parity compliance, or audit findings. Outputs are decision support for follow-up review, not findings on any individual provider.

## What This Project Does

- Builds provider-service and provider-level audit-priority features from public CMS Medicare Part B data.
- Applies anomaly detection plus domain-informed weak supervision to create an audit-priority label.
- Trains Logistic Regression, XGBoost, and LightGBM models to learn the weak-supervision framework under leakage controls.
- Uses the OIG LEIE list only as directional external validation, not as a fraud label or training target.
- Simulates review-capacity tradeoffs and illustrative recovery-potential ranges with Monte Carlo.
- Adds a behavioral health audit-priority module using tightened provider-type and service-concentration logic.

## What This Project Does Not Do

- It does not identify fraud.
- It does not estimate actual recoveries.
- It does not make audit findings.
- It does not determine overpayments, noncompliance, medical necessity, documentation sufficiency, or parity compliance.
- It does not use internal government data, PHI, PII, case files, draft findings, or non-public records.

## Why This Matters

Medicare Part B pays roughly $250B annually to 1.2M+ providers across 6,000+ HCPCS procedure codes. Review teams have to prioritize limited capacity against millions of claim lines, and rules-based filters alone miss combined or contextual patterns. Domain-informed features, weak-supervision modeling, and capacity-scenario simulation give reviewers a defensible starting point for where to look first under public-data review signals — without claiming any individual provider has done anything wrong.

## Current Full-Data Results

| Metric | Value |
|--------|-------|
| CMS Provider-Service rows processed | 9,755,427 |
| Provider-level analytical table | 1,148,873 providers |
| Weak-label positives | 31,207 (2.716% prevalence) |
| LEIE-matched Part B providers | 318 |
| LEIE directional external validation | LEIE-matched providers showed ~2× the weak-label rate of non-LEIE providers (directional check only, not a fraud label) |
| Best supervised model | LightGBM |
| LightGBM PR-AUC | 0.9898 |
| LightGBM ROC AUC | 0.9997 |
| LightGBM Precision@100 / @500 / @1,000 | 1.0000 / 1.0000 / 1.0000 |
| LightGBM Lift@100 | 36.82× |
| Monte Carlo capacity scenarios | Top 100 / 500 / 1,000 / 5,000 / 10,000 providers, scored with `lgb_full_probability` |
| Recovery-potential outputs | Illustrative scenario ranges from analyst-defined assumptions, not actual recoveries |
| Behavioral health cohort (after tightening) | 46,711 providers (4.07% of full provider population) |
| Behavioral health provider-service rows | 302,604 |
| Behavioral health weak-label positives | 249 |
| Behavioral health Top-25 lift vs cohort baseline | 187.6× |

PR-AUC, precision@k, and lift figures reflect how well the supervised model reconstructs the weak-supervision audit-priority framework defined in Notebook 02 — they are not measures of confirmed real-world audit outcomes. See [Important Limitation](#important-limitation) for the full framing.

## Technical Stack

Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, RAPIDS cuDF (with pandas fallback), Plotly / matplotlib / seaborn, joblib, pyarrow / Parquet, Jupyter, WSL2 Ubuntu 24.04, NVIDIA RTX 5080, Git / GitHub.

## Independent Audit Governance

This platform is audited by the AI Audit Framework maintained in the companion cms-policy-rag repository. Nine checks (C-ME-001/002/003, C-LS-001, C-SG-001/002, C-PR-001, C-DL-001/002) validate the platform's model metadata, provenance completeness and README agreement, label-constituent leakage controls, scored-output and downstream-output content integrity, environment provenance, and data lineage. The checks are driven by seven machine-readable declarations tracked in this repository's models/ directory (provenance, leakage constituents, score and downstream output expectations, environment provenance, and two data-lineage declarations), each an owner-signed attestation the framework verifies at runtime. Audit evidence — including the first full clean-sweep register of this platform (2026-07-14-220917) — is custodied in the audit framework's evidence archive, following the principle that the auditor, not the auditee, holds the working papers.

## Reproducibility Note

Large public CMS CSV files, processed Parquet feature tables, model artifacts, and generated report figures are **not committed to this repository**. They are reproducible locally from the public source data using the scripts in [src/](src/) and the notebooks in [notebooks/](notebooks/). See [Quick Start](#quick-start) for the regeneration steps and [Generated Outputs](#generated-outputs) for the full list of artifacts produced by a local run.

---

## Project Status

| Notebook | Status | Description |
|----------|--------|-------------|
| 01 — EDA and Domain-Informed Feature Engineering | ✅ Complete (full run) | Builds ~28 provider-service-level audit-priority features from CMS Part B data |
| 02 — Provider-Level Anomaly Detection and Weak Supervision | ✅ Complete (full run) | Aggregates 9.76M provider-service rows into 1,148,873 provider-level rows × 121 columns; Isolation Forest scoring, weak-supervision audit-priority labels, LEIE cross-reference |
| 03 — Supervised Model Training (XGBoost / LightGBM) | ✅ Complete (full run) | Leakage-controlled supervised models trained on the weak-supervision label across the full provider population |
| 04 — Monte Carlo Audit Capacity & Prioritization Tradeoff Simulation | ✅ Complete (full run) | Audit capacity scenarios, scenario-based recovery-potential ranges, and prioritization tradeoffs simulated from the full 1,148,873-provider population scores |
| 05 — Behavioral Health Audit-Priority Module | ✅ Complete (full run) | Public-data behavioral-health cohort built from CMS specialty text, HCPCS codes, and HCPCS descriptions, with a tightened provider-level qualification rule and a focused review-priority worklist plus capacity scenarios |

> All ✅ Complete entries above reflect the **full production-scale run** against the complete CMS Medicare Part B Provider-Service file (9,755,427 provider-service rows / 1,148,873 providers). An earlier 500,000-row development sample was used during prototyping to keep iteration cycles short; the headline metrics in this README now reflect the full run, not that earlier sample.

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
│   ├── 04_monte_carlo_simulation.py
│   └── 05_behavioral_health_module.py
├── src/
│   ├── __init__.py
│   ├── config.py                   # Centralized paths, hyperparams, feature lists
│   ├── download_data.py            # CMS data download pipeline
│   ├── load_data.py                # GPU/CPU data loading with dtype optimization
│   ├── features.py                 # Domain-informed audit feature builders
│   ├── modeling.py                 # XGBoost + LightGBM GPU ensemble
│   ├── evaluation.py               # Audit-specific metrics (precision@k, AUCPR)
│   └── simulation.py               # Monte Carlo recovery-potential simulation
├── models/                         # Model artifacts (gitignored) + seven tracked governance declarations (JSON)
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

> The current ✅ Complete notebook results in this README were generated from
> the **full CMS Medicare Part B Provider-Service file** (9,755,427
> provider-service rows / 1,148,873 providers). An earlier 500,000-row
> development sample was used during prototyping; see
> [Development Sample vs. Full Production Run](#development-sample-vs-full-production-run)
> for the historical note.

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

The pipeline layers five notebooks so that each stage is auditable on its own and the supervised model never sees the signals that defined its label.

- **Notebook 01 — Domain-Informed Feature Engineering.** Builds approximately **28 provider-service-level** audit-priority features from CMS Part B data (charge-to-allowed ratios, peer specialty z-scores, service-mix concentration, utilization outliers, place-of-service mix, drug revenue share, geographic deviation).
- **Notebook 02 — Anomaly Scores and Weak-Supervision Label.** Aggregates the service-level features into **121 provider-level columns** across the full 1,148,873-provider population, runs Isolation Forest at provider scale, derives a composite review signal, and emits `weak_label_high_audit_priority` — a weak-supervision label that flags providers worth a follow-up review under this framework. Also performs an end-to-end LEIE cross-reference for validation.
- **Notebook 03 — Leakage-Controlled Supervised Models.** Trains Logistic Regression, XGBoost, and LightGBM against the weak-supervision label, with the anomaly-stage signals excluded from the feature set. Selects the best model by PR-AUC and scores the full provider population for downstream simulation.
- **Notebook 04 — Audit Capacity and Scenario-Based Recovery-Potential Simulation.** Consumes the full-population model scores from Notebook 03, ranks all 1,148,873 providers by audit-priority, and simulates capacity scenarios (Top 100 / 500 / 1,000 / 5,000 / 10,000). Reports weak-label precision, recall, and lift for each scenario, then layers analyst-defined recovery-potential assumptions and a Monte Carlo (triangular distribution, 5,000 trials per scenario) to express results as illustrative recovery-potential ranges, not actual recovery estimates.
- **Notebook 05 — Behavioral Health Audit-Priority Module.** Builds a public-data behavioral-health cohort from CMS provider specialty text, HCPCS codes, and HCPCS descriptions, applies a tightened provider-level qualification rule (core specialty OR meaningful BH service concentration), and reuses the Notebook 03 model scores to produce a focused behavioral-health review-priority worklist plus capacity scenarios (Top 25 / 50 / 100 / 250 / 500 / 1,000). Outputs are a public-data approximation, not a clinical or compliance determination.

## Current Results

> ✅ **Full production-scale run against the complete CMS Medicare Part B Provider-Service file.**
> The metrics below were generated against the full 2022 Provider-Service file
> (9,755,427 provider-service rows / 1,148,873 providers), not the earlier
> 500,000-row development sample. The development sample was used during
> prototyping to validate the end-to-end pipeline and is preserved only as
> a historical note — the headline figures here reflect the full run.

### Notebook 02 — Anomaly Detection and Weak Labels (full run)

| Metric | Value (full run) |
|--------|------------------|
| Provider-service rows loaded | 9,755,427 |
| Provider-level rows (after aggregation) | 1,148,873 |
| Provider-level columns (final save) | 121 |
| Weak-label positives (`weak_label_high_audit_priority`) | 31,207 providers (2.716%) |
| LEIE rows loaded | 83,001 |
| LEIE rows with usable NPI | 8,375 |
| Part B providers matched against LEIE | 318 |

#### LEIE validation (full run)

The LEIE cross-reference is held out from training and used purely as an external validity check. Of the 318 Part B providers that matched the OIG LEIE list, the weak-supervision audit-priority signal fired at roughly **twice the rate** seen for non-LEIE providers, and the upstream Isolation Forest anomaly flag showed a similar separation:

| Cohort | `iso_anomaly_flag` rate | `weak_label_high_audit_priority` rate |
|--------|-------------------------|----------------------------------------|
| Non-LEIE providers | 4.9989% | 2.7155% |
| LEIE-matched providers | 9.1195% | 5.6604% |

LEIE inclusion is **not** a fraud label — it is an administrative exclusion list maintained by HHS-OIG covering a wide range of grounds (program-related convictions, license actions, defaulted health-education loans, and others). The validation pattern says only that the unsupervised and weak-supervision review signals built from public CMS billing data tend to flag LEIE-listed providers more often than non-LEIE providers, which is a directional sanity check on the framework, not evidence of fraud, overpayment, noncompliance, or audit findings on any individual provider.

> **Limitation noted in Notebook 02:** Population-level anomaly detection can reflect provider-type structure (specialty mix, billing norms). Future work should evaluate provider-type-specific or specialty-specific anomaly detection so that "unusual" is judged against true peer behavior rather than the overall provider population.

### Notebook 03 — Supervised Model Training (full run)

Dataset (full run): 1,148,873 providers × 121 columns. Features used for modeling: 108. Target: `weak_label_high_audit_priority` (31,207 positives, 2.716% prevalence).

| Model | PR-AUC | ROC AUC | Precision@100 | Precision@500 | Precision@1000 | Lift@100 |
|-------|--------|---------|---------------|---------------|----------------|----------|
| Logistic Regression | 0.8841 | 0.9968 | 0.9800 | 0.9800 | 0.9570 | 36.08× |
| XGBoost | 0.9820 | 0.9994 | 1.0000 | 1.0000 | 1.0000 | 36.82× |
| **LightGBM (best)** | **0.9898** | **0.9997** | **1.0000** | **1.0000** | **1.0000** | **36.82×** |

Best model selected by Average Precision / PR-AUC: **LightGBM**, persisted to `models/best_model_2022_full.joblib` with companion metadata at `models/best_model_2022_full_metadata.json`. These are full-population figures that characterize how well the supervised models reconstruct the weak-supervision audit-priority framework — see [Important Limitation](#important-limitation) for what high PR-AUC and precision@k do and do not mean here.

### Notebook 04 — Monte Carlo Audit Capacity & Recovery-Potential Simulation (full run)

Notebook 04 takes the full-population audit-priority scores from Notebook 03 and turns them into a decision-support view for audit-capacity planning. It is **not** an estimate of actual recoveries — every dollar figure below is an *illustrative recovery-potential range* built from analyst-defined assumption inputs and Monte Carlo resampling.

| Setting | Value |
|---------|-------|
| Input | `data/processed/model_predictions_full_population_2022_full.parquet` |
| Providers scored (full population) | 1,148,873 |
| Model score used (best model from Notebook 03) | `lgb_full_probability` |
| Audit-capacity scenarios | Top 100 / 500 / 1,000 / 5,000 / 10,000 |
| Per-scenario metrics computed | `providers_reviewed`, `weak_label_hits`, `weak_label_precision`, `weak_label_recall`, `lift_vs_random` |
| Illustrative deterministic per-hit recovery-potential assumptions | conservative $2,500 / moderate $7,500 / high $15,000 |
| Monte Carlo distribution (per weak-label hit) | Triangular: low $1,000 / mode $7,500 / high $25,000 |
| Monte Carlo trials per scenario | 5,000 |
| Per-scenario simulation summary | mean, median, p10, p90, min, max |

The capacity table makes the prioritization tradeoff explicit (small worklists are denser in weak-label positives but cover fewer of them; larger worklists trade per-provider density for coverage). The Monte Carlo step then replaces the deterministic per-hit assumptions with a distribution so each capacity scenario carries a p10–p90 uncertainty band rather than a single point.

These outputs are **illustrative recovery-potential ranges**, not actual recoveries, confirmed overpayments, noncompliance findings, or audit findings. They exist to support "where would limited review capacity go furthest under these stated assumptions" decision support, and any provider on any of these worklists would still require independent follow-up review before any real-world conclusion.

### Notebook 05 — Behavioral Health Audit-Priority Module (full run)

Notebook 05 narrows the platform to behavioral-health-related provider-service records and reuses the Notebook 03 full-population model scores to produce a focused review-priority worklist plus capacity scenarios. The behavioral-health classification is a **public-data approximation** built from CMS provider specialty text, HCPCS codes, and HCPCS descriptions — it is not a clinical determination, not a credentialing check, and not a parity or documentation determination.

| Setting | Value |
|---------|-------|
| Inputs | `features_provider_service_2022_full.parquet`, `provider_features_labeled_2022_full.parquet`, `model_predictions_full_population_2022_full.parquet` |
| Behavioral-health provider-service rows | 302,604 (3.10% of all 9,755,427 provider-service rows) |
| Behavioral-health providers (qualified cohort) | **46,711 (4.07% of 1,148,873)** |
| Behavioral-health weak-label positives | 249 |
| Risk score column used | `lgb_full_probability` (LightGBM, best model from Notebook 03) |
| Capacity scenarios | Top 25 / 50 / 100 / 250 / 500 / 1,000 |
| Top-25 capacity-scenario lift vs cohort baseline | **187.6×** |

**Top behavioral-health provider types after qualification (counts):**

| Provider type | Providers in cohort |
|---|---|
| Psychiatry | 18,760 |
| Licensed Clinical Social Worker | 15,203 |
| Psychologist, Clinical | 11,725 |
| Addiction Medicine | 218 |
| Geriatric Psychiatry | 172 |
| Nurse Practitioner | 150 |
| Neuropsychiatry | 139 |
| Internal Medicine | 115 |
| Family Practice | 112 |
| Anesthesiology | 28 |

#### Behavioral-health cohort tightening (why the qualification rule matters)

Public CMS data does not provide a perfect "this provider is a behavioral-health clinician" label, so Notebook 05 builds the cohort in two layers.

- **Loose row-level rule (kept for description, not cohort selection).** A row is `is_behavioral_health_row = 1` if its provider specialty matches a behavioral-health keyword OR its HCPCS_Desc matches a behavioral-health service keyword OR its HCPCS_Cd is in a curated CPT/HCPCS set or the H0001–H2037 H-code range. This is useful for surfacing behavioral-health-related billing activity across the population.
- **Tightened provider-level rule (the actual cohort).** A provider qualifies as behavioral-health (`bh_any_behavioral_health = 1`) only if either:
  - `bh_provider_core_flag` — the provider's specialty itself is a core behavioral-health type (Psychiatry, Clinical Psychologist, Licensed Clinical Social Worker, counselor, addiction medicine, marriage and family therapist, etc.), OR
  - `bh_provider_volume_flag` — the provider has meaningful BH service concentration: `bh_service_rows >= 5` AND `bh_service_share >= 0.20` (i.e., at least 20% of the provider's own services are behavioral-health-tagged).

An earlier broad version of the rule used standalone keywords like `"family"`, `"substance"`, and `"behavioral"`, which inadvertently swept in **Family Practice (78,867 providers)** and similar generalists. The tightened rule removed those bare tokens (replacing them with phrases like `"marriage and family"`, `"substance abuse"`, `"behavioral health"`) and added the provider-level concentration test, dropping the cohort from 223,777 to 46,711 providers and pushing Family Practice to 112 providers (only those whose own billing is meaningfully concentrated in behavioral-health services). The result is a behavioral-health cohort that actually looks like behavioral-health practices, and a Top-25 worklist that is dominated by Psychiatry, Neuropsychiatry, Opioid Treatment Program, and similar specialties rather than by generalist incidental billing.

These outputs are a behavioral-health audit-priority module — **decision support for follow-up review**, not a clinical, parity, documentation, medical-necessity, or compliance determination, and not a fraud, overpayment, or audit-finding determination.

## Development Sample vs. Full Production Run

This repository now reflects the **full production-scale run** against the complete CMS Medicare Part B Provider-Service file.

- **Full run (current headline metrics).** Notebooks 01 → 02 → 03 → 04 → 05 have all been run end-to-end against the complete `provider_service_2022.csv` (9,755,427 provider-service rows / 1,148,873 providers). All counts, prevalence rates, PR-AUC, precision@k, lift, capacity-scenario, and behavioral-health figures in [Current Results](#current-results) come from this run.
- **Earlier development sample (historical note).** During prototyping, the same pipeline was run against a **500,000-row sample parquet** of the Provider-Service file (58,866 providers) to keep iteration cycles short on a single workstation (WSL2 Ubuntu, RTX 5080). That sample was used to validate the end-to-end flow (download → feature engineering → provider aggregation → anomaly detection → weak-supervision label → leakage-controlled supervised models → full-population scoring) and is preserved here only as context. The development-sample numbers have been **superseded** by the full-run figures above and should not be cited as headline metrics.

## Leakage Controls

To keep the supervised models from trivially recovering the weak-supervision label, the following columns are excluded from the modeling feature set in Notebook 03:

- `weak_label_high_audit_priority` — the target itself
- `iso_anomaly_flag`, `iso_anomaly_score` — Isolation Forest outputs that participate in label construction
- `extreme_composite_risk`, `extreme_peer_deviation` — composite review signals that drive the weak label
- `in_leie` — LEIE cross-reference flag (held out for validation, not training)
- `Rndrng_NPI` — provider identifier
- `provider_type`, `provider_state`, `provider_entity_type` — categorical identifiers held out to keep the model from memorizing slices
- All `feat_composite_risk*` columns — composite review signals upstream of the label

A residual consideration remains after these exclusions. The `extreme_peer_deviation` flag is excluded, but the continuous features it is thresholded from — `feat_max_peer_deviation_max` and the `feat_zscore_*` family — are retained in the feature set, since they are also independently meaningful signals of unusual billing. The model therefore never sees the thresholded flag, only the continuous magnitude beneath it. This gives partial purchase on one of the label's two disjunctive arms, but the effect is bounded: the label's required Isolation Forest term is excluded entirely, so the model cannot observe the arm the label requires by logical AND. This residual path is documented and accepted rather than removed; see the feature-leakage assessment in the audit framework's evidence records.

This keeps the reported PR-AUC, precision@k, and lift figures interpretable as "how well the model reconstructs the weak-supervision audit-priority framework from upstream features," not "how well it memorizes its own label."

## Important Limitation

- The supervised model predicts a **weak-supervision audit-priority label**, not confirmed audit outcomes.
- It does **not** predict confirmed fraud, overpayments, noncompliance, or audit findings.
- The high PR-AUC, precision@k, and lift values reflect how well the model learned the weak-supervision framework defined in Notebook 02. They are **not** measures of confirmed real-world audit outcomes.
- Notebook 04 simulates **illustrative recovery-potential ranges** under analyst-defined capacity scenarios and Monte Carlo assumptions. Its outputs are scenario-based decision support, not forecasts of actual recoveries, and not estimates of confirmed overpayments, noncompliance, or audit findings.
- Notebook 05 builds a **behavioral-health audit-priority module** from a public-data approximation (CMS specialty text plus HCPCS codes and descriptions). It is not a clinical determination, not a credentialing check, and not a parity, documentation, medical-necessity, or compliance determination.
- Outputs are intended as **decision support** for prioritizing follow-up review of public CMS billing patterns. Any provider surfaced by the model, by any Notebook 04 capacity scenario, or by the Notebook 05 behavioral-health worklist would still require independent review under the appropriate audit process before any conclusion is drawn.
- Nothing in this project, end to end, determines fraud, overpayment, noncompliance, medical necessity, parity compliance, documentation sufficiency, or audit findings.

## Generated Outputs

These artifacts are produced locally by the notebooks. They are not necessarily committed — generated data, model, and report files are local outputs and may be selectively included later.

**Processed data (full run)**
- `data/processed/features_provider_service_2022_full.parquet`
- `data/processed/provider_features_labeled_2022_full.parquet`
- `data/processed/model_predictions_2022_full.parquet`
- `data/processed/model_predictions_full_population_2022_full.parquet`
- `data/processed/audit_capacity_scenarios_2022_full.parquet`
- `data/processed/monte_carlo_recovery_potential_2022_full.parquet`
- `data/processed/behavioral_health_top_providers_2022_full.parquet`
- `data/processed/behavioral_health_capacity_scenarios_2022_full.parquet`

**Models (full run)**
- `models/best_model_2022_full.joblib`
- `models/best_model_2022_full_metadata.json`

**Reports / figures**
- `reports/isolation_forest_anomaly_scores.png`
- `reports/provider_type_anomaly_rates.png`
- `reports/notebook03_precision_recall_curve.png`
- `reports/notebook03_xgboost_feature_importance.png`
- `reports/notebook03_lightgbm_feature_importance.png`
- `reports/notebook04_capacity_tradeoff.png`
- `reports/notebook04_monte_carlo_recovery_potential_ranges.png`
- `reports/notebook05_behavioral_health_risk_score_distribution.png`
- `reports/notebook05_behavioral_health_provider_type_rates.png`
- `reports/notebook05_behavioral_health_capacity_tradeoff.png`

## Next Phase

Notebooks 01 → 05 now run end-to-end against the full CMS Medicare Part B Provider-Service file. The next phase shifts from "build the pipeline" to "stress-test and present the framework." Candidate work, in rough order of priority:

- **Provider-type-specific anomaly detection.** Re-run the Notebook 02 anomaly stage within specialty cohorts (or other peer groupings) so "unusual" is judged against true peer behavior rather than the overall provider population. Addresses the limitation flagged in Notebook 02.
- **Behavioral-health refinement / submodule expansion.** Iterate on the Notebook 05 cohort logic (e.g., raising the volume-flag concentration threshold, adding `bh_unique_hcpcs` minima, splitting therapy-only vs medication-management vs substance-use submodules) and broaden the behavioral-health module into more focused review-signal sub-cohorts. Each submodule stays inside the same public-data approximation framing as Notebook 05.
- **Dashboard / Streamlit or React portfolio app.** An interactive front-end for the audit-priority worklist, capacity scenarios, Monte Carlo recovery-potential ranges from Notebook 04, and the behavioral-health worklist from Notebook 05 — built so a reviewer can move the capacity slider and see precision/recall and illustrative recovery-potential ranges update live.
- **Final GitHub polish.** Tighten READMEs, notebook docstrings, and figures for portfolio review; add a short repository tour, an at-a-glance results page, and clean up any in-progress scaffolding so a reviewer can land on the repo and read it end to end without context.

Every item above stays inside the same framing: public CMS data, weak-supervision audit-priority signals, decision support for follow-up review. None of it claims fraud, overpayment, noncompliance, medical necessity, parity compliance, documentation sufficiency, or audit findings.
