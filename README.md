# Medicare Claims Audit Intelligence Platform

**GPU-accelerated anomaly detection and audit-priority ranking for Medicare Part B claims**

Built by a federal healthcare auditor (HHS/OIG, ~11 years) to demonstrate how domain knowledge of audit-priority billing patterns can be translated into machine learning features that surface unusual billing behavior from public CMS data at population scale. Outputs are public-data decision-support signals — they do not identify fraud, overpayments, or noncompliance.

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
│  Monte Carlo simulation for recovery estimates (simulation.py)  │
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
python src/download_data.py

# 3. Run EDA notebook
jupyter lab notebooks/01_eda_feature_engineering.py
```

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
