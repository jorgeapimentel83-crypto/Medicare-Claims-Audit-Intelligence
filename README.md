# Medicare Claims Audit Intelligence Platform

**GPU-accelerated anomaly detection and audit targeting for Medicare Part B claims**

Built by a federal healthcare auditor (HHS/OIG, ~11 years) to demonstrate how deep domain knowledge of claims audit red flags translates into machine learning features that surface high-value audit targets from public CMS data at population scale.

---

## Problem Statement

Medicare Part B pays ~$250B annually to 1.2M+ providers across 6,000+ HCPCS procedure codes. OIG and CMS audit teams must prioritize limited resources against millions of claim lines. Traditional audit selection relies on manual referral and rules-based filters that miss complex patterns.

This platform applies gradient-boosted ensemble models and Monte Carlo simulation to rank providers by audit-worthiness using the same red flags experienced federal auditors look for — but across the entire provider population simultaneously.

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
│  Domain-informed audit signals (features.py):                   │
│  • Charge-to-allowed ratio → billing aggressiveness             │
│  • Peer specialty z-scores → outlier vs peers                   │
│  • Herfindahl index → service concentration / "mills"           │
│  • Services-per-beneficiary → unbundling / phantom billing      │
│  • Place-of-service patterns → facility-rate upcoding           │
│  • Drug revenue share → J-code concentration                    │
│  • Geographic deviation → regional outlier detection             │
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
│  Provider audit priority rankings with confidence intervals     │
│  Specialty-level benchmarking visualizations                    │
│  Estimated recoverable overpayments (Monte Carlo bands)         │
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
│   └── simulation.py               # Monte Carlo overpayment estimation
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

The value proposition isn't the algorithms — any ML engineer can run XGBoost. The differentiator is knowing *which features to build and why*. Every feature in `src/features.py` maps to a specific audit red flag pattern from real OIG investigations:

| Feature | Audit Pattern | Real-World Example |
|---------|--------------|-------------------|
| Charge-to-Allowed Ratio | Inflated billing | DME supplier charging $3,000 for a $150 knee brace |
| Peer Specialty Z-Score | Outlier vs peers | "Doc shop" seeing 80+ patients/day with cookie-cutter 99214s |
| HHI Concentration | Service mill | Pain clinic billing 90% facet joint injections (64493-64495) |
| Services/Beneficiary | Unbundling | Lab billing 20+ tests per encounter, clinical need supports 3-4 |
| Facility Ratio | POS upcoding | Billing facility rates (POS 22) for office-rendered services |
| Drug Revenue Share | J-code concentration | Oncologist with 80% revenue from single chemotherapy drug |
