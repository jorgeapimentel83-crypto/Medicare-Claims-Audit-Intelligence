# %% [markdown]
# # Medicare Claims Audit Intelligence Platform
# ## Notebook 01: Exploratory Data Analysis & Domain-Informed Feature Engineering
#
# **Author**: Federal Healthcare Auditor (HHS/OIG, ~11 years)
#
# Every feature below maps to a real red flag pattern from OIG investigations.
# The model will be trained in Notebook 03; this notebook builds the features
# and validates they surface suspicious patterns.
#
# ### Data Source
# CMS Medicare Physician & Other Practitioners PUF — public, de-identified.
# Downloaded via `python src/download_data.py`.

# %%
import sys
import warnings
warnings.filterwarnings("ignore")

# Add project root to path so we can import src modules
from pathlib import Path
PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import PATHS, AUDIT_PALETTE, DEFAULT_YEAR
from load_data import load_provider_service, load_provider_agg, load_geo_service, quality_report, GPU_AVAILABLE
from features import build_all_features, get_feature_metadata

sns.set_palette(AUDIT_PALETTE)
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "figure.dpi": 100,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

YEAR = DEFAULT_YEAR

print(f"{'='*60}")
print(f"  MEDICARE CLAIMS AUDIT INTELLIGENCE PLATFORM")
print(f"  GPU Acceleration: {'ENABLED' if GPU_AVAILABLE else 'DISABLED (CPU)'}")
print(f"  Year: {YEAR}")
print(f"{'='*60}")

# %% [markdown]
# ---
# ## 1. Load Data
#
# Primary dataset: **Provider-Service level** — one row per
# (NPI × HCPCS code × Place of Service). ~10M rows for a full year.
#
# Set `SAMPLE_SIZE` to 500,000 for quick iteration, `None` for full data.

# %%
SAMPLE_SIZE = None  # Set to 500_000 for dev, None for full run

df = load_provider_service(year=YEAR, nrows=SAMPLE_SIZE)
qr = quality_report(df, f"Provider-Service {YEAR}")

# %% [markdown]
# ---
# ## 2. Data Quality Assessment
#
# CMS PUFs have known data quality patterns to be aware of:
# - **Small-cell suppression**: values where <11 beneficiaries → NaN
# - **Entity mix**: Individual providers (I) vs Organizations (O)
# - **Medicare participation**: participating vs non-participating providers

# %%
# Entity type distribution
print("\nEntity Type Distribution:")
print(df["Rndrng_Prvdr_Ent_Cd"].value_counts())

print(f"\nUnique providers (NPIs): {df['Rndrng_NPI'].nunique():,}")
print(f"Unique HCPCS codes: {df['HCPCS_Cd'].nunique():,}")
print(f"Unique specialties: {df['Rndrng_Prvdr_Type'].nunique():,}")
print(f"Unique states: {df['Rndrng_Prvdr_State_Abrvtn'].nunique():,}")

# %% [markdown]
# ---
# ## 3. Specialty and Payment Landscape
#
# Understanding the specialty mix is critical for peer-group benchmarking.
# An orthopedic surgeon billing like a dermatologist is a red flag — but only
# if you know what "normal" looks like for each specialty.

# %%
specialty_stats = (
    df.groupby("Rndrng_Prvdr_Type")
    .agg(
        total_payment=("Tot_Mdcr_Pymt_Amt", "sum"),
        total_services=("Tot_Srvcs", "sum"),
        n_providers=("Rndrng_NPI", "nunique"),
        avg_charge=("Avg_Sbmtd_Chrg", "mean"),
        avg_payment=("Avg_Mdcr_Pymt_Amt", "mean"),
    )
    .sort_values("total_payment", ascending=False)
)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

top_n = 15
top = specialty_stats.head(top_n)

axes[0].barh(range(top_n), top["total_payment"] / 1e9, color=AUDIT_PALETTE[0], alpha=0.85)
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels(top.index, fontsize=9)
axes[0].set_xlabel("Total Medicare Payments ($ Billions)")
axes[0].set_title("Top 15 Specialties by Total Part B Payments")
axes[0].invert_yaxis()

axes[1].barh(range(top_n), top["avg_charge"], color=AUDIT_PALETTE[3], alpha=0.7, label="Avg Submitted Charge")
axes[1].barh(range(top_n), top["avg_payment"], color=AUDIT_PALETTE[2], alpha=0.9, label="Avg Medicare Payment")
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels(top.index, fontsize=9)
axes[1].set_xlabel("Average Amount ($)")
axes[1].set_title("Charge vs Payment Gap by Specialty")
axes[1].legend()
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(PATHS["reports"] / "specialty_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

pct = top["total_payment"].sum() / specialty_stats["total_payment"].sum() * 100
print(f"Top {top_n} specialties = {pct:.1f}% of all Part B payments")

# %% [markdown]
# ---
# ## 4. Build All Audit Features
#
# `features.build_all_features()` applies 10 feature-building steps
# producing ~28 features. Each maps to a real OIG audit red flag.

# %%
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

df = build_all_features(df)

feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
print(f"\n✓ {len(feat_cols)} features built")

# %% [markdown]
# ---
# ## 5. Feature Distributions
#
# ### 5.1 Billing Aggressiveness (Charge-to-Allowed Ratio)
#
# **What it catches**: Providers systematically inflating submitted charges
# above the Medicare fee schedule. Extreme ratios (10x+) indicate potential
# upcoding or charge inflation affecting secondary payers.

# %%
ratio = df["feat_charge_to_allowed"].dropna()
ratio_clipped = ratio.clip(upper=ratio.quantile(0.99))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].hist(ratio_clipped, bins=100, color=AUDIT_PALETTE[0], alpha=0.8, edgecolor="white")
axes[0].axvline(ratio.median(), color=AUDIT_PALETTE[1], linestyle="--", lw=2,
                label=f"Median: {ratio.median():.2f}")
axes[0].axvline(ratio.quantile(0.95), color=AUDIT_PALETTE[3], linestyle="--", lw=2,
                label=f"95th pctl: {ratio.quantile(0.95):.2f}")
axes[0].set_xlabel("Charge-to-Allowed Ratio")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Billing Aggressiveness Distribution")
axes[0].legend()

# By specialty
top10 = df.groupby("Rndrng_Prvdr_Type")["Tot_Srvcs"].sum().nlargest(10).index
spec_med = (
    df[df["Rndrng_Prvdr_Type"].isin(top10)]
    .groupby("Rndrng_Prvdr_Type")["feat_charge_to_allowed"]
    .median()
    .sort_values(ascending=False)
)
axes[1].barh(range(len(spec_med)), spec_med.values, color=AUDIT_PALETTE[0], alpha=0.85)
axes[1].set_yticks(range(len(spec_med)))
axes[1].set_yticklabels(spec_med.index, fontsize=9)
axes[1].set_xlabel("Median Charge-to-Allowed Ratio")
axes[1].set_title("Billing Aggressiveness by Specialty")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(PATHS["reports"] / "charge_to_allowed.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 5.2 Service Concentration (HHI)
#
# **What it catches**: "Mill" operations — providers billing overwhelmingly
# from a single HCPCS code. Legitimate practices are diversified.

# %%
# HHI distribution by unique provider
prov_hhi = df.groupby("Rndrng_NPI")["feat_hhi"].first()

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(prov_hhi.dropna(), bins=100, color=AUDIT_PALETTE[4], alpha=0.8, edgecolor="white")
ax.axvline(0.25, color=AUDIT_PALETTE[1], linestyle="--", lw=2, label="High concentration (0.25)")
ax.axvline(0.50, color=AUDIT_PALETTE[1], linestyle="-", lw=2, label="Very high (0.50)")
ax.set_xlabel("Herfindahl-Hirschman Index (HHI)")
ax.set_ylabel("Providers")
ax.set_title("Service Concentration Distribution (Provider Level)")
ax.legend()
plt.tight_layout()
plt.savefig(PATHS["reports"] / "hhi_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

n_very_high = (prov_hhi > 0.50).sum()
print(f"Providers with HHI > 0.50 (very high concentration): {n_very_high:,}")

# %% [markdown]
# ### 5.3 Composite Risk Score Distribution
#
# Weighted baseline combining all signals. Not a model — a sanity check.

# %%
risk = df["feat_composite_risk"].dropna()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].hist(risk, bins=100, color=AUDIT_PALETTE[0], alpha=0.8, edgecolor="white")
axes[0].axvline(risk.quantile(0.95), color=AUDIT_PALETTE[1], linestyle="--", lw=2,
                label=f"95th pctl: {risk.quantile(0.95):.3f}")
axes[0].axvline(risk.quantile(0.99), color=AUDIT_PALETTE[3], linestyle="--", lw=2,
                label=f"99th pctl: {risk.quantile(0.99):.3f}")
axes[0].set_xlabel("Composite Audit Risk Score")
axes[0].set_title("Risk Score Distribution")
axes[0].legend()

# Top risk by specialty
top_risk = df.nlargest(1000, "feat_composite_risk")
top_risk_spec = top_risk.groupby("Rndrng_Prvdr_Type")["Rndrng_NPI"].nunique().sort_values(ascending=False).head(10)
axes[1].barh(range(len(top_risk_spec)), top_risk_spec.values, color=AUDIT_PALETTE[1], alpha=0.85)
axes[1].set_yticks(range(len(top_risk_spec)))
axes[1].set_yticklabels(top_risk_spec.index, fontsize=9)
axes[1].set_xlabel("Unique Providers in Top 1000 Risk")
axes[1].set_title("Specialties Concentrated in Top Risk Tier")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(PATHS["reports"] / "composite_risk.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 6. Feature Correlation Matrix
#
# Validate features capture different audit signals (low cross-correlation).

# %%
numeric_feats = [c for c in feat_cols if df[c].dtype in ["float32", "float64", "int64", "int32"]]

if numeric_feats:
    corr = df[numeric_feats].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1, annot_kws={"size": 6})
    ax.set_title("Feature Correlation Matrix — Audit Risk Signals", fontsize=13)
    plt.tight_layout()
    plt.savefig(PATHS["reports"] / "feature_correlations.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Flag highly correlated pairs
    high_corr = []
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            if abs(corr.iloc[i, j]) > 0.80:
                high_corr.append((corr.index[i], corr.columns[j], corr.iloc[i, j]))
    if high_corr:
        print(f"\n⚠ Highly correlated pairs (|r| > 0.80) — consider dropping one:")
        for f1, f2, r in high_corr:
            print(f"  {f1} ↔ {f2}: r = {r:.3f}")
    else:
        print("\n✓ No feature pairs with |r| > 0.80")

# %% [markdown]
# ---
# ## 7. Save Feature Set to Parquet
#
# Parquet preserves dtypes, compresses ~5x vs CSV, and loads 10x faster.

# %%
output_path = PATHS["data_processed"] / f"features_provider_service_{YEAR}.parquet"
df.to_parquet(output_path, engine="pyarrow", index=False)
size_mb = output_path.stat().st_size / (1024 ** 2)
print(f"✓ Saved: {output_path.name} ({size_mb:.1f} MB)")

# Save feature metadata
import json
meta = get_feature_metadata()
meta_path = PATHS["data_processed"] / f"feature_metadata_{YEAR}.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"✓ Metadata: {meta_path.name}")

# %% [markdown]
# ---
# ## 8. Summary
#
# | Step | Result |
# |------|--------|
# | Features built | ~28 domain-informed audit signals |
# | Output format | Parquet (fast, compressed, type-safe) |
# | Next notebook | 02_anomaly_detection.py (Isolation Forest, DBSCAN) |
#
# **Key insight for Machinify**: The ML is table stakes. The domain knowledge
# that translates a charge-to-allowed ratio of 15:1 on a J-code billed from
# POS 11 with an HHI of 0.9 into "this is almost certainly a scheme" — that's
# what 11 years of OIG audit work gives you.

# %%
print(f"\n{'='*60}")
print(f"  FEATURE ENGINEERING COMPLETE")
print(f"{'='*60}")
print(f"  Features: {len(feat_cols)}")
print(f"  Rows:     {len(df):,}")
print(f"  Output:   {output_path}")
print(f"\n  Next: notebooks/02_anomaly_detection.py")
