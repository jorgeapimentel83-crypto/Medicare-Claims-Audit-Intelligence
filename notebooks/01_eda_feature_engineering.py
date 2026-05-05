# %% [markdown]
# # Medicare Claims Audit Intelligence Platform
# ## Notebook 01: Exploratory Data Analysis & Domain-Informed Feature Engineering
#
# **Author**: Federal Healthcare Auditor (HHS/OIG, ~11 years)
#
# This notebook builds ~28 audit-priority features from the public CMS
# Medicare Part B Provider & Service file. Each feature is grounded in
# patterns auditors commonly use to prioritize where to look first — not
# as evidence of wrongdoing, but as starting points for review.
#
# The supervised model is trained in Notebook 03. This notebook focuses
# on building the features, sanity-checking their distributions, and
# saving them to Parquet for downstream notebooks.
#
# ### Data Source
# CMS Medicare Physician & Other Practitioners PUF — public, de-identified,
# downloaded via `python src/download_data.py`.
#
# ---
# ### ⚠ Important Limitation — read before continuing
#
# This notebook **does not identify fraud, overpayments, or noncompliance.**
# It builds **audit-priority features** from public summary data.
#
# - The CMS PUF data is aggregated and small-cell-suppressed; it lacks
#   the medical records, modifiers, beneficiary detail, and longitudinal
#   context needed to support any compliance conclusion.
# - High feature scores indicate **patterns that may warrant follow-up
#   review** — they do not indicate wrongdoing. Many high-scoring
#   providers are practicing entirely appropriately given their
#   specialty, geography, or patient panel.
# - A real audit conclusion would require pulling claims-level records,
#   reviewing applicable LCDs/NCDs and CMS policy, gathering provider
#   context (subspecialty, group setting, patient mix), and applying
#   professional auditor judgment.
#
# Treat the outputs of this notebook as a **prioritized worklist for
# educational analysis**, not as findings.

# %% [markdown]
# ### Setup and environment
#
# **Business Purpose** — Wire the notebook into the project so every
# cell pulls paths, palette, and helpers from the same place
# (`src/config.py`). This avoids drift between notebooks.
#
# **Plain-English Logic** — Add `src/` to the Python path, import the
# project's data loaders and feature builders, and configure the
# matplotlib / seaborn defaults.
#
# **Expected Output** — A banner printing whether GPU acceleration is
# available and which CMS year we will analyze.
#
# **Why It Matters** — If the import path or config is wrong, every
# downstream cell will fail. Catching it here saves a lot of confusion
# later.
#
# **What to Check Before Moving On**
# - The banner prints with no `ImportError`.
# - `YEAR` matches the data file you actually downloaded.
# - GPU status matches your machine. CPU is fine for this notebook.

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
# **Business Purpose** — Pull the primary dataset into memory and run a
# one-shot quality report. Everything downstream depends on this frame.
#
# **Plain-English Logic** — `load_provider_service` reads the CSV using
# the dtype map from `src/config.py`, optionally sampling the first
# `nrows`. `quality_report` summarizes shape, missingness, and key
# cardinalities so you can spot something off (wrong year, truncated
# file, schema drift) before building features on it.
#
# **Expected Output** — A dataframe of roughly 10M rows for a full year
# (or `SAMPLE_SIZE` rows if set), and a printed quality report.
#
# **Why It Matters** — The Provider & Service file is the analytical unit
# of this whole notebook. If it loads wrong, everything else is wrong.
#
# **What to Check Before Moving On**
# - Row count is in the expected range (full year ≈ 10M).
# - No "file is HTML" or dtype errors during load.
# - The quality report's missingness columns make sense — heavy NaN on
#   beneficiary-count fields is **expected** (small-cell suppression).
#
# ### Learning note — what is "provider-service level" data?
#
# CMS publishes the same Part B information at three levels of
# aggregation. We are using the most granular one:
#
# | Level | One row per | Use for |
# |---|---|---|
# | Provider-Service (this file) | NPI × HCPCS code × Place of Service | Per-procedure billing patterns, peer comparisons |
# | Provider (aggregate) | NPI | Provider-wide totals, exclusion-list joining |
# | Geography & Service | State × HCPCS | National / state benchmarks |
#
# - **NPI** is the National Provider Identifier — a 10-digit ID that
#   uniquely identifies the rendering provider.
# - **HCPCS** is the procedure code billed (e.g., `99213` for an
#   established-patient office visit).
# - **Place of Service** is the setting (office, hospital outpatient,
#   etc.).
#
# A single NPI usually has dozens of rows in this file — one for each
# procedure they billed in each setting where they billed it.
#
# Iteration tip: set `SAMPLE_SIZE = 500_000` while developing, then
# `None` for the production run.

# %%
SAMPLE_SIZE = None  # Set to 500_000 for dev, None for full run

df = load_provider_service(year=YEAR, nrows=SAMPLE_SIZE)
qr = quality_report(df, f"Provider-Service {YEAR}")

# %% [markdown]
# ---
# ## 2. Data Quality Assessment
#
# **Business Purpose** — Confirm the data has the structure and
# cardinality we expect before building features on top of it.
#
# **Plain-English Logic** — Print the entity-type breakdown (individual
# providers vs organizations) and counts of unique NPIs, HCPCS codes,
# specialties, and states. These are sanity checks, not findings.
#
# **Expected Output** — Several hundred thousand to ~1M unique NPIs,
# thousands of distinct HCPCS codes, ~50 states/territories.
#
# **Why It Matters** — If the unique-NPI count looks unexpectedly low,
# the file may be truncated. If state count is 1, you may be looking at
# a single-state extract rather than the national file.
#
# **What to Check Before Moving On**
# - Entity mix is roughly the expected `I` (individual) heavy split.
# - No single field has implausible cardinality (e.g., 1 specialty).
#
# ### CMS PUF data quality patterns to be aware of
# - **Small-cell suppression**: rows with fewer than 11 beneficiaries
#   have beneficiary-related fields nulled out. This is privacy
#   protection, not missing data — do not impute it.
# - **Entity mix**: Individual providers (`I`) vs Organizations (`O`).
#   Organizations behave differently and are often filtered out of
#   provider-level analytics.
# - **Medicare participation**: participating vs non-participating
#   providers have different fee-schedule behavior — relevant when
#   interpreting submitted-charge ratios later.

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
# **Business Purpose** — Build intuition for which specialties drive
# most Part B spend, and the typical gap between submitted charges and
# Medicare-allowed amounts. This grounds every later peer-comparison
# feature.
#
# **Plain-English Logic** — Group rows by rendering-provider specialty,
# sum total payments and services, and plot the top 15. The second
# panel shows the average submitted charge vs. the average Medicare
# payment for those same specialties.
#
# **Expected Output** — Two horizontal bar charts saved to
# `reports/specialty_distribution.png`, plus a one-line printout of
# what share of total Part B payments the top 15 specialties account
# for.
#
# **Why It Matters** — A peer comparison only makes sense within a
# specialty. Comparing a cardiologist's billing to a podiatrist's
# tells you nothing — comparing a cardiologist to other cardiologists
# tells you everything.
#
# **What to Check Before Moving On**
# - The top specialties (Internal Medicine, Family Practice, etc.)
#   match public CMS reporting.
# - The charge-vs-payment gap is *always* present; that is normal — it
#   reflects the difference between provider-set "list price" and the
#   Medicare fee schedule. We are not flagging the gap itself; we will
#   later flag providers whose ratio is far from their specialty
#   peers.
#
# ### Learning note — peer-group benchmarking
#
# A bare statistic like "this provider billed 8,000 services" means
# nothing on its own. The same number is unremarkable for a
# high-volume primary-care practice and unusual for a niche surgical
# specialty.
#
# Peer-group benchmarking compares each provider against others in
# the **same specialty** (and ideally same state). The benchmarking
# step converts raw values into z-scores, percentiles, or ratios
# *within* the peer group — so the resulting feature is "how
# unusual is this provider vs. their peers", not "how big is their
# raw number". That is the audit-analytics literature's most
# durable signal: outliers defined relative to peers, not against
# the population at large.

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
# ## 4. Build All Audit-Priority Features
#
# **Business Purpose** — Run the feature pipeline that produces the
# ~28 domain-informed review signals downstream notebooks consume.
#
# **Plain-English Logic** — `build_all_features()` (in
# `src/features.py`) applies 10 sequential steps: charge ratios, peer
# deviation, service concentration (HHI), volume anomalies, modifier
# flags, geographic unusualness, and a weighted composite. Each step
# adds columns prefixed with `feat_`.
#
# **Expected Output** — A printed count of how many `feat_` columns
# now exist on `df` (target: ~28).
#
# **Why It Matters** — This is the heart of the notebook. Every later
# visualization, score, and saved Parquet file derives from these
# columns.
#
# **What to Check Before Moving On**
# - The "✓ N features built" line shows roughly 28.
# - No silent warnings about NaN-only columns (would suggest a
#   feature step found nothing to operate on).

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
# This section spot-checks three of the most informative features. The
# goal is to confirm each one has a reasonable distribution shape — not
# to draw conclusions about any specific provider.
#
# ### 5.1 Charge-to-Allowed Ratio
#
# **Business Purpose** — Surface providers whose submitted charges are
# unusually high relative to what Medicare allows for the same
# procedure.
#
# **Plain-English Logic** — For every row, divide the average
# submitted charge by the average Medicare-allowed amount. Then look
# at the distribution overall and by specialty.
#
# **Expected Output** — A right-skewed histogram (most providers
# cluster at modest multiples of allowed; a long tail extends out to
# 10× and beyond), plus a per-specialty median.
#
# **Why It Matters** — Most providers' submitted charges sit at a
# relatively stable multiple of the fee schedule. Providers whose
# ratio is far above their specialty's median may warrant follow-up
# review — often for legitimate reasons (out-of-network billing,
# secondary-payer chargemaster effects), but still worth a closer
# look.
#
# **What to Check Before Moving On**
# - The histogram's median is sensible (Part B providers typically
#   sit in the 2×–5× range; specialty-dependent).
# - The per-specialty bar chart shows variation, not a flat line —
#   that confirms specialty *does* matter as a benchmark.
#
# ### Learning note — what is the charge-to-allowed ratio?
#
# Medicare publishes two amounts for every billed service:
#
# - **Submitted charge**: the provider's stated price (their
#   "chargemaster" rate).
# - **Allowed amount**: what Medicare's fee schedule permits — what
#   they will actually pay (plus the patient's coinsurance).
#
# The ratio = submitted ÷ allowed. A ratio of `3.0` means the
# provider submits charges three times what Medicare will pay. This
# number rarely changes Medicare's own payment (Medicare ignores the
# inflated portion), but it can affect secondary payers and
# coordination-of-benefits calculations. It's a useful **review
# signal** because providers with stable, conservative chargemasters
# look different from providers whose submitted charges fluctuate
# wildly above the fee schedule.

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
axes[0].set_title("Charge-to-Allowed Ratio Distribution")
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
axes[1].set_title("Median Charge-to-Allowed Ratio by Specialty")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(PATHS["reports"] / "charge_to_allowed.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 5.2 Service Concentration (HHI)
#
# **Business Purpose** — Find providers whose billing is heavily
# concentrated in one or two procedure codes — a pattern auditors
# often want to verify is consistent with the provider's stated
# specialty and practice model.
#
# **Plain-English Logic** — For each NPI, compute the
# Herfindahl-Hirschman Index across the share of services in each
# HCPCS code billed. HHI close to 1.0 = single procedure; HHI close
# to 0 = highly diversified.
#
# **Expected Output** — A right-skewed distribution. Most providers
# bill many procedures and sit at low HHI; a smaller tail clusters
# above 0.5.
#
# **Why It Matters** — High-HHI providers are not inherently doing
# anything wrong — many subspecialists legitimately bill almost
# entirely from one code (e.g., a sleep-study reader). The HHI
# feature is most informative when **combined** with specialty
# context and other features in the composite score.
#
# **What to Check Before Moving On**
# - Most mass of the histogram sits below 0.25.
# - The vertical reference lines render at 0.25 and 0.50.
#
# ### Learning note — HHI / service concentration
#
# The Herfindahl-Hirschman Index is most commonly used in antitrust
# to measure market concentration. We are reusing the same formula
# on a single provider's procedure mix:
#
# ```
# HHI = Σ (share_of_services_in_code_i) ²
# ```
#
# - Bill 100 distinct codes equally → HHI ≈ 0.01 (very diversified).
# - Bill 80% of services from one code → HHI ≈ 0.64 (very
#   concentrated).
#
# Concentrated billing is a **review signal**, not a finding. Many
# specialties legitimately concentrate. The signal becomes
# interesting when, for example, a self-described general internist
# has an HHI of 0.95 in a single high-cost code — at that point an
# auditor may want to look at the actual records and policy
# requirements.

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
# **Business Purpose** — Combine the individual signal features into a
# single ranked score so we can sort the worklist top-to-bottom.
#
# **Plain-English Logic** — A weighted sum of normalized features.
# The weights are domain-informed but **not learned** — that is the
# next notebook's job.
#
# **Expected Output** — A right-skewed distribution with the bulk of
# providers near zero and a long tail. Plus a per-specialty count of
# which specialties are over-represented in the top-1,000 risk tier.
#
# **Why It Matters** — This is the simplest possible "ranker" and
# serves as a baseline for the supervised model in Notebook 03. If a
# trained model can't beat this baseline, something is wrong with
# the features, the labels, or the training setup.
#
# **What to Check Before Moving On**
# - No NaNs in the visible histogram range.
# - The top-risk specialty mix is plausible (durable medical
#   equipment, high-utilization codes, etc. tend to surface — that
#   is expected, not a finding).
#
# ### Learning note — the composite score is a baseline, not a model
#
# It is tempting to look at this score and treat it as "the answer."
# It isn't. A composite risk score is a hand-weighted formula
# combining features the analyst already believes are informative.
# It has three big limitations:
#
# 1. **The weights are guesses.** They reflect the author's
#    judgment, not data.
# 2. **No interactions.** A trained model can learn that "high HHI
#    is only interesting when the provider also has a high charge
#    ratio AND a low-volume HCPCS code." A weighted sum cannot.
# 3. **No calibration.** The output isn't a probability of
#    anything — it's a ranking.
#
# Treat the composite as a **baseline** the supervised model in
# Notebook 03 must beat. That is its real job.

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
# **Business Purpose** — Confirm the features carry different
# information, not many redundant restatements of the same signal.
#
# **Plain-English Logic** — Compute the pairwise correlation matrix
# across all numeric features and visualize the lower triangle. Flag
# pairs with |r| > 0.80 as candidates for dropping.
#
# **Expected Output** — A masked heatmap saved to
# `reports/feature_correlations.png`, plus a printed list of
# highly-correlated feature pairs (or a confirmation that none
# exist).
#
# **Why It Matters** — Highly-correlated features inflate model
# variance and waste compute without adding signal. They also make
# downstream feature-importance analysis confusing.
#
# **What to Check Before Moving On**
# - The heatmap renders without saturating (mostly soft colors, not
#   a sea of deep red/blue).
# - Any pair flagged at |r| > 0.80 is one you are willing to act on
#   later (typically: keep one, drop the other before modeling).
#
# ### Learning note — what does a correlation matrix tell you?
#
# Correlation only measures **linear** association between two
# features. Two features with `r ≈ 0.0` are not necessarily
# independent — they may be related non-linearly. Two features
# with `r ≈ 0.9` are essentially the same column with noise.
#
# A practical rule of thumb for audit features:
#
# - `|r| < 0.5` → distinct signals, keep both.
# - `0.5 ≤ |r| < 0.8` → some overlap, but each may add value;
#   defer the decision to model-time importance analysis.
# - `|r| ≥ 0.8` → almost certainly redundant; pick one.

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
# **Business Purpose** — Persist the feature-engineered dataframe so
# downstream notebooks can load it instantly without re-running
# `build_all_features`.
#
# **Plain-English Logic** — Write `df` to a Parquet file, then dump
# the feature metadata (descriptions, weights) to a JSON sidecar.
#
# **Expected Output** — One `.parquet` and one `.json` file in
# `data/processed/`, with their sizes printed.
#
# **Why It Matters** — Notebook 02 onward can read this file in
# seconds. If it's missing or corrupt, every downstream step
# breaks.
#
# **What to Check Before Moving On**
# - The Parquet size is roughly one-fifth of the CSV equivalent.
# - The JSON metadata file exists and parses cleanly.
#
# ### Learning note — why Parquet, not CSV?
#
# CSV is human-readable, but the wrong format for analytical data:
#
# - **Type-safe**: Parquet stores dtypes; CSV does not. A column
#   that is `float32` here loads as `float32` in the next notebook
#   automatically.
# - **Compressed**: Parquet uses columnar compression (Snappy by
#   default) — typically a 5–10× shrink versus CSV.
# - **Fast**: Columnar layout means "read just these 5 columns"
#   only touches those bytes on disk. Loads are often 10× faster.
# - **Standard**: pandas, polars, DuckDB, Spark, BigQuery, and
#   Athena all read Parquet natively.
#
# CSV is for handing data to humans. Parquet is for handing data
# to your next notebook.

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
# | Features built | ~28 domain-informed audit-priority signals |
# | Output format | Parquet (fast, compressed, type-safe) |
# | Next notebook | 02_anomaly_detection.py (Isolation Forest, DBSCAN) |
#
# **Reminder of scope** — None of these features identifies fraud,
# overpayment, or noncompliance. They produce a prioritized list of
# **patterns that may warrant follow-up review**. Translating any
# specific row into an actual audit conclusion requires
# claims-level records, applicable CMS policy review, provider
# context (subspecialty, group setting, patient mix), and the
# professional judgment of a trained auditor.
#
# **Where the domain knowledge actually shows up** — Picking which
# features to build in the first place, choosing peer groups, and
# knowing which combinations of signals are worth a closer look
# (rather than chasing every individual outlier) is the value an
# experienced auditor adds on top of the math.

# %%
print(f"\n{'='*60}")
print(f"  FEATURE ENGINEERING COMPLETE")
print(f"{'='*60}")
print(f"  Features: {len(feat_cols)}")
print(f"  Rows:     {len(df):,}")
print(f"  Output:   {output_path}")
print(f"\n  Next: notebooks/02_anomaly_detection.py")
