# %%
"""
Notebook 02 — Provider-Level Anomaly Detection and Weak Supervision

Business Purpose:
This notebook converts Medicare Part B provider-service level features into a
provider-level audit-priority dataset. The goal is to identify unusual provider
billing patterns that may warrant follow-up review using only public CMS data.

Plain-English Logic:
Notebook 01 created one row per provider-service combination:
    NPI + HCPCS code + place of service

Notebook 02 reshapes those rows into one row per provider:
    one NPI = one provider profile

Then it applies an unsupervised anomaly detection model (Isolation Forest) and
combines that signal with two domain-informed extreme-pattern indicators
(composite risk and peer deviation) to build a conservative weak-supervision
label. That weak label becomes the target variable for the supervised models in
Notebook 03.

Key Concepts Introduced in This Notebook:
- Provider-service level vs provider level data structure
- Why aggregation is needed before anomaly detection
- Why only numeric features are used for modeling
- Why missing values are median-imputed
- Why standardization (z-scaling) matters for distance-based models
- What Isolation Forest does in plain English
- Why a 5% contamination setting means "most unusual 5%," not "wrong providers"
- Why anomaly flags should be read as decision-support signals, not findings
- Why provider-type concentration in anomaly results is a real limitation
- Why a conservative weak label is preferred over a single noisy signal

Important Limitation:
This notebook does not identify fraud, overpayments, noncompliance, or audit
findings. It produces public-data decision-support signals for portfolio
modeling and follow-up-review prioritization. Any provider flagged here would
require independent investigation before any real-world conclusion.
"""

# %%
# Step 1 — Environment Setup and Imports
#
# Business Purpose:
# Load the Python libraries and project settings needed to read the processed
# feature table from Notebook 01.
#
# Plain-English Logic:
# We are setting up the notebook so it can find the project folders, load the
# parquet file, and inspect the feature columns.
#
# Expected Output:
# The notebook should print the project root and confirm the environment is ready.
#
# Why It Matters:
# A clean setup prevents path errors and makes the notebook reproducible.
#
# What to Check Before Moving On:
# Confirm PROJECT_ROOT points to the medicare-claims-audit-intelligence folder.

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import PATHS, DEFAULT_YEAR

YEAR = DEFAULT_YEAR

print("Project root:", PROJECT_ROOT)
print("Year:", YEAR)

# %%
# Step 2 — Load Processed Provider-Service Feature Table
#
# Business Purpose:
# Load the feature table created by Notebook 01.
#
# Plain-English Logic:
# Notebook 01 saved a parquet file containing the original CMS provider-service
# data plus the new feat_ columns. We load that file here instead of rebuilding
# all features from scratch.
#
# Expected Output:
# A dataframe with millions of provider-service rows and the engineered features.
#
# Why It Matters:
# Notebook 02 depends on Notebook 01's output. This step confirms the handoff worked.
#
# What to Check Before Moving On:
# Confirm the file exists, loads successfully, and includes feat_ columns.

feature_path = PATHS["data_processed"] / f"features_provider_service_{YEAR}_full.parquet"

print("Loading:", feature_path)

df = pd.read_parquet(feature_path)

print("Rows:", f"{len(df):,}")
print("Columns:", f"{df.shape[1]:,}")

df.head()

# %%
# Step 3 — Confirm Provider-Service Level Structure
#
# Business Purpose:
# Verify that the loaded table is still at the provider-service level (the grain
# Notebook 01 produced) before we aggregate it up to the provider level.
#
# Plain-English Logic:
# In the CMS Part B public file, "grain" describes what one row represents:
#   - Provider-service level = one row per (NPI + HCPCS code + place of service).
#     A single provider can appear on dozens or hundreds of rows because they
#     bill many service codes in many settings.
#   - Provider level = one row per NPI, summarizing all of that provider's
#     services into a single profile.
# Anomaly detection in this notebook needs the provider-level grain. Before we
# reshape the data we confirm the current structure so the aggregation is
# defensible.
#
# Expected Output:
# Counts of unique NPIs, unique HCPCS codes, unique places of service, unique
# specialties, and total rows.
#
# Why It Matters:
# If we forgot to aggregate, an Isolation Forest would treat each
# provider-service row as an independent observation. A high-volume provider
# with 200 services would influence the model 200 times, while a small provider
# with 2 services would barely register. Confirming the grain upfront prevents
# that bias.
#
# What to Check Before Moving On:
# The number of rows should be much larger than the number of unique NPIs. That
# multiplicity is exactly what the next step collapses.

structure_summary = {
    "rows": len(df),
    "unique_npis": df["Rndrng_NPI"].nunique() if "Rndrng_NPI" in df.columns else None,
    "unique_hcpcs": df["HCPCS_Cd"].nunique() if "HCPCS_Cd" in df.columns else None,
    "unique_pos": df["Place_Of_Srvc"].nunique() if "Place_Of_Srvc" in df.columns else None,
    "unique_specialties": df["Rndrng_Prvdr_Type"].nunique() if "Rndrng_Prvdr_Type" in df.columns else None,
}

pd.Series(structure_summary)

# %%
# Step 4 — Identify Feature Columns and Provider Identifier Columns
#
# Business Purpose:
# Separate the columns we will use for modeling from the columns we will use for
# describing and explaining the providers we flag.
#
# Plain-English Logic:
# Notebook 01 prefixed every engineered feature with feat_, which makes them easy
# to pick out programmatically. Provider identifier columns (NPI, name, specialty,
# state, entity type) are not features in the modeling sense — they describe who
# the provider is. We need both:
#   - feat_ columns for anomaly detection (the inputs to the model)
#   - provider identity columns for interpretation and review tables (the
#     human-readable context that lets us explain a flag)
#
# Expected Output:
# A printed list of engineered features and the provider identity columns that
# exist in this dataframe.
#
# Why It Matters:
# Anomaly detection on identifier strings does not make sense — you cannot
# compute a "z-score" of a provider name or state. Keeping the two groups of
# columns clearly separated prevents accidental leakage of identity into the
# model and keeps the review output readable.
#
# What to Check Before Moving On:
# Confirm that the expected feat_ columns from Notebook 01 are present and that
# at least Rndrng_NPI is in the identifier list.

feature_cols = sorted([col for col in df.columns if col.startswith("feat_")])

provider_id_candidates = [
    "Rndrng_NPI",
    "Rndrng_Prvdr_Last_Org_Name",
    "Rndrng_Prvdr_First_Name",
    "Rndrng_Prvdr_Type",
    "Rndrng_Prvdr_State_Abrvtn",
    "Rndrng_Prvdr_Ent_Cd",
    "Rndrng_Prvdr_Mdcr_Prtcptg_Ind",
]

provider_id_cols = [col for col in provider_id_candidates if col in df.columns]

print(f"Feature columns found: {len(feature_cols)}")
for col in feature_cols:
    print(" -", col)

print("\nProvider identifier columns:")
for col in provider_id_cols:
    print(" -", col)
    
# %%
# Step 5 — Aggregate Provider-Service Rows to Provider Level
#
# Business Purpose:
# Collapse the provider-service feature table into one row per provider so that
# anomaly detection scores providers (the unit of audit-priority review), not
# individual service-line items.
#
# Plain-English Logic:
# Each numeric feat_ column is summarized three ways for every provider:
#   - mean   = the provider's average pattern across their services
#   - max    = the provider's most extreme pattern (often the audit signal)
#   - median = the provider's typical pattern, less affected by one outlier
# We also bring forward core utilization and payment columns as sums and means
# (total services, total beneficiaries, total Medicare paid, etc.) plus simple
# diversity counts (how many distinct HCPCS codes and places of service the
# provider used). Identity columns like provider_type and provider_state are
# carried through with "first" so we can interpret the model output later.
#
# Expected Output:
# provider_features — a dataframe with one row per NPI and many summarized
# numeric columns plus identity columns.
#
# Why It Matters:
# Without aggregation, a high-volume provider would dominate the model simply
# by appearing on more rows. Aggregating first makes every provider a single
# observation, which is what we need so the anomaly model compares providers
# to other providers — not service lines to service lines.
#
# What to Check Before Moving On:
# The number of provider-level rows should equal the number of unique NPIs in
# the provider-service table printed in Step 3.

# Numeric feature columns only
numeric_feature_cols = [
    col for col in feature_cols
    if pd.api.types.is_numeric_dtype(df[col])
]

print(f"Numeric feature columns found: {len(numeric_feature_cols)}")

# Build aggregation rules for engineered features.
# We use multiple summaries because different audit-priority signals matter in
# different ways:
#   mean = provider's average pattern
#   max = provider's most extreme pattern
#   median = provider's typical pattern
agg_dict = {}

for col in numeric_feature_cols:
    agg_dict[f"{col}_mean"] = (col, "mean")
    agg_dict[f"{col}_max"] = (col, "max")
    agg_dict[f"{col}_median"] = (col, "median")

# Add core utilization and payment summaries when available
core_numeric_cols = [
    "Tot_Srvcs",
    "Tot_Benes",
    "Tot_Bene_Day_Srvcs",
    "Avg_Sbmtd_Chrg",
    "Avg_Mdcr_Alowd_Amt",
    "Avg_Mdcr_Pymt_Amt",
    "Avg_Mdcr_Stdzd_Amt",
    "Tot_Mdcr_Pymt_Amt",
]

for col in core_numeric_cols:
    if col in df.columns:
        agg_dict[f"core_{col}_sum"] = (col, "sum")
        agg_dict[f"core_{col}_mean"] = (col, "mean")
        agg_dict[f"core_{col}_max"] = (col, "max")

# Add service diversity and record count
provider_features = (
    df
    .groupby("Rndrng_NPI")
    .agg(
        provider_service_records=("Rndrng_NPI", "size"),
        unique_hcpcs=("HCPCS_Cd", "nunique"),
        unique_pos=("Place_Of_Srvc", "nunique"),
        provider_type=("Rndrng_Prvdr_Type", "first"),
        provider_state=("Rndrng_Prvdr_State_Abrvtn", "first"),
        provider_entity_type=("Rndrng_Prvdr_Ent_Cd", "first"),
        **agg_dict,
    )
    .reset_index()
)

print("Provider-level rows:", f"{len(provider_features):,}")
print("Provider-level columns:", f"{provider_features.shape[1]:,}")
print("Original unique NPIs:", f"{df['Rndrng_NPI'].nunique():,}")

provider_features.head()

# %%
# Step 6 — Select Numeric Features for Anomaly Detection
#
# Business Purpose:
# Build X_provider, the numeric-only matrix that the anomaly model will consume.
#
# Plain-English Logic:
# Isolation Forest (and most anomaly detection methods) operate on numeric
# vectors — they measure how unusual a row is by comparing its numbers to other
# rows. Identifier columns like NPI, provider_type, provider_state, and
# provider_entity_type are categorical strings. They are critical for explaining
# results to a human reviewer, but feeding them directly into the model does
# not make sense and can leak identity into the score. We exclude them from X
# and keep them on provider_features for interpretation.
#
# Expected Output:
# X_provider — a numeric dataframe with the same number of rows as
# provider_features and only numeric columns.
#
# Why It Matters:
# Cleanly separating "model input" from "interpretation context" makes the
# pipeline easier to audit. If a flagged provider's specialty looks structurally
# different from peers, we want to discover that in the explanation step, not
# accidentally encode it as a feature the model can pattern-match on.
#
# What to Check Before Moving On:
# X_provider.shape should equal (len(provider_features), number of numeric
# model columns), and X_provider.dtypes should show no object columns.

exclude_cols = [
    "Rndrng_NPI",
    "provider_type",
    "provider_state",
    "provider_entity_type",
]

candidate_model_cols = [
    col for col in provider_features.columns
    if col not in exclude_cols
]

numeric_model_cols = [
    col for col in candidate_model_cols
    if pd.api.types.is_numeric_dtype(provider_features[col])
]

X_provider = provider_features[numeric_model_cols].copy()

print("Provider rows:", f"{len(provider_features):,}")
print("Numeric model columns:", f"{len(numeric_model_cols):,}")
print("X_provider shape:", X_provider.shape)

X_provider.head()

# %%
# Step 7 — Handle Missing Values for Anomaly Detection
#
# Business Purpose:
# Replace any remaining NaNs in the numeric matrix so the anomaly detection
# model does not error out and so missing values do not silently become outliers.
#
# Plain-English Logic:
# Many engineered features are ratios (charge-to-allowed, drug revenue share,
# services per beneficiary). Ratios can be undefined when their denominator is
# zero or missing, which produces NaN values. We use the column median as the
# fill value because:
#   - the median is robust to skew (most CMS payment columns are very
#     right-skewed), so it is a more honest "typical value" than the mean
#   - it is simple, transparent, and reproducible — anyone reviewing the work
#     can explain it in one sentence
#   - it does not invent extreme values that the anomaly model would then flag
# We then add a final fillna(0) safety net for the rare case where an entire
# column is NaN (its median would also be NaN).
#
# Expected Output:
# X_provider_clean — the same shape as X_provider but with zero missing values.
#
# Why It Matters:
# A more sophisticated imputation (KNN, model-based) would tangle the imputation
# step with the modeling step and make it harder to explain why a provider was
# flagged. For a learning project that emphasizes interpretability, median
# imputation is the right tradeoff.
#
# What to Check Before Moving On:
# Missing values before should be a non-negative integer; missing values after
# should be exactly 0.

missing_before = X_provider.isna().sum().sum()

feature_medians = X_provider.median(numeric_only=True)

X_provider_clean = X_provider.fillna(feature_medians)

# If any column is entirely missing, the median may also be NaN.
# Fill any remaining missing values with 0 as a final safety step.
X_provider_clean = X_provider_clean.fillna(0)

missing_after = X_provider_clean.isna().sum().sum()

print("Missing values before:", f"{missing_before:,}")
print("Missing values after:", f"{missing_after:,}")

# Show columns with the most missing values before imputation
missing_by_col = (
    X_provider
    .isna()
    .sum()
    .sort_values(ascending=False)
    .head(15)
)

missing_by_col

# %%
# Step 8 — Standardize Numeric Features
#
# Business Purpose:
# Put every numeric feature onto the same scale (mean 0, standard deviation 1)
# so no single column dominates the anomaly model just because its raw numbers
# are larger.
#
# Plain-English Logic:
# Our features live on wildly different scales:
#   - core_Tot_Mdcr_Pymt_Amt_sum can be in the millions of dollars
#   - feat_charge_to_allowed_max is a ratio close to 1
#   - feat_hhi_max is a concentration index between 0 and 1
#   - service counts are integers from 1 to tens of thousands
# StandardScaler subtracts each column's mean and divides by its standard
# deviation, so each column ends up centered at 0 with a spread of 1. After
# scaling, "1 unit" means "1 standard deviation" in every column, which is the
# common language Isolation Forest needs to compare features fairly.
#
# Expected Output:
# X_provider_scaled — a NumPy array with the same shape as X_provider_clean.
# Mean of each column ≈ 0, std of each column ≈ 1.
#
# Why It Matters:
# Isolation Forest builds random splits on feature values. A feature with huge
# raw magnitudes will produce huge split values and end up dominating which
# providers look "far from the rest." Scaling neutralizes that bias so the
# model treats a one-standard-deviation move in payments the same as a
# one-standard-deviation move in HCPCS diversity.
#
# What to Check Before Moving On:
# The printed average absolute scaled mean should be very close to 0, and the
# average scaled std should be very close to 1.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_provider_scaled = scaler.fit_transform(X_provider_clean)

print("Original shape:", X_provider_clean.shape)
print("Scaled shape:", X_provider_scaled.shape)

# Quick sanity check: scaled columns should have means near 0.
scaled_means = X_provider_scaled.mean(axis=0)
scaled_stds = X_provider_scaled.std(axis=0)

print("Average absolute scaled mean:", round(float(np.abs(scaled_means).mean()), 6))
print("Average scaled std:", round(float(scaled_stds.mean()), 6))

# %%
# Step 9 — Run Isolation Forest Anomaly Detection
#
# Business Purpose:
# Score every provider on how "unusual" their feature profile looks compared to
# the overall Part B population, and flag the most unusual ones for review.
#
# Plain-English Logic:
# Isolation Forest is an unsupervised method — it does not need labels. The
# core idea is simple:
#   1. Build many random trees by repeatedly splitting the data on random
#      features at random values.
#   2. Count how many splits it takes to "isolate" each provider into its own
#      leaf.
#   3. Providers that are unusual get isolated quickly (few splits). Providers
#      that look like everyone else require many splits before they end up
#      alone.
# The "anomaly score" we compute below is just a transformed version of that
# isolation depth: higher score = unusual, lower score = typical.
#
# About contamination=0.05:
# This parameter tells the model "treat the most unusual 5% as anomalies." It
# is a labeling threshold we choose, not a measurement. It does NOT mean 5% of
# providers are wrong, fraudulent, or noncompliant — it means we are asking the
# model to surface the 5% of providers whose profile differs most from the
# rest. Pick a different contamination value and you simply move the cut line.
#
# Expected Output:
# Two new columns on provider_features:
#   - iso_anomaly_flag (1 = flagged as anomaly, 0 = typical)
#   - iso_anomaly_score (higher = more unusual; we negate sklearn's
#     decision_function so the direction is intuitive)
#
# Why It Matters:
# Public CMS data has no confirmed audit outcomes, so we cannot train a
# supervised model directly. Isolation Forest gives us a transparent,
# label-free starting signal that can later be combined with domain-informed
# rules (Step 13) to create a weak-supervision target.
#
# What to Check Before Moving On:
# About 5% of providers should be flagged (iso_anomaly_flag == 1). If the share
# is dramatically different, re-check the contamination argument.

from sklearn.ensemble import IsolationForest

iso_model = IsolationForest(
    n_estimators=300,
    contamination=0.05,
    random_state=42,
    n_jobs=-1,
)

iso_model.fit(X_provider_scaled)

# sklearn returns:
#   -1 = anomaly
#    1 = normal
iso_pred = iso_model.predict(X_provider_scaled)

# decision_function gives higher values for more normal observations.
# We multiply by -1 so higher scores mean more unusual.
iso_score = -iso_model.decision_function(X_provider_scaled)

provider_features["iso_anomaly_flag"] = (iso_pred == -1).astype(int)
provider_features["iso_anomaly_score"] = iso_score

anomaly_counts = (
    provider_features["iso_anomaly_flag"]
    .value_counts()
    .rename_axis("iso_anomaly_flag")
    .reset_index(name="count")
)

anomaly_counts["pct"] = (
    anomaly_counts["count"] / anomaly_counts["count"].sum() * 100
).round(2)

anomaly_counts

# %%
# Step 9B — Visualize Isolation Forest Anomaly Score Distribution
#
# Business Purpose:
# Show the shape of the iso_anomaly_score distribution so a reader can see
# where the contamination cut line falls and how flagged versus unflagged
# providers separate.
#
# Plain-English Logic:
# We plot a histogram of iso_anomaly_score. Providers below the
# contamination=0.05 threshold form the bulk of the distribution; the right
# tail contains the providers that got iso_anomaly_flag == 1. We split the
# histogram into two overlaid series so the reader can see both populations
# at once.
#
# Expected Output:
# A figure saved to reports/isolation_forest_anomaly_scores.png. Title and
# labels use review-priority language only — no fraud or finding wording.
#
# Why It Matters:
# A distribution plot makes the contamination parameter concrete. It lets the
# reader see that "flagged" is just a chosen tail of a continuous score, not
# a binary determination of unusual behavior.

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))

scores_unflagged = provider_features.loc[
    provider_features["iso_anomaly_flag"] == 0, "iso_anomaly_score"
]
scores_flagged = provider_features.loc[
    provider_features["iso_anomaly_flag"] == 1, "iso_anomaly_score"
]

ax.hist(
    scores_unflagged,
    bins=80,
    color="#4C72B0",
    alpha=0.75,
    edgecolor="white",
    label="Unflagged providers",
)
ax.hist(
    scores_flagged,
    bins=80,
    color="#C44E52",
    alpha=0.75,
    edgecolor="white",
    label="Flagged providers (review-priority signal)",
)

cut_line = scores_flagged.min() if len(scores_flagged) else None
if cut_line is not None:
    ax.axvline(
        cut_line,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Flag threshold ≈ {cut_line:.3f}",
    )

ax.set_xlabel("Isolation Forest Anomaly Score (higher = more unusual)")
ax.set_ylabel("Number of providers")
ax.set_title("Anomaly Score Distribution — Isolation Forest Review-Priority Signal")
ax.legend()

plt.tight_layout()

iso_score_plot_path = PATHS["reports"] / "isolation_forest_anomaly_scores.png"
plt.savefig(iso_score_plot_path, dpi=150, bbox_inches="tight")
plt.show()

print("Saved:", iso_score_plot_path)

# %%
# Step 10 — Review Highest-Scoring Anomaly Providers
#
# Business Purpose:
# Look directly at the top-scoring providers so the anomaly model is not
# treated as a black box.
#
# Plain-English Logic:
# We sort providers by iso_anomaly_score (descending) and pull a small set of
# columns that mix identity (NPI, provider type, state) with interpretable
# feature summaries (composite risk, peer deviation, charge ratios, payments).
# The goal is to be able to answer the question "why was this provider flagged?"
# in plain language.
#
# Expected Output:
# A 25-row table of the highest-scoring providers with identity and
# interpretation columns side by side.
#
# Why It Matters:
# The anomaly flag is a decision-support signal, not a finding. Before treating
# any score as meaningful, a reviewer should be able to read across a flagged
# provider's row and understand which feature patterns drove the score —
# extremely high charges relative to allowed amounts, very concentrated billing,
# unusual service mix, and so on. If the patterns are not explainable from the
# columns shown here, the signal is not yet trustworthy.
#
# What to Check Before Moving On:
# The table should include both identity columns and at least a few feat_*
# summaries. Read a handful of rows and see whether the patterns are
# distinguishable from a typical provider.

review_cols = [
    "Rndrng_NPI",
    "provider_type",
    "provider_state",
    "provider_entity_type",
    "provider_service_records",
    "unique_hcpcs",
    "unique_pos",
    "iso_anomaly_flag",
    "iso_anomaly_score",
]

# Add a small set of interpretable high-value summary features if present.
priority_review_cols = [
    "feat_composite_risk_max",
    "feat_composite_risk_mean",
    "feat_max_peer_deviation_max",
    "feat_charge_to_allowed_max",
    "feat_hhi_max",
    "feat_log_srvcs_per_bene_max",
    "feat_drug_revenue_share_max",
    "feat_facility_ratio_max",
    "core_Tot_Mdcr_Pymt_Amt_sum",
    "core_Tot_Srvcs_sum",
    "core_Tot_Benes_sum",
]

available_review_cols = [
    col for col in review_cols + priority_review_cols
    if col in provider_features.columns
]

top_anomalies = (
    provider_features
    .sort_values("iso_anomaly_score", ascending=False)
    [available_review_cols]
    .head(25)
)

top_anomalies

# %%
# Step 11 — Summarize Anomalies by Provider Type
#
# Business Purpose:
# Look at the anomaly group through the lens of provider type (specialty) to
# spot whether the model is detecting unusual individual behavior or unusual
# provider categories.
#
# Plain-English Logic:
# For each provider_type we count:
#   - total_providers       (how many providers of that type exist overall)
#   - anomaly_providers     (how many of them got an iso_anomaly_flag)
#   - avg / max anomaly score
# Then we sort to see which specialties dominate the anomaly group.
#
# Expected Output:
# A table where each row is a provider type, sorted by raw anomaly counts.
#
# Why It Matters:
# Anomaly detection at the population level cannot tell the difference between
# "this individual provider behaves differently from peers" and "this whole
# specialty operates differently from a typical Part B provider." Independent
# diagnostic testing facilities, ambulance providers, ambulatory surgical
# centers, and clinical labs all have legitimately different billing footprints
# than, say, a primary care physician. If those types dominate the anomaly
# list, the score is partly capturing specialty structure, not provider-level
# unusualness.
#
# What to Check Before Moving On:
# Note which provider types appear most frequently and whether they are
# specialties known to have structurally unusual billing (labs, ambulance,
# ASCs). That observation feeds the limitation note in Step 12.

provider_type_summary = (
    provider_features
    .groupby("provider_type")
    .agg(
        total_providers=("Rndrng_NPI", "count"),
        anomaly_providers=("iso_anomaly_flag", "sum"),
        avg_anomaly_score=("iso_anomaly_score", "mean"),
        max_anomaly_score=("iso_anomaly_score", "max"),
    )
    .reset_index()
)

provider_type_summary["anomaly_rate"] = (
    provider_type_summary["anomaly_providers"] /
    provider_type_summary["total_providers"]
)

provider_type_summary = provider_type_summary.sort_values(
    ["anomaly_providers", "anomaly_rate"],
    ascending=False
)

provider_type_summary.head(20)

# %%
# Step 11B — Review Highest Anomaly Rates by Provider Type
#
# Business Purpose:
# Switch from raw counts to rates so we can see which specialties are
# disproportionately flagged relative to their population size.
#
# Plain-English Logic:
# A specialty with 50,000 providers will produce a lot of anomaly flags simply
# because it is large. Rates (anomaly_providers / total_providers) reveal a
# different signal: which specialties get flagged at a much higher percentage
# than the overall ~5% baseline. We restrict to types with at least 50
# providers so a tiny specialty with 2 of 3 flagged does not dominate the view.
#
# Expected Output:
# Provider types ranked by anomaly_rate, filtered to total_providers >= 50.
#
# Why It Matters:
# A high anomaly rate inside a specialty almost always means the specialty's
# normal billing footprint is structurally different from the population
# average — not that the specialty is doing something wrong. This is the most
# important interpretation guardrail in the notebook.
#
# What to Check Before Moving On:
# Compare the top rates here with the top counts in Step 11. The same
# specialties tend to surface — that pattern is exactly what motivates the
# limitation note in Step 12.

provider_type_summary_rate = (
    provider_type_summary
    .query("total_providers >= 50")
    .sort_values("anomaly_rate", ascending=False)
)

provider_type_summary_rate[
    [
        "provider_type",
        "total_providers",
        "anomaly_providers",
        "anomaly_rate",
        "avg_anomaly_score",
        "max_anomaly_score",
    ]
].head(20)

# %%
# Step 11C — Visualize Provider Types with the Highest Anomaly Rates
#
# Business Purpose:
# Make the structural-difference pattern from Step 11B visible at a glance:
# which provider types get flagged at rates well above the overall ~5%
# contamination baseline.
#
# Plain-English Logic:
# We take provider_type_summary_rate (already filtered to total_providers
# >= 50) and bar-chart the top 15 provider types by anomaly_rate. A
# reference line at 0.05 marks the baseline contamination so the reader can
# see how far each specialty sits above population average.
#
# Expected Output:
# A figure saved to reports/provider_type_anomaly_rates.png with safe,
# review-priority language. No fraud or finding wording.
#
# Why It Matters:
# The chart visually reinforces the limitation note in Step 12: provider
# types with structurally distinct billing footprints (labs, ambulance,
# ASCs, IDTFs) are more likely to be flagged by a population-level model —
# this is a feature of the data, not evidence of provider behavior.

top_rate_n = 15
top_rate_chart_data = (
    provider_type_summary_rate
    .head(top_rate_n)
    .iloc[::-1]
)

fig, ax = plt.subplots(figsize=(12, 8))

ax.barh(
    range(len(top_rate_chart_data)),
    top_rate_chart_data["anomaly_rate"].values,
    color="#4C72B0",
    alpha=0.85,
    edgecolor="white",
)
ax.set_yticks(range(len(top_rate_chart_data)))
ax.set_yticklabels(top_rate_chart_data["provider_type"].values, fontsize=9)
ax.axvline(
    0.05,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="Contamination baseline (0.05)",
)
ax.set_xlabel("Share of providers flagged (anomaly rate)")
ax.set_title(
    "Anomaly Rate by Provider Type — Provider-Type Structural Differences"
)
ax.legend(loc="lower right")

plt.tight_layout()

type_rate_plot_path = PATHS["reports"] / "provider_type_anomaly_rates.png"
plt.savefig(type_rate_plot_path, dpi=150, bbox_inches="tight")
plt.show()

print("Saved:", type_rate_plot_path)

# %%
# Step 12 — Document Interpretation Limitation
#
# Business Purpose:
# Capture, in plain English, the main limitation revealed by Steps 11 and 11B
# so future readers (and future-you) interpret the anomaly flag correctly.
#
# Plain-English Logic:
# The first Isolation Forest model identifies providers that differ from the
# overall Part B population, but some anomaly scores may reflect provider-type
# structure rather than unusual behavior within a true peer group. Future
# versions should consider provider-type-specific or specialty-specific
# anomaly detection so each provider is compared to a meaningful peer group
# rather than to the whole Medicare Part B universe.
#
# Expected Output:
# A printed interpretation note that documents the limitation in writing.
#
# Why It Matters:
# Honest limitations are part of the deliverable. Without this note a reader
# could mistake "high anomaly score" for "high audit finding," which this
# project explicitly does not claim. Anomaly flags here are decision-support
# signals only.
#
# What to Check Before Moving On:
# The note should clearly say (a) anomaly detection compares each provider to
# the overall population, (b) some flags reflect specialty structure, and
# (c) the flag is a decision-support signal, not a fraud, overpayment,
# noncompliance, or audit finding.

interpretation_note = """
Initial Isolation Forest Interpretation

The first Isolation Forest model identifies providers that differ from the overall
Medicare Part B population. In Steps 11 and 11B the anomaly flags concentrated in
structurally distinct provider types such as ambulatory surgical centers, ambulance
service providers, independent diagnostic testing facilities, clinical laboratories,
and oncology-related specialties.

This is useful because it shows the model can identify providers that differ from
the broader population. However, it also reveals an important limitation: some
anomaly scores may reflect provider-type structure rather than unusual behavior
within a true peer group. A primary care physician and a clinical lab have very
different normal billing profiles, and a population-level model will tend to flag
the lab simply because its profile is unlike most providers.

For that reason, the Isolation Forest flag should be interpreted as an
audit-priority decision-support signal, not as evidence of fraud, overpayment,
noncompliance, or an audit finding. A future improvement should evaluate
provider-type-specific or specialty-specific anomaly detection so each provider
is scored against a meaningful peer group.
"""

print(interpretation_note)

# %%
# Step 13 — Create Initial Weak-Supervision Audit-Priority Label
#
# Business Purpose:
# Build a conservative provider-level pseudo-label that Notebook 03 can use as
# a target variable for supervised models (XGBoost, LightGBM).
#
# Plain-English Logic:
# We do not have ground-truth audit outcomes, so we cannot label providers
# directly. Weak supervision is a standard practice for this situation: combine
# multiple imperfect signals so the resulting label is more reliable than any
# single one. We use three signals:
#   1. iso_anomaly_flag           = unsupervised "different from the rest"
#   2. extreme_composite_risk     = composite_risk_max in the top 5%
#                                   (a domain-informed roll-up of audit-priority
#                                    feature signals from Notebook 01)
#   3. extreme_peer_deviation     = max_peer_deviation_max in the top 5%
#                                   (how far a provider's services depart from
#                                    HCPCS-level peer averages)
# The conservative AND/OR rule:
#   weak_label_high_audit_priority = iso_anomaly_flag
#       AND (extreme_composite_risk OR extreme_peer_deviation)
# A provider must look unusual in the unsupervised sense AND also be extreme in
# at least one domain-informed signal. That combination is much harder to
# satisfy by accident than any single rule, which keeps the label
# conservative — fewer positives, but each positive has more independent
# evidence behind it.
#
# Expected Output:
# A new column weak_label_high_audit_priority and a printed prevalence summary.
# The 95th-percentile thresholds are also printed so they are reproducible.
#
# Why It Matters:
# Notebook 03 needs a target column to train supervised models against. A noisy
# single-signal label would teach the model to mimic Isolation Forest. A
# conservative multi-signal label encodes more domain knowledge — providers
# only get labeled when several independent indicators agree.
#
# This label is NOT a fraud label. It is NOT an overpayment, noncompliance, or
# audit finding. It is a weak-supervision target derived from public CMS data
# patterns, intended for portfolio modeling and follow-up-review prioritization.
# Notebooks downstream should never describe it as anything stronger.
#
# What to Check Before Moving On:
# The label prevalence (positives / total) should be small but non-trivial —
# typically a small single-digit percentage. If it is near 0% or near 100%,
# revisit the quantile thresholds or the underlying feature distributions.

# Provider-level composite risk and peer deviation summaries
composite_col = "feat_composite_risk_max"
peer_dev_col = "feat_max_peer_deviation_max"

composite_threshold = provider_features[composite_col].quantile(0.95)
peer_dev_threshold = provider_features[peer_dev_col].quantile(0.95)

provider_features["extreme_composite_risk"] = (
    provider_features[composite_col] >= composite_threshold
).astype(int)

provider_features["extreme_peer_deviation"] = (
    provider_features[peer_dev_col] >= peer_dev_threshold
).astype(int)

# Conservative weak label:
# High audit priority if the provider is an Isolation Forest anomaly AND also
# has either extreme composite risk or extreme peer deviation.
provider_features["weak_label_high_audit_priority"] = (
    (provider_features["iso_anomaly_flag"] == 1)
    & (
        (provider_features["extreme_composite_risk"] == 1)
        | (provider_features["extreme_peer_deviation"] == 1)
    )
).astype(int)

label_summary = (
    provider_features["weak_label_high_audit_priority"]
    .value_counts()
    .rename_axis("weak_label_high_audit_priority")
    .reset_index(name="count")
)

label_summary["pct"] = (
    label_summary["count"] / label_summary["count"].sum() * 100
).round(2)

print("Composite risk 95th percentile:", round(composite_threshold, 4))
print("Peer deviation 95th percentile:", round(peer_dev_threshold, 4))

label_summary

# %%
# Step 14 — Review Weak Label Distribution by Provider Type
#
# Business Purpose:
# Inspect how the weak label is distributed across provider types so we know
# whether the supervised model in Notebook 03 will be learning from a balanced
# set of specialties or a narrow slice.
#
# Plain-English Logic:
# We group provider_features by provider_type and report:
#   - total_providers
#   - weak_label_providers       (count of weak_label_high_audit_priority == 1)
#   - average iso_anomaly_score
#   - average composite risk
#   - average peer deviation
#   - weak_label_rate            (share of that type labeled)
#
# Expected Output:
# A provider-type summary table sorted by weak-label counts and rates.
#
# Why It Matters:
# This is the same limitation lens we used in Step 11/11B applied to the weak
# label. If the label is concentrated in the same structurally different
# specialties, Notebook 03's supervised model will partially learn "what these
# specialties look like" rather than "what unusual provider behavior looks
# like." That is useful to know before training, and it points toward the
# future improvement of stratified or peer-group-specific modeling.
#
# What to Check Before Moving On:
# Look at both counts and rates. High count = many providers labeled. High
# rate = a disproportionate share of that specialty got labeled. Both matter
# for evaluating the label's quality.

weak_label_type_summary = (
    provider_features
    .groupby("provider_type")
    .agg(
        total_providers=("Rndrng_NPI", "count"),
        weak_label_providers=("weak_label_high_audit_priority", "sum"),
        avg_iso_score=("iso_anomaly_score", "mean"),
        avg_composite_risk=("feat_composite_risk_max", "mean"),
        avg_peer_deviation=("feat_max_peer_deviation_max", "mean"),
    )
    .reset_index()
)

weak_label_type_summary["weak_label_rate"] = (
    weak_label_type_summary["weak_label_providers"] /
    weak_label_type_summary["total_providers"]
)

weak_label_type_summary = weak_label_type_summary.sort_values(
    ["weak_label_providers", "weak_label_rate"],
    ascending=False
)

weak_label_type_summary.head(20)

# %%
# Step 14B — Cross-Reference Anomaly Signals With LEIE Exclusions
#
# Business Purpose:
# Compare the unsupervised anomaly score and the weak label against the OIG
# List of Excluded Individuals/Entities (LEIE) so we have at least one
# external public reference point for the signals produced in this notebook.
#
# Plain-English Logic:
# The LEIE is a public list maintained by the HHS Office of Inspector
# General. It identifies individuals and entities excluded from federal
# healthcare programs for a wide variety of reasons — many of which are
# unrelated to Medicare billing patterns. We:
#   1. Load data/raw/leie_exclusions.csv defensively.
#   2. Look for an NPI column. Many LEIE rows have NPI = "0000000000"
#      (no NPI on file). We treat that placeholder as "no NPI" and drop it.
#   3. Normalize both LEIE NPIs and provider Rndrng_NPI to strings and check
#      membership.
#   4. Add provider_features["in_leie"] (1 if matched, 0 otherwise).
#   5. Compare iso_anomaly_score, iso_anomaly_flag rate, and (if present)
#      weak_label_high_audit_priority rate across in_leie groups.
#
# Expected Output:
# - A printed count of LEIE NPIs and the number that match Part B providers.
# - Mean iso_anomaly_score by in_leie.
# - iso_anomaly_flag rate by in_leie.
# - weak_label_high_audit_priority rate by in_leie (if column exists).
# If the LEIE file lacks a usable NPI column, a clear note is printed and
# this cell does not crash.
#
# Why It Matters:
# LEIE is not a fraud, overpayment, or noncompliance label. It is an
# external public reference list that may give a small amount of validation
# context for our anomaly and weak-label signals. A modest lift in average
# anomaly score among LEIE-listed providers would be supportive but not
# conclusive. No lift would not invalidate the signals — most LEIE
# exclusions involve issues outside Medicare Part B billing patterns.
#
# What to Check Before Moving On:
# - The cell ran without errors even if no NPI column was usable.
# - If any LEIE NPIs matched Part B providers, the printed comparison shows
#   group sizes and averages; treat any difference as suggestive only.

leie_path = PATHS["data_raw"] / "leie_exclusions.csv"

print("LEIE source:", leie_path)
print("LEIE file exists:", leie_path.exists())

leie_validation_ran = False

if leie_path.exists():
    leie_df = pd.read_csv(leie_path, dtype=str, low_memory=False)
    print("LEIE rows loaded:", f"{len(leie_df):,}")
    print("LEIE columns:", list(leie_df.columns))

    leie_npi_col = None
    for candidate in ["NPI", "npi", "Npi"]:
        if candidate in leie_df.columns:
            leie_npi_col = candidate
            break

    if leie_npi_col is None:
        print(
            "Note: LEIE file does not contain a recognizable NPI column. "
            "Skipping NPI-based cross-reference."
        )
    else:
        leie_npis_raw = (
            leie_df[leie_npi_col]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        # The LEIE uses "0000000000" as a placeholder when no NPI is on file.
        leie_npis = set(
            npi for npi in leie_npis_raw
            if npi and npi != "0000000000"
        )

        print("LEIE rows with a usable NPI:", f"{len(leie_npis):,}")

        provider_npis = (
            provider_features["Rndrng_NPI"]
            .astype(str)
            .str.strip()
        )

        provider_features["in_leie"] = provider_npis.isin(leie_npis).astype(int)
        leie_validation_ran = True

        n_match = int(provider_features["in_leie"].sum())
        print("Part B providers matched against LEIE:", f"{n_match:,}")

        if n_match == 0:
            print(
                "Note: no Part B providers in this dataset overlap with the "
                "LEIE list. Cross-reference comparisons are not meaningful."
            )
        else:
            score_by_leie = (
                provider_features
                .groupby("in_leie")["iso_anomaly_score"]
                .agg(["count", "mean", "median"])
                .rename(columns={"count": "providers"})
            )
            print("\niso_anomaly_score by in_leie:")
            print(score_by_leie)

            flag_rate_by_leie = (
                provider_features
                .groupby("in_leie")["iso_anomaly_flag"]
                .mean()
                .rename("iso_anomaly_flag_rate")
            )
            print("\niso_anomaly_flag rate by in_leie:")
            print(flag_rate_by_leie)

            if "weak_label_high_audit_priority" in provider_features.columns:
                weak_rate_by_leie = (
                    provider_features
                    .groupby("in_leie")["weak_label_high_audit_priority"]
                    .mean()
                    .rename("weak_label_rate")
                )
                print("\nweak_label_high_audit_priority rate by in_leie:")
                print(weak_rate_by_leie)
else:
    print(
        "Note: LEIE file not found at the expected path. Skipping LEIE "
        "cross-reference validation."
    )

print("\nLEIE validation ran end-to-end:", leie_validation_ran)

# %%
# Step 15 — Save Provider-Level Labeled Feature Table
#
# Business Purpose:
# Persist the final provider-level table — features + anomaly scores + the weak
# label — to disk so Notebook 03 can pick up exactly where this notebook ends.
#
# Plain-English Logic:
# Notebook 02 started with provider-service level data, aggregated to provider
# level, ran Isolation Forest, computed extreme-risk indicators, and built a
# conservative weak-supervision label. We now save the result as a parquet file
# under data/processed/. Parquet is preferred over CSV here because it
# preserves dtypes, is columnar (faster reads in Notebook 03), and compresses
# better for a multi-million-row dataset.
#
# Expected Output:
# A parquet file at:
#   data/processed/provider_features_labeled_<YEAR>.parquet
# plus printed confirmation with row count, column count, and number of
# weak-label positives.
#
# Why It Matters:
# This is the handoff artifact between unsupervised anomaly work (Notebook 02)
# and supervised model training (Notebook 03). Saving it cleanly means
# Notebook 03 starts from a stable, reproducible target — no need to rerun
# Isolation Forest or recompute the weak label.
#
# What to Check Before Moving On:
# - File exists at the expected path
# - Saved row count == len(provider_features)
# - weak_label_high_audit_priority column is present and has both 0s and 1s
# - Weak-label positive count matches what Step 13 reported

output_path = PATHS["data_processed"] / f"provider_features_labeled_{YEAR}_full.parquet"

provider_features.to_parquet(output_path, engine="pyarrow", index=False)

print("Saved:", output_path)
print("Rows saved:", f"{len(provider_features):,}")
print("Columns saved:", f"{provider_features.shape[1]:,}")
print(
    "Weak-label positives:",
    f"{provider_features['weak_label_high_audit_priority'].sum():,}"
)
print("File exists:", output_path.exists())
