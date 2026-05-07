# %%
"""
Notebook 05 — Behavioral Health Audit-Priority Module

Business Purpose:
This notebook is a focused extension of the Medicare Claims Audit
Intelligence Platform that narrows attention to behavioral-health-
related provider-service records. The goal is to surface
audit-priority review signals for the behavioral-health slice of the
2022 Medicare Part B population using the same public CMS data and
the same model outputs already produced by Notebooks 01–04.

Plain-English Logic:
Behavioral health (psychiatry, psychology, clinical social work,
counseling, addiction medicine, and similar specialties) has its own
billing patterns, peer norms, and HCPCS code families. Population-
wide anomaly detection treats those rows alongside cardiology,
oncology, lab, DME, and every other specialty, so behavioral-health
review signals can get washed out by the larger population.

This module:
  1. Identifies behavioral-health-related provider-service rows from
     public CMS fields only (provider specialty text, HCPCS codes,
     HCPCS descriptions).
  2. Aggregates those rows up to a per-provider behavioral-health
     summary and merges it onto the existing provider-level labeled
     table from Notebook 02 and the full-population model scores
     from Notebook 03.
  3. Produces a behavioral-health top-ranked review-priority worklist
     and a capacity-tradeoff view comparable to Notebook 04, scoped
     to behavioral-health providers.

Important Limitation:
This notebook is an audit-priority module, not a clinical or
compliance determination. It does NOT determine fraud, overpayments,
noncompliance, medical necessity, parity compliance, documentation
sufficiency, or audit findings. The behavioral-health classification
is a public-data approximation built from provider specialty text,
HCPCS codes, and HCPCS descriptions — it is not a clinical
classification, not a credentialing check, and not a service-line
billing audit. Any provider surfaced here would still require
independent review under the appropriate audit process before any
real-world conclusion.
"""

# %%
# Step 1 — Environment Setup and Imports
#
# Business Purpose:
# Load the libraries and project paths needed to read the full-run
# provider-service, provider-level, and model-prediction tables
# produced by Notebooks 01–03.
#
# Plain-English Logic:
# We resolve PROJECT_ROOT defensively so the notebook works whether it
# is executed as a script (__file__ defined) or interactively in
# Jupyter (__file__ undefined). Then we add src/ to sys.path and
# import shared project paths from config.
#
# Expected Output:
# Printed PROJECT_ROOT and the year being analyzed.
#
# Why It Matters:
# Reproducibility starts with a clean import path. If PROJECT_ROOT is
# wrong, every downstream cell will silently load the wrong file.
#
# What to Check Before Moving On:
# Confirm PROJECT_ROOT points at medicare-claims-audit-intelligence
# and YEAR matches the parquet files produced by earlier notebooks.

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

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
# Step 2 — Load Required Full-Run Datasets
#
# Business Purpose:
# Load the three full-run inputs this module operates on:
#   - the provider-service feature table (9.76M rows)
#   - the provider-level labeled table (1.15M rows)
#   - the full-population model prediction file (1.15M rows)
#
# Plain-English Logic:
# We read each parquet, print its shape and the columns we will use
# downstream, and confirm the model probability columns that are
# actually present in the prediction file. The provider-service file
# uses CMS-original column names (Rndrng_Prvdr_Type,
# Rndrng_Prvdr_State_Abrvtn) while the provider-level files use the
# project's renamed columns (provider_type, provider_state). We
# preserve both so step-by-step joins are explicit.
#
# Expected Output:
# Three loaded dataframes plus printed shape, key columns, provider
# count, and available model score columns.
#
# Why It Matters:
# Every downstream step in this notebook reads from one of these three
# tables. Verifying their shapes here catches any upstream rebuild
# mismatch before behavioral-health filtering hides it.
#
# What to Check Before Moving On:
# - features_df has 9,755,427 rows and contains HCPCS_Cd, HCPCS_Desc,
#   Rndrng_Prvdr_Type, Rndrng_Prvdr_State_Abrvtn
# - provider_df has 1,148,873 rows and contains weak_label_high_audit_priority
# - predictions_df has 1,148,873 rows and at least one *_full_probability column

features_path = (
    PATHS["data_processed"]
    / f"features_provider_service_{YEAR}_full.parquet"
)
provider_path = (
    PATHS["data_processed"]
    / f"provider_features_labeled_{YEAR}_full.parquet"
)
predictions_path = (
    PATHS["data_processed"]
    / f"model_predictions_full_population_{YEAR}_full.parquet"
)

print("Loading:", features_path)
features_df = pd.read_parquet(features_path)
print("  shape:", features_df.shape)

print("Loading:", provider_path)
provider_df = pd.read_parquet(provider_path)
print("  shape:", provider_df.shape)

print("Loading:", predictions_path)
predictions_df = pd.read_parquet(predictions_path)
print("  shape:", predictions_df.shape)

target_col = "weak_label_high_audit_priority"

probability_cols = [
    col for col in predictions_df.columns
    if col.endswith("_full_probability")
]

print("\nKey provider-service columns present:", [
    c for c in [
        "Rndrng_NPI",
        "Rndrng_Prvdr_Type",
        "Rndrng_Prvdr_State_Abrvtn",
        "HCPCS_Cd",
        "HCPCS_Desc",
        "Tot_Benes",
        "Tot_Srvcs",
        "Tot_Mdcr_Pymt_Amt",
        "feat_charge_to_allowed",
        "feat_max_peer_deviation",
        "feat_composite_risk",
    ] if c in features_df.columns
])
print("Key provider-level columns present:", [
    c for c in [
        "Rndrng_NPI",
        "provider_type",
        "provider_state",
        target_col,
        "feat_composite_risk_max",
    ] if c in provider_df.columns
])
print("Available model probability columns:", probability_cols)
print(
    "Unique providers in provider_df:",
    f"{provider_df['Rndrng_NPI'].nunique():,}"
)

# %%
# Step 3 — Define Behavioral Health Identification Rules
#
# Business Purpose:
# Tag each provider-service row in features_df with a behavioral-
# health flag built from public CMS fields only (provider specialty
# text, HCPCS code, HCPCS description).
#
# Plain-English Logic:
# We build three independent flags and OR them together:
#   - is_bh_provider_type : Rndrng_Prvdr_Type contains a behavioral-
#                           health specialty keyword (tightened — see
#                           below)
#   - is_bh_service_desc  : HCPCS_Desc contains a behavioral-health
#                           service keyword
#   - is_bh_hcpcs_code    : HCPCS_Cd is in a curated list of common
#                           behavioral-health CPT/HCPCS codes, or it
#                           matches the H-code range (H0001–H2037)
# A row qualifies as behavioral-health-related if ANY of the three
# fires. Keyword matching is case-insensitive substring contains.
#
# Tightening Note (provider-type keywords):
# An earlier version of this notebook used standalone tokens like
# "family" and "substance" and "behavioral", which inadvertently
# matched non-behavioral-health specialties (most notably "Family
# Practice"). We have tightened the provider-type keyword list to
# require more specific phrases:
#   - "marriage and family", "marriage & family", "family therapist",
#     "marriage family therapist" replace bare "family" / "marriage"
#   - "substance abuse" replaces bare "substance"
#   - "behavioral health" replaces bare "behavioral"
# We keep "clinical social worker" but drop bare "social worker" so
# the provider-type rule does not pull in non-clinical social workers.
# The HCPCS-code and HCPCS-description lists stay as-is — those are
# genuinely service-level signals where a wider net is appropriate.
#
# Important Framing:
# `is_behavioral_health_row` here is row-level only. A single isolated
# behavioral-health-tagged row does NOT make a provider a behavioral-
# health provider — that is determined later in Step 5 with a stricter
# provider-level rule (core specialty OR meaningful BH service
# concentration). This is a public-data approximation, not a clinical
# classification, and CMS specialty text plus HCPCS descriptions
# describe billing, not the clinical context.
#
# Expected Output:
# - features_df gains four columns:
#     is_bh_provider_type, is_bh_service_desc,
#     is_bh_hcpcs_code, is_behavioral_health_row
# - Printed counts for each individual flag and for the combined flag
#
# Why It Matters:
# Behavioral-health rows differ enough from the rest of Medicare Part B
# that scoping the worklist matters more than getting the boundary
# perfect. We err on the side of inclusion, then disclose how many
# rows each rule contributed.
#
# What to Check Before Moving On:
# - is_behavioral_health_row count is non-zero
# - is_behavioral_health_row >= max(individual flag counts)
# - The top behavioral-health provider types include Psychiatry,
#   Clinical Psychologist, Clinical Social Worker

bh_provider_type_keywords = [
    "psychiatry",
    "psychiatrist",
    "clinical psychologist",
    "psychologist",
    "clinical social worker",
    "mental health",
    "addiction",
    "substance abuse",
    "counselor",
    "behavioral health",
    "neuropsychiatry",
    "marriage and family",
    "marriage & family",
    "family therapist",
    "marriage family therapist",
]

bh_service_desc_keywords = [
    "psychotherapy",
    "psychiatric",
    "behavioral",
    "mental health",
    "counseling",
    "substance",
    "addiction",
    "alcohol",
    "opioid",
    "depression",
    "anxiety",
    "crisis",
    "group therapy",
    "family therapy",
    "psychological",
    "psychoanalysis",
]

bh_hcpcs_code_set = {
    # Diagnostic interview / evaluation
    "90791", "90792",
    # Individual psychotherapy
    "90832", "90834", "90837",
    # Family / group / interactive
    "90846", "90847", "90849", "90853",
    # Psychological / neuropsychological testing
    "96130", "96131", "96132", "96133",
    "96136", "96137", "96138", "96139",
    # Behavioral health integration / collaborative care
    "99484", "99492", "99493", "99494",
    # SBIRT (G-codes)
    "G0396", "G0397",
    # Group / behavioral counseling (G-codes)
    "G0410", "G0411",
    # Behavioral health intervention (G-codes)
    "G2086", "G2087", "G2088",
    # FQHC / RHC behavioral health visit (G-codes)
    "G0469", "G0470",
}

# H0001–H2037 H-code range covers a wide swath of behavioral-health
# and substance-use treatment HCPCS Level II codes. We match by
# pattern rather than enumerating every value: an "H" followed by
# digits where the integer body is in [1, 2037].
_h_code_pattern = re.compile(r"^H(\d{4})$", flags=re.IGNORECASE)


def _is_bh_h_code(code: str) -> bool:
    if not isinstance(code, str):
        return False
    m = _h_code_pattern.match(code)
    if not m:
        return False
    n = int(m.group(1))
    return 1 <= n <= 2037


# Build a single regex per text field for efficiency on 9.76M rows.
bh_provider_type_regex = "|".join(
    re.escape(kw) for kw in bh_provider_type_keywords
)
bh_service_desc_regex = "|".join(
    re.escape(kw) for kw in bh_service_desc_keywords
)

prov_type_str = features_df["Rndrng_Prvdr_Type"].fillna("").astype(str)
hcpcs_desc_str = features_df["HCPCS_Desc"].fillna("").astype(str)
hcpcs_code_str = features_df["HCPCS_Cd"].fillna("").astype(str)

features_df["is_bh_provider_type"] = (
    prov_type_str.str.contains(
        bh_provider_type_regex, case=False, regex=True, na=False
    )
).astype(np.int8)

features_df["is_bh_service_desc"] = (
    hcpcs_desc_str.str.contains(
        bh_service_desc_regex, case=False, regex=True, na=False
    )
).astype(np.int8)

is_explicit_bh_code = hcpcs_code_str.isin(bh_hcpcs_code_set)
is_h_range = hcpcs_code_str.map(_is_bh_h_code)

features_df["is_bh_hcpcs_code"] = (
    is_explicit_bh_code | is_h_range
).astype(np.int8)

features_df["is_behavioral_health_row"] = (
    (features_df["is_bh_provider_type"] == 1)
    | (features_df["is_bh_service_desc"] == 1)
    | (features_df["is_bh_hcpcs_code"] == 1)
).astype(np.int8)

print(
    "Provider-service rows tagged behavioral-health by each rule:"
)
print(
    "  is_bh_provider_type      :",
    f"{int(features_df['is_bh_provider_type'].sum()):,}"
)
print(
    "  is_bh_service_desc       :",
    f"{int(features_df['is_bh_service_desc'].sum()):,}"
)
print(
    "  is_bh_hcpcs_code         :",
    f"{int(features_df['is_bh_hcpcs_code'].sum()):,}"
)
print(
    "  is_behavioral_health_row :",
    f"{int(features_df['is_behavioral_health_row'].sum()):,}"
)
print(
    "\nReminder: this is a public-data approximation, not a clinical "
    "classification."
)

# %%
# Step 4 — Behavioral Health Subset Summary
#
# Business Purpose:
# Describe the behavioral-health slice quantitatively so a reader can
# see what the rules actually selected before any modeling work
# happens on top of it.
#
# Plain-English Logic:
# We filter features_df to is_behavioral_health_row == 1 and report:
#   - row count
#   - unique NPIs
#   - unique HCPCS codes
#   - share of the full provider-service population
#   - top 10 provider types by row count
#   - top 10 HCPCS codes (with their description) by row count
#
# Expected Output:
# Printed counts and two short top-10 tables.
#
# Why It Matters:
# These prints are the sanity check on Step 3. If the top provider
# types are obviously not behavioral-health, or the top HCPCS codes
# are E&M visits, the rules need adjustment before anything
# downstream is trustworthy.
#
# What to Check Before Moving On:
# Because the row-level rule still includes is_bh_service_desc and
# is_bh_hcpcs_code, the top provider types here will include
# generalist specialties (Family Practice, Internal Medicine, NP, PA)
# whose providers occasionally bill a behavioral-health code or
# screening — that is fine at the row level. The stricter
# provider-level cohort in Step 5 narrows this down. Top HCPCS codes
# should include therapy codes (90834 / 90837 / 90791) and common
# behavioral-health screenings (e.g., G0444 depression screening).

bh_service_df = features_df[
    features_df["is_behavioral_health_row"] == 1
].copy()

total_rows = int(len(features_df))
bh_rows = int(len(bh_service_df))
bh_share = (bh_rows / total_rows) if total_rows else 0.0

print("Behavioral-health provider-service rows :", f"{bh_rows:,}")
print(
    "Share of total provider-service rows     :",
    f"{bh_share:.4%}"
)
print(
    "Unique behavioral-health NPIs            :",
    f"{bh_service_df['Rndrng_NPI'].nunique():,}"
)
print(
    "Unique behavioral-health HCPCS codes     :",
    f"{bh_service_df['HCPCS_Cd'].nunique():,}"
)

print("\nTop 10 behavioral-health provider types by row count:")
print(
    bh_service_df["Rndrng_Prvdr_Type"]
    .value_counts()
    .head(10)
    .to_string()
)

print("\nTop 10 behavioral-health HCPCS codes by row count:")
top_hcpcs = (
    bh_service_df.groupby("HCPCS_Cd")
    .agg(rows=("HCPCS_Cd", "size"),
         description=("HCPCS_Desc", "first"))
    .sort_values("rows", ascending=False)
    .head(10)
)
print(top_hcpcs.to_string())

# %%
# Step 5 — Provider-Level Behavioral Health Qualification
#
# Business Purpose:
# Decide which providers actually count as behavioral-health providers
# for the rest of this notebook. The row-level rule from Step 3 is
# inclusive (good for surfacing behavioral-health-related services);
# the provider-level rule here is stricter (good for building a
# focused review-priority cohort).
#
# Plain-English Logic:
# Public CMS data does not give us a perfect behavioral-health
# provider label — there is no "this provider IS a behavioral-health
# clinician" field — so we approximate it from two complementary
# signals:
#
#   (A) Aggregate behavioral-health activity per provider. Within
#       bh_service_df (rows where is_behavioral_health_row == 1) we
#       group by Rndrng_NPI and compute:
#         - bh_service_rows           : count of BH-tagged rows
#         - bh_unique_hcpcs           : distinct BH-tagged HCPCS codes
#         - bh_total_services         : sum of Tot_Srvcs on those rows
#         - bh_total_benes            : sum of Tot_Benes on those rows
#         - bh_total_medicare_payment : sum of Tot_Mdcr_Pymt_Amt
#         - bh_avg_charge_to_allowed  : mean of feat_charge_to_allowed
#         - bh_max_peer_deviation     : max of feat_max_peer_deviation
#         - bh_max_composite_risk     : max of feat_composite_risk
#       We left-merge those onto provider_df.
#
#   (B) Compute concentration ratios using the provider's *total*
#       activity from provider_df:
#         - bh_service_share     = bh_total_services / core_Tot_Srvcs_sum
#         - bh_service_row_share = bh_service_rows / provider_service_records
#       These tell us how much of the provider's billing is BH-tagged.
#
#   (C) Two-layer qualification:
#         - bh_provider_core_flag   = 1 if provider_type matches the
#                                     tightened BH specialty regex
#                                     (Psychiatry, Clinical Psychologist,
#                                     LCSW, counselor, addiction
#                                     medicine, marriage & family
#                                     therapist, etc.)
#         - bh_provider_volume_flag = 1 if the provider has meaningful
#                                     BH concentration:
#                                       bh_service_rows >= 5
#                                       AND bh_service_share >= 0.20
#         - bh_any_behavioral_health = 1 if EITHER flag is 1
#       This excludes providers who only billed one or two incidental
#       behavioral-health-tagged services from the cohort.
#
# Important Framing:
# The row-level subset in Step 4 (bh_service_df) remains useful for
# describing behavioral-health-related billing activity in the
# population. The cohort that the rest of this notebook builds on
# (worklist, capacity scenarios, charts) uses the stricter
# provider-level rule defined here. This is still a public-data
# approximation, not a clinical determination.
#
# Expected Output:
# - provider_df_bh dataframe (same row count as provider_df) with
#   bh_* aggregate columns, bh_service_share, bh_service_row_share,
#   bh_provider_core_flag, bh_provider_volume_flag, and the final
#   bh_any_behavioral_health flag.
# - Printed "before vs after" cohort sizes so the tightening effect
#   is visible.
#
# Why It Matters:
# Without the stricter rule, generalist specialties that occasionally
# bill a behavioral-health code (Family Practice depression
# screenings, cardiology incidental codes, etc.) would dominate the
# behavioral-health worklist. The two-layer rule keeps the cohort to
# providers whose specialty *is* behavioral health, plus mid-level
# clinicians (NP/PA) whose own billing is meaningfully concentrated
# in behavioral-health services.
#
# What to Check Before Moving On:
# - len(provider_df_bh) == len(provider_df)
# - bh_any_behavioral_health.sum() is materially smaller than the
#   "any BH-tagged row" count printed in Step 4
# - The top behavioral-health provider types after qualification are
#   dominated by Psychiatry, Clinical Psychologist, LCSW, counseling
#   specialties (not Family Practice / Internal Medicine).

bh_agg = (
    bh_service_df.groupby("Rndrng_NPI")
    .agg(
        bh_service_rows=("Rndrng_NPI", "size"),
        bh_unique_hcpcs=("HCPCS_Cd", "nunique"),
        bh_total_services=("Tot_Srvcs", "sum"),
        bh_total_benes=("Tot_Benes", "sum"),
        bh_total_medicare_payment=("Tot_Mdcr_Pymt_Amt", "sum"),
        bh_avg_charge_to_allowed=("feat_charge_to_allowed", "mean"),
        bh_max_peer_deviation=("feat_max_peer_deviation", "max"),
        bh_max_composite_risk=("feat_composite_risk", "max"),
    )
    .reset_index()
)

print("Behavioral-health provider activity summary shape:", bh_agg.shape)
print(
    "Providers with at least one BH-tagged row        :",
    f"{len(bh_agg):,}"
)

provider_df_bh = provider_df.merge(bh_agg, on="Rndrng_NPI", how="left")

# Fill count-style columns with 0 for providers with no BH-tagged
# rows; leave the mean/max columns as NaN (they have no signal there).
count_cols = [
    "bh_service_rows",
    "bh_unique_hcpcs",
    "bh_total_services",
    "bh_total_benes",
    "bh_total_medicare_payment",
]
for col in count_cols:
    provider_df_bh[col] = provider_df_bh[col].fillna(0)

# Concentration ratios — use provider-level totals from provider_df
# so the denominator is the provider's full billing footprint, not
# just BH-tagged activity.
provider_total_services = provider_df_bh["core_Tot_Srvcs_sum"].astype(
    float
)
provider_total_rows = provider_df_bh["provider_service_records"].astype(
    float
)

provider_df_bh["bh_service_share"] = np.where(
    provider_total_services > 0,
    provider_df_bh["bh_total_services"].astype(float)
    / provider_total_services.replace(0, np.nan),
    0.0,
)
provider_df_bh["bh_service_share"] = (
    provider_df_bh["bh_service_share"].fillna(0.0)
)

provider_df_bh["bh_service_row_share"] = np.where(
    provider_total_rows > 0,
    provider_df_bh["bh_service_rows"].astype(float)
    / provider_total_rows.replace(0, np.nan),
    0.0,
)
provider_df_bh["bh_service_row_share"] = (
    provider_df_bh["bh_service_row_share"].fillna(0.0)
)

# Provider-type "core" flag: provider_type matches the same tightened
# behavioral-health specialty regex used at the row level in Step 3.
provider_type_str = (
    provider_df_bh["provider_type"].fillna("").astype(str)
)
provider_df_bh["bh_provider_core_flag"] = (
    provider_type_str.str.contains(
        bh_provider_type_regex, case=False, regex=True, na=False
    )
).astype(np.int8)

# Volume flag: the provider has a meaningful BH service concentration.
provider_df_bh["bh_provider_volume_flag"] = (
    (provider_df_bh["bh_service_rows"] >= 5)
    & (provider_df_bh["bh_service_share"] >= 0.20)
).astype(np.int8)

# Final qualification: core specialty OR meaningful volume.
provider_df_bh["bh_any_behavioral_health"] = (
    (provider_df_bh["bh_provider_core_flag"] == 1)
    | (provider_df_bh["bh_provider_volume_flag"] == 1)
).astype(np.int8)

n_before = int(len(bh_agg))
n_core = int(provider_df_bh["bh_provider_core_flag"].sum())
n_volume = int(provider_df_bh["bh_provider_volume_flag"].sum())
n_after = int(provider_df_bh["bh_any_behavioral_health"].sum())

print()
print(
    "Behavioral-health provider qualification (before vs after "
    "tightening):"
)
print(
    f"  Providers with ANY BH-tagged row (loose rule): {n_before:,}"
)
print(
    f"  bh_provider_core_flag (specialty match)      : {n_core:,}"
)
print(
    f"  bh_provider_volume_flag (>=5 rows & >=20%)   : {n_volume:,}"
)
print(
    f"  bh_any_behavioral_health (core OR volume)    : {n_after:,}"
)
print("provider_df_bh shape                            :",
      provider_df_bh.shape)

print("provider_df_bh shape                     :", provider_df_bh.shape)
print(
    "Providers with bh_any_behavioral_health=1 :",
    f"{int(provider_df_bh['bh_any_behavioral_health'].sum()):,}"
)

# %%
# Step 6 — Merge Full-Population Model Scores
#
# Business Purpose:
# Attach the model-predicted audit-priority probabilities from
# Notebook 03 to every provider in provider_df_bh and choose a single
# risk_score column the rest of the notebook will use.
#
# Plain-English Logic:
# predictions_df already shares Rndrng_NPI, provider_type,
# provider_state, and the weak label with provider_df_bh. We merge in
# only the *_full_probability columns (those are the new information
# this file adds) and then build risk_score with the same fallback
# order Notebook 04 used:
#   lgb_full_probability  ->  xgb_full_probability  ->  lr_full_probability
# so behavioral-health rankings come from the same model that drove
# the full-population analysis.
#
# Expected Output:
# - provider_df_bh gains the available *_full_probability columns and
#   a unified risk_score column.
# - Printed name of the score column used and its min/mean/max.
#
# Why It Matters:
# Downstream behavioral-health rankings should match the model
# selected by the rest of the platform. Picking the score column once
# here keeps the worklist and capacity scenarios from accidentally
# using two different models.
#
# What to Check Before Moving On:
# - score_col is one of the *_full_probability columns
# - risk_score is in [0, 1] and has no NaN
# - len(provider_df_bh) matches the original provider_df length

predictions_subset = predictions_df[
    ["Rndrng_NPI", *probability_cols]
].copy()

provider_df_bh = provider_df_bh.merge(
    predictions_subset, on="Rndrng_NPI", how="left"
)

preferred_score_columns = [
    "lgb_full_probability",
    "xgb_full_probability",
    "lr_full_probability",
]
score_col = next(
    (c for c in preferred_score_columns if c in provider_df_bh.columns),
    None,
)
if score_col is None:
    raise ValueError(
        "No model probability column found in the prediction file. "
        "Expected one of: " + ", ".join(preferred_score_columns)
    )

provider_df_bh["risk_score"] = provider_df_bh[score_col].astype(float)

print("Score column used :", score_col)
print(
    "risk_score min / mean / max:",
    round(float(provider_df_bh["risk_score"].min()), 6),
    "/",
    round(float(provider_df_bh["risk_score"].mean()), 6),
    "/",
    round(float(provider_df_bh["risk_score"].max()), 6),
)
print(
    "Reminder: higher risk_score = higher model-predicted audit "
    "priority. Not fraud, not overpayments, not findings."
)

# %%
# Step 7 — Behavioral Health Provider Risk Summary
#
# Business Purpose:
# Quantify the behavioral-health slice on the metrics that drive the
# rest of the platform: weak-label prevalence, average risk_score,
# and how the slice compares to non-behavioral-health providers.
#
# Plain-English Logic:
# We split provider_df_bh into two cohorts on bh_any_behavioral_health,
# then compute the same headline numbers for each:
#   - provider count
#   - weak-label positives
#   - weak-label rate
#   - mean / median risk_score
# We also surface the top behavioral-health provider types and states
# by raw count, plus the weak-label rate by top behavioral-health
# provider type so a reader can see which specialties the platform
# is signaling on most.
#
# Expected Output:
# - Printed cohort comparison
# - Printed top provider types and states for behavioral health
# - Printed weak-label rate by top behavioral-health provider type
#
# Why It Matters:
# These are the prints a reviewer reads first. They show whether the
# behavioral-health slice differs from the rest of the population in
# a way that justifies a dedicated module, and which specialties
# inside the slice are doing most of the work.
#
# What to Check Before Moving On:
# Behavioral-health cohort size > 0, weak-label rate is sensible,
# and the top-provider-type table has familiar specialty names.

bh_mask = provider_df_bh["bh_any_behavioral_health"] == 1
bh_providers = provider_df_bh[bh_mask].copy()
nonbh_providers = provider_df_bh[~bh_mask].copy()


def cohort_summary(name, frame):
    n = int(len(frame))
    pos = int(frame[target_col].sum()) if n else 0
    rate = (pos / n) if n else float("nan")
    mean_rs = float(frame["risk_score"].mean()) if n else float("nan")
    median_rs = (
        float(frame["risk_score"].median()) if n else float("nan")
    )
    return {
        "cohort": name,
        "providers": n,
        "weak_label_positives": pos,
        "weak_label_rate": rate,
        "risk_score_mean": mean_rs,
        "risk_score_median": median_rs,
    }


cohort_rows = [
    cohort_summary("Behavioral health", bh_providers),
    cohort_summary("Non-behavioral health", nonbh_providers),
]
cohort_df = pd.DataFrame(cohort_rows)

print("Cohort comparison (behavioral-health vs non-behavioral-health):")
print(cohort_df.to_string(index=False))

print("\nTop 10 behavioral-health provider types by count:")
print(
    bh_providers["provider_type"].value_counts().head(10).to_string()
)

print("\nTop 10 behavioral-health provider states by count:")
print(
    bh_providers["provider_state"].value_counts().head(10).to_string()
)

print(
    "\nWeak-label rate by top behavioral-health provider type "
    "(types with >=100 providers):"
)
type_counts = bh_providers["provider_type"].value_counts()
eligible_types = type_counts[type_counts >= 100].index.tolist()
type_rate = (
    bh_providers[bh_providers["provider_type"].isin(eligible_types)]
    .groupby("provider_type")
    .agg(
        providers=("Rndrng_NPI", "size"),
        weak_label_positives=(target_col, "sum"),
        weak_label_rate=(target_col, "mean"),
        risk_score_mean=("risk_score", "mean"),
    )
    .sort_values("providers", ascending=False)
    .head(15)
)
print(type_rate.to_string())

# %%
# Step 8 — Behavioral Health Top-Ranked Review-Priority Worklist
#
# Business Purpose:
# Produce a per-provider behavioral-health review-priority worklist
# ordered by model-predicted audit priority.
#
# Plain-English Logic:
# We restrict to behavioral-health providers, sort descending by
# risk_score, and keep the columns a reviewer would want when reading
# the worklist:
#   Rndrng_NPI, provider_type, provider_state,
#   risk_score, weak_label_high_audit_priority,
#   bh_service_rows, bh_unique_hcpcs,
#   bh_total_services, bh_total_benes, bh_total_medicare_payment,
#   bh_avg_charge_to_allowed, bh_max_peer_deviation
# We save the full ranked table as parquet (not just the top N) so
# downstream tools can apply their own capacity cutoff.
#
# Important Framing:
# This is a review-priority worklist, NOT a suspect list. Inclusion
# here means a public-data audit-priority signal fired against this
# provider's behavioral-health activity, not that the provider has
# committed any wrongdoing.
#
# Expected Output:
# A parquet file at:
#   data/processed/behavioral_health_top_providers_<YEAR>_full.parquet
#
# Why It Matters:
# A persisted, ranked behavioral-health worklist is the artifact
# downstream notebooks, dashboards, or analyst follow-up would
# actually consume. Saving it once here keeps the rest of the
# notebook fast and gives every later view a single source of truth.

worklist_columns = [
    "Rndrng_NPI",
    "provider_type",
    "provider_state",
    "risk_score",
    target_col,
    "bh_service_rows",
    "bh_unique_hcpcs",
    "bh_total_services",
    "bh_total_benes",
    "bh_total_medicare_payment",
    "bh_avg_charge_to_allowed",
    "bh_max_peer_deviation",
]

bh_worklist = (
    bh_providers[worklist_columns]
    .sort_values("risk_score", ascending=False)
    .reset_index(drop=True)
    .copy()
)

bh_worklist_path = (
    PATHS["data_processed"]
    / f"behavioral_health_top_providers_{YEAR}_full.parquet"
)
bh_worklist.to_parquet(
    bh_worklist_path, engine="pyarrow", index=False
)

print("Saved review-priority worklist:", bh_worklist_path)
print("  rows:", f"{len(bh_worklist):,}")
print("  columns:", list(bh_worklist.columns))
print("\nTop 10 behavioral-health review-priority providers:")
print(bh_worklist.head(10).to_string(index=False))
print(
    "\nReminder: this is a review-priority worklist, not a suspect "
    "list. Any provider listed here would still require independent "
    "follow-up review before any conclusion."
)

# %%
# Step 9 — Behavioral Health Capacity Scenarios
#
# Business Purpose:
# Quantify how a finite review budget translates into weak-label
# precision, recall, and lift inside the behavioral-health cohort.
#
# Plain-English Logic:
# For each capacity K in {25, 50, 100, 250, 500, 1,000} we take the
# top-K behavioral-health providers from bh_worklist and compute:
#   - providers_reviewed     = K (capped at the cohort size)
#   - weak_label_hits        = positives caught in the top K
#   - weak_label_precision   = hits / K
#   - weak_label_recall      = hits / total positives in the cohort
#   - lift_vs_random         = precision_at_K / cohort positive rate
# Lift is computed against the behavioral-health cohort baseline,
# not the full population, so the number reads as "how many times
# better than random within the behavioral-health slice."
#
# Expected Output:
# A capacity_scenarios_df dataframe with one row per capacity scenario,
# saved to:
#   data/processed/behavioral_health_capacity_scenarios_<YEAR>_full.parquet
#
# Why It Matters:
# Behavioral-health audit programs run at a different capacity scale
# than the full-population view in Notebook 04. The K values here
# (25–1,000) are sized for a focused module, not a national worklist.
#
# What to Check Before Moving On:
# - weak_label_precision is monotonically non-increasing as K grows
# - weak_label_recall is monotonically non-decreasing as K grows
# - lift_vs_random >= 1.0 at small K

bh_capacities = [25, 50, 100, 250, 500, 1_000]

cohort_size = int(len(bh_worklist))
cohort_positives = int(bh_worklist[target_col].sum())
cohort_positive_rate = (
    (cohort_positives / cohort_size) if cohort_size else 0.0
)

scenario_rows = []
for k in bh_capacities:
    k_eff = min(k, cohort_size)
    top_k = bh_worklist.head(k_eff)
    hits = int(top_k[target_col].sum())
    precision_k = (hits / k_eff) if k_eff else float("nan")
    recall_k = (
        (hits / cohort_positives) if cohort_positives else float("nan")
    )
    lift_k = (
        (precision_k / cohort_positive_rate)
        if cohort_positive_rate else float("nan")
    )
    scenario_rows.append({
        "capacity_label": f"Top {k:,}",
        "providers_reviewed": int(k_eff),
        "weak_label_hits": hits,
        "weak_label_precision": float(precision_k),
        "weak_label_recall": float(recall_k),
        "lift_vs_random": float(lift_k),
    })

bh_capacity_df = pd.DataFrame(scenario_rows)

bh_capacity_path = (
    PATHS["data_processed"]
    / f"behavioral_health_capacity_scenarios_{YEAR}_full.parquet"
)
bh_capacity_df.to_parquet(
    bh_capacity_path, engine="pyarrow", index=False
)

print("Behavioral-health capacity scenarios:")
print(bh_capacity_df.to_string(index=False))
print("\nSaved:", bh_capacity_path)
print(
    "Reminder: weak_label_hits counts providers flagged by the "
    "weak-supervision label, not confirmed audit findings."
)

# %%
# Step 10 — Visualizations
#
# Business Purpose:
# Save three portfolio-safe charts that summarize the behavioral-
# health module:
#   A. Behavioral Health Audit-Priority Score Distribution
#   B. Behavioral Health Weak-Label Rate by Provider Type
#   C. Behavioral Health Review Capacity Tradeoff
#
# Plain-English Logic:
# (A) Histogram of risk_score for behavioral-health providers, with a
#     reference line at the cohort mean. The y-axis is log-scaled so
#     the long tail at the high-priority end is readable.
# (B) Bar chart of weak-label rate by behavioral-health provider type
#     for types with >= 100 providers (same eligibility as Step 7).
# (C) Capacity tradeoff plot: providers_reviewed on a log x-axis,
#     precision and recall on the y-axis, mirroring Notebook 04 but
#     scoped to the behavioral-health cohort.
#
# Expected Output:
# Three PNGs saved to reports/.
#
# Why It Matters:
# Charts convert the tables in Steps 7–9 into something a reviewer
# can scan in seconds. Keeping them muted (single-color bars, dashed
# reference line) keeps the visual emphasis on the data, not styling.

bh_score_hist_path = (
    PATHS["reports"]
    / "notebook05_behavioral_health_risk_score_distribution.png"
)
bh_type_rate_path = (
    PATHS["reports"]
    / "notebook05_behavioral_health_provider_type_rates.png"
)
bh_capacity_plot_path = (
    PATHS["reports"]
    / "notebook05_behavioral_health_capacity_tradeoff.png"
)

# A. Risk-score distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(
    bh_providers["risk_score"].values,
    bins=60,
    color="#1a5276",
    alpha=0.85,
)
mean_score = float(bh_providers["risk_score"].mean())
ax.axvline(
    mean_score,
    color="#b9770e",
    linestyle="--",
    linewidth=1.5,
    label=f"Cohort mean = {mean_score:.4f}",
)
ax.set_yscale("log")
ax.set_xlabel("Model-predicted audit-priority score (risk_score)")
ax.set_ylabel("Provider count (log scale)")
ax.set_title("Behavioral Health Audit-Priority Score Distribution")
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(bh_score_hist_path, dpi=150, bbox_inches="tight")
plt.show()
print("Saved:", bh_score_hist_path)

# B. Weak-label rate by behavioral-health provider type
type_plot_df = (
    type_rate.sort_values("weak_label_rate", ascending=True)
    .copy()
)
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(
    type_plot_df.index,
    type_plot_df["weak_label_rate"].values,
    color="#1a5276",
    alpha=0.85,
)
ax.set_xlabel("Weak-label rate (audit-priority signal prevalence)")
ax.set_title("Behavioral Health Weak-Label Rate by Provider Type")
ax.grid(True, axis="x", linestyle=":", alpha=0.5)
plt.tight_layout()
plt.savefig(bh_type_rate_path, dpi=150, bbox_inches="tight")
plt.show()
print("Saved:", bh_type_rate_path)

# C. Capacity tradeoff
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    bh_capacity_df["providers_reviewed"],
    bh_capacity_df["weak_label_precision"],
    marker="o",
    linewidth=2,
    color="#1a5276",
    label="Weak-label precision",
)
ax.plot(
    bh_capacity_df["providers_reviewed"],
    bh_capacity_df["weak_label_recall"],
    marker="s",
    linewidth=2,
    color="#b9770e",
    label="Weak-label recall",
)
ax.set_xscale("log")
ax.set_xlabel("Providers reviewed (behavioral-health capacity)")
ax.set_ylabel("Rate")
ax.set_title("Behavioral Health Review Capacity Tradeoff")
ax.set_ylim(0, 1.05)
ax.grid(True, which="both", linestyle=":", alpha=0.5)
ax.legend(loc="center right")
plt.tight_layout()
plt.savefig(bh_capacity_plot_path, dpi=150, bbox_inches="tight")
plt.show()
print("Saved:", bh_capacity_plot_path)

# %%
# Step 11 — Notebook Summary
#
# Business Purpose:
# Print a short, copy-pasteable summary of the run so the notebook
# ends with a clear, auditable record of what was produced.
#
# Plain-English Logic:
# We surface the input row counts, the behavioral-health slice size,
# the score column used, the saved output paths, and the closing
# limitation reminder so anyone reusing the outputs is reminded of
# what they are and are not.
#
# Expected Output:
# Plain printed text. No file writes here.
#
# Why It Matters:
# A consistent end-of-notebook summary makes it easy to compare runs
# across versions or years and to lift key numbers into a write-up
# without losing the limitation framing.

bh_pos = int(bh_providers[target_col].sum())
bh_n = int(len(bh_providers))
bh_rate = (bh_pos / bh_n) if bh_n else float("nan")

print("=" * 60)
print("Notebook 05 Summary — Behavioral Health Audit-Priority Module")
print("=" * 60)
print(
    f"Provider-service rows loaded            : "
    f"{len(features_df):,}"
)
print(
    f"Behavioral-health provider-service rows : "
    f"{len(bh_service_df):,}  ({bh_share:.4%} of total)"
)
print(
    f"Behavioral-health providers             : "
    f"{bh_n:,}"
)
print(
    f"Behavioral-health weak-label positives  : "
    f"{bh_pos:,}"
)
print(
    f"Behavioral-health weak-label prevalence : "
    f"{bh_rate:.5f}"
)
print(f"Risk score column used                  : {score_col}")
print("-" * 60)
print(f"Worklist saved to             : {bh_worklist_path}")
print(f"Capacity scenarios saved to   : {bh_capacity_path}")
print(f"Risk-score histogram saved to : {bh_score_hist_path}")
print(f"Provider-type rates saved to  : {bh_type_rate_path}")
print(f"Capacity tradeoff saved to    : {bh_capacity_plot_path}")
print("=" * 60)
print(
    "Reminder: this is a public-data audit-priority module. It does "
    "not determine fraud, overpayments, noncompliance, medical "
    "necessity, parity compliance, documentation sufficiency, or "
    "audit findings. The behavioral-health classification is a "
    "public-data approximation — provider specialty text plus HCPCS "
    "codes and descriptions — not a clinical classification. Treat "
    "every output as decision support for follow-up review, not as "
    "a finding."
)

# %%
