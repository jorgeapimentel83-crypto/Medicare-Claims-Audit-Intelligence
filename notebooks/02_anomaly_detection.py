# %%
"""
Notebook 02 — Provider-Level Anomaly Detection and Weak Supervision

Business Purpose:
This notebook converts Medicare Part B provider-service level features into a
provider-level audit-priority dataset. The goal is to identify unusual provider
billing patterns that may warrant follow-up review using public CMS data.

Plain-English Logic:
Notebook 01 created one row per provider-service combination:
    NPI + HCPCS code + place of service

Notebook 02 will summarize those rows into one row per provider:
    one NPI = one provider profile

Then we will use anomaly detection and weak supervision to create a provider-level
audit-priority label for later supervised modeling.

Important Limitation:
This notebook does not identify fraud, overpayments, noncompliance, or audit
findings. It creates public-data decision-support signals for portfolio modeling
and follow-up-review prioritization.
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

PROJECT_ROOT = Path.cwd().resolve()
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

feature_path = PATHS["data_processed"] / f"features_provider_service_{YEAR}.parquet"

print("Loading:", feature_path)

df = pd.read_parquet(feature_path)

print("Rows:", f"{len(df):,}")
print("Columns:", f"{df.shape[1]:,}")

df.head()

# %%
# Step 3 — Confirm Provider-Service Level Structure
#
# Business Purpose:
# Verify that the data is still at the provider-service level before we aggregate.
#
# Plain-English Logic:
# A single provider can appear on many rows because each row represents a provider,
# a service code, and a place of service. Before creating provider-level anomaly
# scores, we need to confirm that structure.
#
# Expected Output:
# Counts of unique providers, HCPCS codes, place-of-service values, and total rows.
#
# Why It Matters:
# Anomaly detection will be done at the provider level, so we need to understand
# the current row structure first.
#
# What to Check Before Moving On:
# The number of rows should be much larger than the number of unique NPIs.

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
# Identify which columns are engineered model features and which columns identify
# the provider.
#
# Plain-English Logic:
# Feature columns start with feat_. Provider identifier columns describe the NPI,
# provider name, specialty, state, and entity type. We need both:
#   - feat_ columns for anomaly detection
#   - provider identity columns for interpretation and review tables
#
# Expected Output:
# A list of engineered features and available provider identity columns.
#
# Why It Matters:
# This sets up the next step: aggregating provider-service rows into one provider
# profile per NPI.
#
# What to Check Before Moving On:
# Confirm that the expected feat_ columns from Notebook 01 are present.

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