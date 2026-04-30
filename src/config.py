"""
Medicare Claims Audit Intelligence Platform — Configuration
=============================================================
Centralized paths, hyperparameters, feature definitions, and
CMS dataset URLs. Import this module everywhere.

Usage:
    from src.config import PATHS, CMS_DATASETS, MODEL_PARAMS
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATHS = {
    "project_root": PROJECT_ROOT,
    "data_raw": PROJECT_ROOT / "data" / "raw",
    "data_interim": PROJECT_ROOT / "data" / "interim",
    "data_processed": PROJECT_ROOT / "data" / "processed",
    "notebooks": PROJECT_ROOT / "notebooks",
    "models": PROJECT_ROOT / "models",
    "reports": PROJECT_ROOT / "reports",
    "outputs": PROJECT_ROOT / "outputs",
}

# Ensure directories exist
for p in PATHS.values():
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CMS Public Use File Download URLs
# ---------------------------------------------------------------------------
# All datasets are public, de-identified — no DUA, no authentication.
#
# These are the verified static CSV URLs published under
# data.cms.gov/sites/default/files/. The /data/{year} dataset-page URLs
# return HTML, not CSV — do not use them as download sources.
# ---------------------------------------------------------------------------
CMS_DATASETS = {
    # NPI × HCPCS × POS level — primary audit dataset (~10M rows, ~2GB)
    "provider_service": {
        "name": "Medicare Physician & Other Practitioners — by Provider and Service",
        "urls": {
            "2022": (
                "https://data.cms.gov/sites/default/files/2025-11/"
                "53fb2bae-4913-48dc-a6d4-d8c025906567/"
                "MUP_PHY_R25_P05_V20_D22_Prov_Svc.csv"
            ),
        },
        "filename": "provider_service_{year}.csv",
    },

    # One row per NPI — aggregate provider summary (~1.2M rows)
    "provider_agg": {
        "name": "Medicare Physician & Other Practitioners — by Provider",
        "urls": {
            "2022": (
                "https://data.cms.gov/sites/default/files/2025-11/"
                "adcd20c5-4534-43cd-8dfa-881ebe7bacfd/"
                "MUP_PHY_R25_P07_V20_D22_Prov.csv"
            ),
        },
        "filename": "provider_agg_{year}.csv",
    },

    # State/national benchmarks by HCPCS — for peer comparison features
    "geo_service": {
        "name": "Medicare Physician & Other Practitioners — by Geography and Service",
        "urls": {
            "2022": (
                "https://data.cms.gov/sites/default/files/2024-05/"
                "3167b4d9-10c0-48f0-a680-1165f4eec064/"
                "MUP_PHY_R24_P05_V10_D22_Geo.csv"
            ),
        },
        "filename": "geo_service_{year}.csv",
    },
}

# OIG LEIE — known excluded providers for weak supervision labels
LEIE_URL = "https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv"
LEIE_FILENAME = "leie_exclusions.csv"


# ---------------------------------------------------------------------------
# Column Dtype Map — optimized types for CMS PUF columns
# ---------------------------------------------------------------------------
# Using float32 instead of float64 halves memory; string cols stay object.
# CMS column names follow their published data dictionary.
CMS_DTYPE_MAP = {
    "Rndrng_NPI": "str",
    "Rndrng_Prvdr_Last_Org_Name": "str",
    "Rndrng_Prvdr_First_Name": "str",
    "Rndrng_Prvdr_MI": "str",
    "Rndrng_Prvdr_Crdntls": "str",
    "Rndrng_Prvdr_Gndr": "str",
    "Rndrng_Prvdr_Ent_Cd": "str",
    "Rndrng_Prvdr_St1": "str",
    "Rndrng_Prvdr_St2": "str",
    "Rndrng_Prvdr_City": "str",
    "Rndrng_Prvdr_State_Abrvtn": "str",
    "Rndrng_Prvdr_State_FIPS": "str",
    "Rndrng_Prvdr_Zip5": "str",
    "Rndrng_Prvdr_RUCA": "str",
    "Rndrng_Prvdr_RUCA_Desc": "str",
    "Rndrng_Prvdr_Cntry": "str",
    "Rndrng_Prvdr_Type": "str",
    "Rndrng_Prvdr_Mdcr_Prtcptg_Ind": "str",
    "HCPCS_Cd": "str",
    "HCPCS_Desc": "str",
    "HCPCS_Drug_Ind": "str",
    "Place_Of_Srvc": "str",
    "Tot_Benes": "float32",
    "Tot_Srvcs": "float32",
    "Tot_Sbmtd_Chrg": "float64",
    "Tot_Mdcr_Alowd_Amt": "float64",
    "Tot_Mdcr_Pymt_Amt": "float64",
    "Tot_Mdcr_Stdzd_Amt": "float64",
    "Avg_Sbmtd_Chrg": "float32",
    "Avg_Mdcr_Alowd_Amt": "float32",
    "Avg_Mdcr_Pymt_Amt": "float32",
    "Avg_Mdcr_Stdzd_Amt": "float32",
}


# ---------------------------------------------------------------------------
# Feature Engineering Parameters
# ---------------------------------------------------------------------------
FEATURE_PARAMS = {
    "charge_ratio": {
        "outlier_percentile": 0.95,
    },
    "peer_deviation": {
        "group_col": "Rndrng_Prvdr_Type",
        "metrics": ["Tot_Srvcs", "Avg_Sbmtd_Chrg", "Avg_Mdcr_Pymt_Amt"],
        "zscore_threshold": 3.0,
    },
    "service_concentration": {
        "hhi_high_threshold": 0.50,
    },
    "volume_anomaly": {
        "p95_flag": 0.95,
        "p99_flag": 0.99,
    },
    "modifier_flags": ["26", "TC", "59", "25", "76", "77"],
}


# ---------------------------------------------------------------------------
# Model Hyperparameters
# ---------------------------------------------------------------------------
MODEL_PARAMS = {
    "xgboost": {
        "device": "cuda",
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "early_stopping_rounds": 50,
        "random_state": 42,
    },
    "lightgbm": {
        "device": "gpu",
        "objective": "binary",
        "metric": "average_precision",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "early_stopping_rounds": 50,
        "verbose": -1,
        "random_state": 42,
    },
    "ensemble": {
        "xgb_weight": 0.55,
        "lgb_weight": 0.45,
    },
}


# ---------------------------------------------------------------------------
# Monte Carlo Simulation Parameters
# ---------------------------------------------------------------------------
MONTE_CARLO_PARAMS = {
    "n_simulations": 10_000,
    "confidence_levels": [0.80, 0.90, 0.95],
    "random_state": 42,
}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
AUDIT_PALETTE = ["#1a5276", "#c0392b", "#27ae60", "#f39c12", "#8e44ad", "#2c3e50"]

DEFAULT_YEAR = "2022"
