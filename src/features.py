"""
Medicare Claims Audit — Domain-Informed Feature Engineering
=============================================================
Every feature maps to an audit-priority pattern commonly used to
prioritize where reviewers look first when working from public CMS
summary data. These are public-data decision-support features — they
do not identify fraud, overpayments, or noncompliance.

The value here isn't in the algorithms — it's in knowing which features
to build. Recognizing that a charge-to-allowed ratio of 15:1 on a J-code
billed from POS 11 with an HHI of 0.9 is the kind of combined pattern
that may warrant follow-up review draws on years of audit experience.

Usage:
    from src.features import build_all_features
    df = build_all_features(df_provider_service)

Feature Categories:
    1. Charge-to-Allowed Ratio  — submitted charge vs allowed amount
    2. Peer Specialty Deviation — z-scores vs peer group
    3. Service Concentration    — Herfindahl-Hirschman Index
    4. Volume Anomaly           — services per beneficiary
    5. Place of Service         — facility vs office patterns
    6. Payment Variance         — charge/payment gaps
    7. Drug Billing             — J-code revenue concentration
    8. Geographic Deviation     — provider vs state benchmarks
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import FEATURE_PARAMS

log = logging.getLogger(__name__)


# ===================================================================
# Feature 1: Charge-to-Allowed Ratio
# ===================================================================
# REVIEW SIGNAL: Ratio of what a provider submits vs what Medicare allows.
# Typical range is 1.5-4x (most providers mark up above fee schedule).
# Ratios of 10x+ are unusual billing behavior worth a closer look — this
# can affect secondary payer calculations, patient liability, and MSP
# recoveries. Out-of-network billing and chargemaster effects can also
# produce high ratios for legitimate reasons.
#
# Illustrative example: DME suppliers submitting charges of $3,000 for
# items with a $150 allowed amount would produce a 20:1 ratio.
# ===================================================================
def charge_to_allowed_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute charge-to-allowed ratio signals (audit-priority feature)."""
    df = df.copy()

    mask = df["Avg_Mdcr_Alowd_Amt"] > 0
    df["feat_charge_to_allowed"] = np.where(
        mask, df["Avg_Sbmtd_Chrg"] / df["Avg_Mdcr_Alowd_Amt"], np.nan
    )

    # Log-transform stabilizes extreme outliers for tree models
    df["feat_log_charge_ratio"] = np.log1p(
        df["feat_charge_to_allowed"].clip(lower=0)
    )

    # Binary flag: charge-to-allowed ratio above the configured outlier percentile
    pctl = FEATURE_PARAMS["charge_ratio"]["outlier_percentile"]
    threshold = df["feat_charge_to_allowed"].quantile(pctl)
    df["feat_extreme_billing"] = (df["feat_charge_to_allowed"] > threshold).astype(int)

    return df


# ===================================================================
# Feature 2: Peer Specialty Deviation (Z-Scores)
# ===================================================================
# REVIEW SIGNAL: How far does this provider deviate from specialty peers?
# A dermatologist performing 10x the skin biopsies of peer dermatologists
# is an unusual billing pattern that may warrant follow-up review, even
# if each individual claim looks fine.
#
# Illustrative example: high-throughput practices billing E&M codes
# (99214/99215) at rates 5x the specialty median surface as outliers
# under this signal — context determines whether the pattern is
# legitimate (panel mix, subspecialty) or worth a closer look.
# ===================================================================
def peer_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Z-scores of key metrics vs same-specialty peer group."""
    df = df.copy()
    group_col = FEATURE_PARAMS["peer_deviation"]["group_col"]
    metrics = FEATURE_PARAMS["peer_deviation"]["metrics"]

    feat_names = []
    for col in metrics:
        feat_name = f"feat_zscore_{col.lower().replace('avg_', '').replace('tot_', '')}"
        feat_names.append(feat_name)

        stats = df.groupby(group_col)[col].agg(["mean", "std"])
        stats.columns = ["_pmean", "_pstd"]
        df = df.merge(stats, left_on=group_col, right_index=True, how="left")

        df[feat_name] = np.where(
            df["_pstd"] > 0,
            (df[col] - df["_pmean"]) / df["_pstd"],
            0,
        )
        df.drop(columns=["_pmean", "_pstd"], inplace=True)

    # Composite: max absolute z-score across all metrics
    df["feat_max_peer_deviation"] = df[feat_names].abs().max(axis=1)

    return df


# ===================================================================
# Feature 3: Service Concentration (Herfindahl-Hirschman Index)
# ===================================================================
# REVIEW SIGNAL: Revenue concentration across HCPCS codes. Most
# practices have diversified service mixes; a provider with 90% of
# revenue from a single code shows a highly concentrated billing
# pattern. Many subspecialists legitimately concentrate, so this is
# only informative in combination with specialty context.
#
# Illustrative example: pain management practices that bill almost
# exclusively for facet joint injections (64493-64495) or trigger
# point injections (20552-20553) appear at the high end of HHI.
#
# HHI interpretation:
#   0.00 - 0.15 : Diversified
#   0.15 - 0.25 : Moderate concentration
#   0.25 - 0.50 : High concentration (may warrant follow-up review)
#   0.50 - 1.00 : Very high concentration (audit-priority pattern)
# ===================================================================
def service_concentration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Herfindahl-Hirschman Index of HCPCS revenue concentration per provider."""
    df = df.copy()

    # Revenue share per code per provider
    prov_total = df.groupby("Rndrng_NPI")["Tot_Mdcr_Pymt_Amt"].sum().rename("_prov_total")
    df = df.merge(prov_total, left_on="Rndrng_NPI", right_index=True, how="left")

    df["_share_sq"] = np.where(
        df["_prov_total"] > 0,
        (df["Tot_Mdcr_Pymt_Amt"] / df["_prov_total"]) ** 2,
        0,
    )

    # HHI = sum of squared revenue shares
    hhi = df.groupby("Rndrng_NPI")["_share_sq"].sum().rename("feat_hhi")
    df = df.merge(hhi, left_on="Rndrng_NPI", right_index=True, how="left")

    # Code diversity: number of distinct HCPCS per provider
    n_codes = df.groupby("Rndrng_NPI")["HCPCS_Cd"].nunique().rename("feat_n_hcpcs_codes")
    df = df.merge(n_codes, left_on="Rndrng_NPI", right_index=True, how="left")

    # Log of code count (better for models than raw count)
    df["feat_log_n_hcpcs"] = np.log1p(df["feat_n_hcpcs_codes"])

    # Concentration tier
    df["feat_concentration_tier"] = pd.cut(
        df["feat_hhi"],
        bins=[0, 0.15, 0.25, 0.50, 1.01],
        labels=["Diversified", "Moderate", "High", "Very_High"],
    )

    df.drop(columns=["_prov_total", "_share_sq"], inplace=True)
    return df


# ===================================================================
# Feature 4: Volume Anomaly (Services per Beneficiary)
# ===================================================================
# REVIEW SIGNAL: How many services per unique patient? Extreme values
# can be associated with unbundling or overutilization patterns and
# may warrant follow-up review. High values can also reflect
# legitimate panel mix or subspecialty practice.
#
# Illustrative example: lab providers billing 20+ tests per patient
# encounter where the clinical indication typically supports fewer
# tests would surface as outliers under this signal.
# ===================================================================
def volume_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Services per beneficiary at line and provider levels."""
    df = df.copy()

    mask = df["Tot_Benes"] > 0
    df["feat_srvcs_per_bene"] = np.where(
        mask, df["Tot_Srvcs"] / df["Tot_Benes"], np.nan
    )
    df["feat_log_srvcs_per_bene"] = np.log1p(
        df["feat_srvcs_per_bene"].clip(lower=0)
    )

    # Provider-level aggregate
    prov = df.groupby("Rndrng_NPI").agg(
        _tsrvcs=("Tot_Srvcs", "sum"),
        _tbenes=("Tot_Benes", "sum"),
    )
    prov["feat_prov_srvcs_per_bene"] = np.where(
        prov["_tbenes"] > 0, prov["_tsrvcs"] / prov["_tbenes"], np.nan
    )
    df = df.merge(
        prov[["feat_prov_srvcs_per_bene"]],
        left_on="Rndrng_NPI", right_index=True, how="left",
    )

    # Volume tier flags
    p95 = df["feat_srvcs_per_bene"].quantile(0.95)
    p99 = df["feat_srvcs_per_bene"].quantile(0.99)
    df["feat_volume_flag"] = np.select(
        [df["feat_srvcs_per_bene"] > p99, df["feat_srvcs_per_bene"] > p95],
        [2, 1],
        default=0,
    )

    return df


# ===================================================================
# Feature 5: Place of Service Patterns
# ===================================================================
# REVIEW SIGNAL: Facility rates (POS 22 = outpatient hospital) are
# typically ~60% higher than non-facility rates (POS 11 = office) for
# the same procedure. Providers whose facility/non-facility mix looks
# unusual for their specialty may warrant follow-up review of the
# place-of-service codes used.
#
# Key POS codes:
#   F/22 = Outpatient Hospital (facility rate)
#   11   = Office (non-facility rate)
#   21   = Inpatient Hospital
#   23   = Emergency Room
#   31   = Skilled Nursing Facility
#   81   = Independent Lab
# ===================================================================
def pos_features(df: pd.DataFrame) -> pd.DataFrame:
    """Place of Service billing pattern features."""
    df = df.copy()

    # Facility indicator
    facility_codes = {"F", "21", "22", "23", "24", "31"}
    df["_is_fac"] = df["Place_Of_Srvc"].isin(facility_codes).astype(int)

    # Provider-level facility ratio
    prov_pos = df.groupby("Rndrng_NPI").agg(
        _fac_lines=("_is_fac", "sum"),
        _tot_lines=("_is_fac", "count"),
    )
    prov_pos["feat_facility_ratio"] = prov_pos["_fac_lines"] / prov_pos["_tot_lines"]
    df = df.merge(
        prov_pos[["feat_facility_ratio"]],
        left_on="Rndrng_NPI", right_index=True, how="left",
    )

    # POS diversity
    pos_div = df.groupby("Rndrng_NPI")["Place_Of_Srvc"].nunique().rename("feat_n_pos_types")
    df = df.merge(pos_div, left_on="Rndrng_NPI", right_index=True, how="left")

    df.drop(columns=["_is_fac"], inplace=True)
    return df


# ===================================================================
# Feature 6: Payment Variance
# ===================================================================
# REVIEW SIGNAL: Large gaps between submitted, allowed, and paid amounts.
# Also: the standardized-vs-raw payment ratio reflects geographic cost
# adjustments. Providers whose geographic payment profile looks unusual
# relative to their location may warrant follow-up review of GPCI
# locality assignments.
# ===================================================================
def payment_variance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Payment gap, payment rate, and standardization variance."""
    df = df.copy()

    # Absolute gap: what was billed minus what was allowed
    df["feat_charge_gap"] = df["Avg_Sbmtd_Chrg"] - df["Avg_Mdcr_Alowd_Amt"]

    # Payment rate: fraction of allowed amount actually paid
    # (values < 1.0 indicate coinsurance/deductible; values << 1.0 are unusual)
    mask_a = df["Avg_Mdcr_Alowd_Amt"] > 0
    df["feat_payment_rate"] = np.where(
        mask_a, df["Avg_Mdcr_Pymt_Amt"] / df["Avg_Mdcr_Alowd_Amt"], np.nan
    )

    # Standardized-to-raw payment ratio
    # CMS standardizes payments to remove geographic adjustments.
    # If standardized >> raw, the provider is in a low-payment area.
    # If standardized << raw, they benefit from high geographic adjustors.
    mask_p = df["Avg_Mdcr_Pymt_Amt"] > 0
    df["feat_std_payment_ratio"] = np.where(
        mask_p, df["Avg_Mdcr_Stdzd_Amt"] / df["Avg_Mdcr_Pymt_Amt"], np.nan
    )

    return df


# ===================================================================
# Feature 7: Drug Billing Concentration
# ===================================================================
# REVIEW SIGNAL: Part B drugs (~$40B/year) are a long-standing OIG
# focus area. HCPCS J-codes represent injectable/infused drugs
# administered in clinical settings. High drug revenue concentration
# is an audit-priority pattern that may warrant follow-up review for:
#   - Average Sales Price (ASP) reporting accuracy
#   - Documentation of administered drugs
#   - Off-label use considerations
#   - Buy-and-bill markup behavior
# ===================================================================
def drug_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drug billing concentration and J-code features."""
    df = df.copy()

    # CMS drug indicator flag
    df["feat_is_drug"] = (df["HCPCS_Drug_Ind"] == "Y").astype(int)

    # J-code heuristic (HCPCS codes starting with J = injectable drugs)
    df["feat_is_jcode"] = df["HCPCS_Cd"].str.startswith("J", na=False).astype(int)

    # Provider-level drug revenue share
    drug_rev = (
        df[df["feat_is_drug"] == 1]
        .groupby("Rndrng_NPI")["Tot_Mdcr_Pymt_Amt"]
        .sum()
    )
    total_rev = df.groupby("Rndrng_NPI")["Tot_Mdcr_Pymt_Amt"].sum()
    drug_share = (drug_rev / total_rev).rename("feat_drug_revenue_share").fillna(0)

    df = df.merge(drug_share, left_on="Rndrng_NPI", right_index=True, how="left")
    df["feat_drug_revenue_share"] = df["feat_drug_revenue_share"].fillna(0)

    return df


# ===================================================================
# Feature 8: Geographic Deviation
# ===================================================================
# REVIEW SIGNAL: Provider metrics compared to state-level benchmarks.
# A provider whose submitted charges deviate sharply from state-level
# medians is an unusual billing pattern that may warrant follow-up
# review. This optionally uses a geographic benchmark dataset
# (geo_service); otherwise benchmarks are computed from the input.
# ===================================================================
def geographic_deviation_features(
    df: pd.DataFrame,
    df_geo: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Provider vs state-level benchmark deviations.
    If geographic benchmark data is not provided, computes state-level
    benchmarks from the provider data itself.
    """
    df = df.copy()

    # Compute state-level benchmarks from provider data
    state_col = "Rndrng_Prvdr_State_Abrvtn"
    if state_col not in df.columns:
        log.warning("No state column found — skipping geographic features")
        return df

    state_stats = df.groupby(state_col).agg(
        state_med_charge=("Avg_Sbmtd_Chrg", "median"),
        state_med_payment=("Avg_Mdcr_Pymt_Amt", "median"),
        state_med_srvcs=("Tot_Srvcs", "median"),
    )

    df = df.merge(state_stats, left_on=state_col, right_index=True, how="left")

    # Deviation from state median
    df["feat_geo_charge_dev"] = np.where(
        df["state_med_charge"] > 0,
        df["Avg_Sbmtd_Chrg"] / df["state_med_charge"],
        np.nan,
    )
    df["feat_geo_payment_dev"] = np.where(
        df["state_med_payment"] > 0,
        df["Avg_Mdcr_Pymt_Amt"] / df["state_med_payment"],
        np.nan,
    )

    df.drop(columns=["state_med_charge", "state_med_payment", "state_med_srvcs"], inplace=True)
    return df


# ===================================================================
# Feature 9: Entity Type Indicators
# ===================================================================
# REVIEW SIGNAL: Individual providers (Ent_Cd = "I") and organizations
# (Ent_Cd = "O") have different billing norms — e.g., high HHI is more
# typical for an individual specialist than for a large group practice.
# Downstream models can use these indicators to learn entity-aware
# behavior; this builder produces indicators only.
# ===================================================================
def entity_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """Entity type and Medicare-participation indicator features."""
    df = df.copy()

    df["feat_is_organization"] = (df["Rndrng_Prvdr_Ent_Cd"] == "O").astype(int)
    df["feat_is_individual"] = (df["Rndrng_Prvdr_Ent_Cd"] == "I").astype(int)

    # Medicare participating indicator
    df["feat_is_participating"] = (df["Rndrng_Prvdr_Mdcr_Prtcptg_Ind"] == "Y").astype(int)

    return df


# ===================================================================
# Composite Audit-Priority Score (Unsupervised Baseline)
# ===================================================================
# This is NOT a model — it's a simple weighted percentile-rank composite
# that serves as a baseline ranking. It blends the individual review
# signals into a single ordering for prioritization. If providers known
# to be of audit interest do not rank high on this, the underlying
# features likely need rethinking.
# ===================================================================
def composite_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Weighted composite of audit-priority signals (baseline, not a model)."""
    df = df.copy()

    weights = {
        "feat_log_charge_ratio": 0.15,
        "feat_max_peer_deviation": 0.25,
        "feat_hhi": 0.15,
        "feat_log_srvcs_per_bene": 0.15,
        "feat_charge_gap": 0.10,
        "feat_drug_revenue_share": 0.10,
        "feat_facility_ratio": 0.05,
        "feat_geo_charge_dev": 0.05,
    }

    # Percentile-rank normalize each feature to [0, 1]
    for feat in weights:
        if feat in df.columns:
            df[f"_norm_{feat}"] = df[feat].rank(pct=True, na_option="keep")

    # Weighted sum
    df["feat_composite_risk"] = 0.0
    total_weight = 0.0
    for feat, w in weights.items():
        norm_col = f"_norm_{feat}"
        if norm_col in df.columns:
            df["feat_composite_risk"] += w * df[norm_col].fillna(0)
            total_weight += w

    # Re-normalize if some features were missing
    if total_weight > 0 and total_weight < 1.0:
        df["feat_composite_risk"] /= total_weight

    # Drop temp columns
    norm_cols = [c for c in df.columns if c.startswith("_norm_")]
    df.drop(columns=norm_cols, inplace=True)

    return df


# ===================================================================
# Master Pipeline
# ===================================================================
def build_all_features(
    df: pd.DataFrame,
    df_geo: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Apply all domain-informed audit-priority feature builders in sequence.

    Parameters
    ----------
    df : pd.DataFrame
        Provider-Service level CMS data.
    df_geo : pd.DataFrame, optional
        Geographic benchmark data for state-level comparisons.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ~25 new feature columns prefixed 'feat_'.
    """
    pipeline = [
        ("Charge-to-Allowed Ratio", lambda d: charge_to_allowed_features(d)),
        ("Peer Specialty Deviation", lambda d: peer_deviation_features(d)),
        ("Service Concentration (HHI)", lambda d: service_concentration_features(d)),
        ("Volume Anomaly", lambda d: volume_anomaly_features(d)),
        ("Place of Service Patterns", lambda d: pos_features(d)),
        ("Payment Variance", lambda d: payment_variance_features(d)),
        ("Drug Billing Concentration", lambda d: drug_features(d)),
        ("Geographic Deviation", lambda d: geographic_deviation_features(d, df_geo)),
        ("Entity Type", lambda d: entity_type_features(d)),
        ("Composite Audit-Priority Score", lambda d: composite_risk_score(d)),
    ]

    log.info("Building audit-priority features...")
    for name, func in pipeline:
        log.info(f"  → {name}")
        df = func(df)

    feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    log.info(f"\n✓ Feature engineering complete: {len(feat_cols)} features")
    for col in feat_cols:
        log.info(f"    {col}")

    return df


def get_feature_metadata() -> dict:
    """Return feature names, descriptions, and audit-priority rationale."""
    return {
        "feat_charge_to_allowed": "Ratio of submitted charge to Medicare allowed amount (charge-to-allowed ratio)",
        "feat_log_charge_ratio": "Log-transformed charge-to-allowed ratio (stabilized for models)",
        "feat_extreme_billing": "Binary flag: charge-to-allowed ratio above the configured outlier percentile",
        "feat_zscore_sbmtd_chrg": "Z-score of avg submitted charge vs specialty peers",
        "feat_zscore_mdcr_pymt_amt": "Z-score of avg Medicare payment vs specialty peers",
        "feat_zscore_srvcs": "Z-score of total services vs specialty peers",
        "feat_max_peer_deviation": "Maximum absolute z-score across all peer metrics",
        "feat_hhi": "Herfindahl-Hirschman Index of HCPCS revenue concentration",
        "feat_n_hcpcs_codes": "Count of distinct HCPCS codes billed by provider",
        "feat_log_n_hcpcs": "Log-transformed HCPCS code count",
        "feat_concentration_tier": "Categorical: Diversified/Moderate/High/Very_High",
        "feat_srvcs_per_bene": "Services per beneficiary at line level",
        "feat_log_srvcs_per_bene": "Log-transformed services per beneficiary",
        "feat_prov_srvcs_per_bene": "Provider-level aggregate services per beneficiary",
        "feat_volume_flag": "Ordinal: 0=Normal, 1=Elevated (>95th), 2=Extreme (>99th)",
        "feat_facility_ratio": "Fraction of claim lines billed at facility place of service",
        "feat_n_pos_types": "Count of distinct Place of Service codes used by provider",
        "feat_charge_gap": "Dollar gap between submitted charge and allowed amount",
        "feat_payment_rate": "Ratio of Medicare payment to allowed amount",
        "feat_std_payment_ratio": "Ratio of standardized to raw Medicare payment",
        "feat_is_drug": "Binary: HCPCS code flagged as drug by CMS",
        "feat_is_jcode": "Binary: HCPCS code starts with J (injectable drug)",
        "feat_drug_revenue_share": "Fraction of provider revenue from drug codes",
        "feat_geo_charge_dev": "Provider charge deviation from state median",
        "feat_geo_payment_dev": "Provider payment deviation from state median",
        "feat_is_organization": "Binary: provider is an organization (vs individual)",
        "feat_is_individual": "Binary: provider is an individual",
        "feat_is_participating": "Binary: Medicare participating provider",
        "feat_composite_risk": "Weighted composite of audit-priority signals (baseline score, not a model)",
    }
