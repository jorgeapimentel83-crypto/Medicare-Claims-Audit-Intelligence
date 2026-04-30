"""
Medicare Claims Audit Intelligence Platform
=============================================
GPU-accelerated Medicare Part B claims audit targeting.

Modules:
    config          Centralized paths, hyperparams, CMS URLs
    download_data   CMS public data download pipeline
    load_data       GPU/CPU data loading with dtype optimization
    features        Domain-informed audit feature engineering
    modeling        XGBoost + LightGBM GPU ensemble
    evaluation      Audit-specific metrics (precision@k, AUCPR)
    simulation      Monte Carlo overpayment estimation
"""
