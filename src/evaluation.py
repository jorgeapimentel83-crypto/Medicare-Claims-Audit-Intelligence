"""
Medicare Claims Audit — Evaluation Metrics
=============================================
Audit-specific model evaluation. Standard ML metrics (accuracy, F1)
are misleading for audit targeting because:
  1. Base rate is low (~1-5% of providers are truly problematic)
  2. We care about PRECISION at the top of the ranked list
  3. Cost of false negatives (missed fraud) >> cost of false positives (wasted audit)

Key metrics:
  - AUCPR (Average Precision): primary metric for imbalanced audit detection
  - Precision@K: what fraction of our top-K targets are actually hits?
  - Lift: how much better is the model vs random audit selection?
  - Estimated Recovery: expected dollar recovery from top-K audits

Usage:
    from src.evaluation import full_evaluation_report
    report = full_evaluation_report(y_true, y_proba, dollar_amounts)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

log = logging.getLogger(__name__)


def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Precision at K: of the top-K highest-scored providers,
    what fraction are true positives?

    This is the most operationally relevant metric:
    "If I audit the top 500 providers from this model, how many
    will actually have findings?"
    """
    if len(y_scores) < k:
        k = len(y_scores)
    top_k_idx = np.argsort(y_scores)[-k:]
    return y_true[top_k_idx].mean()


def recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Recall at K: of all true positives, what fraction appear
    in the top-K ranked list?
    """
    if y_true.sum() == 0:
        return 0.0
    top_k_idx = np.argsort(y_scores)[-k:]
    return y_true[top_k_idx].sum() / y_true.sum()


def lift_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Lift at K: how many times better is the model vs random selection?

    Lift = precision_at_k / base_rate
    A lift of 10x means the model finds 10x more hits than random auditing.
    """
    base_rate = y_true.mean()
    if base_rate == 0:
        return 0.0
    return precision_at_k(y_true, y_scores, k) / base_rate


def estimated_recovery_at_k(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    dollar_amounts: np.ndarray,
    k: int,
    recovery_rate: float = 0.65,
) -> float:
    """
    Estimated dollar recovery from auditing the top-K providers.

    Parameters
    ----------
    dollar_amounts : array
        Total Medicare payments per provider (proxy for recovery potential).
    recovery_rate : float
        Estimated fraction of payments recoverable in a successful audit.
        OIG historically recovers ~$6 for every $1 spent on audits.
    """
    top_k_idx = np.argsort(y_scores)[-k:]
    # Only count recoveries from true positives
    hits = y_true[top_k_idx].astype(bool)
    return dollar_amounts[top_k_idx][hits].sum() * recovery_rate


def full_evaluation_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    dollar_amounts: Optional[np.ndarray] = None,
    k_values: list = None,
) -> dict:
    """
    Generate a comprehensive audit evaluation report.

    Returns a dict suitable for logging, serialization, or dashboard display.
    """
    k_values = k_values or [50, 100, 250, 500, 1000]

    report = {
        "n_total": len(y_true),
        "n_positive": int(y_true.sum()),
        "base_rate": float(y_true.mean()),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "precision_at_k": {},
        "recall_at_k": {},
        "lift_at_k": {},
    }

    if dollar_amounts is not None:
        report["recovery_at_k"] = {}

    for k in k_values:
        if k > len(y_true):
            continue
        report["precision_at_k"][k] = precision_at_k(y_true, y_proba, k)
        report["recall_at_k"][k] = recall_at_k(y_true, y_proba, k)
        report["lift_at_k"][k] = lift_at_k(y_true, y_proba, k)
        if dollar_amounts is not None:
            report["recovery_at_k"][k] = estimated_recovery_at_k(
                y_true, y_proba, dollar_amounts, k
            )

    # Print formatted report
    log.info(f"\n{'='*60}")
    log.info(f"  AUDIT MODEL EVALUATION REPORT")
    log.info(f"{'='*60}")
    log.info(f"  Total providers: {report['n_total']:,}")
    log.info(f"  True positives:  {report['n_positive']:,} ({report['base_rate']:.2%})")
    log.info(f"  ROC AUC:         {report['roc_auc']:.4f}")
    log.info(f"  Avg Precision:   {report['average_precision']:.4f}")
    log.info(f"\n  {'K':>6}  {'P@K':>8}  {'R@K':>8}  {'Lift':>8}")
    log.info(f"  {'─'*38}")
    for k in k_values:
        if k in report["precision_at_k"]:
            p = report["precision_at_k"][k]
            r = report["recall_at_k"][k]
            l = report["lift_at_k"][k]
            log.info(f"  {k:>6,}  {p:>8.3f}  {r:>8.3f}  {l:>7.1f}x")

    if dollar_amounts is not None:
        log.info(f"\n  Estimated Recovery (at 65% recovery rate):")
        for k, amount in report["recovery_at_k"].items():
            log.info(f"    Top {k:,}: ${amount:,.0f}")

    return report
