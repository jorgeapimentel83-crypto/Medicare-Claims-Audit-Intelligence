"""
Medicare Claims Audit — XGBoost + LightGBM GPU Ensemble
=========================================================
GPU-accelerated gradient boosting ensemble for audit risk scoring.
Same stack as the BNSF Rail Forecasting project: new domain.

Design rationale:
- XGBoost: strong on feature interactions (audit patterns are combinatorial)
- LightGBM: efficient on large datasets (10M+ rows, leaf-wise growth)
- Weighted ensemble: reduces variance without ensembling complexity
- AUCPR primary metric: audit is rare-event detection (precision matters)

Usage:
    from src.modeling import AuditEnsemble
    model = AuditEnsemble()
    model.fit(X_train, y_train, X_val, y_val, feature_names)
    scores = model.predict_proba(X_test)
    importance = model.feature_importance()
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

from config import MODEL_PARAMS, PATHS

log = logging.getLogger(__name__)


class AuditEnsemble:
    """
    Weighted XGBoost + LightGBM ensemble for audit risk scoring.

    Both models use GPU acceleration (XGBoost device='cuda',
    LightGBM device='gpu') on RTX 5080.
    """

    def __init__(
        self,
        xgb_params: Optional[dict] = None,
        lgb_params: Optional[dict] = None,
        xgb_weight: float = None,
        lgb_weight: float = None,
    ):
        cfg = MODEL_PARAMS
        self.xgb_params = xgb_params or {
            k: v for k, v in cfg["xgboost"].items()
            if k not in ("n_estimators", "early_stopping_rounds")
        }
        self.lgb_params = lgb_params or {
            k: v for k, v in cfg["lightgbm"].items()
            if k not in ("n_estimators", "early_stopping_rounds")
        }
        self.n_estimators = cfg["xgboost"]["n_estimators"]
        self.early_stopping = cfg["xgboost"]["early_stopping_rounds"]
        self.xgb_weight = xgb_weight or cfg["ensemble"]["xgb_weight"]
        self.lgb_weight = lgb_weight or cfg["ensemble"]["lgb_weight"]

        self.xgb_model = None
        self.lgb_model = None
        self.feature_names = None
        self._fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> "AuditEnsemble":
        """Train both models with early stopping on AUCPR."""
        self.feature_names = feature_names
        log.info(f"Training ensemble: {X_train.shape[0]:,} train, {X_val.shape[0]:,} val")
        log.info(f"  Positive rate: {y_train.mean():.4f} (train), {y_val.mean():.4f} (val)")

        # --- XGBoost (CUDA) ---
        log.info(f"\n{'─'*50}")
        log.info(f"  XGBoost (device={self.xgb_params.get('device', 'cpu')})")
        log.info(f"{'─'*50}")
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

        self.xgb_model = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=self.early_stopping,
            verbose_eval=100,
        )
        xgb_best = self.xgb_model.best_iteration
        log.info(f"  XGBoost best iteration: {xgb_best}")

        # --- LightGBM (GPU) ---
        log.info(f"\n{'─'*50}")
        log.info(f"  LightGBM (device={self.lgb_params.get('device', 'cpu')})")
        log.info(f"{'─'*50}")
        ltrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        lval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=ltrain)

        self.lgb_model = lgb.train(
            self.lgb_params,
            ltrain,
            num_boost_round=self.n_estimators,
            valid_sets=[ltrain, lval],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(self.early_stopping),
                lgb.log_evaluation(100),
            ],
        )
        lgb_best = self.lgb_model.best_iteration
        log.info(f"  LightGBM best iteration: {lgb_best}")

        self._fitted = True

        # Evaluate ensemble on validation
        val_metrics = self.evaluate(X_val, y_val)
        log.info(f"\n  Ensemble validation metrics:")
        for k, v in val_metrics.items():
            log.info(f"    {k}: {v:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted ensemble probability scores."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        xgb_pred = self.xgb_model.predict(
            xgb.DMatrix(X, feature_names=self.feature_names)
        )
        lgb_pred = self.lgb_model.predict(X)

        return self.xgb_weight * xgb_pred + self.lgb_weight * lgb_pred

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Compute audit-relevant metrics."""
        proba = self.predict_proba(X)
        metrics = {
            "average_precision": average_precision_score(y, proba),
            "roc_auc": roc_auc_score(y, proba),
        }

        # Precision at K — how many of the top-K scored are actually positive?
        for k in [100, 500, 1000]:
            if len(proba) >= k:
                top_k_idx = np.argsort(proba)[-k:]
                metrics[f"precision_at_{k}"] = y[top_k_idx].mean()

        return metrics

    def feature_importance(self, top_n: int = 20) -> dict:
        """Combined feature importance (gain-based) from both models."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        xgb_imp = self.xgb_model.get_score(importance_type="gain")
        lgb_imp = dict(zip(
            self.feature_names or [f"f{i}" for i in range(self.lgb_model.num_feature())],
            self.lgb_model.feature_importance(importance_type="gain"),
        ))

        # Normalize and combine
        xgb_total = sum(xgb_imp.values()) or 1
        lgb_total = sum(lgb_imp.values()) or 1

        combined = {}
        all_feats = set(list(xgb_imp.keys()) + list(lgb_imp.keys()))
        for f in all_feats:
            xgb_norm = xgb_imp.get(f, 0) / xgb_total
            lgb_norm = lgb_imp.get(f, 0) / lgb_total
            combined[f] = self.xgb_weight * xgb_norm + self.lgb_weight * lgb_norm

        return dict(sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_n])

    def save(self, name: str = "audit_ensemble") -> Path:
        """Save both models to the models/ directory."""
        models_dir = PATHS["models"]
        self.xgb_model.save_model(str(models_dir / f"{name}_xgb.json"))
        self.lgb_model.save_model(str(models_dir / f"{name}_lgb.txt"))

        meta = {
            "xgb_weight": self.xgb_weight,
            "lgb_weight": self.lgb_weight,
            "feature_names": self.feature_names,
            "xgb_best_iteration": self.xgb_model.best_iteration,
            "lgb_best_iteration": self.lgb_model.best_iteration,
        }
        with open(models_dir / f"{name}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        log.info(f"✓ Models saved to {models_dir}/{name}_*")
        return models_dir

    @classmethod
    def load(cls, name: str = "audit_ensemble") -> "AuditEnsemble":
        """Load a saved ensemble."""
        models_dir = PATHS["models"]
        with open(models_dir / f"{name}_meta.json") as f:
            meta = json.load(f)

        model = cls(
            xgb_weight=meta["xgb_weight"],
            lgb_weight=meta["lgb_weight"],
        )
        model.xgb_model = xgb.Booster()
        model.xgb_model.load_model(str(models_dir / f"{name}_xgb.json"))
        model.lgb_model = lgb.Booster(model_file=str(models_dir / f"{name}_lgb.txt"))
        model.feature_names = meta["feature_names"]
        model._fitted = True

        return model
