# %%
"""
Notebook 03 — Supervised Audit-Priority Modeling on Weak Labels

Business Purpose:
This notebook trains a first set of supervised models against the weak
audit-priority label produced by Notebook 02. The goal is not to predict
fraud, overpayments, noncompliance, or audit findings. The goal is to
learn the patterns the weak label captures, so the model can rank-order
providers for follow-up review using public CMS Medicare Part B data.

Plain-English Logic:
Notebook 02 produced one row per provider (NPI), engineered summary
features, ran an Isolation Forest, and combined that signal with two
domain-informed extreme-pattern indicators (composite risk and peer
deviation) into a conservative weak-supervision label called
weak_label_high_audit_priority.

In this notebook we treat that weak label as the target variable and
train three supervised models:
  1. Logistic Regression baseline (interpretable, fast)
  2. XGBoost (gradient boosted trees) if installed
  3. LightGBM (gradient boosted trees) if installed

We then evaluate using metrics that match how an audit-priority worklist
is actually used in practice — top-K precision, recall, and lift —
because picking which providers to review is a ranking problem, not a
fixed-threshold classification problem.

Important Limitation:
The model in this notebook is learning the weak-supervision label from
Notebook 02. The target column weak_label_high_audit_priority is NOT a
fraud label, NOT an overpayment label, NOT a noncompliance label, and
NOT an audit finding. It is a research signal derived from public CMS
data patterns. Predictions from this notebook are decision-support
outputs intended for audit-priority research and portfolio modeling.
Any provider ranked highly here would still require independent
investigation before any real-world conclusion.
"""

# %%
# Step 1 — Environment Setup and Imports
#
# Business Purpose:
# Load the libraries and project paths needed to read the provider-level
# labeled dataset from Notebook 02 and train supervised models.
#
# Plain-English Logic:
# We resolve PROJECT_ROOT defensively so the notebook works whether it is
# executed as a script (__file__ defined) or interactively in Jupyter
# (__file__ undefined). Then we add src/ to sys.path and import shared
# project paths from config.
#
# Expected Output:
# Printed PROJECT_ROOT and the year being modeled.
#
# Why It Matters:
# Reproducibility starts with a clean import path. If PROJECT_ROOT is
# wrong, every downstream cell will silently load the wrong file.
#
# What to Check Before Moving On:
# Confirm PROJECT_ROOT points at medicare-claims-audit-intelligence and
# YEAR matches the parquet file produced by Notebook 02.

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
# Step 2 — Load Provider-Level Labeled Dataset
#
# Business Purpose:
# Load the provider-level table that Notebook 02 produced — features,
# anomaly scores, extreme-risk indicators, and the weak label.
#
# Plain-English Logic:
# Notebook 02 saved one row per NPI to:
#   data/processed/provider_features_labeled_<YEAR>_full.parquet
# We read it back here. We also print the prevalence of the target so the
# class balance is visible before any modeling decisions.
#
# Full-Run Note:
# This full run uses the complete provider-level labeled table created
# from 9,755,427 provider-service rows and 1,148,873 providers. It
# replaces the earlier 500,000-row development sample metrics.
#
# Expected Output:
# - Dataframe shape (rows, columns)
# - Target value counts
# - Target percentage
#
# Why It Matters:
# The weak label is heavily imbalanced (a small minority of providers are
# labeled positive by design). Knowing the prevalence upfront drives the
# choice of evaluation metrics (Average Precision over plain Accuracy)
# and class-weighting strategy in each model.
#
# What to Check Before Moving On:
# Confirm the target column exists, has both 0s and 1s, and that the
# positive rate matches what Notebook 02's Step 13 reported.

labeled_path = PATHS["data_processed"] / f"provider_features_labeled_{YEAR}_full.parquet"

print("Loading:", labeled_path)

provider_df = pd.read_parquet(labeled_path)

print("Shape:", provider_df.shape)

target_col = "weak_label_high_audit_priority"

target_counts = provider_df[target_col].value_counts().sort_index()
target_pct = (target_counts / target_counts.sum() * 100).round(3)

print("\nTarget distribution:")
print(target_counts)
print("\nTarget percentage:")
print(target_pct)

# %%
# Step 3 — Define Target and Exclude Leakage Columns
#
# Business Purpose:
# Build a clean feature matrix X and target vector y by removing columns
# that would leak the answer to the model.
#
# Plain-English Logic:
# Notebook 02 constructed weak_label_high_audit_priority from three
# inputs:
#   - iso_anomaly_flag and iso_anomaly_score
#   - extreme_composite_risk (built from feat_composite_risk_max)
#   - extreme_peer_deviation
# If we leave any of those in X, the supervised model will simply
# memorize the rule that defines the label. The model would look great on
# paper and learn nothing useful.
#
# We also drop:
#   - in_leie (an external reference signal, not a model feature)
#   - identifier columns (Rndrng_NPI, provider_type, provider_state,
#     provider_entity_type) — categorical strings used for interpretation,
#     not modeling
#   - all columns starting with "feat_composite_risk" — composite risk
#     was a heavy ingredient in the weak label, so excluding the raw
#     composite_risk summaries reduces target leakage
#
# After exclusions we keep only numeric columns. Tree-based models can
# technically use object dtypes if encoded, but the project convention
# is to keep modeling simple and interpretable: numeric features only.
#
# Expected Output:
# - Number of features used
# - Number of leakage columns excluded
# - X.shape and y.shape
#
# Why It Matters:
# Target leakage is the most common silent bug in supervised learning on
# weak labels. The label-construction columns are easy to forget because
# they look like ordinary features. Excluding them explicitly here keeps
# the metrics in Step 9 honest.
#
# What to Check Before Moving On:
# - feat_composite_risk_* columns appear in the excluded list (not in X)
# - iso_anomaly_flag, iso_anomaly_score, extreme_composite_risk,
#   extreme_peer_deviation are excluded
# - X is entirely numeric

leakage_or_id_cols = [
    target_col,
    "iso_anomaly_flag",
    "iso_anomaly_score",
    "extreme_composite_risk",
    "extreme_peer_deviation",
    "in_leie",
    "Rndrng_NPI",
    "provider_type",
    "provider_state",
    "provider_entity_type",
]

composite_risk_cols = [
    col for col in provider_df.columns
    if col.startswith("feat_composite_risk")
]

excluded_cols = sorted(set(leakage_or_id_cols) | set(composite_risk_cols))

candidate_cols = [
    col for col in provider_df.columns
    if col not in excluded_cols
]

feature_cols = [
    col for col in candidate_cols
    if pd.api.types.is_numeric_dtype(provider_df[col])
]

X = provider_df[feature_cols].copy()
y = provider_df[target_col].astype(int).copy()

# Final NaN safety net — Notebook 02 already imputed but a saved/loaded
# parquet round-trip can occasionally re-introduce NaNs in derived cols.
X = X.fillna(X.median(numeric_only=True)).fillna(0)

print("Features used:", len(feature_cols))
print("Excluded columns (leakage / id):")
for col in excluded_cols:
    if col in provider_df.columns:
        print(" -", col)

print("\nX shape:", X.shape)
print("y shape:", y.shape)
print("Positive class rate:", round(float(y.mean()), 5))

# %%
# Step 4 — Train/Test Split
#
# Business Purpose:
# Hold out a stratified 20% test set so every model is evaluated on
# providers it has never seen.
#
# Plain-English Logic:
# We use sklearn's train_test_split with stratify=y so the positive rate
# is preserved across the train and test sets. random_state=42 keeps the
# split reproducible.
#
# Expected Output:
# - X_train.shape, X_test.shape
# - Train target rate, test target rate (should be nearly identical)
#
# Why It Matters:
# With a heavily imbalanced label, an unstratified split can produce a
# test set with very few positives, making metrics noisy. Stratification
# guarantees a stable evaluation baseline.
#
# What to Check Before Moving On:
# Train and test target rates should be within ~0.001 of each other.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train target rate:", round(float(y_train.mean()), 5))
print("Test target rate:", round(float(y_test.mean()), 5))

# %%
# Step 5 — Logistic Regression Baseline
#
# Business Purpose:
# Train an interpretable, well-understood baseline that any later model
# must outperform to justify added complexity.
#
# Plain-English Logic:
# Logistic Regression on a high-dimensional feature set needs the inputs
# on a common scale, so we wrap StandardScaler and LogisticRegression in
# a Pipeline. class_weight="balanced" tells the model to up-weight the
# minority positive class so it does not collapse into "always predict
# 0." max_iter=1000 gives the solver enough room to converge.
#
# Expected Output:
# A fitted lr_pipeline plus its predicted probabilities on the test set.
#
# Why It Matters:
# A baseline anchors interpretation. If a much fancier gradient boosted
# model only barely beats Logistic Regression, the extra complexity is
# probably not earning its keep on this dataset.
#
# What to Check Before Moving On:
# The pipeline fits without warnings about non-convergence and produces
# probability arrays of length len(X_test).

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

lr_pipeline.fit(X_train, y_train)

lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]

print("Logistic Regression baseline trained.")
print("lr_probs shape:", lr_probs.shape)
print("lr_probs min/max:", round(float(lr_probs.min()), 4), "/",
      round(float(lr_probs.max()), 4))

# %%
# Step 6 — XGBoost Model (defensive)
#
# Business Purpose:
# Train an XGBoost classifier that can capture non-linear interactions
# between provider features.
#
# Plain-English Logic:
# We start conservatively for the first notebook:
#   - n_estimators=300
#   - max_depth=4
#   - learning_rate=0.05
#   - subsample=0.8 / colsample_bytree=0.8
#   - eval_metric="aucpr" (PR-AUC matches our imbalanced-target focus)
#   - tree_method="hist" (CPU-friendly histogram learner; portable)
#   - scale_pos_weight = negative_count / positive_count
#     (mathematical equivalent of class_weight="balanced" for XGBoost)
# We do NOT force GPU. Portability across machines matters more than
# training speed at this stage.
#
# We import XGBoost inside a try/except so this notebook still runs end
# to end on a machine where xgboost is not installed.
#
# Expected Output:
# A fitted XGBoost classifier and predicted probabilities on the test
# set, OR a clear printed note that XGBoost is unavailable.
#
# Why It Matters:
# Tree-based models often outperform Logistic Regression on tabular
# data with non-linear effects, but only if their hyperparameters are
# reasonable. A conservative first model is also faster to train and
# easier to reason about.
#
# What to Check Before Moving On:
# Either xgb_probs has shape (len(X_test),) and looks like real
# probabilities, or the printed note explains why XGBoost is skipped.

xgb_model = None
xgb_probs = None

try:
    from xgboost import XGBClassifier

    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    scale_pos_weight = neg_count / max(pos_count, 1)

    # GPU portability note:
    # This notebook intentionally uses CPU-compatible tree_method="hist"
    # so the portfolio runs reproducibly on machines without a GPU.
    # A later performance-tuning notebook can enable GPU acceleration
    # (e.g. tree_method="gpu_hist" or device="cuda") on compatible
    # hardware. We do not force GPU here because portability across
    # reviewer machines matters more than training speed for a
    # baseline-modeling notebook.
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
    )

    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    print("XGBoost trained.")
    print("scale_pos_weight:", round(scale_pos_weight, 3))
    print("xgb_probs shape:", xgb_probs.shape)
except ImportError:
    print(
        "Note: xgboost is not installed. Skipping XGBoost. "
        "Install with `pip install xgboost` to enable this model."
    )
except Exception as exc:
    print("Note: XGBoost training failed with:", repr(exc))
    xgb_model = None
    xgb_probs = None

# %%
# Step 7 — LightGBM Model (defensive)
#
# Business Purpose:
# Train a LightGBM classifier as a second gradient-boosted comparison.
# LightGBM uses a leaf-wise growth strategy that often differs in
# behavior from XGBoost's level-wise default, so the two together give a
# fuller picture of what tree-based models can do here.
#
# Plain-English Logic:
# We use:
#   - n_estimators=300
#   - num_leaves=31
#   - learning_rate=0.05
#   - subsample=0.8 / colsample_bytree=0.8
#   - class_weight="balanced"
# Like Step 6, we wrap the import in try/except so the notebook still
# runs end-to-end if lightgbm is not installed.
#
# Expected Output:
# A fitted LightGBM classifier and predicted probabilities on the test
# set, OR a clear printed note that LightGBM is unavailable.
#
# Why It Matters:
# Comparing two boosted-tree implementations helps detect whether any
# specific model's quirks are driving results, versus the pattern being
# a property of the data and label.
#
# What to Check Before Moving On:
# Either lgb_probs has shape (len(X_test),) or the skip note prints.

lgb_model = None
lgb_probs = None

try:
    from lightgbm import LGBMClassifier

    # num_leaves note:
    # We use num_leaves=31 as a conservative first-pass LightGBM setting
    # for this baseline notebook. A later performance-tuning notebook can
    # compare more aggressive values against the settings in
    # src/config.py. The goal of Notebook 03 is leakage-safe baseline
    # modeling, not final hyperparameter optimization, so we keep this
    # value stable for now to make run-to-run comparisons clean.
    lgb_model = LGBMClassifier(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    lgb_model.fit(X_train, y_train)
    lgb_probs = lgb_model.predict_proba(X_test)[:, 1]

    print("LightGBM trained.")
    print("lgb_probs shape:", lgb_probs.shape)
except ImportError:
    print(
        "Note: lightgbm is not installed. Skipping LightGBM. "
        "Install with `pip install lightgbm` to enable this model."
    )
except Exception as exc:
    print("Note: LightGBM training failed with:", repr(exc))
    lgb_model = None
    lgb_probs = None

# %%
# Step 8 — Reusable Evaluation Function
#
# Business Purpose:
# Provide one function that scores any model with metrics that match how
# an audit-priority worklist actually gets used: top-K ranking metrics,
# not a fixed-threshold confusion matrix.
#
# Plain-English Logic:
# For each model we compute:
#   - Average Precision (PR-AUC)   — overall ranking quality on a heavily
#                                    imbalanced label
#   - ROC AUC                       — alternate ranking metric, less
#                                    sensitive to imbalance
#   - Precision@K, Recall@K, Lift@K for K in {100, 500, 1000}
#       Precision@K = of the top-K ranked providers, what fraction are
#                     positive in the held-out test set?
#       Recall@K    = of all positives in the test set, what fraction
#                     appear in the top K ranks?
#       Lift@K      = Precision@K / overall positive rate, i.e. how many
#                     times better than random the top K is
# We also print the confusion matrix at threshold 0.5 — but only as a
# reference, with an explicit note that for audit-priority work the
# top-K metrics are more relevant. A reviewer's worklist is "look at the
# top N providers," not "look at everyone above 0.5."
#
# Expected Output:
# A dict with all metrics, suitable for collection into a comparison
# dataframe in Step 9.
#
# Why It Matters:
# Accuracy on an imbalanced dataset is misleading (always-predict-0
# already scores 95%+). PR-AUC and top-K metrics cut through that and
# describe the ranking quality directly.
#
# What to Check Before Moving On:
# Test the function on lr_probs once; metrics should be finite numbers
# (no NaN, no inf) and Precision@K should fall within [0, 1].

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)


def evaluate_model(name, y_true, y_score, ks=(100, 500, 1000), threshold=0.5):
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)

    metrics = {
        "model": name,
        "n_test": int(len(y_true_arr)),
        "n_positives_test": int(y_true_arr.sum()),
        "average_precision": float(average_precision_score(y_true_arr, y_score_arr)),
        "roc_auc": float(roc_auc_score(y_true_arr, y_score_arr)),
    }

    overall_rate = y_true_arr.mean() if len(y_true_arr) else 0.0
    order = np.argsort(-y_score_arr)
    y_sorted = y_true_arr[order]

    total_positives = int(y_true_arr.sum())

    for k in ks:
        k_eff = min(k, len(y_sorted))
        if k_eff == 0:
            metrics[f"precision_at_{k}"] = float("nan")
            metrics[f"recall_at_{k}"] = float("nan")
            metrics[f"lift_at_{k}"] = float("nan")
            continue
        top_k_positives = int(y_sorted[:k_eff].sum())
        precision_k = top_k_positives / k_eff
        recall_k = (top_k_positives / total_positives) if total_positives else float("nan")
        lift_k = (precision_k / overall_rate) if overall_rate else float("nan")
        metrics[f"precision_at_{k}"] = float(precision_k)
        metrics[f"recall_at_{k}"] = float(recall_k)
        metrics[f"lift_at_{k}"] = float(lift_k)

    y_pred_threshold = (y_score_arr >= threshold).astype(int)
    cm = confusion_matrix(y_true_arr, y_pred_threshold, labels=[0, 1])

    print(f"--- {name} ---")
    print("Average Precision (PR-AUC):", round(metrics["average_precision"], 4))
    print("ROC AUC:", round(metrics["roc_auc"], 4))
    for k in ks:
        print(
            f"Precision@{k}: {metrics[f'precision_at_{k}']:.4f}  "
            f"Recall@{k}: {metrics[f'recall_at_{k}']:.4f}  "
            f"Lift@{k}: {metrics[f'lift_at_{k}']:.2f}x"
        )
    print(f"Confusion matrix at threshold={threshold} (rows=true, cols=pred):")
    print(pd.DataFrame(
        cm,
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    ))
    print(
        "Note: for audit-priority research, top-K precision/recall/lift "
        "are more relevant than a fixed-threshold confusion matrix. "
        "Reviewers work from a ranked worklist, not a 0.5 cutoff."
    )
    print()

    return metrics


# Quick sanity test on the baseline so the function is exercised once.
_ = evaluate_model("Logistic Regression (sanity)", y_test, lr_probs)

# %%
# Step 9 — Compare Models
#
# Business Purpose:
# Collect every available model's metrics into one dataframe so we can
# pick the best performer by Average Precision.
#
# Plain-English Logic:
# Each fitted model contributes a row to the comparison table. Models
# that were skipped (xgboost or lightgbm not installed) are absent —
# defensive design from Steps 6 and 7. We sort by Average Precision
# descending because PR-AUC is the most appropriate single ranking
# metric for our heavily imbalanced label.
#
# Expected Output:
# A results dataframe with one row per available model, sorted by
# Average Precision.
#
# Why It Matters:
# Picking a "best" model from a single number can hide tradeoffs. The
# full table also shows ROC AUC and top-K metrics so a reader can see
# whether the winner is uniformly better or only better at certain
# operating points.
#
# What to Check Before Moving On:
# - The dataframe contains every successfully trained model
# - Average Precision values are between 0 and 1
# - Lift@K values are >= 1 for any model better than random ranking

results_rows = []

results_rows.append(evaluate_model("Logistic Regression", y_test, lr_probs))

if xgb_probs is not None:
    results_rows.append(evaluate_model("XGBoost", y_test, xgb_probs))

if lgb_probs is not None:
    results_rows.append(evaluate_model("LightGBM", y_test, lgb_probs))

results_df = (
    pd.DataFrame(results_rows)
    .sort_values("average_precision", ascending=False)
    .reset_index(drop=True)
)

results_df

# %%
# Step 10 — Plot Precision-Recall Curve
#
# Business Purpose:
# Visualize the precision-recall tradeoff for every available model on
# one chart so a reader can see how each model behaves across thresholds.
#
# Plain-English Logic:
# Average Precision summarizes the area under the PR curve as a single
# number; the curve itself shows where each model is precise but
# low-recall (top-left) versus where it covers most positives but with
# more false positives (bottom-right). Audit-priority worklists usually
# care about the top-left corner — high precision in the highest-ranked
# slice — so the shape of the curve there matters more than the overall
# AP value alone.
#
# Expected Output:
# A figure saved to:
#   reports/notebook03_precision_recall_curve.png
#
# Why It Matters:
# A flat tail and a steep top-left mean a model is great for short
# review lists. A model with similar AP but a smoother curve might be
# better for longer worklists. The chart makes that tradeoff visible.

from sklearn.metrics import precision_recall_curve

fig, ax = plt.subplots(figsize=(10, 7))

model_probs = [("Logistic Regression", lr_probs)]
if xgb_probs is not None:
    model_probs.append(("XGBoost", xgb_probs))
if lgb_probs is not None:
    model_probs.append(("LightGBM", lgb_probs))

for name, probs in model_probs:
    precision, recall, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    ax.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.3f})")

baseline = float(y_test.mean())
ax.axhline(
    baseline,
    color="black",
    linestyle="--",
    linewidth=1.0,
    label=f"Baseline positive rate = {baseline:.3f}",
)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve — Audit-Priority Weak Label (Test Set)")
ax.legend(loc="upper right")

plt.tight_layout()

pr_curve_path = PATHS["reports"] / "notebook03_precision_recall_curve.png"
plt.savefig(pr_curve_path, dpi=150, bbox_inches="tight")
plt.show()

print("Saved:", pr_curve_path)

# %%
# Step 11 — Plot Top 20 Feature Importances
#
# Business Purpose:
# Show which features each tree-based model relied on most. This is the
# first glimpse of "why is the model ranking these providers high?"
#
# Plain-English Logic:
# XGBoost and LightGBM expose a feature_importances_ array aligned with
# the input columns. We rank features by importance and bar-chart the
# top 20 for whichever models are available. We do not interpret these
# as causal — feature importances describe what the model used, not why
# any individual provider was flagged.
#
# Expected Output:
# Up to two figures saved to:
#   reports/notebook03_xgboost_feature_importance.png
#   reports/notebook03_lightgbm_feature_importance.png
#
# Why It Matters:
# A model that relies almost entirely on identity-correlated features
# (e.g. counts that proxy for provider type) is suspect. A model that
# spreads importance across multiple peer-deviation, charge-ratio, and
# concentration features is using the engineered audit-priority signals
# the project was designed to surface.


def plot_top_feature_importance(model, name, feature_names, save_path, top_n=20):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        print(f"Note: {name} has no feature_importances_, skipping.")
        return

    imp_df = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        })
        .sort_values("importance", ascending=False)
        .head(top_n)
        .iloc[::-1]
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(imp_df["feature"], imp_df["importance"], color="#1a5276", alpha=0.85)
    ax.set_xlabel("Feature importance")
    ax.set_title(f"{name} — Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", save_path)


if xgb_model is not None:
    plot_top_feature_importance(
        xgb_model,
        "XGBoost",
        feature_cols,
        PATHS["reports"] / "notebook03_xgboost_feature_importance.png",
    )
else:
    print("Note: XGBoost model unavailable; skipping its importance plot.")

if lgb_model is not None:
    plot_top_feature_importance(
        lgb_model,
        "LightGBM",
        feature_cols,
        PATHS["reports"] / "notebook03_lightgbm_feature_importance.png",
    )
else:
    print("Note: LightGBM model unavailable; skipping its importance plot.")

# %%
# Step 12 — Save Prediction Output
#
# Business Purpose:
# Persist a per-provider prediction file so downstream notebooks (e.g.
# financial simulation, dashboarding) can pick up exactly the rankings
# this notebook produced.
#
# Plain-English Logic:
# We rebuild a prediction frame for the held-out test set, keyed by the
# same provider identifiers used elsewhere in the project:
#   - Rndrng_NPI
#   - provider_type
#   - provider_state
#   - y_true (the weak-supervision target)
#   - probability columns for each available model
# We save as parquet for dtype preservation and fast columnar reads.
#
# Expected Output:
# A parquet file at:
#   data/processed/model_predictions_<YEAR>_full.parquet
#
# Why It Matters:
# Saving the predictions decouples scoring from training. A future
# notebook can rank providers, simulate review costs, or compare
# rankings across years without retraining anything.

test_index = X_test.index

# Index alignment safety check:
# train_test_split preserves the original pandas index on X_test and
# y_test, so X_test.index points back at exactly the same rows in
# provider_df that y_test came from. The assertion below is a cheap
# guard that confirms the held-out provider rows still align with the
# y_test labels before we glue probabilities onto identifiers.
assert len(test_index) == len(y_test), (
    "Index length mismatch between test features and target."
)

predictions_df = provider_df.loc[test_index, [
    "Rndrng_NPI",
    "provider_type",
    "provider_state",
]].copy()

predictions_df["y_true"] = y_test.values
predictions_df["lr_proba"] = lr_probs

if xgb_probs is not None:
    predictions_df["xgb_proba"] = xgb_probs
if lgb_probs is not None:
    predictions_df["lgb_proba"] = lgb_probs

predictions_path = PATHS["data_processed"] / f"model_predictions_{YEAR}_full.parquet"
predictions_df.to_parquet(predictions_path, engine="pyarrow", index=False)

print("Saved:", predictions_path)
print("Rows saved:", f"{len(predictions_df):,}")
print("Columns saved:", list(predictions_df.columns))

# %%
# Step 13 — Full-Population Audit-Priority Scoring
#
# Business Purpose:
# Produce one model-predicted audit-priority score per provider for the
# entire 2022 Medicare Part B population, not just the held-out test
# set. Downstream artifacts (financial review-cost simulations, state
# and specialty rollups, dashboards) need scores for every provider in
# the file, including those used to fit the model.
#
# Plain-English Logic:
# We call predict_proba on the full feature matrix X for each model
# that successfully trained:
#   - Logistic Regression (always available)
#   - XGBoost (if installed and trained)
#   - LightGBM (if installed and trained)
# We then attach those probabilities to the provider identifier
# columns (Rndrng_NPI, provider_type, provider_state) and the weak
# label, and persist the result as parquet for downstream use.
#
# Important Framing:
# These are model-predicted audit-priority scores, not fraud scores,
# overpayment scores, noncompliance findings, or audit findings. They
# are research signals derived from public CMS Medicare Part B data
# and the weak-supervision label from Notebook 02.
#
# Note on in-sample scoring:
# Because X includes rows the models trained on, scores for those
# specific providers are in-sample and will look optimistic relative
# to held-out performance. The honest performance numbers are the
# test-set metrics in Step 9 / Step 12. The full-population file
# exists to provide a complete worklist, not to re-evaluate accuracy.
#
# Expected Output:
# A parquet file at:
#   data/processed/model_predictions_full_population_<YEAR>_full.parquet
# with columns:
#   Rndrng_NPI, provider_type, provider_state,
#   weak_label_high_audit_priority,
#   lr_full_probability,
#   xgb_full_probability (if XGBoost trained),
#   lgb_full_probability (if LightGBM trained)
#
# Why It Matters:
# The test-set prediction file (Step 12) is for model evaluation.
# Downstream simulations and rollups need a score for every provider,
# which is what this step produces. Keeping the two files separate
# preserves a clean evaluation artifact while still supporting full
# population analytics.
#
# What to Check Before Moving On:
# - Row count equals provider_df row count
# - All probability columns are within [0, 1]
# - Identifier columns match provider_df exactly (no row reordering)

full_population_df = provider_df[[
    "Rndrng_NPI",
    "provider_type",
    "provider_state",
    target_col,
]].copy()

full_population_df["lr_full_probability"] = lr_pipeline.predict_proba(X)[:, 1]

if xgb_model is not None:
    full_population_df["xgb_full_probability"] = xgb_model.predict_proba(X)[:, 1]

if lgb_model is not None:
    full_population_df["lgb_full_probability"] = lgb_model.predict_proba(X)[:, 1]

full_population_path = (
    PATHS["data_processed"] / f"model_predictions_full_population_{YEAR}_full.parquet"
)
full_population_df.to_parquet(full_population_path, engine="pyarrow", index=False)

print("Saved:", full_population_path)
print("Rows saved:", f"{len(full_population_df):,}")
print("Columns saved:", list(full_population_df.columns))
print(
    "Reminder: these are model-predicted audit-priority scores, not "
    "fraud, overpayment, noncompliance, or audit-finding scores."
)

# %%
# Step 14 — Save Best Model and Metadata
#
# Business Purpose:
# Persist the single best-performing model from Step 9's comparison so
# downstream code (scoring scripts, simulation notebooks, future
# notebooks) can load it without retraining.
#
# Plain-English Logic:
# We pick the row at the top of results_df (already sorted by Average
# Precision descending in Step 9) and serialize the matching fitted
# estimator with joblib:
#   - Logistic Regression  -> save lr_pipeline (the full Pipeline so
#                             the StandardScaler is included)
#   - XGBoost              -> save xgb_model
#   - LightGBM             -> save lgb_model
# Alongside the model we save a small JSON metadata file describing
# the target column, feature list, evaluation metrics, and an explicit
# note that the target is a weak-supervision audit-priority label —
# not a fraud, overpayment, noncompliance, or audit finding label.
#
# Expected Output:
# - models/best_model_<YEAR>_full.joblib
# - models/best_model_<YEAR>_full_metadata.json
#
# Why It Matters:
# Saving the model decouples training from scoring. Later work can
# reload the exact estimator that produced the reported metrics
# without rerunning any of the training cells. The metadata file
# preserves the framing language so anyone reusing the model is
# reminded of what the target really is.
#
# What to Check Before Moving On:
# - The joblib file exists and is non-empty
# - The metadata JSON lists the same number of features used in
#   training (len(feature_cols))
# - best_model_name matches results_df.iloc[0]["model"]

import json
import joblib

PATHS["models"].mkdir(parents=True, exist_ok=True)

best_row_for_save = results_df.iloc[0]
best_model_name = str(best_row_for_save["model"])

if best_model_name == "Logistic Regression":
    best_estimator = lr_pipeline
elif best_model_name == "XGBoost":
    best_estimator = xgb_model
elif best_model_name == "LightGBM":
    best_estimator = lgb_model
else:
    best_estimator = lr_pipeline

best_model_path = PATHS["models"] / f"best_model_{YEAR}_full.joblib"
joblib.dump(best_estimator, best_model_path)

best_model_metadata = {
    "best_model_name": best_model_name,
    "target_column": target_col,
    "number_of_features": int(len(feature_cols)),
    "feature_columns": list(feature_cols),
    "target_prevalence": float(y.mean()),
    "average_precision": float(best_row_for_save["average_precision"]),
    "roc_auc": float(best_row_for_save["roc_auc"]),
    "note": (
        "The target is a weak-supervision audit-priority label, not a "
        "fraud, overpayment, noncompliance, or audit finding label."
    ),
}

best_model_metadata_path = PATHS["models"] / f"best_model_{YEAR}_full_metadata.json"
with open(best_model_metadata_path, "w", encoding="utf-8") as f:
    json.dump(best_model_metadata, f, indent=2)

print("Saved best model     :", best_model_path)
print("Saved best metadata  :", best_model_metadata_path)
print("Best model name      :", best_model_name)
print("Average Precision    :", round(best_model_metadata["average_precision"], 4))
print("ROC AUC              :", round(best_model_metadata["roc_auc"], 4))

# %%
# Step 15 — Notebook Summary
#
# Business Purpose:
# Print a short, copy-pasteable summary of the run so the notebook ends
# with a clear, auditable record of what was produced.
#
# Plain-English Logic:
# We surface:
#   - the best model by Average Precision
#   - the number of features used
#   - the target prevalence (full dataset and test set)
#   - the saved prediction path
#
# Expected Output:
# Plain printed text. No file writes here.
#
# Why It Matters:
# A consistent end-of-notebook summary makes it easier to compare runs
# across versions or years, and to lift key numbers into a write-up.

best_row = results_df.iloc[0]

print("=" * 60)
print("Notebook 03 Summary — FULL-DATA RUN")
print("=" * 60)
print(
    "Run type                        : FULL provider-level labeled dataset"
)
print(
    "Source                          : "
    "provider_features_labeled_<YEAR>_full.parquet "
    "(built from 9,755,427 provider-service rows / 1,148,873 providers)"
)
print(
    "Replaces                        : earlier 500,000-row development "
    "sample metrics"
)
print("-" * 60)
print(f"Best model by Average Precision : {best_row['model']}")
print(f"  Average Precision (PR-AUC)    : {best_row['average_precision']:.4f}")
print(f"  ROC AUC                       : {best_row['roc_auc']:.4f}")
print(f"  Precision@100                 : {best_row['precision_at_100']:.4f}")
print(f"  Lift@100                      : {best_row['lift_at_100']:.2f}x")
print(f"Features used                   : {len(feature_cols)}")
print(f"Target prevalence (all rows)    : {float(y.mean()):.5f}")
print(f"Target prevalence (test set)    : {float(y_test.mean()):.5f}")
print(f"Test predictions saved to       : {predictions_path}")
print(f"Full-population scores saved to : {full_population_path}")
print(f"Best model saved to             : {best_model_path}")
print(f"Best-model metadata saved to    : {best_model_metadata_path}")
print("=" * 60)
print(
    "Reminder: weak_label_high_audit_priority is a weak-supervision "
    "label and a review signal derived from public CMS data. The "
    "model output is decision support for audit-priority research. "
    "Treat predictions as a ranked review signal, not a finding."
)

# %%
