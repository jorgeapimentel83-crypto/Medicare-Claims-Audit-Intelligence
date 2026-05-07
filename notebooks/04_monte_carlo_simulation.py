# %%
"""
Notebook 04 — Monte Carlo Audit-Capacity and Recovery-Potential Simulation

Business Purpose:
This notebook turns the full-population audit-priority scores from
Notebook 03 into a decision-support view for audit-capacity planning.
It simulates how prioritization tradeoffs behave under different
review-capacity assumptions and how illustrative recovery-potential
ranges respond to that prioritization.

Plain-English Logic:
Notebook 03 produced one model-predicted audit-priority score per
provider for the full 2022 Medicare Part B provider population. In a
real audit program, reviewers can only look at a finite number of
providers per period, so the practical question is:
    "If we can review the top N providers, what does that buy us?"

This notebook answers that question with three layers:
  1. Audit-capacity scenarios — Top 100 / 500 / 1,000 / 5,000 / 10,000.
     For each scenario we measure weak-label precision, recall, and
     lift versus a random worklist.
  2. Illustrative recovery-potential assumptions — three deterministic
     per-provider amounts (conservative / moderate / high). These are
     scenario inputs the user controls, not estimates of real recoveries.
  3. Monte Carlo simulation — for each capacity scenario we draw a
     triangular distribution per weak-label hit and roll up the
     simulated totals across 5,000 trials. We report mean, median,
     p10, and p90 to express uncertainty as a range, not a point.

Important Limitation:
This notebook does NOT estimate confirmed recoveries, fraud,
overpayments, noncompliance, or audit findings. The numbers it
produces are illustrative scenario-based recovery-potential ranges
built from public CMS data, model-predicted audit-priority scores,
and analyst-defined assumption inputs. Any provider in any of these
worklists would still require independent investigation before any
real-world conclusion. Treat every output as decision support for
audit-priority research and capacity planning, not as a finding.
"""

# %%
# Step 1 — Environment Setup and Imports
#
# Business Purpose:
# Load the libraries and project paths needed to read the
# full-population prediction file from Notebook 03 and run the
# capacity / Monte Carlo simulations.
#
# Plain-English Logic:
# We resolve PROJECT_ROOT defensively so the notebook works whether it
# is executed as a script (__file__ defined) or interactively in
# Jupyter (__file__ undefined). Then we add src/ to sys.path and import
# shared project paths from config.
#
# Expected Output:
# Printed PROJECT_ROOT and the year being simulated.
#
# Why It Matters:
# Reproducibility starts with a clean import path. If PROJECT_ROOT is
# wrong, every downstream cell will silently load the wrong file.
#
# What to Check Before Moving On:
# Confirm PROJECT_ROOT points at medicare-claims-audit-intelligence
# and YEAR matches the prediction file produced by Notebook 03.

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
# Step 2 — Load Full-Population Prediction File
#
# Business Purpose:
# Load the per-provider model scores Notebook 03 produced for the
# entire 2022 Medicare Part B provider population. Every downstream
# step in this notebook works off this single table.
#
# Plain-English Logic:
# Notebook 03 saved one row per NPI to:
#   data/processed/model_predictions_full_population_<YEAR>_full.parquet
# We read it back here. We print its shape, columns, the weak-label
# distribution, and which model probability columns are available so
# the next step can pick a score deterministically.
#
# Expected Output:
# - Dataframe shape (rows, columns)
# - Column list
# - Weak-label value counts
# - Available *_full_probability columns
#
# Why It Matters:
# Notebook 03 only writes a model probability column when the
# corresponding model trained successfully. Listing what is actually
# present here makes the score selection in Step 4 deterministic and
# transparent.
#
# What to Check Before Moving On:
# - File loads without error
# - weak_label_high_audit_priority has both 0s and 1s
# - At least one of {lgb_full_probability, xgb_full_probability,
#   lr_full_probability} is present

predictions_path = (
    PATHS["data_processed"]
    / f"model_predictions_full_population_{YEAR}_full.parquet"
)

print("Loading:", predictions_path)

predictions_df = pd.read_parquet(predictions_path)

print("Shape:", predictions_df.shape)
print("Columns:", list(predictions_df.columns))

target_col = "weak_label_high_audit_priority"

print("\nWeak-label distribution:")
print(predictions_df[target_col].value_counts().sort_index())
print(
    "Weak-label positive rate:",
    round(float(predictions_df[target_col].mean()), 5),
)

probability_cols = [
    col for col in predictions_df.columns
    if col.endswith("_full_probability")
]
print("\nAvailable model probability columns:", probability_cols)

# %%
# Step 3 — Select Model Score
#
# Business Purpose:
# Pick a single model-predicted audit-priority score column and rename
# it to risk_score so the rest of the notebook is model-agnostic.
#
# Plain-English Logic:
# Notebook 03's full-data run identified LightGBM as the best model by
# Average Precision, so we prefer lgb_full_probability. If that column
# is not present (e.g. LightGBM was skipped for some reason), we fall
# back to xgb_full_probability, then to lr_full_probability so the
# notebook still runs end to end.
#
# Important Framing:
# A higher risk_score means the model judged the provider's pattern
# more similar to the weak-label positives from Notebook 02. It does
# NOT mean the provider committed fraud, has overpayments, has any
# noncompliance, or has any audit finding. It is a research signal
# derived from public CMS data.
#
# Expected Output:
# - Printed name of the score column used
# - Printed min / mean / max of risk_score
#
# Why It Matters:
# Pinning the score column up front avoids accidental reuse of two
# different models inside the same simulation. The renamed risk_score
# column is what every downstream step references.
#
# What to Check Before Moving On:
# - score_col is one of the available columns from Step 2
# - risk_score is fully populated (no NaN) and bounded in [0, 1]

preferred_score_columns = [
    "lgb_full_probability",
    "xgb_full_probability",
    "lr_full_probability",
]

score_col = None
for candidate in preferred_score_columns:
    if candidate in predictions_df.columns:
        score_col = candidate
        break

if score_col is None:
    raise ValueError(
        "No model probability column found in the prediction file. "
        "Expected one of: " + ", ".join(preferred_score_columns)
    )

predictions_df["risk_score"] = predictions_df[score_col].astype(float)

print("Score column used :", score_col)
print(
    "risk_score min / mean / max:",
    round(float(predictions_df["risk_score"].min()), 6),
    "/",
    round(float(predictions_df["risk_score"].mean()), 6),
    "/",
    round(float(predictions_df["risk_score"].max()), 6),
)
print(
    "Reminder: higher risk_score = higher model-predicted audit "
    "priority. Not fraud, not overpayments, not findings."
)

# %%
# Step 4 — Rank Providers by Risk Score
#
# Business Purpose:
# Build a ranked worklist so audit-capacity scenarios are simply
# "take the top N rows."
#
# Plain-English Logic:
# We sort the prediction frame by risk_score descending so rank 1 is
# the highest model-predicted audit priority. We add risk_rank (1, 2,
# 3, ...) and risk_percentile (rank / total) so a reviewer can read the
# worklist either way: "this is the 47th-highest provider" or "this
# provider is in the top 0.004% of model-predicted priority."
#
# Expected Output:
# A printed top-10 preview of the ranked worklist with the columns:
#   Rndrng_NPI, provider_type, provider_state,
#   weak_label_high_audit_priority, risk_score, risk_rank
#
# Why It Matters:
# Every capacity scenario in Step 5 is just .head(N) on this ranked
# frame. Building it once here keeps the Monte Carlo simulation in
# Step 8 fast and consistent.
#
# What to Check Before Moving On:
# - The top-ranked row has the highest risk_score in the frame
# - risk_rank starts at 1 with no gaps
# - risk_percentile is in (0, 1]

ranked_df = (
    predictions_df
    .sort_values("risk_score", ascending=False)
    .reset_index(drop=True)
    .copy()
)

ranked_df["risk_rank"] = np.arange(1, len(ranked_df) + 1, dtype=np.int64)
ranked_df["risk_percentile"] = (
    ranked_df["risk_rank"] / float(len(ranked_df))
)

preview_cols = [
    "Rndrng_NPI",
    "provider_type",
    "provider_state",
    target_col,
    "risk_score",
    "risk_rank",
]

print("Top 10 ranked providers:")
print(ranked_df[preview_cols].head(10).to_string(index=False))

# %%
# Step 5 — Audit-Capacity Scenarios
#
# Business Purpose:
# Quantify how a finite review budget translates into weak-label
# precision, recall, and lift over a random worklist.
#
# Plain-English Logic:
# For each capacity K in {100, 500, 1,000, 5,000, 10,000} we take the
# top-K providers from the ranked frame and compute:
#   - providers_reviewed       = K (capped at the population)
#   - weak_label_hits          = positives caught in the top K
#   - weak_label_precision     = hits / K
#   - weak_label_recall        = hits / total weak-label positives
#   - lift_vs_random           = precision_at_K / overall positive rate
# A lift of 10x means that, at this capacity, the ranked worklist is
# ten times denser in weak-label positives than picking K providers
# at random would be.
#
# Expected Output:
# A capacity_scenarios_df dataframe with one row per scenario and the
# columns above, printed in full.
#
# Why It Matters:
# Audit programs almost never review every provider. The capacity
# table makes the prioritization tradeoff explicit: tiny lists are
# very precise but cover few positives; big lists cover more positives
# but the per-provider hit rate falls. This is the foundation for
# every recovery-potential calculation that follows.
#
# What to Check Before Moving On:
# - weak_label_precision is monotonically non-increasing as K grows
# - weak_label_recall is monotonically non-decreasing as K grows
# - lift_vs_random >= 1.0 at small K (otherwise the ranking is no
#   better than random)

audit_capacities = [100, 500, 1_000, 5_000, 10_000]

total_providers = int(len(ranked_df))
total_positives = int(ranked_df[target_col].sum())
overall_positive_rate = (
    float(total_positives) / float(total_providers)
    if total_providers else 0.0
)

scenario_rows = []
for k in audit_capacities:
    k_eff = min(k, total_providers)
    top_k = ranked_df.head(k_eff)
    hits = int(top_k[target_col].sum())
    precision_k = (hits / k_eff) if k_eff else float("nan")
    recall_k = (
        (hits / total_positives) if total_positives else float("nan")
    )
    lift_k = (
        (precision_k / overall_positive_rate)
        if overall_positive_rate else float("nan")
    )
    scenario_rows.append({
        "capacity_label": f"Top {k:,}",
        "providers_reviewed": int(k_eff),
        "weak_label_hits": hits,
        "weak_label_precision": float(precision_k),
        "weak_label_recall": float(recall_k),
        "lift_vs_random": float(lift_k),
    })

capacity_scenarios_df = pd.DataFrame(scenario_rows)

print("Audit-capacity scenarios:")
print(capacity_scenarios_df.to_string(index=False))
print(
    "\nReminder: weak_label_hits counts providers flagged by the "
    "weak-supervision label, not confirmed audit findings."
)

# %%
# Step 6 — Plot Audit-Capacity Tradeoff
#
# Business Purpose:
# Visualize how weak-label precision and recall move against each
# other as audit capacity grows.
#
# Plain-English Logic:
# We put providers_reviewed on the x-axis (log scale, since the
# scenarios span 100 -> 10,000) and overlay precision and recall on
# the y-axis. The reader can see both the precision drop and the
# recall climb in the same picture and pick a capacity that matches
# their program's tolerance.
#
# Expected Output:
# A figure saved to:
#   reports/notebook04_capacity_tradeoff.png
#
# Why It Matters:
# A static table answers "at K=500 what is precision?" but the plot
# makes the tradeoff shape obvious. If precision falls slowly and
# recall climbs fast, larger worklists are cheap; if precision
# collapses early, small worklists are the only sensible choice.

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    capacity_scenarios_df["providers_reviewed"],
    capacity_scenarios_df["weak_label_precision"],
    marker="o",
    linewidth=2,
    color="#1a5276",
    label="Weak-label precision",
)
ax.plot(
    capacity_scenarios_df["providers_reviewed"],
    capacity_scenarios_df["weak_label_recall"],
    marker="s",
    linewidth=2,
    color="#b9770e",
    label="Weak-label recall",
)

ax.set_xscale("log")
ax.set_xlabel("Providers reviewed (audit capacity)")
ax.set_ylabel("Rate")
ax.set_title(
    "Audit-Capacity Tradeoff — Weak-Label Precision vs Recall"
)
ax.set_ylim(0, 1.05)
ax.grid(True, which="both", linestyle=":", alpha=0.5)
ax.legend(loc="center right")

plt.tight_layout()

capacity_plot_path = (
    PATHS["reports"] / "notebook04_capacity_tradeoff.png"
)
plt.savefig(capacity_plot_path, dpi=150, bbox_inches="tight")
plt.show()

print("Saved:", capacity_plot_path)

# %%
# Step 7 — Illustrative Recovery-Potential Assumptions
#
# Business Purpose:
# Convert weak-label hits into illustrative dollar ranges using three
# analyst-controlled assumption levels.
#
# Plain-English Logic:
# We define three deterministic per-provider amounts:
#   - conservative : $2,500
#   - moderate     : $7,500
#   - high         : $15,000
# For each capacity scenario we multiply weak_label_hits by each
# amount. The result is a transparent, "if every weak-label hit
# returned $X, the program total would be ..." view.
#
# Important Framing:
# These are NOT estimated real recoveries. They are illustrative
# scenario assumptions the analyst sets and the notebook reports
# back. The Monte Carlo step that follows replaces these point
# numbers with a distribution to show uncertainty more honestly.
#
# Expected Output:
# A recovery_assumption_df dataframe with one row per capacity
# scenario and three illustrative recovery-potential columns.
#
# Why It Matters:
# Decision-makers think in dollars, not in lift. A simple deterministic
# table is the cleanest way to translate hit counts into program-scale
# ranges without claiming any modeling sophistication that does not
# exist.

recovery_assumptions = {
    "conservative_recovery_potential": 2_500.0,
    "moderate_recovery_potential": 7_500.0,
    "high_recovery_potential": 15_000.0,
}

recovery_assumption_df = capacity_scenarios_df[
    ["capacity_label", "providers_reviewed", "weak_label_hits"]
].copy()

for col_name, per_hit_amount in recovery_assumptions.items():
    recovery_assumption_df[col_name] = (
        recovery_assumption_df["weak_label_hits"].astype(float)
        * per_hit_amount
    )

print("Illustrative recovery-potential assumptions per weak-label hit:")
for col_name, amount in recovery_assumptions.items():
    print(f"  {col_name:<35} = ${amount:,.0f}")

print("\nIllustrative recovery-potential by audit-capacity scenario:")
print(recovery_assumption_df.to_string(index=False))
print(
    "\nReminder: these are illustrative recovery-potential assumptions, "
    "not actual recoveries or confirmed amounts."
)

# %%
# Step 8 — Monte Carlo Simulation
#
# Business Purpose:
# Replace the deterministic per-hit amounts from Step 7 with a
# distribution so each capacity scenario carries an uncertainty range
# instead of a single number.
#
# Plain-English Logic:
# For each capacity scenario we run 5,000 trials. In each trial we
# draw one triangular sample per weak-label hit:
#   - low  = $1,000
#   - mode = $7,500
#   - high = $25,000
# and sum across hits to get a simulated trial total. Across the
# 5,000 trial totals we report:
#   - simulated_recovery_potential_mean
#   - simulated_recovery_potential_median
#   - simulated_recovery_potential_p10
#   - simulated_recovery_potential_p90
#   - simulated_recovery_potential_min
#   - simulated_recovery_potential_max
# The triangular distribution is intentionally simple and easy to
# explain. It is a scenario input, not a fitted estimator.
#
# Important Framing:
# The output is a simulated_recovery_potential range, not an actual
# recovery estimate. The label is deliberate to keep the framing
# honest in any downstream chart or write-up.
#
# Expected Output:
# A monte_carlo_df dataframe with one row per capacity scenario and
# the simulated_recovery_potential_* summary columns.
#
# Why It Matters:
# Point estimates implied by Step 7 hide the fact that even granting
# the model and the weak label as correct, per-hit recovery potential
# is uncertain. The p10/p90 spread communicates that uncertainty in a
# format a decision-maker can read directly.
#
# What to Check Before Moving On:
# - p10 < median < p90 for every scenario
# - mean and median are reasonably close (triangular distributions are
#   not strongly skewed)
# - min, max bracket the p10 and p90 values

n_trials = 5_000
triangular_low = 1_000.0
triangular_mode = 7_500.0
triangular_high = 25_000.0

rng = np.random.default_rng(seed=42)

mc_rows = []
for _, scenario in capacity_scenarios_df.iterrows():
    capacity_label = scenario["capacity_label"]
    providers_reviewed = int(scenario["providers_reviewed"])
    hits = int(scenario["weak_label_hits"])

    if hits == 0:
        trial_totals = np.zeros(n_trials, dtype=float)
    else:
        samples = rng.triangular(
            left=triangular_low,
            mode=triangular_mode,
            right=triangular_high,
            size=(n_trials, hits),
        )
        trial_totals = samples.sum(axis=1)

    mc_rows.append({
        "capacity_label": capacity_label,
        "providers_reviewed": providers_reviewed,
        "weak_label_hits": hits,
        "n_trials": int(n_trials),
        "simulated_recovery_potential_mean": float(trial_totals.mean()),
        "simulated_recovery_potential_median": float(
            np.median(trial_totals)
        ),
        "simulated_recovery_potential_p10": float(
            np.percentile(trial_totals, 10)
        ),
        "simulated_recovery_potential_p90": float(
            np.percentile(trial_totals, 90)
        ),
        "simulated_recovery_potential_min": float(trial_totals.min()),
        "simulated_recovery_potential_max": float(trial_totals.max()),
    })

monte_carlo_df = pd.DataFrame(mc_rows)

print(
    f"Monte Carlo settings: n_trials={n_trials}, "
    f"triangular(low=${triangular_low:,.0f}, "
    f"mode=${triangular_mode:,.0f}, high=${triangular_high:,.0f}) "
    "per weak-label hit."
)
print("\nMonte Carlo simulated recovery-potential summary:")
print(monte_carlo_df.to_string(index=False))
print(
    "\nReminder: simulated_recovery_potential_* values are scenario "
    "outputs from analyst-defined inputs, not actual recoveries."
)

# %%
# Step 9 — Plot Monte Carlo Recovery-Potential Ranges
#
# Business Purpose:
# Show the p10 / median / p90 simulated recovery-potential range
# alongside each audit-capacity scenario.
#
# Plain-English Logic:
# We bar-chart the median simulated recovery potential for each
# capacity scenario and overlay error bars from p10 to p90. The bar
# heights communicate the central scenario; the error bars communicate
# the uncertainty band that the Monte Carlo produced.
#
# Expected Output:
# A figure saved to:
#   reports/notebook04_monte_carlo_recovery_potential_ranges.png
#
# Why It Matters:
# A point estimate ("if we review the top 1,000 we recover $X") tends
# to be over-read. The p10/p90 band is harder to misinterpret because
# it explicitly displays the range the simulation produced under the
# stated assumptions.

fig, ax = plt.subplots(figsize=(10, 6))

x_positions = np.arange(len(monte_carlo_df))
medians = monte_carlo_df["simulated_recovery_potential_median"].values
lower_err = (
    medians
    - monte_carlo_df["simulated_recovery_potential_p10"].values
)
upper_err = (
    monte_carlo_df["simulated_recovery_potential_p90"].values
    - medians
)

ax.bar(
    x_positions,
    medians,
    color="#1a5276",
    alpha=0.85,
    label="Median simulated recovery potential",
)
ax.errorbar(
    x_positions,
    medians,
    yerr=[lower_err, upper_err],
    fmt="none",
    ecolor="#b9770e",
    elinewidth=2,
    capsize=6,
    label="p10 – p90 range",
)

ax.set_xticks(x_positions)
ax.set_xticklabels(monte_carlo_df["capacity_label"].tolist())
ax.set_ylabel("Simulated recovery potential ($)")
ax.set_title(
    "Monte Carlo Simulated Recovery-Potential Ranges by Audit Capacity"
)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"${v:,.0f}")
)
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
ax.legend(loc="upper left")

plt.tight_layout()

mc_plot_path = (
    PATHS["reports"]
    / "notebook04_monte_carlo_recovery_potential_ranges.png"
)
plt.savefig(mc_plot_path, dpi=150, bbox_inches="tight")
plt.show()

print("Saved:", mc_plot_path)

# %%
# Step 10 — Save Outputs
#
# Business Purpose:
# Persist the capacity scenarios and Monte Carlo summary as parquet
# so downstream artifacts (dashboards, write-ups, future notebooks)
# can reference them without rerunning the simulation.
#
# Plain-English Logic:
# We write two files:
#   - data/processed/audit_capacity_scenarios_<YEAR>_full.parquet
#       (capacity_scenarios_df joined with the deterministic
#       recovery-potential assumption columns from Step 7)
#   - data/processed/monte_carlo_recovery_potential_<YEAR>_full.parquet
#       (the Monte Carlo summary from Step 8)
# Parquet keeps dtypes intact and is fast to reload.
#
# Expected Output:
# Two parquet files at the paths above, plus a printed confirmation
# of row counts and column names.
#
# Why It Matters:
# Any later artifact that wants to show "if we review the top 1,000
# providers, the simulated recovery-potential band is $A to $B" can
# read these files directly instead of re-running the simulation.
#
# What to Check Before Moving On:
# - Both parquet files exist and are non-empty
# - Their row counts equal len(capacity_scenarios_df)

capacity_output = capacity_scenarios_df.merge(
    recovery_assumption_df.drop(
        columns=["providers_reviewed", "weak_label_hits"]
    ),
    on="capacity_label",
    how="left",
)

capacity_output_path = (
    PATHS["data_processed"]
    / f"audit_capacity_scenarios_{YEAR}_full.parquet"
)
capacity_output.to_parquet(
    capacity_output_path, engine="pyarrow", index=False
)

monte_carlo_output_path = (
    PATHS["data_processed"]
    / f"monte_carlo_recovery_potential_{YEAR}_full.parquet"
)
monte_carlo_df.to_parquet(
    monte_carlo_output_path, engine="pyarrow", index=False
)

print("Saved:", capacity_output_path)
print("  rows:", len(capacity_output))
print("  columns:", list(capacity_output.columns))
print("Saved:", monte_carlo_output_path)
print("  rows:", len(monte_carlo_df))
print("  columns:", list(monte_carlo_df.columns))

# %%
# Step 11 — Notebook Summary
#
# Business Purpose:
# Print a short, copy-pasteable summary of what this notebook
# produced so the run ends with a clear, auditable record.
#
# Plain-English Logic:
# We surface the input file, the model score column used, the number
# of providers scored, the highest-capacity scenario, the
# best-precision scenario, and the saved output paths. The closing
# reminder repeats the framing language so anyone reusing the outputs
# is reminded that they are scenario-based decision support, not
# findings or actual recoveries.
#
# Expected Output:
# Plain printed text. No file writes here.
#
# Why It Matters:
# A consistent end-of-notebook summary makes it easy to compare runs
# across versions or years and to lift key numbers into a write-up
# without losing the limitation framing.

best_precision_row = capacity_scenarios_df.loc[
    capacity_scenarios_df["weak_label_precision"].idxmax()
]
highest_capacity_row = capacity_scenarios_df.loc[
    capacity_scenarios_df["providers_reviewed"].idxmax()
]

print("=" * 60)
print("Notebook 04 Summary — Monte Carlo Audit-Capacity Simulation")
print("=" * 60)
print(f"Input file                        : {predictions_path}")
print(f"Model score column used           : {score_col}")
print(f"Providers scored                  : {len(ranked_df):,}")
print(f"Total weak-label positives        : {total_positives:,}")
print(f"Overall weak-label positive rate  : {overall_positive_rate:.5f}")
print("-" * 60)
print(
    f"Highest-capacity scenario         : "
    f"{highest_capacity_row['capacity_label']} "
    f"(precision {highest_capacity_row['weak_label_precision']:.4f}, "
    f"recall {highest_capacity_row['weak_label_recall']:.4f}, "
    f"lift {highest_capacity_row['lift_vs_random']:.2f}x)"
)
print(
    f"Best-precision scenario           : "
    f"{best_precision_row['capacity_label']} "
    f"(precision {best_precision_row['weak_label_precision']:.4f}, "
    f"recall {best_precision_row['weak_label_recall']:.4f}, "
    f"lift {best_precision_row['lift_vs_random']:.2f}x)"
)
print("-" * 60)
print(f"Capacity scenarios saved to       : {capacity_output_path}")
print(f"Monte Carlo summary saved to      : {monte_carlo_output_path}")
print(f"Capacity tradeoff chart saved to  : {capacity_plot_path}")
print(f"Monte Carlo chart saved to        : {mc_plot_path}")
print("=" * 60)
print(
    "Reminder: every output in this notebook is scenario-based "
    "decision support derived from public CMS data, model-predicted "
    "audit-priority scores, and analyst-defined recovery-potential "
    "assumptions. Nothing here represents actual recoveries, "
    "confirmed fraud, overpayments, noncompliance, or audit findings."
)

# %%
