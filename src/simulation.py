"""
Medicare Claims Audit — Monte Carlo Simulation
=================================================
Uncertainty quantification for audit recovery estimates.

In real OIG audits, we project overpayments from a sample to the universe
of claims. The confidence interval around that projection determines whether
a recovery demand is defensible. This module replicates that logic using
Monte Carlo simulation on model risk scores.

Approach:
    1. For each flagged provider, model outputs a risk probability
    2. Simulate N trials: does this provider have findings? (Bernoulli draw)
    3. If yes, sample a recovery amount from the provider's payment distribution
    4. Aggregate across all flagged providers to get portfolio-level estimates
    5. Report point estimate + confidence intervals (80%, 90%, 95%)

Usage:
    from src.simulation import MonteCarloAudit
    mc = MonteCarloAudit(n_simulations=10_000)
    results = mc.simulate(risk_scores, payment_amounts)
    mc.plot_results(results)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import MONTE_CARLO_PARAMS

log = logging.getLogger(__name__)


@dataclass
class SimulationResults:
    """Container for Monte Carlo simulation outputs."""
    point_estimate: float
    mean_estimate: float
    std_estimate: float
    ci_80: tuple  # (lower, upper)
    ci_90: tuple
    ci_95: tuple
    n_simulations: int
    n_providers: int
    distribution: np.ndarray  # Full distribution of simulated recoveries

    def summary(self) -> str:
        return (
            f"Monte Carlo Recovery Estimate ({self.n_simulations:,} simulations)\n"
            f"  Providers flagged: {self.n_providers:,}\n"
            f"  Point estimate:    ${self.point_estimate:,.0f}\n"
            f"  Mean (simulated):  ${self.mean_estimate:,.0f}\n"
            f"  Std deviation:     ${self.std_estimate:,.0f}\n"
            f"  80% CI: ${self.ci_80[0]:,.0f} — ${self.ci_80[1]:,.0f}\n"
            f"  90% CI: ${self.ci_90[0]:,.0f} — ${self.ci_90[1]:,.0f}\n"
            f"  95% CI: ${self.ci_95[0]:,.0f} — ${self.ci_95[1]:,.0f}\n"
        )


class MonteCarloAudit:
    """
    Monte Carlo simulation engine for Medicare audit recovery estimation.

    Parameters
    ----------
    n_simulations : int
        Number of Monte Carlo trials. 10,000 gives stable CI estimates.
    recovery_rate_mean : float
        Mean recovery rate on confirmed overpayments (default: 0.65).
        OIG historically recovers ~$6 per $1 invested in audit.
    recovery_rate_std : float
        Std of recovery rate to model uncertainty in actual recovery.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_simulations: int = None,
        recovery_rate_mean: float = 0.65,
        recovery_rate_std: float = 0.15,
        random_state: int = None,
    ):
        cfg = MONTE_CARLO_PARAMS
        self.n_simulations = n_simulations or cfg["n_simulations"]
        self.recovery_rate_mean = recovery_rate_mean
        self.recovery_rate_std = recovery_rate_std
        self.rng = np.random.default_rng(random_state or cfg["random_state"])
        self.confidence_levels = cfg["confidence_levels"]

    def simulate(
        self,
        risk_scores: np.ndarray,
        payment_amounts: np.ndarray,
        threshold: float = 0.5,
    ) -> SimulationResults:
        """
        Run Monte Carlo simulation on flagged providers.

        Parameters
        ----------
        risk_scores : array, shape (n_providers,)
            Model-predicted probability that a provider has audit findings.
        payment_amounts : array, shape (n_providers,)
            Total Medicare payments per provider (recovery ceiling).
        threshold : float
            Only simulate providers above this risk score.

        Returns
        -------
        SimulationResults
            Point estimate and confidence intervals for portfolio recovery.
        """
        # Filter to flagged providers
        mask = risk_scores >= threshold
        n_flagged = mask.sum()
        scores = risk_scores[mask]
        payments = payment_amounts[mask]

        log.info(f"Monte Carlo simulation: {n_flagged:,} providers above threshold {threshold}")
        log.info(f"  Total payments at risk: ${payments.sum():,.0f}")
        log.info(f"  Running {self.n_simulations:,} simulations...")

        # Point estimate: E[recovery] = sum(risk * payment * recovery_rate)
        point_estimate = (scores * payments * self.recovery_rate_mean).sum()

        # Monte Carlo: simulate the full distribution
        simulated_totals = np.zeros(self.n_simulations)

        for i in range(self.n_simulations):
            # For each provider: does the audit find anything? (Bernoulli)
            has_findings = self.rng.random(n_flagged) < scores

            # If findings, how much can we recover? (Beta-distributed rate)
            recovery_rates = np.clip(
                self.rng.normal(self.recovery_rate_mean, self.recovery_rate_std, n_flagged),
                0.1, 0.95,
            )

            # Total recovery this simulation
            simulated_totals[i] = (has_findings * payments * recovery_rates).sum()

        # Compute confidence intervals
        ci_80 = (np.percentile(simulated_totals, 10), np.percentile(simulated_totals, 90))
        ci_90 = (np.percentile(simulated_totals, 5), np.percentile(simulated_totals, 95))
        ci_95 = (np.percentile(simulated_totals, 2.5), np.percentile(simulated_totals, 97.5))

        results = SimulationResults(
            point_estimate=point_estimate,
            mean_estimate=simulated_totals.mean(),
            std_estimate=simulated_totals.std(),
            ci_80=ci_80,
            ci_90=ci_90,
            ci_95=ci_95,
            n_simulations=self.n_simulations,
            n_providers=n_flagged,
            distribution=simulated_totals,
        )

        log.info(results.summary())
        return results

    def plot_results(self, results: SimulationResults, save_path: Optional[str] = None):
        """Plot the Monte Carlo recovery distribution with CI bands."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.hist(
            results.distribution / 1e6,
            bins=80,
            color="#1a5276",
            alpha=0.8,
            edgecolor="white",
            density=True,
        )

        # CI bands
        for ci, color, label in [
            (results.ci_95, "#c0392b", "95% CI"),
            (results.ci_90, "#f39c12", "90% CI"),
            (results.ci_80, "#27ae60", "80% CI"),
        ]:
            ax.axvline(ci[0] / 1e6, color=color, linestyle="--", linewidth=1.5, alpha=0.7)
            ax.axvline(ci[1] / 1e6, color=color, linestyle="--", linewidth=1.5, alpha=0.7,
                       label=f"{label}: ${ci[0]/1e6:.1f}M — ${ci[1]/1e6:.1f}M")

        # Point estimate
        ax.axvline(
            results.point_estimate / 1e6,
            color="black", linestyle="-", linewidth=2,
            label=f"Point est: ${results.point_estimate/1e6:.1f}M",
        )

        ax.set_xlabel("Estimated Total Recovery ($ Millions)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(
            f"Monte Carlo Audit Recovery Distribution\n"
            f"({results.n_providers:,} flagged providers, {results.n_simulations:,} simulations)",
            fontsize=13,
        )
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info(f"✓ Plot saved: {save_path}")
        plt.show()


    def sensitivity_analysis(
        self,
        risk_scores: np.ndarray,
        payment_amounts: np.ndarray,
        thresholds: list = None,
    ) -> pd.DataFrame:
        """
        Run simulations across multiple risk thresholds to find
        the optimal audit cutoff (cost-benefit analysis).
        """
        import pandas as _pd

        thresholds = thresholds or [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        rows = []
        for t in thresholds:
            n_flagged = (risk_scores >= t).sum()
            if n_flagged == 0:
                continue
            results = self.simulate(risk_scores, payment_amounts, threshold=t)
            rows.append({
                "threshold": t,
                "n_providers": n_flagged,
                "point_estimate": results.point_estimate,
                "ci_90_lower": results.ci_90[0],
                "ci_90_upper": results.ci_90[1],
                "recovery_per_audit": results.point_estimate / n_flagged,
            })

        return _pd.DataFrame(rows)
