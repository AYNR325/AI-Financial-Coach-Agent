"""Monte Carlo Simulation Agent — vectorized risk futures + narration."""

from __future__ import annotations

from typing import Any

from core.currency import inr
from core.schemas import BudgetBaseline, DebtPlanResult, IncomeProfile, SimulationResult
from core.simulation import run_monte_carlo
from llm.client import complete
from llm.prompts import MONTE_CARLO_SYSTEM, monte_carlo_user_prompt


def _fallback_narrative(sim: SimulationResult) -> str:
    pct = sim.solvency_probability * 100
    buffer_msg = (
        f" Building an emergency buffer of about {inr(sim.recommended_buffer)} "
        f"would push you toward a 90% chance of staying solvent."
        if sim.solvency_probability < 0.90
        else f" Your current buffer target of {inr(sim.recommended_buffer)} already supports ~90% solvency."
    )
    debt_free = ""
    if sim.debt_free_month_median is not None:
        debt_free = (
            f" Across successful paths, debt-free lands around month "
            f"{sim.debt_free_month_median:.0f} "
            f"(range {sim.debt_free_month_p10:.0f}–{sim.debt_free_month_p90:.0f})."
        )
    tone = (
        "That's a solid cushion for irregular income."
        if pct >= 75
        else "That's a real risk — a slow season could break the plan without a bigger buffer."
    )
    return (
        f"In {sim.n_sims:,} simulated futures over {sim.horizon_months} months, "
        f"you stay solvent in about {pct:.0f}% of scenarios. {tone}{buffer_msg}{debt_free}"
    )


def run_monte_carlo_agent(
    profile: IncomeProfile,
    budget: BudgetBaseline,
    debt_plan: DebtPlanResult,
    *,
    initial_buffer: float = 0.0,
    horizon: int = 24,
    n_sims: int = 2000,
    seed: int | None = 42,
    use_llm: bool = True,
) -> dict[str, Any]:
    strategy = (
        debt_plan.avalanche
        if debt_plan.recommended_strategy == "avalanche"
        else debt_plan.snowball
    )
    fixed_expenses = budget.essential_monthly_expenses
    sim = run_monte_carlo(
        profile=profile,
        monthly_expenses=fixed_expenses,
        debt_monthly_payments=strategy.monthly_payment_schedule,
        initial_buffer=initial_buffer,
        horizon=horizon,
        n_sims=n_sims,
        seed=seed,
    )
    payload = sim.to_dict()
    narrate_payload = {
        k: payload[k]
        for k in (
            "solvency_probability",
            "debt_free_month_p10",
            "debt_free_month_median",
            "debt_free_month_p90",
            "recommended_buffer",
            "n_sims",
            "horizon_months",
            "initial_buffer",
        )
    }

    narrative = ""
    if use_llm:
        narrative = complete(
            MONTE_CARLO_SYSTEM,
            monte_carlo_user_prompt(narrate_payload, horizon),
        )
    if not narrative:
        narrative = _fallback_narrative(sim)

    return {"simulation": sim, "simulation_dict": payload, "narrative": narrative}
