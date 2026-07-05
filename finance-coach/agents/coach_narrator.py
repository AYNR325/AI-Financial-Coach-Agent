"""Coach/Narrator Agent — combines all agent outputs into coaching + safe-to-spend."""

from __future__ import annotations

from typing import Any

from core.calculations import compute_safe_to_spend
from core.currency import inr
from core.schemas import (
    BudgetBaseline,
    CoachOutput,
    DebtPlanResult,
    IncomeProfile,
    SimulationResult,
)
from llm.client import complete
from llm.prompts import COACH_SYSTEM, coach_user_prompt


def _fallback_message(
    profile: IncomeProfile,
    debt_plan: DebtPlanResult,
    budget: BudgetBaseline,
    simulation: SimulationResult,
    weekly: float,
    monthly: float,
) -> str:
    pct = simulation.solvency_probability * 100
    rec = debt_plan.recommended_strategy
    months = (
        debt_plan.avalanche.months_to_debt_free
        if rec == "avalanche"
        else debt_plan.snowball.months_to_debt_free
    )
    buffer_line = (
        f" Building your emergency fund toward {inr(simulation.recommended_buffer)} "
        f"would raise solvency toward 90%."
        if simulation.solvency_probability < 0.90
        else " Your buffer target already supports a strong solvency outlook."
    )
    return (
        f"Based on your income pattern, you can safely spend about {inr(weekly)} this week "
        f"({inr(monthly)}/month) without risking your debt plan. "
        f"Your buffer covers {pct:.0f}% of possible bad-month scenarios over the next "
        f"{simulation.horizon_months} months.{buffer_line} "
        f"Stick with the {rec} debt strategy — about {months} months to debt-free — "
        f"and protect essentials ({inr(budget.essential_monthly_expenses)}/mo) first "
        f"when income dips toward {inr(profile.p10_income)}."
    )


def run_coach_narrator(
    profile: IncomeProfile,
    debt_plan: DebtPlanResult,
    budget: BudgetBaseline,
    simulation: SimulationResult,
    use_llm: bool = True,
) -> dict[str, Any]:
    monthly, weekly = compute_safe_to_spend(
        profile,
        budget,
        debt_plan,
        simulation.recommended_buffer,
        initial_buffer=simulation.initial_buffer,
    )

    message = ""
    if use_llm:
        message = complete(
            COACH_SYSTEM,
            coach_user_prompt(
                income_profile=profile.to_dict(),
                debt_plan=debt_plan.to_dict(),
                budget=budget.to_dict(),
                simulation={
                    k: simulation.to_dict()[k]
                    for k in (
                        "solvency_probability",
                        "debt_free_month_median",
                        "recommended_buffer",
                        "horizon_months",
                    )
                },
                safe_to_spend_weekly=weekly,
                safe_to_spend_monthly=monthly,
            ),
        )
    if not message:
        message = _fallback_message(profile, debt_plan, budget, simulation, weekly, monthly)

    highlights = [
        f"Safe to spend this week: {inr(weekly, 2)}",
        f"Solvency probability: {simulation.solvency_probability * 100:.0f}%",
        f"Recommended strategy: {debt_plan.recommended_strategy}",
        f"Buffer for ~90% solvency: {inr(simulation.recommended_buffer, 2)}",
    ]
    coach = CoachOutput(
        message=message,
        safe_to_spend_weekly=weekly,
        safe_to_spend_monthly=monthly,
        highlights=highlights,
    )
    return {"coach": coach, "coach_dict": coach.to_dict()}
