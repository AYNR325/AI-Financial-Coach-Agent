"""Debt Analyzer Agent — deterministic payoff math + LLM narration."""

from __future__ import annotations

from typing import Any, Literal

from core.calculations import compare_strategies
from core.currency import inr
from core.schemas import Debt, DebtPlanResult
from llm.client import complete
from llm.prompts import DEBT_SYSTEM, debt_user_prompt


def _fallback_narrative(plan: DebtPlanResult) -> str:
    rec = plan.avalanche if plan.recommended_strategy == "avalanche" else plan.snowball
    other = plan.snowball if plan.recommended_strategy == "avalanche" else plan.avalanche
    monthly = plan.total_minimum_payments + plan.extra_monthly_payment_budget
    return (
        f"We recommend the {plan.recommended_strategy} strategy. "
        f"{plan.recommendation_reason} "
        f"Under that plan you reach debt-free in about {rec.months_to_debt_free} months, "
        f"paying {inr(rec.total_interest_paid, 2)} in interest "
        f"(versus {inr(other.total_interest_paid, 2)} with the alternative). "
        f"A concrete next step: automate at least "
        f"{inr(monthly, 2)} "
        f"toward debts each month, even in slower income months."
    )


def run_debt_analyzer(
    debts: list[Debt],
    extra_monthly: float,
    preference: Literal["avalanche", "snowball", "auto"] = "auto",
    use_llm: bool = True,
) -> dict[str, Any]:
    plan = compare_strategies(debts, extra_monthly, preference=preference)
    payload = plan.to_dict()

    # Slim payload for narration (drop long trajectories)
    narrate_payload = {
        "recommended_strategy": payload["recommended_strategy"],
        "recommendation_reason": payload["recommendation_reason"],
        "extra_monthly_payment_budget": payload["extra_monthly_payment_budget"],
        "total_minimum_payments": payload["total_minimum_payments"],
        "avalanche": {
            "months_to_debt_free": payload["avalanche"]["months_to_debt_free"],
            "total_interest_paid": payload["avalanche"]["total_interest_paid"],
            "total_paid": payload["avalanche"]["total_paid"],
        },
        "snowball": {
            "months_to_debt_free": payload["snowball"]["months_to_debt_free"],
            "total_interest_paid": payload["snowball"]["total_interest_paid"],
            "total_paid": payload["snowball"]["total_paid"],
        },
    }

    narrative = ""
    if use_llm:
        narrative = complete(DEBT_SYSTEM, debt_user_prompt(narrate_payload))
    if not narrative:
        narrative = _fallback_narrative(plan)

    return {"debt_plan": plan, "debt_plan_dict": payload, "narrative": narrative}
