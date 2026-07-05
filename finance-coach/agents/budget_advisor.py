"""Budget Advisor Agent — category aggregation + optional LLM narration."""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.calculations import budget_baseline
from core.currency import inr
from core.schemas import BudgetBaseline
from llm.client import complete
from llm.prompts import BUDGET_SYSTEM, budget_user_prompt


def _fallback_narrative(budget: BudgetBaseline) -> str:
    flagged = (
        f" Watch {', '.join(budget.flagged_categories)} — spending is trending up."
        if budget.flagged_categories
        else ""
    )
    return (
        f"Your baseline monthly spend is about {inr(budget.total_monthly_expenses)}, "
        f"with {inr(budget.essential_monthly_expenses)} in needs and "
        f"{inr(budget.discretionary_monthly_expenses)} in wants.{flagged}"
    )


def run_budget_advisor(tx_df: pd.DataFrame, use_llm: bool = True) -> dict[str, Any]:
    budget = budget_baseline(tx_df)
    payload = budget.to_dict()

    narrative = ""
    if use_llm:
        narrative = complete(BUDGET_SYSTEM, budget_user_prompt(payload))
    if not narrative:
        narrative = _fallback_narrative(budget)

    return {"budget": budget, "budget_dict": payload, "narrative": narrative}
