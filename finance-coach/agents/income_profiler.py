"""Income Profiler Agent — distribution stats + LLM summary."""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.calculations import income_profile
from core.currency import inr
from core.schemas import IncomeProfile
from llm.client import complete
from llm.prompts import INCOME_SYSTEM, income_user_prompt


def _fallback_narrative(profile: IncomeProfile) -> str:
    if profile.months_of_data == 0:
        return "We don't have enough income history yet to profile your cash flow."
    variance_pct = (
        (profile.std_dev / profile.mean_monthly_income * 100)
        if profile.mean_monthly_income
        else 0
    )
    return (
        f"Your income looks {profile.income_type}, averaging "
        f"{inr(profile.mean_monthly_income)}/month over {profile.months_of_data} months "
        f"with about {variance_pct:.0f}% month-to-month variation. "
        f"A bad month (10th percentile) is around {inr(profile.p10_income)}, "
        f"so plans should survive that floor—not just your average month."
    )


def run_income_profiler(tx_df: pd.DataFrame, use_llm: bool = True) -> dict[str, Any]:
    profile = income_profile(tx_df)
    payload = profile.to_dict()
    # Don't send full monthly series to LLM
    narrate_payload = {k: v for k, v in payload.items() if k != "monthly_incomes"}

    narrative = ""
    if use_llm:
        narrative = complete(INCOME_SYSTEM, income_user_prompt(narrate_payload))
    if not narrative:
        narrative = _fallback_narrative(profile)

    return {"income_profile": profile, "income_profile_dict": payload, "narrative": narrative}
