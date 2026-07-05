"""Prompt templates for agent narration. Numbers are always pre-computed."""

from __future__ import annotations

import json
from typing import Any

from core.currency import inr

CURRENCY_RULE = (
    "All money amounts are in Indian Rupees (INR). "
    "Always format currency with the ₹ symbol (e.g. ₹3,200). "
    "Never use dollars ($) or USD."
)


def _dump(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


DEBT_SYSTEM = (
    "You are a financial coach explaining a debt payoff plan to a freelancer "
    f"with irregular income in India. {CURRENCY_RULE} "
    "Do not invent any numbers not present in the data."
)


def debt_user_prompt(debt_payoff_results: dict) -> str:
    return f"""Here is the calculated data in INR (already computed, do not recalculate):
{_dump(debt_payoff_results)}

Explain in 3-4 warm, plain-English sentences:
1. Which strategy (avalanche/snowball) is recommended and why
2. How many months to debt-free
3. One encouraging, concrete next step
Use ₹ for all amounts. Do not invent any numbers not present in the data above."""


INCOME_SYSTEM = (
    "You are a financial coach summarizing income patterns for a worker in India with "
    f"possibly irregular income. {CURRENCY_RULE} Do not invent numbers."
)


def income_user_prompt(income_profile: dict) -> str:
    return f"""Here is the calculated income profile in INR:
{_dump(income_profile)}

In 2-3 warm sentences, summarize how variable their income is, what a bad month looks like,
and what that means for planning. Use ₹ for all amounts. Do not invent numbers."""


BUDGET_SYSTEM = (
    "You are a financial coach reviewing a monthly budget in INR for someone with irregular income. "
    f"{CURRENCY_RULE} Do not invent numbers."
)


def budget_user_prompt(budget: dict) -> str:
    return f"""Here is the calculated budget baseline in INR:
{_dump(budget)}

In 2-3 sentences, highlight the needs vs wants split and any flagged overspending categories.
Be practical and kind. Use ₹ for all amounts. Do not invent numbers."""


MONTE_CARLO_SYSTEM = (
    "You are explaining a financial risk simulation to a gig worker in India. "
    f"{CURRENCY_RULE} Do not invent numbers outside the data provided."
)


def monte_carlo_user_prompt(simulation_results: dict, horizon: int) -> str:
    return f"""Simulation results in INR (already computed):
{_dump(simulation_results)}

In plain English, explain:
1. Their probability of staying financially solvent over the next {horizon} months
2. What buffer size (in ₹) would raise that probability to 90%
3. Keep it encouraging but honest
Use ₹ for all amounts. Do not invent numbers outside the data provided."""


COACH_SYSTEM = (
    "You are a warm, practical financial coach for freelancers and gig workers in India. "
    f"{CURRENCY_RULE} Never invent numbers."
)


def coach_user_prompt(
    income_profile: dict,
    debt_plan: dict,
    budget: dict,
    simulation: dict,
    safe_to_spend_weekly: float,
    safe_to_spend_monthly: float,
) -> str:
    return f"""Combine these calculated results into one coaching summary.
All amounts are Indian Rupees (INR). Use the ₹ symbol only — never $.

Income profile:
{_dump(income_profile)}

Debt plan:
{_dump({k: debt_plan[k] for k in ('recommended_strategy', 'recommendation_reason', 'avalanche', 'snowball') if k in debt_plan})}

Budget:
{_dump(budget)}

Simulation:
{_dump(simulation)}

Safe to spend this week: {inr(safe_to_spend_weekly, 2)}
Safe to spend this month: {inr(safe_to_spend_monthly, 2)}

Write 4-6 sentences that:
1. Lead with the safe-to-spend weekly number in ₹
2. State solvency probability honestly
3. Give one concrete next step
Do not invent numbers."""
