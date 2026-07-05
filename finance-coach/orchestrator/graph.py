"""LangGraph orchestrator wiring all financial coach agents."""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents.budget_advisor import run_budget_advisor
from agents.coach_narrator import run_coach_narrator
from agents.debt_analyzer import run_debt_analyzer
from agents.income_profiler import run_income_profiler
from agents.monte_carlo_agent import run_monte_carlo_agent
from core.schemas import BudgetBaseline, Debt, DebtPlanResult, IncomeProfile, SimulationResult


class CoachState(TypedDict, total=False):
    transactions: Any  # pd.DataFrame
    debts: list[Debt]
    extra_monthly: float
    preference: Literal["avalanche", "snowball", "auto"]
    initial_buffer: float
    horizon: int
    n_sims: int
    seed: Optional[int]
    use_llm: bool

    income_profile: IncomeProfile
    income_profile_dict: dict
    income_narrative: str

    debt_plan: DebtPlanResult
    debt_plan_dict: dict
    debt_narrative: str

    budget: BudgetBaseline
    budget_dict: dict
    budget_narrative: str

    simulation: SimulationResult
    simulation_dict: dict
    simulation_narrative: str

    coach_message: str
    safe_to_spend_weekly: float
    safe_to_spend_monthly: float
    coach_highlights: list[str]
    coach_dict: dict


def _analyze_node(state: CoachState) -> dict:
    """Run independent analyzers (income, debt, budget)."""
    use_llm = state.get("use_llm", True)
    income = run_income_profiler(state["transactions"], use_llm=use_llm)
    debt = run_debt_analyzer(
        debts=state["debts"],
        extra_monthly=state.get("extra_monthly", 0.0),
        preference=state.get("preference", "auto"),
        use_llm=use_llm,
    )
    budget = run_budget_advisor(state["transactions"], use_llm=use_llm)
    return {
        "income_profile": income["income_profile"],
        "income_profile_dict": income["income_profile_dict"],
        "income_narrative": income["narrative"],
        "debt_plan": debt["debt_plan"],
        "debt_plan_dict": debt["debt_plan_dict"],
        "debt_narrative": debt["narrative"],
        "budget": budget["budget"],
        "budget_dict": budget["budget_dict"],
        "budget_narrative": budget["narrative"],
    }


def _monte_carlo_node(state: CoachState) -> dict:
    result = run_monte_carlo_agent(
        profile=state["income_profile"],
        budget=state["budget"],
        debt_plan=state["debt_plan"],
        initial_buffer=state.get("initial_buffer", 0.0),
        horizon=state.get("horizon", 24),
        n_sims=state.get("n_sims", 2000),
        seed=state.get("seed", 42),
        use_llm=state.get("use_llm", True),
    )
    return {
        "simulation": result["simulation"],
        "simulation_dict": result["simulation_dict"],
        "simulation_narrative": result["narrative"],
    }


def _coach_node(state: CoachState) -> dict:
    result = run_coach_narrator(
        profile=state["income_profile"],
        debt_plan=state["debt_plan"],
        budget=state["budget"],
        simulation=state["simulation"],
        use_llm=state.get("use_llm", True),
    )
    coach = result["coach"]
    return {
        "coach_message": coach.message,
        "safe_to_spend_weekly": coach.safe_to_spend_weekly,
        "safe_to_spend_monthly": coach.safe_to_spend_monthly,
        "coach_highlights": coach.highlights,
        "coach_dict": result["coach_dict"],
    }


def build_graph():
    graph = StateGraph(CoachState)
    graph.add_node("analyze", _analyze_node)
    graph.add_node("monte_carlo", _monte_carlo_node)
    graph.add_node("coach_narrator", _coach_node)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "monte_carlo")
    graph.add_edge("monte_carlo", "coach_narrator")
    graph.add_edge("coach_narrator", END)
    return graph.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_coach_pipeline(
    transactions: pd.DataFrame,
    debts: list[Debt],
    extra_monthly: float,
    *,
    preference: Literal["avalanche", "snowball", "auto"] = "auto",
    initial_buffer: float = 0.0,
    horizon: int = 24,
    n_sims: int = 2000,
    seed: int | None = 42,
    use_llm: bool = True,
) -> CoachState:
    """Run the full multi-agent coaching pipeline."""
    graph = get_graph()
    initial: CoachState = {
        "transactions": transactions,
        "debts": debts,
        "extra_monthly": extra_monthly,
        "preference": preference,
        "initial_buffer": initial_buffer,
        "horizon": horizon,
        "n_sims": n_sims,
        "seed": seed,
        "use_llm": use_llm,
    }
    return graph.invoke(initial)
