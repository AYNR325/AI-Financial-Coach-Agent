"""Canonical data structures for the financial coach."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class Debt:
    name: str
    balance: float
    apr: float
    minimum_payment: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IncomeProfile:
    mean_monthly_income: float
    std_dev: float
    p10_income: float
    p90_income: float
    income_type: str
    months_of_data: int
    monthly_incomes: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyResult:
    strategy: str
    months_to_debt_free: int
    total_interest_paid: float
    total_paid: float
    monthly_payment_schedule: list[float] = field(default_factory=list)
    balance_trajectory: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DebtPlanResult:
    avalanche: StrategyResult
    snowball: StrategyResult
    recommended_strategy: str
    recommendation_reason: str
    extra_monthly_payment_budget: float
    total_minimum_payments: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "avalanche": self.avalanche.to_dict(),
            "snowball": self.snowball.to_dict(),
            "recommended_strategy": self.recommended_strategy,
            "recommendation_reason": self.recommendation_reason,
            "extra_monthly_payment_budget": self.extra_monthly_payment_budget,
            "total_minimum_payments": self.total_minimum_payments,
        }


@dataclass
class BudgetBaseline:
    monthly_spend_by_category: dict[str, float]
    total_monthly_expenses: float
    essential_monthly_expenses: float
    discretionary_monthly_expenses: float
    needs_vs_wants: dict[str, float]
    flagged_categories: list[str] = field(default_factory=list)
    months_of_data: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationResult:
    solvency_probability: float
    debt_free_month_p10: float | None
    debt_free_month_median: float | None
    debt_free_month_p90: float | None
    recommended_buffer: float
    percentile_trajectories: dict[str, list[float]] = field(default_factory=dict)
    n_sims: int = 0
    horizon_months: int = 0
    initial_buffer: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CoachOutput:
    message: str
    safe_to_spend_weekly: float
    safe_to_spend_monthly: float
    highlights: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
