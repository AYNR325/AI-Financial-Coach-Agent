"""Deterministic finance math — income stats, debt payoff, budget baseline."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .currency import inr
from .schemas import (
    BudgetBaseline,
    Debt,
    DebtPlanResult,
    IncomeProfile,
    StrategyResult,
)

ESSENTIAL_CATEGORIES = {"rent", "groceries", "utilities", "transport"}
DISCRETIONARY_CATEGORIES = {"eating_out", "subscriptions", "misc", "entertainment"}

MAX_PAYOFF_MONTHS = 600


def monthly_income_series(tx_df: pd.DataFrame) -> pd.Series:
    income = tx_df[tx_df["type"] == "income"].copy()
    if income.empty:
        return pd.Series(dtype=float)
    income["month"] = income["date"].dt.to_period("M")
    return income.groupby("month")["amount"].sum().astype(float)


def income_profile(tx_df: pd.DataFrame) -> IncomeProfile:
    """Characterize income as a distribution from transaction history."""
    series = monthly_income_series(tx_df)
    if series.empty:
        return IncomeProfile(
            mean_monthly_income=0.0,
            std_dev=0.0,
            p10_income=0.0,
            p90_income=0.0,
            income_type="unknown",
            months_of_data=0,
            monthly_incomes=[],
        )

    values = series.to_numpy(dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    p10 = float(np.percentile(values, 10))
    p90 = float(np.percentile(values, 90))
    cv = (std / mean) if mean > 0 else 0.0
    # ~15%+ month-to-month variation is treated as irregular/gig-style income
    income_type = "gig/freelance" if cv >= 0.15 else "stable/paycheck"

    return IncomeProfile(
        mean_monthly_income=round(mean, 2),
        std_dev=round(std, 2),
        p10_income=round(p10, 2),
        p90_income=round(p90, 2),
        income_type=income_type,
        months_of_data=len(values),
        monthly_incomes=[round(float(v), 2) for v in values],
    )


def _payoff_schedule(
    debts: list[Debt],
    extra_monthly: float,
    order: Literal["avalanche", "snowball"],
) -> StrategyResult:
    """Month-by-month debt payoff under avalanche or snowball."""
    if not debts:
        return StrategyResult(
            strategy=order,
            months_to_debt_free=0,
            total_interest_paid=0.0,
            total_paid=0.0,
            monthly_payment_schedule=[],
            balance_trajectory=[0.0],
        )

    balances = [float(d.balance) for d in debts]
    aprs = [float(d.apr) / 100.0 for d in debts]
    minimums = [float(d.minimum_payment) for d in debts]
    names = [d.name for d in debts]

    if order == "avalanche":
        priority = sorted(range(len(debts)), key=lambda i: (-aprs[i], balances[i]))
    else:
        priority = sorted(range(len(debts)), key=lambda i: (balances[i], -aprs[i]))

    total_interest = 0.0
    total_paid = 0.0
    monthly_payments: list[float] = []
    balance_trajectory: list[float] = [sum(balances)]
    months = 0

    while sum(balances) > 0.01 and months < MAX_PAYOFF_MONTHS:
        months += 1
        # Accrue interest
        for i in range(len(balances)):
            if balances[i] <= 0:
                continue
            interest = balances[i] * (aprs[i] / 12.0)
            balances[i] += interest
            total_interest += interest

        # Minimum payments
        payment_this_month = 0.0
        remaining_extra = extra_monthly
        for i in range(len(balances)):
            if balances[i] <= 0:
                continue
            pay = min(minimums[i], balances[i])
            balances[i] -= pay
            payment_this_month += pay
            total_paid += pay
            if balances[i] < 0.01:
                balances[i] = 0.0

        # Extra toward priority debt(s)
        for i in priority:
            if remaining_extra <= 0:
                break
            if balances[i] <= 0:
                continue
            pay = min(remaining_extra, balances[i])
            balances[i] -= pay
            remaining_extra -= pay
            payment_this_month += pay
            total_paid += pay
            if balances[i] < 0.01:
                balances[i] = 0.0

        monthly_payments.append(round(payment_this_month, 2))
        balance_trajectory.append(round(sum(balances), 2))

        # Guard: if we cannot reduce principal (payments < interest), break
        if months > 1 and balance_trajectory[-1] >= balance_trajectory[-2] - 0.01:
            # Still allow progress if extra is applied; only break if no payment capacity
            if payment_this_month <= 0:
                break

    return StrategyResult(
        strategy=order,
        months_to_debt_free=months if sum(balances) <= 0.01 else MAX_PAYOFF_MONTHS,
        total_interest_paid=round(total_interest, 2),
        total_paid=round(total_paid, 2),
        monthly_payment_schedule=monthly_payments,
        balance_trajectory=balance_trajectory,
    )


def avalanche_payoff(debts: list[Debt], extra_monthly: float) -> StrategyResult:
    return _payoff_schedule(debts, extra_monthly, "avalanche")


def snowball_payoff(debts: list[Debt], extra_monthly: float) -> StrategyResult:
    return _payoff_schedule(debts, extra_monthly, "snowball")


def compare_strategies(
    debts: list[Debt],
    extra_monthly: float,
    preference: Literal["avalanche", "snowball", "auto"] = "auto",
) -> DebtPlanResult:
    """Compare avalanche vs snowball and recommend a strategy."""
    avalanche = avalanche_payoff(debts, extra_monthly)
    snowball = snowball_payoff(debts, extra_monthly)
    total_minimum = sum(d.minimum_payment for d in debts)

    interest_saved = snowball.total_interest_paid - avalanche.total_interest_paid

    if preference == "snowball":
        recommended = "snowball"
        reason = "You preferred snowball for motivation — paying smallest balances first."
    elif preference == "avalanche":
        recommended = "avalanche"
        reason = "You preferred avalanche — attacking highest interest rates first."
    elif interest_saved > 1.0:
        recommended = "avalanche"
        reason = (
            f"Avalanche saves {inr(interest_saved, 2)} in interest versus snowball "
            f"and reaches debt-free in {avalanche.months_to_debt_free} months."
        )
    elif snowball.months_to_debt_free < avalanche.months_to_debt_free:
        recommended = "snowball"
        reason = (
            f"Snowball reaches debt-free sooner ({snowball.months_to_debt_free} vs "
            f"{avalanche.months_to_debt_free} months) with similar interest cost."
        )
    else:
        recommended = "avalanche"
        reason = "Avalanche and snowball are similar; avalanche is the default for interest savings."

    return DebtPlanResult(
        avalanche=avalanche,
        snowball=snowball,
        recommended_strategy=recommended,
        recommendation_reason=reason,
        extra_monthly_payment_budget=extra_monthly,
        total_minimum_payments=round(total_minimum, 2),
    )


def budget_baseline(tx_df: pd.DataFrame) -> BudgetBaseline:
    """Aggregate spending into a monthly budget baseline."""
    expenses = tx_df[tx_df["type"] == "expense"].copy()
    if expenses.empty:
        return BudgetBaseline(
            monthly_spend_by_category={},
            total_monthly_expenses=0.0,
            essential_monthly_expenses=0.0,
            discretionary_monthly_expenses=0.0,
            needs_vs_wants={"needs": 0.0, "wants": 0.0},
            flagged_categories=[],
            months_of_data=0,
        )

    expenses["month"] = expenses["date"].dt.to_period("M")
    months = max(expenses["month"].nunique(), 1)

    by_cat = expenses.groupby("category")["amount"].sum() / months
    monthly_spend = {str(k): round(float(v), 2) for k, v in by_cat.items()}

    essential = sum(v for k, v in monthly_spend.items() if k in ESSENTIAL_CATEGORIES)
    discretionary = sum(v for k, v in monthly_spend.items() if k in DISCRETIONARY_CATEGORIES)
    other = sum(
        v
        for k, v in monthly_spend.items()
        if k not in ESSENTIAL_CATEGORIES and k not in DISCRETIONARY_CATEGORIES
    )
    # Treat unmapped categories as discretionary
    discretionary += other

    # Flag categories trending upward (last 3 months avg > prior months avg by >15%)
    flagged: list[str] = []
    month_order = sorted(expenses["month"].unique())
    if len(month_order) >= 4:
        recent = month_order[-3:]
        prior = month_order[:-3]
        for cat in monthly_spend:
            cat_df = expenses[expenses["category"] == cat]
            recent_avg = cat_df[cat_df["month"].isin(recent)].groupby("month")["amount"].sum().mean()
            prior_avg = cat_df[cat_df["month"].isin(prior)].groupby("month")["amount"].sum().mean()
            if prior_avg and recent_avg > prior_avg * 1.15:
                flagged.append(cat)

    total = sum(monthly_spend.values())
    return BudgetBaseline(
        monthly_spend_by_category=monthly_spend,
        total_monthly_expenses=round(total, 2),
        essential_monthly_expenses=round(essential, 2),
        discretionary_monthly_expenses=round(discretionary, 2),
        needs_vs_wants={
            "needs": round(essential, 2),
            "wants": round(discretionary, 2),
        },
        flagged_categories=flagged,
        months_of_data=months,
    )


def compute_safe_to_spend(
    profile: IncomeProfile,
    budget: BudgetBaseline,
    debt_plan: DebtPlanResult,
    recommended_buffer: float,
    *,
    initial_buffer: float = 0.0,
    buffer_months: int = 6,
) -> tuple[float, float]:
    """Return (monthly_safe, weekly_safe).

    Prefers a bad-month (p10) budget. If a bad month has no room but an average
    month does, uses half of average-month surplus (still cautious). Truly
    underwater plans (average month also negative) stay at ₹0.

    Buffer savings apply only to the gap above cash already saved (initial_buffer).
    Contribution is capped to half of leftover room so a large target cannot
    wipe safe-to-spend by itself.
    """
    debt_payment = debt_plan.total_minimum_payments
    fixed = budget.essential_monthly_expenses + debt_payment
    p10_room = profile.p10_income - fixed
    mean_room = profile.mean_monthly_income - fixed

    if p10_room > 0:
        room = p10_room
    elif mean_room > 0:
        room = mean_room * 0.5
    else:
        return 0.0, 0.0

    buffer_gap = max(0.0, recommended_buffer - initial_buffer)
    ideal_contribution = buffer_gap / max(buffer_months, 1)
    buffer_contribution = min(ideal_contribution, room * 0.5)
    monthly_safe = max(0.0, room - buffer_contribution)
    weekly_safe = monthly_safe / 4.0
    return round(monthly_safe, 2), round(weekly_safe, 2)
