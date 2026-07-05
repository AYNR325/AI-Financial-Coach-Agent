from pathlib import Path

from core.calculations import (
    avalanche_payoff,
    budget_baseline,
    compare_strategies,
    income_profile,
    snowball_payoff,
)
from core.parsing import load_debts, load_transactions
from core.schemas import Debt

DATA = Path(__file__).resolve().parents[1] / "data"


def test_income_profile_gig_worker():
    df = load_transactions(DATA / "sample_gig_worker.csv")
    profile = income_profile(df)
    assert profile.months_of_data == 12
    assert profile.mean_monthly_income > 0
    assert profile.std_dev > 0
    assert profile.p10_income < profile.mean_monthly_income < profile.p90_income
    assert profile.income_type == "gig/freelance"


def test_income_profile_paycheck_dataset():
    df = load_transactions(DATA / "personal_transactions.csv")
    profile = income_profile(df)
    assert profile.months_of_data >= 3
    assert profile.mean_monthly_income > 0
    # Paycheck credits are present in the source file
    assert df["merchant"].str.contains("Paycheck", case=False, na=False).any()


def test_simple_debt_payoff_months():
    """Single debt: ₹1000 at 0% APR, ₹100/mo min, ₹0 extra → 10 months."""
    debts = [Debt(name="Card", balance=1000.0, apr=0.0, minimum_payment=100.0)]
    result = avalanche_payoff(debts, extra_monthly=0.0)
    assert result.months_to_debt_free == 10
    assert result.total_interest_paid == 0.0


def test_avalanche_prefers_high_interest():
    debts = [
        Debt(name="High APR", balance=500.0, apr=30.0, minimum_payment=25.0),
        Debt(name="Low APR", balance=500.0, apr=5.0, minimum_payment=25.0),
    ]
    plan = compare_strategies(debts, extra_monthly=50.0)
    assert plan.recommended_strategy == "avalanche"
    assert plan.avalanche.total_interest_paid <= plan.snowball.total_interest_paid + 0.01


def test_snowball_preference():
    debts = load_debts(DATA / "sample_debts.json")
    plan = compare_strategies(debts["debts"], debts["extra_monthly_payment_budget"], preference="snowball")
    assert plan.recommended_strategy == "snowball"


def test_budget_baseline():
    df = load_transactions(DATA / "sample_gig_worker.csv")
    budget = budget_baseline(df)
    assert budget.total_monthly_expenses > 0
    assert "rent" in budget.monthly_spend_by_category
    assert budget.essential_monthly_expenses > 0
    assert budget.needs_vs_wants["needs"] > 0

