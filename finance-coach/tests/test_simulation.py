from pathlib import Path

from core.calculations import budget_baseline, compare_strategies, compute_safe_to_spend, income_profile
from core.parsing import load_debts, load_transactions
from core.simulation import run_monte_carlo
from orchestrator.graph import run_coach_pipeline

DATA = Path(__file__).resolve().parents[1] / "data"


def _run_scenario(tx_path: str, debts_path: str, seed: int = 42):
    tx = load_transactions(DATA / tx_path)
    debts = load_debts(DATA / debts_path)
    profile = income_profile(tx)
    budget = budget_baseline(tx)
    plan = compare_strategies(debts["debts"], debts["extra_monthly_payment_budget"])
    strategy = plan.avalanche if plan.recommended_strategy == "avalanche" else plan.snowball
    sim = run_monte_carlo(
        profile=profile,
        monthly_expenses=budget.essential_monthly_expenses,
        debt_monthly_payments=strategy.monthly_payment_schedule,
        initial_buffer=0.0,
        horizon=24,
        n_sims=2000,
        seed=seed,
    )
    return sim


def test_monte_carlo_outputs_shape():
    sim = _run_scenario("sample_gig_worker.csv", "sample_debts.json")
    assert 0.0 <= sim.solvency_probability <= 1.0
    assert sim.n_sims == 2000
    assert sim.horizon_months == 24
    assert len(sim.percentile_trajectories["p50"]) == 24
    assert sim.recommended_buffer >= 0


def test_bad_case_solvency_lower_than_gig():
    gig = _run_scenario("sample_gig_worker.csv", "sample_debts.json")
    bad = _run_scenario("sample_bad_case.csv", "sample_bad_case_debts.json")
    assert bad.solvency_probability < gig.solvency_probability
    # Bad case should look materially riskier for the demo
    assert bad.solvency_probability < 0.85


def test_higher_buffer_improves_solvency():
    tx = load_transactions(DATA / "sample_bad_case.csv")
    debts = load_debts(DATA / "sample_bad_case_debts.json")
    profile = income_profile(tx)
    budget = budget_baseline(tx)
    plan = compare_strategies(debts["debts"], debts["extra_monthly_payment_budget"])
    strategy = plan.avalanche if plan.recommended_strategy == "avalanche" else plan.snowball

    low = run_monte_carlo(
        profile,
        budget.essential_monthly_expenses,
        strategy.monthly_payment_schedule,
        initial_buffer=0.0,
        seed=7,
    )
    high = run_monte_carlo(
        profile,
        budget.essential_monthly_expenses,
        strategy.monthly_payment_schedule,
        initial_buffer=low.recommended_buffer,
        seed=7,
    )
    assert high.solvency_probability >= low.solvency_probability


def test_scaled_kaggle_presets_have_safe_to_spend():
    presets = [
        ("personal_transactions.csv", "sample_paycheck_debts.json"),
        ("Personal_Finance_Dataset.csv", "sample_kaggle_pf_debts.json"),
    ]
    for tx_path, debts_path in presets:
        tx = load_transactions(DATA / tx_path)
        debts = load_debts(DATA / debts_path)
        result = run_coach_pipeline(
            tx,
            debts["debts"],
            debts["extra_monthly_payment_budget"],
            use_llm=False,
            seed=42,
        )
        assert result["safe_to_spend_weekly"] > 0
        assert result["simulation"].recommended_buffer < 100_000


def test_existing_buffer_does_not_reduce_safe_to_spend():
    tx = load_transactions(DATA / "sample_gig_worker.csv")
    debts = load_debts(DATA / "sample_debts.json")
    baseline = run_coach_pipeline(
        tx,
        debts["debts"],
        debts["extra_monthly_payment_budget"],
        initial_buffer=0.0,
        use_llm=False,
        seed=42,
    )
    funded = run_coach_pipeline(
        tx,
        debts["debts"],
        debts["extra_monthly_payment_budget"],
        initial_buffer=baseline["simulation"].recommended_buffer or 5000.0,
        use_llm=False,
        seed=42,
    )
    assert funded["safe_to_spend_weekly"] >= baseline["safe_to_spend_weekly"] * 0.95


def test_bad_case_stays_at_zero_safe_to_spend():
    tx = load_transactions(DATA / "sample_bad_case.csv")
    debts = load_debts(DATA / "sample_bad_case_debts.json")
    result = run_coach_pipeline(
        tx,
        debts["debts"],
        debts["extra_monthly_payment_budget"],
        use_llm=False,
        seed=42,
    )
    assert result["safe_to_spend_weekly"] == 0.0
    assert result["simulation"].solvency_probability < 0.05
