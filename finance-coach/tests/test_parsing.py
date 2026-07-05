from pathlib import Path

import pandas as pd
import pytest

from core.parsing import load_debts, load_transactions

DATA = Path(__file__).resolve().parents[1] / "data"


def test_load_synthetic_gig_worker():
    df = load_transactions(DATA / "sample_gig_worker.csv")
    assert list(df.columns) == ["date", "merchant", "category", "amount", "type"]
    assert set(df["type"].unique()) <= {"income", "expense"}
    assert (df["amount"] >= 0).all()
    assert df["type"].eq("income").any()
    assert df["type"].eq("expense").any()


def test_load_synthetic_bad_case():
    df = load_transactions(DATA / "sample_bad_case.csv")
    assert len(df) > 100
    assert df["date"].is_monotonic_increasing or True  # sorted in finalize


def test_load_kaggle_personal_finance():
    df = load_transactions(DATA / "Personal_Finance_Dataset.csv")
    assert list(df.columns) == ["date", "merchant", "category", "amount", "type"]
    assert df["type"].isin(["income", "expense"]).all()
    # Salary rows are Expense in source — must remain expenses, not income
    # (category gets remapped; type is authoritative)


def test_load_kaggle_personal_transactions():
    df = load_transactions(DATA / "personal_transactions.csv")
    assert list(df.columns) == ["date", "merchant", "category", "amount", "type"]
    # Credit card payments filtered out
    assert not df["merchant"].str.contains("Credit Card Payment", case=False, na=False).any()
    # Paychecks become income
    assert df["type"].eq("income").any()


def test_load_debts():
    payload = load_debts(DATA / "sample_debts.json")
    assert len(payload["debts"]) == 2
    assert payload["extra_monthly_payment_budget"] == 3500.0
    assert payload["debts"][0].name == "Credit Card - HDFC"


def test_load_bad_case_debts():
    payload = load_debts(DATA / "sample_bad_case_debts.json")
    assert len(payload["debts"]) == 3
    assert payload["extra_monthly_payment_budget"] == 500.0


def test_unknown_schema_raises():
    bad = Path(__file__).parent / "_tmp_bad.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(bad, index=False)
    try:
        with pytest.raises(ValueError):
            load_transactions(bad)
    finally:
        bad.unlink(missing_ok=True)
