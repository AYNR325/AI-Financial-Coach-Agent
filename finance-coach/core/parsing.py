"""Multi-schema CSV/JSON ingestion into the canonical transaction + debts model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import Debt

CANONICAL_COLUMNS = ["date", "merchant", "category", "amount", "type"]

# Map external category labels onto a small internal set.
CATEGORY_MAP: dict[str, str] = {
    # synthetic
    "rent": "rent",
    "groceries": "groceries",
    "transport": "transport",
    "subscriptions": "subscriptions",
    "utilities": "utilities",
    "eating_out": "eating_out",
    "misc": "misc",
    "income": "income",
    # Personal_Finance_Dataset
    "food & drink": "eating_out",
    "food and drink": "eating_out",
    "utilities": "utilities",
    "rent": "rent",
    "investment": "income",
    "shopping": "misc",
    "entertainment": "eating_out",
    "health & fitness": "misc",
    "health and fitness": "misc",
    "salary": "misc",
    "travel": "transport",
    "other": "misc",
    # personal_transactions
    "mortgage & rent": "rent",
    "mortgage and rent": "rent",
    "restaurants": "eating_out",
    "movies & dvds": "subscriptions",
    "movies and dvds": "subscriptions",
    "music": "subscriptions",
    "mobile phone": "utilities",
    "gas & fuel": "transport",
    "gas and fuel": "transport",
    "home improvement": "misc",
    "fast food": "eating_out",
    "coffee shops": "eating_out",
    "internet": "utilities",
    "paycheck": "income",
    "credit card payment": "transfer",
    "shopping": "misc",
    "groceries": "groceries",
}


def _normalize_category(raw: str) -> str:
    key = str(raw).strip().lower()
    return CATEGORY_MAP.get(key, key.replace(" ", "_") or "misc")


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], format="mixed", dayfirst=False)
    out["merchant"] = out["merchant"].astype(str)
    out["category"] = out["category"].map(_normalize_category)
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0).abs()
    out["type"] = out["type"].astype(str).str.lower().str.strip()
    out = out[out["type"].isin(["income", "expense"])]
    # Income rows should use category "income"
    out.loc[out["type"] == "income", "category"] = "income"
    out = out.sort_values("date").reset_index(drop=True)
    return out[CANONICAL_COLUMNS]


def _is_synthetic(columns: set[str]) -> bool:
    return {"date", "merchant", "category", "amount", "type"}.issubset(columns)


def _is_kaggle_personal_finance(columns: set[str]) -> bool:
    return {"date", "transaction description", "category", "amount", "type"}.issubset(columns)


def _is_kaggle_personal_transactions(columns: set[str]) -> bool:
    return {"date", "description", "amount", "transaction type", "category"}.issubset(columns)


def _parse_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    return _finalize(
        pd.DataFrame(
            {
                "date": df[cols["date"]],
                "merchant": df[cols["merchant"]],
                "category": df[cols["category"]],
                "amount": df[cols["amount"]],
                "type": df[cols["type"]],
            }
        )
    )


def _parse_kaggle_personal_finance(df: pd.DataFrame) -> pd.DataFrame:
    """Adapter for Personal_Finance_Dataset.csv.

    Classify by Type (Income/Expense), not category name — Salary rows in this
    dataset are labeled Expense and must stay expenses.
    """
    cols = {c.lower(): c for c in df.columns}
    type_raw = df[cols["type"]].astype(str).str.lower().str.strip()
    type_norm = type_raw.map(lambda t: "income" if t == "income" else "expense")
    return _finalize(
        pd.DataFrame(
            {
                "date": df[cols["date"]],
                "merchant": df[cols["transaction description"]],
                "category": df[cols["category"]],
                "amount": df[cols["amount"]],
                "type": type_norm,
            }
        )
    )


def _parse_kaggle_personal_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Adapter for personal_transactions.csv (debit/credit + account transfers)."""
    cols = {c.lower(): c for c in df.columns}
    category = df[cols["category"]].astype(str)
    # Drop internal credit-card payment transfers to avoid double-counting.
    mask = ~category.str.lower().str.contains("credit card payment", na=False)
    filtered = df.loc[mask].copy()

    tx_type = filtered[cols["transaction type"]].astype(str).str.lower().str.strip()
    type_norm = tx_type.map(lambda t: "income" if t == "credit" else "expense")

    return _finalize(
        pd.DataFrame(
            {
                "date": filtered[cols["date"]],
                "merchant": filtered[cols["description"]],
                "category": filtered[cols["category"]],
                "amount": filtered[cols["amount"]],
                "type": type_norm,
            }
        )
    )


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load a transactions CSV and normalize to the canonical schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Transaction file not found: {path}")

    df = pd.read_csv(path)
    columns = {c.lower().strip() for c in df.columns}

    if _is_synthetic(columns):
        return _parse_synthetic(df)
    if _is_kaggle_personal_finance(columns):
        return _parse_kaggle_personal_finance(df)
    if _is_kaggle_personal_transactions(columns):
        return _parse_kaggle_personal_transactions(df)

    raise ValueError(
        f"Unrecognized transaction CSV schema. Columns: {list(df.columns)}. "
        "Expected synthetic, Personal_Finance_Dataset, or personal_transactions format."
    )


def load_transactions_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize an in-memory DataFrame (e.g. from Streamlit upload)."""
    columns = {c.lower().strip() for c in df.columns}
    if _is_synthetic(columns):
        return _parse_synthetic(df)
    if _is_kaggle_personal_finance(columns):
        return _parse_kaggle_personal_finance(df)
    if _is_kaggle_personal_transactions(columns):
        return _parse_kaggle_personal_transactions(df)
    raise ValueError(f"Unrecognized transaction schema. Columns: {list(df.columns)}")


def load_debts(path: str | Path) -> dict[str, Any]:
    """Load debts JSON: {debts: [...], extra_monthly_payment_budget: float}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Debts file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return normalize_debts(data)


def load_debts_from_dict(data: dict[str, Any]) -> dict[str, Any]:
    return normalize_debts(data)


def normalize_debts(data: dict[str, Any]) -> dict[str, Any]:
    if "debts" not in data:
        raise ValueError("Debts JSON must contain a 'debts' list")

    debts: list[Debt] = []
    for item in data["debts"]:
        debts.append(
            Debt(
                name=str(item["name"]),
                balance=float(item["balance"]),
                apr=float(item["apr"]),
                minimum_payment=float(item["minimum_payment"]),
            )
        )

    return {
        "debts": debts,
        "extra_monthly_payment_budget": float(data.get("extra_monthly_payment_budget", 0.0)),
        "notes": data.get("notes", ""),
    }


def debts_to_serializable(debts_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "debts": [d.to_dict() if isinstance(d, Debt) else d for d in debts_payload["debts"]],
        "extra_monthly_payment_budget": debts_payload["extra_monthly_payment_budget"],
        "notes": debts_payload.get("notes", ""),
    }
