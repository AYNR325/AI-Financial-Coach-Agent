"""Smoke-test the full coach pipeline on preset datasets."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.parsing import load_debts, load_transactions
from orchestrator.graph import run_coach_pipeline

DATA = ROOT / "data"

SCENARIOS = [
    ("gig", "sample_gig_worker.csv", "sample_debts.json"),
    ("paycheck", "personal_transactions.csv", "sample_paycheck_debts.json"),
    ("kaggle_pf", "Personal_Finance_Dataset.csv", "sample_kaggle_pf_debts.json"),
    ("bad", "sample_bad_case.csv", "sample_bad_case_debts.json"),
]


def main() -> None:
    for name, tx, debts in SCENARIOS:
        tx_df = load_transactions(DATA / tx)
        d = load_debts(DATA / debts)
        r = run_coach_pipeline(
            tx_df,
            d["debts"],
            d["extra_monthly_payment_budget"],
            use_llm=False,
            seed=42,
        )
        print(f"=== {name} ===")
        print(
            f"  solvency={r['simulation'].solvency_probability:.2%} "
            f"buffer={r['simulation'].recommended_buffer}"
        )
        coach = (r.get("coach_message") or "")[:100].encode("ascii", "replace").decode("ascii")
        print(f"  coach: {coach}...")


if __name__ == "__main__":
    main()
