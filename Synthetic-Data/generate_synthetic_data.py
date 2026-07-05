"""
Synthetic data generator for the AI Financial Coach (gig/irregular-income) project.

Generates 3 datasets:
1. sample_gig_worker.csv   - 12 months of irregular income + categorized transactions (clean case)
2. sample_debts.json       - multi-debt profile for the Debt Analyzer Agent
3. sample_bad_case.csv     - a riskier income/spending profile where the current plan
                             should show LOW solvency probability in the Monte Carlo demo

Run: python generate_synthetic_data.py
Outputs land in ./synthetic_data/
"""

import json
import os
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
np.random.seed(42)

OUT_DIR = "synthetic_data"
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES = {
    "rent": (1200, 1200, 0),          # (mean, min, std) — fixed, no variance
    "groceries": (380, 250, 60),
    "transport": (150, 60, 45),        # gig workers: gas/rideshare fees, more variable
    "subscriptions": (45, 45, 0),
    "utilities": (140, 90, 25),
    "eating_out": (220, 50, 90),
    "misc": (160, 20, 70),
}

MERCHANTS = {
    "rent": ["Landlord Property Mgmt", "Riverside Apartments LLC"],
    "groceries": ["Whole Foods Market", "Trader Joe's", "Local Grocery Co-op", "Costco Wholesale"],
    "transport": ["Shell Gas Station", "Uber", "Lyft", "City Transit Authority"],
    "subscriptions": ["Netflix", "Spotify", "Adobe Creative Cloud", "Gym Membership"],
    "utilities": ["City Power & Water", "Comcast Internet", "State Gas Co"],
    "eating_out": ["Chipotle", "Local Diner", "Starbucks", "DoorDash"],
    "misc": ["Amazon", "Target", "CVS Pharmacy", "Home Depot"],
}


def generate_transactions_for_month(month_start, num_days=30, tightness=1.0):
    """Generate a list of categorized transactions for one month.
    tightness > 1.0 inflates spending (used for the bad-case dataset)."""
    rows = []
    for category, (mean, min_amt, std) in CATEGORIES.items():
        # number of transactions per category per month varies by category
        n_tx = {"rent": 1, "subscriptions": 3, "utilities": 2}.get(category, np.random.randint(3, 9))
        for _ in range(n_tx):
            amt = max(min_amt, np.random.normal(mean, std) * tightness / n_tx * (1 if category in ("rent",) else n_tx))
            # simpler: just spread mean across n_tx transactions with noise
            amt = max(5, (mean * tightness / n_tx) + np.random.normal(0, std / max(n_tx, 1)))
            date = month_start + pd.Timedelta(days=int(np.random.randint(0, num_days)))
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "merchant": np.random.choice(MERCHANTS[category]),
                "category": category,
                "amount": round(amt, 2),
                "type": "expense",
            })
    return rows


def generate_income_for_month(month_start, mean_income, std_income, income_type="gig"):
    """Simulate 1-4 irregular income deposits per month (gig/freelance style)."""
    rows = []
    n_payments = np.random.randint(2, 5)
    remaining = max(200, np.random.normal(mean_income, std_income))
    splits = np.random.dirichlet(np.ones(n_payments)) * remaining
    for amt in splits:
        date = month_start + pd.Timedelta(days=int(np.random.randint(0, 28)))
        source = np.random.choice(
            ["Uber Driver Payout", "Freelance Client Invoice", "DoorDash Payout",
             "Upwork Payment", "Etsy Shop Sales", "Consulting Client Payment"]
        )
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "merchant": source,
            "category": "income",
            "amount": round(float(amt), 2),
            "type": "income",
        })
    return rows


def build_dataset(months=12, mean_income=3200, std_income=950, spend_tightness=1.0, start="2025-01-01"):
    all_rows = []
    month_starts = pd.date_range(start, periods=months, freq="MS")
    for m in month_starts:
        all_rows.extend(generate_income_for_month(m, mean_income, std_income))
        all_rows.extend(generate_transactions_for_month(m, tightness=spend_tightness))
    df = pd.DataFrame(all_rows).sort_values("date").reset_index(drop=True)
    return df


# -----------------------------------------------------------------------
# Dataset 1: Clean gig worker case — realistic variance, manageable spending
# -----------------------------------------------------------------------
gig_worker_df = build_dataset(
    months=12, mean_income=3200, std_income=950, spend_tightness=1.0
)
gig_worker_df.to_csv(f"{OUT_DIR}/sample_gig_worker.csv", index=False)

# -----------------------------------------------------------------------
# Dataset 2: Multi-debt profile (JSON, for Debt Analyzer Agent)
# -----------------------------------------------------------------------
debts = {
    "debts": [
        {
            "name": "Credit Card - Visa",
            "balance": 4200.00,
            "apr": 24.99,
            "minimum_payment": 105.00,
        },
        {
            "name": "Credit Card - Store Card",
            "balance": 1150.00,
            "apr": 27.99,
            "minimum_payment": 40.00,
        },
        {
            "name": "Personal Loan",
            "balance": 6800.00,
            "apr": 11.5,
            "minimum_payment": 210.00,
        },
        {
            "name": "Student Loan",
            "balance": 15200.00,
            "apr": 5.8,
            "minimum_payment": 165.00,
        },
    ],
    "extra_monthly_payment_budget": 300.00,
    "notes": "extra_monthly_payment_budget is the amount available above minimums, to be allocated by avalanche/snowball strategy."
}
with open(f"{OUT_DIR}/sample_debts.json", "w") as f:
    json.dump(debts, f, indent=2)

# -----------------------------------------------------------------------
# Dataset 3: Bad case — high income variance + tight/overspending budget
#            This should produce a LOW solvency probability in the Monte
#            Carlo simulation, to demonstrate the tool catching real risk.
# -----------------------------------------------------------------------
bad_case_df = build_dataset(
    months=12, mean_income=2600, std_income=1400, spend_tightness=1.35, start="2025-01-01"
)
bad_case_df.to_csv(f"{OUT_DIR}/sample_bad_case.csv", index=False)

# Matching (heavier) debt file for the bad case
bad_case_debts = {
    "debts": [
        {"name": "Credit Card - Visa", "balance": 7800.00, "apr": 26.99, "minimum_payment": 195.00},
        {"name": "Credit Card - Store Card", "balance": 2400.00, "apr": 29.99, "minimum_payment": 80.00},
        {"name": "Payday-style Loan", "balance": 1500.00, "apr": 36.0, "minimum_payment": 150.00},
    ],
    "extra_monthly_payment_budget": 60.00,
    "notes": "Low extra budget relative to income variance — designed to show low solvency probability."
}
with open(f"{OUT_DIR}/sample_bad_case_debts.json", "w") as f:
    json.dump(bad_case_debts, f, indent=2)

# -----------------------------------------------------------------------
# Summary printout
# -----------------------------------------------------------------------
print("Generated datasets in ./synthetic_data/:\n")

for name, df in [("sample_gig_worker.csv", gig_worker_df), ("sample_bad_case.csv", bad_case_df)]:
    income = df[df["type"] == "income"]["amount"]
    expense = df[df["type"] == "expense"]["amount"]
    monthly_income = income.groupby(pd.to_datetime(df.loc[income.index, "date"]).dt.to_period("M")).sum()
    print(f"{name}")
    print(f"  rows: {len(df)}")
    print(f"  monthly income  -> mean: {monthly_income.mean():.2f}, std: {monthly_income.std():.2f}")
    print(f"  total expenses/mo (avg): {expense.sum() / 12:.2f}")
    print()

print("sample_debts.json  (clean case debts)")
print("sample_bad_case_debts.json  (bad case debts)")
