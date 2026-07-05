# """
# Synthetic data generator for the AI Financial Coach (gig/irregular-income) project.

# Generates 3 datasets:
# 1. sample_gig_worker.csv   - 12 months of irregular income + categorized transactions (clean case)
# 2. sample_debts.json       - multi-debt profile for the Debt Analyzer Agent
# 3. sample_bad_case.csv     - a riskier income/spending profile where the current plan
#                              should show LOW solvency probability in the Monte Carlo demo

# Run: python generate_synthetic_data.py
# Outputs land in ./synthetic_data/
# """

# import json
# import os
# import numpy as np
# import pandas as pd
# from faker import Faker

# fake = Faker()
# Faker.seed(42)
# np.random.seed(42)

# OUT_DIR = "synthetic_data"
# os.makedirs(OUT_DIR, exist_ok=True)

# CATEGORIES = {
#     "rent": (1200, 1200, 0),          # (mean, min, std) — fixed, no variance
#     "groceries": (380, 250, 60),
#     "transport": (150, 60, 45),        # gig workers: gas/rideshare fees, more variable
#     "subscriptions": (45, 45, 0),
#     "utilities": (140, 90, 25),
#     "eating_out": (220, 50, 90),
#     "misc": (160, 20, 70),
# }

# MERCHANTS = {
#     "rent": ["Landlord Property Mgmt", "Riverside Apartments LLC"],
#     "groceries": ["Whole Foods Market", "Trader Joe's", "Local Grocery Co-op", "Costco Wholesale"],
#     "transport": ["Shell Gas Station", "Uber", "Lyft", "City Transit Authority"],
#     "subscriptions": ["Netflix", "Spotify", "Adobe Creative Cloud", "Gym Membership"],
#     "utilities": ["City Power & Water", "Comcast Internet", "State Gas Co"],
#     "eating_out": ["Chipotle", "Local Diner", "Starbucks", "DoorDash"],
#     "misc": ["Amazon", "Target", "CVS Pharmacy", "Home Depot"],
# }


# def generate_transactions_for_month(month_start, num_days=30, tightness=1.0):
#     """Generate a list of categorized transactions for one month.
#     tightness > 1.0 inflates spending (used for the bad-case dataset)."""
#     rows = []
#     for category, (mean, min_amt, std) in CATEGORIES.items():
#         # number of transactions per category per month varies by category
#         n_tx = {"rent": 1, "subscriptions": 3, "utilities": 2}.get(category, np.random.randint(3, 9))
#         for _ in range(n_tx):
#             amt = max(min_amt, np.random.normal(mean, std) * tightness / n_tx * (1 if category in ("rent",) else n_tx))
#             # simpler: just spread mean across n_tx transactions with noise
#             amt = max(5, (mean * tightness / n_tx) + np.random.normal(0, std / max(n_tx, 1)))
#             date = month_start + pd.Timedelta(days=int(np.random.randint(0, num_days)))
#             rows.append({
#                 "date": date.strftime("%Y-%m-%d"),
#                 "merchant": np.random.choice(MERCHANTS[category]),
#                 "category": category,
#                 "amount": round(amt, 2),
#                 "type": "expense",
#             })
#     return rows


# def generate_income_for_month(month_start, mean_income, std_income, income_type="gig"):
#     """Simulate 1-4 irregular income deposits per month (gig/freelance style)."""
#     rows = []
#     n_payments = np.random.randint(2, 5)
#     remaining = max(200, np.random.normal(mean_income, std_income))
#     splits = np.random.dirichlet(np.ones(n_payments)) * remaining
#     for amt in splits:
#         date = month_start + pd.Timedelta(days=int(np.random.randint(0, 28)))
#         source = np.random.choice(
#             ["Uber Driver Payout", "Freelance Client Invoice", "DoorDash Payout",
#              "Upwork Payment", "Etsy Shop Sales", "Consulting Client Payment"]
#         )
#         rows.append({
#             "date": date.strftime("%Y-%m-%d"),
#             "merchant": source,
#             "category": "income",
#             "amount": round(float(amt), 2),
#             "type": "income",
#         })
#     return rows


# def build_dataset(months=12, mean_income=3200, std_income=950, spend_tightness=1.0, start="2025-01-01"):
#     all_rows = []
#     month_starts = pd.date_range(start, periods=months, freq="MS")
#     for m in month_starts:
#         all_rows.extend(generate_income_for_month(m, mean_income, std_income))
#         all_rows.extend(generate_transactions_for_month(m, tightness=spend_tightness))
#     df = pd.DataFrame(all_rows).sort_values("date").reset_index(drop=True)
#     return df


# # -----------------------------------------------------------------------
# # Dataset 1: Clean gig worker case — realistic variance, manageable spending
# # -----------------------------------------------------------------------
# gig_worker_df = build_dataset(
#     months=12, mean_income=3200, std_income=950, spend_tightness=1.0
# )
# gig_worker_df.to_csv(f"{OUT_DIR}/sample_gig_worker.csv", index=False)

# # -----------------------------------------------------------------------
# # Dataset 2: Multi-debt profile (JSON, for Debt Analyzer Agent)
# # -----------------------------------------------------------------------
# debts = {
#     "debts": [
#         {
#             "name": "Credit Card - Visa",
#             "balance": 4200.00,
#             "apr": 24.99,
#             "minimum_payment": 105.00,
#         },
#         {
#             "name": "Credit Card - Store Card",
#             "balance": 1150.00,
#             "apr": 27.99,
#             "minimum_payment": 40.00,
#         },
#         {
#             "name": "Personal Loan",
#             "balance": 6800.00,
#             "apr": 11.5,
#             "minimum_payment": 210.00,
#         },
#         {
#             "name": "Student Loan",
#             "balance": 15200.00,
#             "apr": 5.8,
#             "minimum_payment": 165.00,
#         },
#     ],
#     "extra_monthly_payment_budget": 300.00,
#     "notes": "extra_monthly_payment_budget is the amount available above minimums, to be allocated by avalanche/snowball strategy."
# }
# with open(f"{OUT_DIR}/sample_debts.json", "w") as f:
#     json.dump(debts, f, indent=2)

# # -----------------------------------------------------------------------
# # Dataset 3: Bad case — high income variance + tight/overspending budget
# #            This should produce a LOW solvency probability in the Monte
# #            Carlo simulation, to demonstrate the tool catching real risk.
# # -----------------------------------------------------------------------
# bad_case_df = build_dataset(
#     months=12, mean_income=2600, std_income=1400, spend_tightness=1.35, start="2025-01-01"
# )
# bad_case_df.to_csv(f"{OUT_DIR}/sample_bad_case.csv", index=False)

# # Matching (heavier) debt file for the bad case
# bad_case_debts = {
#     "debts": [
#         {"name": "Credit Card - Visa", "balance": 7800.00, "apr": 26.99, "minimum_payment": 195.00},
#         {"name": "Credit Card - Store Card", "balance": 2400.00, "apr": 29.99, "minimum_payment": 80.00},
#         {"name": "Payday-style Loan", "balance": 1500.00, "apr": 36.0, "minimum_payment": 150.00},
#     ],
#     "extra_monthly_payment_budget": 60.00,
#     "notes": "Low extra budget relative to income variance — designed to show low solvency probability."
# }
# with open(f"{OUT_DIR}/sample_bad_case_debts.json", "w") as f:
#     json.dump(bad_case_debts, f, indent=2)

# # -----------------------------------------------------------------------
# # Summary printout
# # -----------------------------------------------------------------------
# print("Generated datasets in ./synthetic_data/:\n")

# for name, df in [("sample_gig_worker.csv", gig_worker_df), ("sample_bad_case.csv", bad_case_df)]:
#     income = df[df["type"] == "income"]["amount"]
#     expense = df[df["type"] == "expense"]["amount"]
#     monthly_income = income.groupby(pd.to_datetime(df.loc[income.index, "date"]).dt.to_period("M")).sum()
#     print(f"{name}")
#     print(f"  rows: {len(df)}")
#     print(f"  monthly income  -> mean: {monthly_income.mean():.2f}, std: {monthly_income.std():.2f}")
#     print(f"  total expenses/mo (avg): {expense.sum() / 12:.2f}")
#     print()

# print("sample_debts.json  (clean case debts)")
# print("sample_bad_case_debts.json  (bad case debts)")


"""
Synthetic data generator for the AI Financial Coach (gig/irregular-income) project.
INR-denominated, matching the template_transactions.csv / template_debts.json format.

Generates 4 files (overwrites in place):
1. sample_gig_worker.csv        - happy path: irregular but manageable gig income (INR)
2. sample_debts.json             - modest multi-debt profile matched to the happy path (INR)
3. sample_bad_case.csv           - risky case: high variance + expenses close to/above income (INR)
4. sample_bad_case_debts.json    - heavier debt load matched to the bad case (INR)

Run: python generate_synthetic_data.py
Outputs land in this directory (data/), overwriting the previous versions.
"""

import json
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

CATEGORIES = {
    "rent": (12000, 12000, 0),
    "groceries": (3000, 1800, 450),
    "transport": (1400, 500, 400),
    "subscriptions": (699, 699, 0),
    "utilities": (1900, 1200, 300),
    "eating_out": (2200, 400, 800),
    "misc": (1500, 300, 600),
}

MERCHANTS = {
    "rent": ["Landlord Property Mgmt", "Shanti Nagar Apartments"],
    "groceries": ["Local Grocery Store", "DMart", "Reliance Fresh", "Big Bazaar"],
    "transport": ["Uber", "Ola Cabs", "Local Metro Card Recharge", "Petrol Pump"],
    "subscriptions": ["Netflix", "Spotify", "Hotstar", "Gym Membership"],
    "utilities": ["City Power & Water", "Airtel Broadband", "Piped Gas Co"],
    "eating_out": ["Cafe Coffee Day", "Zomato Order", "Swiggy Order", "Local Dhaba"],
    "misc": ["Amazon", "Flipkart", "Local Pharmacy", "Big Bazaar"],
}

INCOME_SOURCES = [
    "Freelance Client Invoice", "Upwork Payment", "Fiverr Payout",
    "Zomato Delivery Payout", "Swiggy Delivery Payout", "Consulting Client Payment",
]


def generate_month_transactions(month_start, rng, spend_tightness=1.0):
    rows = []
    for category, (mean, min_amt, std) in CATEGORIES.items():
        n_tx = {"rent": 1, "subscriptions": 2, "utilities": 2}.get(category, int(rng.integers(2, 6)))
        for _ in range(n_tx):
            amt = max(
                min_amt / max(n_tx, 1),
                (mean * spend_tightness / n_tx) + rng.normal(0, std / max(n_tx, 1)),
            )
            date = month_start + pd.Timedelta(days=int(rng.integers(0, 27)))
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "merchant": rng.choice(MERCHANTS[category]),
                "category": category,
                "amount": int(round(amt, -1)),
                "type": "expense",
            })
    return rows


def generate_month_income(month_start, mean_income, std_income, rng):
    rows = []
    n_payments = int(rng.integers(2, 4))
    total = max(2000, rng.normal(mean_income, std_income))
    splits = rng.dirichlet(np.ones(n_payments)) * total
    for amt in splits:
        date = month_start + pd.Timedelta(days=int(rng.integers(0, 25)))
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "merchant": rng.choice(INCOME_SOURCES),
            "category": "income",
            "amount": int(round(float(amt), -2)),
            "type": "income",
        })
    return rows


def build_dataset(mean_income, std_income, spend_tightness, seed, months=12, start="2025-01-01"):
    rng = np.random.default_rng(seed)
    all_rows = []
    for m in pd.date_range(start, periods=months, freq="MS"):
        all_rows.extend(generate_month_income(m, mean_income, std_income, rng))
        all_rows.extend(generate_month_transactions(m, rng, spend_tightness))
    df = pd.DataFrame(all_rows).sort_values("date").reset_index(drop=True)
    return df[["date", "merchant", "category", "amount", "type"]]


def main():
    # Happy path: gig worker, irregular but manageable income.
    gig_df = build_dataset(mean_income=32000, std_income=11000, spend_tightness=1.0, seed=11)
    gig_df.to_csv(os.path.join(HERE, "sample_gig_worker.csv"), index=False)

    gig_debts = {
        "debts": [
            {"name": "Credit Card - HDFC", "balance": 35000, "apr": 34.0, "minimum_payment": 1750},
            {"name": "Personal Loan", "balance": 60000, "apr": 13.0, "minimum_payment": 2200},
        ],
        "extra_monthly_payment_budget": 3500,
        "notes": "extra_monthly_payment_budget is extra money above minimums used for avalanche/snowball payoff.",
    }
    with open(os.path.join(HERE, "sample_debts.json"), "w") as f:
        json.dump(gig_debts, f, indent=2)

    # Bad case: higher income variance, expenses close to/above income, heavier debt.
    bad_df = build_dataset(mean_income=22000, std_income=9500, spend_tightness=1.3, seed=99)
    bad_df.to_csv(os.path.join(HERE, "sample_bad_case.csv"), index=False)

    bad_debts = {
        "debts": [
            {"name": "Credit Card - ICICI", "balance": 78000, "apr": 42.0, "minimum_payment": 3900},
            {"name": "Credit Card - SBI", "balance": 24000, "apr": 38.0, "minimum_payment": 1200},
            {"name": "Personal Loan", "balance": 65000, "apr": 16.0, "minimum_payment": 2600},
        ],
        "extra_monthly_payment_budget": 500,
        "notes": "Low extra budget relative to income variance -- designed to show low solvency probability.",
    }
    with open(os.path.join(HERE, "sample_bad_case_debts.json"), "w") as f:
        json.dump(bad_debts, f, indent=2)

    for name, df in [("sample_gig_worker.csv", gig_df), ("sample_bad_case.csv", bad_df)]:
        income = df[df["type"] == "income"]["amount"]
        expense = df[df["type"] == "expense"]["amount"]
        monthly_income = income.groupby(pd.to_datetime(df.loc[income.index, "date"]).dt.to_period("M")).sum()
        print(f"{name}")
        print(f"  rows: {len(df)}")
        print(f"  monthly income -> mean: Rs.{monthly_income.mean():.0f}, std: Rs.{monthly_income.std():.0f}")
        print(f"  avg monthly expenses: Rs.{expense.sum()/12:.0f}")
        print()

    print(f"total debt (happy path): Rs.{sum(d['balance'] for d in gig_debts['debts'])}")
    print(f"total debt (bad case):   Rs.{sum(d['balance'] for d in bad_debts['debts'])}")


if __name__ == "__main__":
    main()
