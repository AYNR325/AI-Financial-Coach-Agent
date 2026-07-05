# AI Financial Coach for Irregular-Income Workers

> Every budgeting app assumes you get paid the same amount every month. A huge and growing share of workers now have irregular income — freelance, gig, commission, seasonal. A bad month isn't an edge case for them, it's a certainty. **This coach doesn't give you one plan; it tells you the odds your plan actually survives a bad month.**

A multi-agent financial coach, built specifically for gig workers and freelancers with irregular income, that models income **variance** (not just an average), compares debt payoff strategies with real amortization math, and runs thousands of **Monte Carlo** simulated futures so you see a *probability* of staying solvent — not a single static budget.

**Core design rule the whole project follows:** the LLM never does arithmetic. Every number (income stats, debt payoff schedules, solvency probability) is computed in plain Python (numpy/pandas). The LLM's only job is to explain those pre-computed numbers in plain English. This means the app gives numerically correct results even if the LLM narration fails, times out, or isn't configured at all.

---

## Table of Contents
- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Running the app](#running-the-app)
- [Running tests](#running-tests)
- [How to use / test it](#how-to-use--test-it)
- [Understanding the dashboard](#understanding-the-dashboard)
- [Uploading your own data](#uploading-your-own-data)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

---

## What it does

You give it two things:
1. **A transaction history** (CSV — income + expenses)
2. **A list of debts** (JSON — balances, APRs, minimum payments)

It runs 5 agents in sequence and gives you back:

| Agent | What it computes |
|---|---|
| **Income Profiler** | Mean/std-dev monthly income, a "bad month" (p10) and "good month" (p90) estimate, and classifies you as `gig/freelance` or `stable/paycheck` based on income variance |
| **Debt Analyzer** | Full avalanche (highest-interest-first) vs. snowball (smallest-balance-first) payoff schedules — exact months-to-debt-free and total interest for each, plus a recommendation |
| **Budget Advisor** | Categorized monthly spend, a needs-vs-wants split, and which categories are trending upward |
| **Monte Carlo Agent** | Runs thousands of simulated financial futures using bootstrap-resampling from your *actual* historical income months (not an assumed bell curve), and reports: probability you stay solvent, the range of months until you're debt-free (p10/median/p90), and the buffer size needed to hit ~90% solvency |
| **Coach Narrator** | Combines everything into one plain-English summary, leading with a concrete **safe-to-spend this week** number |

All of this is shown on an interactive Streamlit dashboard with charts (a solvency gauge, an uncertainty "fan chart" for debt payoff, income/budget breakdowns), and can be exported as a self-contained HTML or Markdown report.

---

## Architecture

```
                 Transactions CSV + Debts JSON
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Parsing layer      │  core/parsing.py
                 │ (auto-detects 3      │
                 │  CSV schemas)        │
                 └──────────┬──────────┘
                            ▼
        ┌───────────────────────────────────────┐
        │         LangGraph Orchestrator          │  orchestrator/graph.py
        └───────────────────┬───────────────────┘
                            ▼
                 ┌─────────────────────┐
                 │   "analyze" node      │
                 │  runs 3 agents:       │
                 │  • Income Profiler    │  agents/income_profiler.py
                 │  • Debt Analyzer      │  agents/debt_analyzer.py
                 │  • Budget Advisor     │  agents/budget_advisor.py
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │  "monte_carlo" node    │  agents/monte_carlo_agent.py
                 │  (vectorized numpy    │  core/simulation.py
                 │   simulation, no LLM  │
                 │   math involved)      │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ "coach_narrator" node │  agents/coach_narrator.py
                 │ combines everything   │
                 │ into final summary    │
                 └──────────┬──────────┘
                            ▼
                 Streamlit Dashboard (frontend/app.py)
                 + HTML/Markdown report export (frontend/report.py)
```

Every agent follows the same internal pattern: **calculate first (deterministic Python), narrate second (optional LLM call)**. If the LLM call fails or is disabled, each agent falls back to a template-based plain-English narrative built directly from the same numbers — so the app never breaks and never invents a number, regardless of LLM availability.

### LLM provider chain
`llm/client.py` tries **Groq** first (fast, generous free tier), then falls back to **Google Gemini** if Groq is unavailable or a key isn't set, and falls back further to the deterministic template narratives if neither is configured. See `llm/prompts.py` for the exact prompts — every one of them explicitly instructs the model to use ₹ (INR) and never invent numbers outside the data it's given.

---

## Project structure

```
finance-coach/
├── agents/                    # One file per agent — calculation call + optional LLM narration
│   ├── income_profiler.py
│   ├── debt_analyzer.py
│   ├── budget_advisor.py
│   ├── monte_carlo_agent.py
│   └── coach_narrator.py
├── core/                       # All deterministic logic — no LLM calls anywhere in this folder
│   ├── schemas.py              # Typed data structures (Debt, IncomeProfile, SimulationResult, etc.)
│   ├── parsing.py              # CSV/JSON loading + schema auto-detection
│   ├── calculations.py         # Income stats, avalanche/snowball amortization, budget baseline
│   ├── simulation.py           # Vectorized Monte Carlo engine (bootstrap resampling)
│   └── currency.py             # INR formatting helpers (₹ symbol, comma grouping)
├── llm/
│   ├── client.py               # Groq → Gemini → none fallback chain
│   └── prompts.py              # All prompt templates (currency rules baked in)
├── orchestrator/
│   └── graph.py                # LangGraph pipeline wiring all agents together
├── frontend/
│   ├── app.py                  # Streamlit dashboard (sidebar controls, charts, tabs)
│   └── report.py               # Downloadable HTML + Markdown report builder
├── data/                       # Sample datasets + templates
│   ├── template_transactions.csv
│   ├── template_debts.json
│   ├── sample_gig_worker.csv / sample_debts.json          (preset: Gig worker — happy path)
│   ├── sample_bad_case.csv / sample_bad_case_debts.json   (preset: Bad case — risky plan)
│   ├── personal_transactions.csv / sample_paycheck_debts.json      (preset: Kaggle paycheck)
│   ├── Personal_Finance_Dataset.csv / sample_kaggle_pf_debts.json  (preset: Kaggle personal finance)
│   └── generate_synthetic_data.py                          # Regenerate the synthetic presets
├── scripts/
│   └── smoke_pipeline.py       # Runs all 4 presets end-to-end from the command line
├── tests/
│   ├── test_calculations.py    # Debt payoff + income stats correctness
│   ├── test_parsing.py         # Schema auto-detection + debt loading
│   └── test_simulation.py      # Monte Carlo behavior (e.g. bad case < gig case solvency)
├── .env.example                # Template for API keys (copy to .env, never commit .env itself)
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.10+
- pip

### 1. Create a virtual environment and install dependencies
```bash
cd finance-coach
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. (Optional) Set up free LLM narration
The app works fully without this step — it'll use built-in template narration instead. To enable AI-written narration:

```bash
cp .env.example .env      # macOS/Linux
copy .env.example .env    # Windows
```

Edit `.env` and add one or both:
```
GROQ_API_KEY=your_key_here       # https://console.groq.com — free, no card required
GOOGLE_API_KEY=your_key_here     # https://aistudio.google.com/apikey — free, no card required
```
Groq is tried first (fast, generous daily free limit); Gemini is the fallback if Groq isn't configured or hits a rate limit. `.env` is already in `.gitignore` — never commit real keys.

---

## Running the app

```bash
streamlit run frontend/app.py
```
This opens the dashboard in your browser (default: `http://localhost:8501`).

### Running tests
```bash
pytest -q
```
Expect `19 passed`. These check debt-payoff math correctness, income classification, schema parsing, and that the "bad case" preset produces meaningfully lower solvency than the "gig worker" preset (the core property the whole pitch depends on).

### Running the smoke pipeline (no UI, just terminal output)
```bash
python scripts/smoke_pipeline.py
```
Runs all 4 preset datasets through the full pipeline with LLM narration disabled and prints solvency/buffer/coach-message for each — useful for a fast sanity check after any code change, without opening Streamlit.

---

## How to use / test it

### Fastest path (no upload needed)
1. Run `streamlit run frontend/app.py`
2. In the sidebar, pick a **Preset demo** — start with **"Gig worker (happy path)"**
3. (Optional) Adjust: debt strategy preference, existing emergency savings, simulation horizon, whether to use LLM narration
4. Click **Run coach**
5. Explore the 5 tabs: **Overview**, **Income**, **Debt**, **Budget**, **Details**
6. Switch the preset to **"Bad case (risky plan)"** and re-run — compare the solvency gauge and fan chart side by side; this contrast is the core of the pitch

### The 4 built-in presets
| Preset | Story it tells |
|---|---|
| **Gig worker (happy path)** | Irregular but manageable income, ~91% solvency — the primary "our tool works" demo |
| **Bad case (risky plan)** | Expenses close to/above income, heavy debt, ~0% solvency — shows the tool catching real risk |
| **Kaggle paycheck (stable income)** | A steady biweekly-paycheck dataset — contrast case showing the tool also handles traditional stable income correctly (classified as `stable/paycheck`, not `gig/freelance`) |
| **Kaggle personal finance** | An alternate real-world CSV schema, to demonstrate the parser's schema auto-detection |

### Testing with your own / additional data
Use the **"Download transactions template"** / **"Download debts template"** buttons in the sidebar to get the exact expected format, fill in your own numbers, and upload via the **Transactions CSV** / **Debts JSON** file uploaders. See [Uploading your own data](#uploading-your-own-data) below for the exact schema.

---

## Understanding the dashboard

- **Overview tab** — solvency gauge (probability of staying financially solvent over your chosen horizon), buffer progress (how close your current savings are to the recommended buffer), and the headline safe-to-spend number.
- **Income tab** — monthly income history chart, income range (p10/p90) visualization, and the gig/freelance vs. stable/paycheck classification.
- **Debt tab** — avalanche vs. snowball comparison chart, full payoff schedule, and which strategy is recommended and why.
- **Budget tab** — needs-vs-wants donut chart and per-category monthly spend breakdown, with any categories flagged as trending upward.
- **Details tab** — raw computed numbers behind every chart, useful for verifying the math directly against your input data.

**Download full report** (top of the page, once you've run the coach): exports everything above as a single self-contained HTML file (styled, works offline, light/dark mode aware) or a plain Markdown file — useful for judges to take away after your demo, or for a real user to save/print.

---

## Uploading your own data

### Transactions CSV
Recommended format (matches `data/template_transactions.csv`):

| Column | Meaning | Example |
|---|---|---|
| `date` | Transaction date | `2025-01-05` |
| `merchant` | Who paid you / who you paid | `Upwork Payment` |
| `category` | Spend category | `rent`, `groceries`, `income`, etc. |
| `amount` | Amount in ₹ (positive number) | `15000` |
| `type` | `income` or `expense` | `income` |

Two other schemas are auto-detected if you have real exported data in these shapes:
- **Kaggle-style:** `Date, Transaction Description, Category, Amount, Type`
- **Bank-style:** `Date, Description, Amount, Transaction Type, Category` (where `credit` = income, `debit` = expense)

At least ~3 months of transaction history is recommended — the Income Profiler needs enough data points to compute a meaningful variance estimate.

### Debts JSON
Exact shape (matches `data/template_debts.json`):
```json
{
  "debts": [
    {
      "name": "Credit Card - HDFC",
      "balance": 45000,
      "apr": 36.0,
      "minimum_payment": 2250
    }
  ],
  "extra_monthly_payment_budget": 3000
}
```
- `balance`, `minimum_payment`, `extra_monthly_payment_budget` are in ₹
- `apr` is the annual interest rate as a percentage (e.g. `36.0` means 36%)
- `extra_monthly_payment_budget` is money available *above* the sum of all minimum payments, to be allocated toward extra payoff
- If you skip uploading debts, the app uses the selected preset's sample debts instead

---

## Known limitations

- **Currency is fixed to INR (₹)** throughout the app, including in LLM prompts. The two Kaggle-derived presets ("Kaggle paycheck", "Kaggle personal finance") are real third-party datasets originally recorded in USD; their debt files were rescaled to roughly match, but the underlying transaction amounts are not true INR figures — treat these two presets as schema-adapter demonstrations, not real financial scenarios. The two synthetic presets ("Gig worker", "Bad case") and the templates are fully INR-native and safe to use as your primary demo/test data.
- **No real bank integration** — all data is CSV/JSON upload only, by design, for a hackathon scope.
- **Simulation assumes independence between months** beyond the historical resampling — it doesn't model longer-term trends like a permanently lost client (see the "shock scenarios" idea in future work below).
- **LLM narration quality depends on the free-tier model used** (Llama 3.3 70B via Groq, or Gemini Flash) — narration is generally solid but is not a substitute for professional financial advice, and the app should be framed as educational/illustrative in any pitch.

### Possible future additions (not implemented in this build)
Named shock scenarios (e.g. "monsoon slowdown", "medical emergency"), a two-agent aggressive-vs-safety debate before the coach's final recommendation, a negotiation script generator for your highest-APR debt, and a debt-consolidation what-if calculator were considered but intentionally left out of this submission to keep the core pipeline focused and fully tested.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Pipeline failed: ...` error in the UI | Usually a malformed CSV/JSON upload — check it matches the template format exactly (see above), or download and inspect the template files for comparison |
| LLM narration toggle shows a warning about no API keys | `.env` doesn't exist or is empty — copy `.env.example` to `.env` and add a Groq and/or Gemini key, then restart Streamlit |
| App still uses template narration even with keys set | Restart Streamlit after editing `.env` — environment variables are only read at startup |
| `pytest` fails after editing sample data | If you regenerate `data/sample_*.csv` or `*_debts.json`, check `tests/test_parsing.py` and `tests/test_calculations.py` for hardcoded values (debt counts, income classification) that may need updating to match |
| Solvency shows 100% or 0% and looks suspicious | Check your income/expense ratio in the uploaded data — a 100% or 0% result is mathematically correct at the extremes (e.g. expenses far below or far above income for every simulated month), not a bug |

---

