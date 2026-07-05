# AI Financial Coach for Irregular-Income Workers

> Every budgeting app assumes you get paid the same amount every month. 41%+ of workers now have irregular income (freelance, gig, commission, seasonal). A bad month isn't an edge case for them — it's a certainty. **Our coach doesn't give you one plan; it tells you the odds your plan actually survives a bad month.**

A multi-agent financial coach that models income **variance**, compares debt strategies deterministically, and runs thousands of **Monte Carlo** futures so you see probability of success — not a single static number.

**Key design rule:** the LLM never does arithmetic. Python (numpy/pandas) computes every number; the LLM only narrates (and optionally categorizes merchants).

## Features

- **Income Profiler** — mean, std, p10/p90, gig vs paycheck classification
- **Debt Analyzer** — avalanche vs snowball, interest and months-to-free
- **Budget Advisor** — category spend, needs vs wants, trend flags
- **Monte Carlo Agent** — vectorized solvency probability + recommended buffer
- **Coach Narrator** — plain-English advice + **safe-to-spend** this week (₹)

## Quick start

```bash
cd finance-coach
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Optional LLM narration (Groq primary, Gemini fallback):

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
# edit .env with GROQ_API_KEY and/or GOOGLE_API_KEY
```

Without API keys the app still runs using built-in narrative templates.

### Run the dashboard

```bash
streamlit run frontend/app.py
```

### Run tests

```bash
pytest -q
```

## Demo path (pitch)

1. Open with the irregular-income problem statement above.
2. Load **Gig worker (happy path)** → show safe-to-spend, fan chart, solid solvency.
3. Switch to **Bad case (risky plan)** → solvency drops; coach recommends a larger emergency buffer.
4. Optional: **Kaggle paycheck** — low variance contrast (“traditional tools work for this person; ours shines on gig data”).
5. Point at the **fan chart** — uncertainty cone, not a single line.

## Data

| Preset | Files |
|---|---|
| Gig worker | `data/sample_gig_worker.csv` + `data/sample_debts.json` |
| Bad case | `data/sample_bad_case.csv` + `data/sample_bad_case_debts.json` |
| Kaggle paycheck | `data/personal_transactions.csv` + `data/sample_paycheck_debts.json` |
| Kaggle personal finance | `data/Personal_Finance_Dataset.csv` + `data/sample_kaggle_pf_debts.json` |

Upload any matching CSV + debts JSON from the sidebar. Parsers auto-detect synthetic, Personal Finance, and personal_transactions schemas.

## Architecture

```
CSV/JSON → parsing → LangGraph orchestrator
  → Income / Debt / Budget agents (deterministic core + narration)
  → Monte Carlo agent (vectorized numpy)
  → Coach narrator (safe-to-spend + summary)
  → Streamlit dashboard
```

## Project layout

```
finance-coach/
├── agents/           # Agent wrappers (calc + optional LLM)
├── core/             # schemas, parsing, calculations, simulation
├── orchestrator/     # LangGraph pipeline
├── llm/              # Groq → Gemini client + prompts
├── data/             # Sample datasets
├── frontend/app.py   # Streamlit UI
└── tests/            # Unit tests for math + parsing
```

## Pitch line

Unlike traditional budgeting tools that assume a fixed paycheck, this coach models your actual income variance and runs thousands of simulated futures — so instead of one static plan, you get a probability: your real chance of staying solvent, and exactly what buffer to build to improve those odds.
