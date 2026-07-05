"""Streamlit dashboard for the AI Financial Coach — visual-first UI."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.currency import inr  # noqa: E402
from core.parsing import (  # noqa: E402
    debts_to_serializable,
    load_debts,
    load_debts_from_dict,
    load_transactions,
    load_transactions_from_df,
)
try:
    from frontend.report import build_report_html, build_report_markdown  # noqa: E402
except ImportError:  # when launched as streamlit run frontend/app.py
    from report import build_report_html, build_report_markdown  # noqa: E402
from llm.client import llm_available  # noqa: E402
from orchestrator.graph import run_coach_pipeline  # noqa: E402

DATA = ROOT / "data"

PRESETS = {
    "Gig worker (happy path)": {
        "tx": DATA / "sample_gig_worker.csv",
        "debts": DATA / "sample_debts.json",
        "blurb": "High income variance, manageable debts — primary demo.",
    },
    "Bad case (risky plan)": {
        "tx": DATA / "sample_bad_case.csv",
        "debts": DATA / "sample_bad_case_debts.json",
        "blurb": "Intentionally underwater — expect ~0% solvency and ₹0 safe-to-spend.",
    },
    "Kaggle paycheck (stable income)": {
        "tx": DATA / "personal_transactions.csv",
        "debts": DATA / "sample_paycheck_debts.json",
        "blurb": "Biweekly paycheck contrast — debts scaled to paycheck income.",
    },
    "Kaggle personal finance": {
        "tx": DATA / "Personal_Finance_Dataset.csv",
        "debts": DATA / "sample_kaggle_pf_debts.json",
        "blurb": "Alternate schema adapter — debts scaled to this dataset's income.",
    },
}

COLORS = {
    "green": "#10b981",
    "amber": "#f59e0b",
    "red": "#ef4444",
    "blue": "#3b82f6",
    "indigo": "#6366f1",
    "violet": "#8b5cf6",
    "slate": "#94a3b8",
    "needs": "#0ea5e9",
    "wants": "#f472b6",
}


def _is_dark_theme() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def _theme() -> dict:
    """Theme tokens for charts — adapts to Streamlit light/dark mode."""
    dark = _is_dark_theme()
    text = "#e2e8f0" if dark else "#0f172a"
    muted = "#94a3b8" if dark else "#64748b"
    return {
        "dark": dark,
        "text": text,
        "muted": muted,
        "gauge_bg": "#1e293b" if dark else "#f1f5f9",
        "step_red": "#7f1d1d" if dark else "#fee2e2",
        "step_amber": "#78350f" if dark else "#fef3c7",
        "step_green": "#14532d" if dark else "#d1fae5",
        "grid": "rgba(148,163,184,0.2)" if dark else "rgba(100,116,139,0.15)",
        "layout": dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans, Segoe UI, sans-serif", color=text, size=13),
            margin=dict(l=40, r=20, t=50, b=40),
            xaxis=dict(gridcolor="rgba(148,163,184,0.15)", zerolinecolor=muted),
            yaxis=dict(gridcolor="rgba(148,163,184,0.15)", zerolinecolor=muted),
        ),
    }


def _layout(**overrides) -> dict:
    """Merge theme layout with overrides without duplicate xaxis/yaxis kwargs."""
    base = _theme()["layout"]
    merged = {**base, **overrides}
    for axis in ("xaxis", "yaxis"):
        if axis in base and axis in overrides and isinstance(overrides[axis], dict):
            merged[axis] = {**base[axis], **overrides[axis]}
    return merged


def _status(solvency: float) -> tuple[str, str, str]:
    if solvency >= 0.75:
        return "Healthy", COLORS["green"], "Your plan holds up in most bad-month scenarios."
    if solvency >= 0.50:
        return "Caution", COLORS["amber"], "Plan works often, but a slow season can still hurt."
    return "At risk", COLORS["red"], "Most simulated futures go negative — act on the buffer."


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'DM Sans', 'Segoe UI', sans-serif;
        }
        /* Extra top padding so content clears Streamlit's fixed header / Deploy bar */
        .block-container {
            padding-top: 3.5rem !important;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        /* Keep download buttons from sitting under the top-right Deploy menu */
        div[data-testid="stDownloadButton"] {
            margin-bottom: 0.5rem;
        }
        div[data-testid="stDownloadButton"] button {
            white-space: nowrap;
        }

        /* Hero stays high-contrast on both themes */
        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #0284c7 140%);
            border-radius: 20px;
            padding: 1.4rem 1.6rem;
            color: #ffffff;
            margin-bottom: 1rem;
            box-shadow: 0 12px 40px rgba(15, 23, 42, 0.25);
        }
        .hero-label, .hero-value, .hero-sub, .hero-hint {
            color: #ffffff !important;
        }
        .hero-label { font-size: 0.85rem; opacity: 0.9; letter-spacing: 0.04em; text-transform: uppercase; }
        .hero-value { font-size: 2.8rem; font-weight: 700; line-height: 1.1; margin: 0.2rem 0; }
        .hero-sub { font-size: 1rem; opacity: 0.92; }
        .hero-hint { margin-top: 0.7rem; opacity: 0.9; max-width: 280px; }
        .status-pill {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.85rem;
            background: rgba(15, 23, 42, 0.45);
            border: 1px solid currentColor;
        }

        /* Cards use Streamlit theme tokens so light/dark both stay readable */
        .metric-card, .action-card {
            background: var(--secondary-background-color);
            color: var(--text-color);
            border: 1px solid rgba(128,128,128,0.28);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
            height: 100%;
        }
        .metric-card .label {
            color: var(--text-color);
            opacity: 0.65;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .metric-card .value {
            color: var(--text-color);
            font-size: 1.6rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }
        .metric-card .hint, .metric-card .value span {
            color: var(--text-color) !important;
            opacity: 0.6;
            font-size: 0.8rem;
            margin-top: 0.15rem;
        }
        .action-card { margin-bottom: 0.6rem; }
        .action-card .num {
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--primary-color);
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-color);
            margin: 0.4rem 0 0.6rem 0;
        }
        .top-bar {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
            flex-wrap: wrap;
            margin-bottom: 0.4rem;
        }
        div[data-testid="stTabs"] button { font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str, hint: str = "") -> str:
    hint_html = f'<div class="hint">{hint}</div>' if hint else ""
    return (
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{hint_html}</div>'
    )


def _solvency_gauge(probability: float) -> go.Figure:
    t = _theme()
    pct = probability * 100
    color = COLORS["green"] if pct >= 75 else COLORS["amber"] if pct >= 50 else COLORS["red"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 42, "color": t["text"]}},
            title={"text": "Chance you stay solvent", "font": {"size": 14, "color": t["muted"]}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": t["muted"]},
                "bar": {"color": color, "thickness": 0.75},
                "bgcolor": t["gauge_bg"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": t["step_red"]},
                    {"range": [50, 75], "color": t["step_amber"]},
                    {"range": [75, 100], "color": t["step_green"]},
                ],
                "threshold": {
                    "line": {"color": t["text"], "width": 3},
                    "thickness": 0.85,
                    "value": 90,
                },
            },
        )
    )
    fig.update_layout(height=300, **t["layout"])
    return fig


def _fan_chart(trajectories: dict) -> go.Figure:
    months = trajectories.get("months", list(range(1, len(trajectories.get("p50", [])) + 1)))
    p10 = trajectories.get("p10", [])
    p50 = trajectories.get("p50", [])
    p90 = trajectories.get("p90", [])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=months + months[::-1],
            y=p90 + p10[::-1],
            fill="toself",
            fillcolor="rgba(59, 130, 246, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Baseline band",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=months, y=p90, mode="lines",
            line=dict(color="#93c5fd", width=1.5, dash="dot"), name="Best case (p90)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=months, y=p50, mode="lines",
            line=dict(color=COLORS["blue"], width=3.5), name="Typical path (p50)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=months, y=p10, mode="lines",
            line=dict(color="#93c5fd", width=1.5, dash="dot"), name="Bad case (p10)",
        )
    )
    t = _theme()
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["red"], annotation_text="Broke")
    fig.update_layout(
        title="Your money over time (2,000 possible futures)",
        xaxis_title="Month",
        yaxis_title="Balance (₹)",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        **t["layout"],
    )
    return fig


def _income_monthly_chart(monthly_incomes: list[float]) -> go.Figure:
    months = list(range(1, len(monthly_incomes) + 1))
    mean = sum(monthly_incomes) / len(monthly_incomes) if monthly_incomes else 0
    colors = [COLORS["green"] if v >= mean else COLORS["amber"] for v in monthly_incomes]
    fig = go.Figure(
        go.Bar(
            x=months,
            y=monthly_incomes,
            marker_color=colors,
            hovertemplate="Month %{x}<br>₹%{y:,.0f}<extra></extra>",
        )
    )
    t = _theme()
    fig.add_hline(y=mean, line_dash="dash", line_color=COLORS["slate"], annotation_text="Average")
    fig.update_layout(
        title="Monthly income history",
        xaxis_title="Month",
        yaxis_title="Income (₹)",
        height=320,
        **t["layout"],
    )
    return fig


def _income_range_chart(ip: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=["Income range"],
            y=[ip["p90_income"]],
            mode="markers",
            marker=dict(size=0.1, color="rgba(0,0,0,0)"),
            error_y=dict(
                type="data",
                symmetric=False,
                array=[0],
                arrayminus=[ip["p90_income"] - ip["p10_income"]],
                thickness=18,
                width=0,
                color="rgba(59,130,246,0.35)",
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=["Income range"] * 3,
            y=[ip["p10_income"], ip["mean_monthly_income"], ip["p90_income"]],
            mode="markers+text",
            marker=dict(
                size=[14, 18, 14],
                color=[COLORS["red"], COLORS["blue"], COLORS["green"]],
            ),
            text=["Bad month", "Average", "Good month"],
            textposition="middle right",
            hovertemplate="%{text}: ₹%{y:,.0f}<extra></extra>",
            showlegend=False,
        )
    )
    t = _theme()
    layout = {**t["layout"], "xaxis": {**t["layout"].get("xaxis", {}), "showticklabels": False}}
    fig.update_layout(
        title="Bad month → average → good month",
        yaxis_title="Monthly income (₹)",
        height=280,
        **layout,
    )
    return fig


def _cashflow_chart(ip: dict, budget: dict, debt_mins: float) -> go.Figure:
    labels = ["Avg income", "Essentials", "Wants", "Min debt", "Left over"]
    income = ip["mean_monthly_income"]
    essentials = budget["essential_monthly_expenses"]
    wants = budget["discretionary_monthly_expenses"]
    leftover = income - essentials - wants - debt_mins
    values = [income, -essentials, -wants, -debt_mins, leftover]
    colors = [
        COLORS["green"],
        COLORS["needs"],
        COLORS["wants"],
        COLORS["violet"],
        COLORS["green"] if leftover >= 0 else COLORS["red"],
    ]
    fig = go.Figure(
        go.Waterfall(
            x=labels,
            y=values,
            measure=["absolute", "relative", "relative", "relative", "total"],
            connector={"line": {"color": "#cbd5e1"}},
            decreasing={"marker": {"color": COLORS["red"]}},
            increasing={"marker": {"color": COLORS["green"]}},
            totals={"marker": {"color": COLORS["blue"]}},
            text=[inr(abs(v)) for v in values],
            textposition="outside",
        )
    )
    t = _theme()
    fig.update_layout(
        title="Where an average month goes",
        yaxis_title="Amount (₹)",
        height=340,
        **t["layout"],
    )
    _ = colors
    return fig


def _budget_donut(budget: dict) -> go.Figure:
    needs = budget["essential_monthly_expenses"]
    wants = budget["discretionary_monthly_expenses"]
    fig = go.Figure(
        go.Pie(
            labels=["Needs", "Wants"],
            values=[needs, wants],
            hole=0.62,
            marker_colors=[COLORS["needs"], COLORS["wants"]],
            textinfo="label+percent",
            hovertemplate="%{label}: ₹%{value:,.0f}<extra></extra>",
        )
    )
    t = _theme()
    fig.update_layout(
        title="Needs vs wants",
        height=300,
        annotations=[
            dict(
                text=f"{inr(needs + wants)}<br><span style='font-size:12px'>/month</span>",
                x=0.5, y=0.5, showarrow=False, font=dict(size=16, color=t["text"]),
            )
        ],
        showlegend=False,
        **t["layout"],
    )
    return fig


def _budget_category_chart(monthly_spend: dict) -> go.Figure:
    if not monthly_spend:
        return go.Figure()
    items = sorted(monthly_spend.items(), key=lambda x: x[1], reverse=True)
    cats = [c.replace("_", " ").title() for c, _ in items]
    vals = [v for _, v in items]
    palette = [
        COLORS["blue"], COLORS["indigo"], COLORS["violet"], COLORS["needs"],
        COLORS["wants"], COLORS["amber"], COLORS["green"], COLORS["slate"],
    ]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=cats,
            orientation="h",
            marker_color=palette[: len(cats)],
            text=[inr(v) for v in vals],
            textposition="outside",
            hovertemplate="%{y}: ₹%{x:,.0f}<extra></extra>",
        )
    )
    t = _theme()
    layout = {**t["layout"], "yaxis": {**t["layout"].get("yaxis", {}), "autorange": "reversed"}}
    fig.update_layout(
        title="Spend by category",
        xaxis_title="Monthly avg (₹)",
        height=max(280, 40 * len(cats) + 80),
        **layout,
    )
    return fig


def _debt_compare_chart(debt_plan: dict) -> go.Figure:
    strategies = ["Avalanche", "Snowball"]
    interest = [
        debt_plan["avalanche"]["total_interest_paid"],
        debt_plan["snowball"]["total_interest_paid"],
    ]
    months = [
        debt_plan["avalanche"]["months_to_debt_free"],
        debt_plan["snowball"]["months_to_debt_free"],
    ]
    rec = debt_plan["recommended_strategy"].title()
    bar_colors = [
        COLORS["green"] if s.lower() == rec.lower() else COLORS["slate"]
        for s in strategies
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Interest paid",
            x=strategies,
            y=interest,
            marker_color=bar_colors,
            text=[inr(v) for v in interest],
            textposition="outside",
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Months to free",
            x=strategies,
            y=months,
            mode="markers+lines+text",
            marker=dict(size=12, color=COLORS["amber"]),
            line=dict(color=COLORS["amber"], width=2),
            text=[f"{m} mo" for m in months],
            textposition="top center",
            yaxis="y2",
        )
    )
    t = _theme()
    fig.update_layout(
        **_layout(
            title=f"Debt strategies (recommended: {rec})",
            yaxis=dict(title="Total interest (₹)", side="left", gridcolor=t["grid"]),
            yaxis2=dict(
                title="Months",
                overlaying="y",
                side="right",
                showgrid=False,
                color=t["muted"],
            ),
            height=340,
            legend=dict(orientation="h", y=1.12),
        )
    )
    return fig


def _debt_payoff_chart(debt_plan: dict) -> go.Figure:
    fig = go.Figure()
    for key, color in (("avalanche", COLORS["blue"]), ("snowball", COLORS["violet"])):
        traj = debt_plan[key].get("balance_trajectory") or []
        if not traj:
            continue
        months = list(range(len(traj)))
        is_rec = debt_plan["recommended_strategy"] == key
        fig.add_trace(
            go.Scatter(
                x=months,
                y=traj,
                mode="lines",
                name=key.title() + (" ★" if is_rec else ""),
                line=dict(width=3.5 if is_rec else 2, color=color, dash="solid" if is_rec else "dot"),
            )
        )
    t = _theme()
    fig.update_layout(
        title="Debt balance over time",
        xaxis_title="Month",
        yaxis_title="Total debt (₹)",
        height=340,
        legend=dict(orientation="h", y=1.1),
        **t["layout"],
    )
    return fig


def _debt_free_range_chart(sim) -> go.Figure:
    t = _theme()
    if sim.debt_free_month_median is None:
        fig = go.Figure()
        fig.update_layout(
            title="Debt-free timing (not reached in most futures)",
            height=220,
            **t["layout"],
        )
        return fig

    p10 = sim.debt_free_month_p10 or sim.debt_free_month_median
    p50 = sim.debt_free_month_median
    p90 = sim.debt_free_month_p90 or sim.debt_free_month_median
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[p10, p90],
            y=["Debt-free"],
            mode="lines",
            line=dict(color="rgba(59,130,246,0.35)", width=18),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[p10, p50, p90],
            y=["Debt-free"] * 3,
            mode="markers+text",
            marker=dict(size=[12, 16, 12], color=[COLORS["blue"], COLORS["indigo"], COLORS["blue"]]),
            text=[f"Fast<br>{p10:.0f} mo", f"Typical<br>{p50:.0f} mo", f"Slow<br>{p90:.0f} mo"],
            textposition="top center",
            textfont=dict(color=t["text"]),
            showlegend=False,
        )
    )
    layout = {**t["layout"], "yaxis": {**t["layout"].get("yaxis", {}), "showticklabels": False}}
    fig.update_layout(
        title="When you become debt-free (across successful futures)",
        xaxis_title="Month",
        height=240,
        **layout,
    )
    return fig


def _buffer_progress_chart(current: float, target: float) -> go.Figure:
    t = _theme()
    target = max(target, 1.0)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=current,
            number={"prefix": "₹", "font": {"size": 28, "color": t["text"]}},
            delta={
                "reference": target,
                "relative": False,
                "valueformat": ",.0f",
                "increasing": {"color": COLORS["green"]},
                "decreasing": {"color": COLORS["red"]},
            },
            title={
                "text": f"Buffer progress → {inr(target)} for ~90% safety",
                "font": {"color": t["muted"]},
            },
            gauge={
                "axis": {
                    "range": [0, max(target * 1.1, current * 1.1, 100)],
                    "tickcolor": t["muted"],
                },
                "bar": {"color": COLORS["blue"]},
                "bgcolor": t["gauge_bg"],
                "steps": [
                    {"range": [0, target * 0.5], "color": t["step_red"]},
                    {"range": [target * 0.5, target], "color": t["step_amber"]},
                    {"range": [target, target * 1.1], "color": t["step_green"]},
                ],
                "threshold": {
                    "line": {"color": COLORS["green"], "width": 3},
                    "value": target,
                },
            },
        )
    )
    fig.update_layout(height=280, **t["layout"])
    return fig


def main():
    st.set_page_config(
        page_title="AI Financial Coach",
        page_icon="💸",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    if "result" not in st.session_state:
        st.session_state.result = None

    # Sidebar + pipeline first so download buttons use the latest result
    with st.sidebar:
        st.header("Setup")
        preset_name = st.selectbox("Preset demo", list(PRESETS.keys()))
        st.caption(PRESETS[preset_name]["blurb"])

        st.divider()
        st.subheader("Upload your own")
        with st.expander("Required file formats", expanded=False):
            st.markdown(
                """
**1. Transactions CSV** (required columns — recommended format):

| Column | Meaning | Example |
|---|---|---|
| `date` | Transaction date | `2025-01-05` |
| `merchant` | Who paid / who you paid | `Upwork Payment` |
| `category` | Spend category | `rent`, `groceries`, `income` |
| `amount` | Amount in ₹ (positive) | `15000` |
| `type` | `income` or `expense` | `income` |

Also accepted (auto-detected):
- Kaggle style: `Date, Transaction Description, Category, Amount, Type`
- Bank style: `Date, Description, Amount, Transaction Type, Category`
  (`credit` = income, `debit` = expense)

**2. Debts JSON** (exact shape):

```json
{
  "debts": [
    {
      "name": "Credit Card",
      "balance": 45000,
      "apr": 36.0,
      "minimum_payment": 2250
    }
  ],
  "extra_monthly_payment_budget": 3000
}
```

- `balance` / `minimum_payment` / `extra_monthly_payment_budget` → ₹
- `apr` → annual interest rate % (e.g. `36.0` for 36%)

If you skip debts upload, the app uses the preset's sample debts.
                """
            )
            st.download_button(
                "Download transactions template",
                data=(DATA / "template_transactions.csv").read_text(encoding="utf-8"),
                file_name="template_transactions.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download debts template",
                data=(DATA / "template_debts.json").read_text(encoding="utf-8"),
                file_name="template_debts.json",
                mime="application/json",
                use_container_width=True,
            )

        tx_upload = st.file_uploader(
            "Transactions CSV",
            type=["csv"],
            help="Must include date, merchant/description, amount, and income/expense type. See 'Required file formats'.",
        )
        debts_upload = st.file_uploader(
            "Debts JSON",
            type=["json"],
            help="JSON with a debts list (name, balance, apr, minimum_payment) and extra_monthly_payment_budget.",
        )

        st.divider()
        preference = st.selectbox(
            "How should we pay off debt?",
            options=["auto", "avalanche", "snowball"],
            format_func=lambda x: {
                "auto": "Auto (pick the cheaper plan)",
                "avalanche": "Avalanche (highest interest first)",
                "snowball": "Snowball (smallest balance first)",
            }[x],
            help=(
                "Avalanche pays the costliest loan first (saves interest). "
                "Snowball pays the smallest loan first (quick wins). "
                "Auto chooses avalanche when it saves meaningful interest."
            ),
        )
        initial_buffer = st.number_input(
            "Emergency savings you already have (₹)",
            min_value=0.0,
            value=0.0,
            step=500.0,
            help=(
                "Cash you can use if a month goes badly (savings / emergency fund). "
                "Leave at 0 if you are starting from scratch. "
                "A higher buffer usually improves your solvency odds."
            ),
        )
        horizon = st.slider(
            "How far ahead should we simulate?",
            min_value=6,
            max_value=48,
            value=24,
            format="%d months",
            help=(
                "We run thousands of possible futures for this many months "
                "(default 24 = 2 years) and report how often you stay financially safe."
            ),
        )
        use_llm = st.toggle("Use LLM narration", value=llm_available())
        if use_llm and not llm_available():
            st.warning(
                "No API keys loaded. Put keys in "
                f"`{ROOT / '.env'}`, then restart Streamlit."
            )
            use_llm = False

        run_clicked = st.button("Run coach", type="primary", use_container_width=True)

    if run_clicked:
        with st.spinner("Running agents & simulating futures..."):
            try:
                if tx_upload is not None:
                    tx_df = load_transactions_from_df(pd.read_csv(tx_upload))
                else:
                    tx_df = load_transactions(PRESETS[preset_name]["tx"])

                if debts_upload is not None:
                    debts_payload = load_debts_from_dict(json.load(debts_upload))
                else:
                    debts_payload = load_debts(PRESETS[preset_name]["debts"])

                result = run_coach_pipeline(
                    transactions=tx_df,
                    debts=debts_payload["debts"],
                    extra_monthly=debts_payload["extra_monthly_payment_budget"],
                    preference=preference,
                    initial_buffer=initial_buffer,
                    horizon=horizon,
                    n_sims=2000,
                    seed=42,
                    use_llm=use_llm,
                )
                result["_tx_preview"] = tx_df.head(20)
                result["_tx_full"] = tx_df
                result["_debts_preview"] = debts_to_serializable(debts_payload)
                result["_debts_objects"] = debts_payload["debts"]
                result["_extra_monthly"] = debts_payload["extra_monthly_payment_budget"]
                result["_preset"] = preset_name if tx_upload is None else "Custom upload"
                result["_initial_buffer"] = initial_buffer
                st.session_state.result = result
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                st.session_state.result = None

    result = st.session_state.result

    # ---------- Title (full width, clear of navbar) ----------
    st.markdown("## AI Financial Coach")
    st.caption("Odds, not one static plan — built for irregular income. Amounts in ₹.")

    # ---------- Download toolbar on its own row (below title, not under Deploy menu) ----------
    if result is not None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        html_report = build_report_html(result)
        md_report = build_report_markdown(result)
        st.markdown("**Download full report**")
        d1, d2, _ = st.columns([1.1, 1.1, 1.8])
        with d1:
            st.download_button(
                label="Download HTML report",
                data=html_report,
                file_name=f"financial_coach_report_{stamp}.html",
                mime="text/html",
                use_container_width=True,
                type="primary",
                help="Full visual report (opens in any browser)",
            )
        with d2:
            st.download_button(
                label="Download Markdown",
                data=md_report,
                file_name=f"financial_coach_report_{stamp}.md",
                mime="text/markdown",
                use_container_width=True,
                help="Markdown report for docs / sharing",
            )
    else:
        st.caption("Run coach to enable report download.")
    if result is None:
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            _metric_card("Step 1", "Pick a preset", "Try Gig worker first"),
            unsafe_allow_html=True,
        )
        c2.markdown(
            _metric_card("Step 2", "Hit Run coach", "Agents + Monte Carlo"),
            unsafe_allow_html=True,
        )
        c3.markdown(
            _metric_card("Step 3", "Read the visuals", "Charts over walls of text"),
            unsafe_allow_html=True,
        )
        st.info(
            "Tip: compare **Gig worker** vs **Bad case** — the fan chart and solvency gauge "
            "show why irregular income needs probability, not a single plan."
        )
        return

    # ---------- Unpack ----------
    sim = result["simulation"]
    ip = result["income_profile_dict"]
    budget = result["budget_dict"]
    debt_plan = result["debt_plan_dict"]
    solvency = sim.solvency_probability
    status_label, status_color, status_hint = _status(solvency)
    rec = debt_plan["recommended_strategy"]
    rec_months = debt_plan[rec]["months_to_debt_free"]
    rec_interest = debt_plan[rec]["total_interest_paid"]

    # ---------- Hero ----------
    st.markdown(
        f"""
        <div class="hero-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; flex-wrap:wrap;">
                <div>
                    <div class="hero-label">Safe to spend this week</div>
                    <div class="hero-value">{inr(result['safe_to_spend_weekly'])}</div>
                    <div class="hero-sub">{inr(result['safe_to_spend_monthly'])} / month without risking your debt plan</div>
                </div>
                <div style="text-align:right;">
                    <span class="status-pill" style="color:{status_color};">
                        {status_label} · {solvency * 100:.0f}% solvent
                    </span>
                    <div class="hero-hint">{status_hint}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- KPI row ----------
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        _metric_card("Solvency odds", f"{solvency * 100:.0f}%", "of futures stay afloat"),
        unsafe_allow_html=True,
    )
    k2.markdown(
        _metric_card(
            "Buffer target",
            inr(sim.recommended_buffer),
            "already saved" if solvency >= 0.90 and result.get("_initial_buffer", 0) >= sim.recommended_buffer
            else "stretch goal" if solvency < 0.50
            else "for ~90% safety",
        ),
        unsafe_allow_html=True,
    )
    k3.markdown(
        _metric_card("Debt strategy", rec.title(), f"{rec_months} months · {inr(rec_interest)} interest"),
        unsafe_allow_html=True,
    )
    k4.markdown(
        _metric_card("Income type", ip.get("income_type", "unknown"), f"avg {inr(ip['mean_monthly_income'])}/mo"),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Coach summary</div>', unsafe_allow_html=True)
    st.info(result.get("coach_message", ""))

    # ---------- 3 action cards ----------
    st.markdown('<div class="section-title">What to do next</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(
            f"""
            <div class="action-card">
                <div>1. Protect a weekly spend cap</div>
                <div class="num">{inr(result['safe_to_spend_weekly'])}</div>
                <div>Stay under this in slow weeks.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            f"""
            <div class="action-card">
                <div>2. Build emergency buffer</div>
                <div class="num">{inr(sim.recommended_buffer)}</div>
                <div>Raises solvency toward 90%.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a3:
        other = "snowball" if rec == "avalanche" else "avalanche"
        saved = abs(debt_plan[other]["total_interest_paid"] - rec_interest)
        st.markdown(
            f"""
            <div class="action-card">
                <div>3. Stick to {rec.title()}</div>
                <div class="num">{rec_months} mo</div>
                <div>Saves ~{inr(saved)} vs {other}.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------- Tabs ----------
    tab_overview, tab_income, tab_debt, tab_budget, tab_details = st.tabs(
        ["Overview", "Income", "Debt", "Budget", "Details"]
    )

    with tab_overview:
        left, right = st.columns([1, 1.25])
        with left:
            st.plotly_chart(_solvency_gauge(solvency), use_container_width=True, theme="streamlit")
            st.plotly_chart(
                _buffer_progress_chart(result.get("_initial_buffer", 0.0), sim.recommended_buffer),
                use_container_width=True,
                theme="streamlit",
            )
        with right:
            st.plotly_chart(
                _fan_chart(sim.percentile_trajectories),
                use_container_width=True,
                theme="streamlit",
            )
            st.plotly_chart(_debt_free_range_chart(sim), use_container_width=True, theme="streamlit")

        st.plotly_chart(
            _cashflow_chart(ip, budget, debt_plan["total_minimum_payments"]),
            use_container_width=True,
            theme="streamlit",
        )

    with tab_income:
        c1, c2 = st.columns([1.3, 1])
        with c1:
            incomes = ip.get("monthly_incomes") or []
            if incomes:
                st.plotly_chart(_income_monthly_chart(incomes), use_container_width=True, theme="streamlit")
            else:
                st.info("Not enough monthly income history to chart.")
        with c2:
            st.plotly_chart(_income_range_chart(ip), use_container_width=True, theme="streamlit")
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="label">Snapshot</div>
                    <div class="value" style="font-size:1.2rem;">
                        {inr(ip['mean_monthly_income'])}
                        <span style="font-size:0.9rem;"> avg</span>
                    </div>
                    <div class="hint">
                        Swing ±{inr(ip['std_dev'])} · {ip['months_of_data']} months of data
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_debt:
        d1, d2 = st.columns(2)
        with d1:
            st.plotly_chart(_debt_compare_chart(debt_plan), use_container_width=True, theme="streamlit")
        with d2:
            st.plotly_chart(_debt_payoff_chart(debt_plan), use_container_width=True, theme="streamlit")

        debts = result["_debts_preview"].get("debts", [])
        if debts:
            debt_df = pd.DataFrame(debts)
            fig = go.Figure(
                go.Bar(
                    x=debt_df["name"],
                    y=debt_df["balance"],
                    marker_color=COLORS["indigo"],
                    text=[f"{inr(b)}<br>{a:.1f}% APR" for b, a in zip(debt_df["balance"], debt_df["apr"])],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="Balances by debt",
                yaxis_title="Balance (₹)",
                height=320,
                **_theme()["layout"],
            )
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    with tab_budget:
        b1, b2 = st.columns([1, 1.2])
        with b1:
            st.plotly_chart(_budget_donut(budget), use_container_width=True, theme="streamlit")
        with b2:
            st.plotly_chart(
                _budget_category_chart(budget.get("monthly_spend_by_category", {})),
                use_container_width=True,
                theme="streamlit",
            )
        if budget.get("flagged_categories"):
            st.warning(
                "Trending up: "
                + ", ".join(c.replace("_", " ").title() for c in budget["flagged_categories"])
            )

    with tab_details:
        st.caption(f"Scenario: **{result.get('_preset', '')}**")
        with st.expander("Coach notes (full text)", expanded=False):
            st.write(result["coach_message"])
            st.write("**Income:**", result["income_narrative"])
            st.write("**Debt:**", result["debt_narrative"])
            st.write("**Budget:**", result["budget_narrative"])
            st.write("**Risk:**", result["simulation_narrative"])
        with st.expander("Raw transactions (preview)"):
            st.dataframe(result["_tx_preview"], use_container_width=True)
        with st.expander("Debts JSON"):
            st.json(result["_debts_preview"])


if __name__ == "__main__":
    main()
