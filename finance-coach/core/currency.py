"""Indian Rupee formatting helpers."""

from __future__ import annotations


def inr(amount: float | int, decimals: int = 0) -> str:
    """Format an amount as Indian Rupees, e.g. ₹3,220."""
    try:
        value = float(amount)
    except (TypeError, ValueError):
        value = 0.0
    if decimals <= 0:
        return f"₹{value:,.0f}"
    return f"₹{value:,.{decimals}f}"


CURRENCY_LABEL = "₹"
CURRENCY_NAME = "INR"
AXIS_SUFFIX = "(₹)"
