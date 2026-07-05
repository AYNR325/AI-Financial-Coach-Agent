"""Vectorized Monte Carlo risk simulation for irregular-income futures."""

from __future__ import annotations

import numpy as np

from .schemas import IncomeProfile, SimulationResult


def _sample_incomes(
    profile: IncomeProfile,
    n_sims: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap from historical months when possible; else normal draws."""
    history = np.asarray(profile.monthly_incomes, dtype=float)
    if len(history) >= 3:
        idx = rng.integers(0, len(history), size=(n_sims, horizon))
        return history[idx]

    mean = max(profile.mean_monthly_income, 0.0)
    std = max(profile.std_dev, mean * 0.05 if mean > 0 else 100.0)
    samples = rng.normal(mean, std, size=(n_sims, horizon))
    return np.clip(samples, 0.0, None)


def _debt_payment_path(monthly_payments: list[float], horizon: int) -> np.ndarray:
    """Pad or truncate the debt payment schedule to the simulation horizon."""
    if not monthly_payments:
        return np.zeros(horizon, dtype=float)
    path = np.zeros(horizon, dtype=float)
    n = min(len(monthly_payments), horizon)
    path[:n] = monthly_payments[:n]
    return path


def _run_once(
    income_draws: np.ndarray,
    monthly_expenses: float,
    debt_payments: np.ndarray,
    initial_buffer: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run balance dynamics for all simulations.

    income_draws: (n_sims, horizon)
    """
    n_sims, horizon = income_draws.shape
    net = income_draws - monthly_expenses - debt_payments[np.newaxis, :]
    balances = np.empty((n_sims, horizon + 1), dtype=float)
    balances[:, 0] = initial_buffer
    balances[:, 1:] = initial_buffer + np.cumsum(net, axis=1)

    insolvent = (balances[:, 1:] < 0).any(axis=1)

    debt_done = debt_payments <= 0.01
    if debt_done.all():
        debt_free_month = np.zeros(n_sims)
        debt_free_month[insolvent] = np.nan
    else:
        if (~debt_done).any():
            last_pay_idx = int(np.max(np.where(~debt_done)[0])) + 1
        else:
            last_pay_idx = 0
        debt_free_month = np.full(n_sims, float(last_pay_idx), dtype=float)
        debt_free_month[insolvent] = np.nan
        if last_pay_idx >= horizon and debt_payments[-1] > 0.01:
            debt_free_month[:] = np.nan
            debt_free_month[~insolvent] = np.nan

    return balances, insolvent, debt_free_month


def run_monte_carlo(
    profile: IncomeProfile,
    monthly_expenses: float,
    debt_monthly_payments: list[float],
    *,
    initial_buffer: float = 0.0,
    horizon: int = 24,
    n_sims: int = 2000,
    seed: int | None = 42,
    target_solvency: float = 0.90,
) -> SimulationResult:
    """
    Simulate thousands of financial futures given income variance.

    Uses bootstrap resampling of historical monthly incomes when available.
    """
    rng = np.random.default_rng(seed)
    debt_payments = _debt_payment_path(debt_monthly_payments, horizon)
    income_draws = _sample_incomes(profile, n_sims, horizon, rng)

    balances, insolvent, debt_free_month = _run_once(
        income_draws, monthly_expenses, debt_payments, initial_buffer
    )

    solvency = float((~insolvent).mean())
    valid_df = debt_free_month[~np.isnan(debt_free_month)]
    if len(valid_df) > 0:
        p10 = float(np.percentile(valid_df, 10))
        median = float(np.percentile(valid_df, 50))
        p90 = float(np.percentile(valid_df, 90))
    else:
        p10 = median = p90 = None

    traj = balances[:, 1:]
    percentile_trajectories = {
        "p10": [round(float(x), 2) for x in np.percentile(traj, 10, axis=0)],
        "p50": [round(float(x), 2) for x in np.percentile(traj, 50, axis=0)],
        "p90": [round(float(x), 2) for x in np.percentile(traj, 90, axis=0)],
        "months": list(range(1, horizon + 1)),
    }

    recommended_buffer = _find_recommended_buffer(
        profile=profile,
        monthly_expenses=monthly_expenses,
        debt_payments=debt_payments,
        horizon=horizon,
        n_sims=n_sims,
        seed=seed,
        target_solvency=target_solvency,
        current_buffer=initial_buffer,
        current_solvency=solvency,
    )

    return SimulationResult(
        solvency_probability=round(solvency, 4),
        debt_free_month_p10=round(p10, 1) if p10 is not None else None,
        debt_free_month_median=round(median, 1) if median is not None else None,
        debt_free_month_p90=round(p90, 1) if p90 is not None else None,
        recommended_buffer=round(recommended_buffer, 2),
        percentile_trajectories=percentile_trajectories,
        n_sims=n_sims,
        horizon_months=horizon,
        initial_buffer=initial_buffer,
    )


def _find_recommended_buffer(
    profile: IncomeProfile,
    monthly_expenses: float,
    debt_payments: np.ndarray,
    horizon: int,
    n_sims: int,
    seed: int | None,
    target_solvency: float,
    current_buffer: float,
    current_solvency: float,
) -> float:
    """Binary-search the emergency buffer needed to hit target solvency.

    Capped so insolvent plans don't recommend absurd multi-lakh buffers.
    """
    if current_solvency >= target_solvency:
        return max(current_buffer, 0.0)

    max_buffer = max(
        monthly_expenses * 6.0,
        profile.mean_monthly_income * 3.0,
        1000.0,
    )

    gap = max(profile.mean_monthly_income - profile.p10_income, monthly_expenses * 0.5)
    lo = current_buffer
    hi = min(max(current_buffer + gap * 6, monthly_expenses * 3, 500.0), max_buffer)

    sol_hi = _solvency_at_buffer(
        profile, monthly_expenses, debt_payments, hi, horizon, n_sims, seed
    )
    if sol_hi < target_solvency:
        return round(max_buffer, 2)

    for _ in range(20):
        mid = (lo + hi) / 2.0
        sol = _solvency_at_buffer(
            profile, monthly_expenses, debt_payments, mid, horizon, n_sims, seed
        )
        if sol >= target_solvency:
            hi = mid
        else:
            lo = mid

    return max(min(hi, max_buffer), 0.0)


def _solvency_at_buffer(
    profile: IncomeProfile,
    monthly_expenses: float,
    debt_payments: np.ndarray,
    buffer: float,
    horizon: int,
    n_sims: int,
    seed: int | None,
) -> float:
    rng = np.random.default_rng(seed)
    income_draws = _sample_incomes(profile, n_sims, horizon, rng)
    _, insolvent, _ = _run_once(income_draws, monthly_expenses, debt_payments, buffer)
    return float((~insolvent).mean())
