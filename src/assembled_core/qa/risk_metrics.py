"""Portfolio risk metrics computation.

This module provides a focused set of portfolio risk metrics that complement
the performance metrics in qa.metrics. It focuses on risk-specific measures
like daily volatility, annualized volatility, Value at Risk (VaR), and Expected Shortfall (ES).

Key features:
- Historical VaR/ES computation (simple, stable, testable)
- Daily and annualized volatility
- Reuses existing drawdown computation from qa.metrics
- Robust handling of edge cases (empty data, insufficient observations)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.assembled_core.qa.metrics import (
    PERIODS_PER_YEAR_1D,
    PERIODS_PER_YEAR_5MIN,
    _compute_returns,
    compute_drawdown,
)


def _get_periods_per_year(freq: Literal["1d", "5min"]) -> int:
    """Get periods per year for a given frequency.

    Args:
        freq: Frequency string ("1d" or "5min")

    Returns:
        Number of periods per year
    """
    if freq == "1d":
        return PERIODS_PER_YEAR_1D
    elif freq == "5min":
        return PERIODS_PER_YEAR_5MIN
    else:
        # Default to daily
        return PERIODS_PER_YEAR_1D


def compute_portfolio_risk_metrics(
    equity: pd.Series | pd.DataFrame, freq: Literal["1d", "5min"] = "1d"
) -> dict[str, float]:
    """Compute portfolio risk metrics from equity curve.

    This function computes a focused set of risk metrics:
    - daily_vol: Daily volatility (standard deviation of daily returns)
    - ann_vol: Annualized volatility
    - max_drawdown: Maximum drawdown (absolute, negative value)
    - var_95: Value at Risk at 95% confidence (historical, in absolute terms)
    - es_95: Expected Shortfall at 95% confidence (historical, in absolute terms)

    Args:
        equity: Series of equity values, or DataFrame with 'equity' column
        freq: Trading frequency ("1d" or "5min") for annualization

    Returns:
        Dictionary with risk metrics:
        - daily_vol: float | None (daily volatility, None if insufficient data)
        - ann_vol: float | None (annualized volatility, None if insufficient data)
        - max_drawdown: float (maximum drawdown, always present, 0.0 if no drawdown)
        - var_95: float | None (VaR 95%, None if insufficient data)
        - es_95: float | None (ES 95%, None if insufficient data)

    Example:
        >>> import pandas as pd
        >>> equity = pd.Series([10000, 10100, 10050, 10200, 10100])
        >>> risk_metrics = compute_portfolio_risk_metrics(equity, freq="1d")
        >>> print(risk_metrics["ann_vol"])
        >>> print(risk_metrics["var_95"])
    """
    # Extract equity series if DataFrame provided
    if isinstance(equity, pd.DataFrame):
        if "equity" not in equity.columns:
            raise ValueError("DataFrame must have 'equity' column")
        equity_series = equity["equity"].copy()
    else:
        equity_series = equity.copy()

    # Handle empty or insufficient data
    if equity_series.empty or len(equity_series) < 2:
        return {
            "daily_vol": None,
            "ann_vol": None,
            "max_drawdown": 0.0,
            "var_95": None,
            "es_95": None,
        }

    # Sanitize equity values
    equity_series = equity_series.replace([np.inf, -np.inf], np.nan)
    equity_series = equity_series.ffill().bfill()

    if len(equity_series.dropna()) < 2:
        return {
            "daily_vol": None,
            "ann_vol": None,
            "max_drawdown": 0.0,
            "var_95": None,
            "es_95": None,
        }

    # Compute returns
    returns = _compute_returns(equity_series)

    if len(returns) < 2:
        # Compute drawdown even with minimal data
        _, max_drawdown, _, _ = compute_drawdown(equity_series)
        return {
            "daily_vol": None,
            "ann_vol": None,
            "max_drawdown": max_drawdown,
            "var_95": None,
            "es_95": None,
        }

    # Daily volatility (standard deviation of daily returns)
    daily_vol = float(returns.std()) if len(returns) >= 2 else None
    if daily_vol is not None and (np.isnan(daily_vol) or daily_vol <= 0):
        daily_vol = None

    # Annualized volatility
    ann_vol = None
    if daily_vol is not None:
        periods_per_year = _get_periods_per_year(freq)
        ann_vol = float(daily_vol * np.sqrt(periods_per_year))
        if np.isnan(ann_vol) or ann_vol <= 0:
            ann_vol = None

    # Max drawdown (reuse from qa.metrics)
    _, max_drawdown, _, _ = compute_drawdown(equity_series)

    # VaR and ES (95% confidence, historical)
    var_95 = None
    es_95 = None

    if len(returns) >= 5:  # Need at least 5 observations for meaningful VaR/ES
        # Get current equity value for scaling
        current_equity = float(equity_series.iloc[-1])

        # VaR: 5th percentile of returns (worst 5% of outcomes)
        var_95_pct = float(np.percentile(returns, 5))
        if not np.isnan(var_95_pct):
            # VaR in absolute terms (negative value = loss)
            var_95 = var_95_pct * current_equity

        # ES: Expected value of returns below VaR threshold (Conditional VaR)
        # ES = mean of returns in the worst 5% tail
        var_threshold = np.percentile(returns, 5)
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) > 0:
            es_95_pct = float(tail_returns.mean())
            if not np.isnan(es_95_pct):
                # ES in absolute terms (negative value = expected loss)
                es_95 = es_95_pct * current_equity

    return {
        "daily_vol": daily_vol,
        "ann_vol": ann_vol,
        "max_drawdown": max_drawdown,
        "var_95": var_95,
        "es_95": es_95,
    }
