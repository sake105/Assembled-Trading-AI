"""Antifragility Score (Plan 7.10).

Measures whether a portfolio benefits from market stress:
``correlation(returns, abs(market_returns))`` — positive = antifragile.
"""

from __future__ import annotations

import pandas as pd


def compute_antifragility_score(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Compute rolling antifragility score.

    Positive = portfolio benefits from market volatility (antifragile).
    Negative = portfolio suffers from market volatility (fragile).

    Args:
        portfolio_returns: Portfolio daily returns.
        market_returns: Market benchmark returns.
        window: Rolling window.

    Returns:
        Series of antifragility scores.
    """
    abs_market = market_returns.abs()
    return portfolio_returns.rolling(window, min_periods=20).corr(abs_market)


def compute_portfolio_antifragility(
    weights: dict[str, float],
    asset_returns: pd.DataFrame,
    market_returns: pd.Series,
) -> float:
    """Compute portfolio-level antifragility score.

    Args:
        weights: Symbol → weight.
        asset_returns: Returns DataFrame (columns = symbols).
        market_returns: Market returns series.

    Returns:
        Float antifragility score.
    """
    portfolio_ret = sum(
        asset_returns[sym] * w for sym, w in weights.items()
        if sym in asset_returns.columns
    )
    if isinstance(portfolio_ret, (int, float)):
        return 0.0

    score = compute_antifragility_score(portfolio_ret, market_returns)
    return round(float(score.iloc[-1]) if pd.notna(score.iloc[-1]) else 0.0, 4)


__all__ = ["compute_antifragility_score", "compute_portfolio_antifragility"]
