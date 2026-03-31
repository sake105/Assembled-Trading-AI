"""Portfolio return and volatility attribution: per-symbol contribution analysis.

Decomposes portfolio-level return and volatility into per-symbol contributions
using standard portfolio analytics:

  - Return attribution: weight_i * return_i
  - Vol attribution (marginal): weight_i * (Sigma @ w)_i / portfolio_vol

This module is stateless and has no side effects. It is designed for use in
reporting pipelines and backtests, not in the trading cycle hot path.

M6-T08: implement attribution reports.
"""

from __future__ import annotations

from typing import Any

import math

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Return attribution
# ---------------------------------------------------------------------------


def compute_symbol_return_contributions(
    weights: dict[str, float],
    returns: dict[str, float],
) -> dict[str, float]:
    """Compute per-symbol return contribution: weight_i * return_i.

    Args:
        weights: symbol -> weight mapping (fractions, e.g. 0.10 = 10%).
        returns: symbol -> period return mapping (e.g. 0.02 = 2% gain).

    Returns:
        symbol -> contribution dict. Missing symbols contribute 0.0.
        Empty dict if either input is empty.
    """
    if not weights or not returns:
        return {}
    return {sym: weights[sym] * returns.get(sym, 0.0) for sym in weights}


def compute_portfolio_return(
    weights: dict[str, float],
    returns: dict[str, float],
) -> float:
    """Compute total portfolio return as sum of per-symbol contributions.

    Args:
        weights: symbol -> weight mapping.
        returns: symbol -> period return mapping.

    Returns:
        Total portfolio return (float). Returns 0.0 if inputs are empty.
    """
    contribs = compute_symbol_return_contributions(weights, returns)
    return sum(contribs.values()) if contribs else 0.0


# ---------------------------------------------------------------------------
# Volatility attribution
# ---------------------------------------------------------------------------


def compute_covariance_matrix(
    prices: pd.DataFrame,
    symbols: list[str],
    lookback_days: int = 60,
    annualize_factor: float = 252.0,
) -> pd.DataFrame:
    """Compute annualized covariance matrix from price history.

    Args:
        prices: DataFrame with columns ``timestamp``, ``symbol``, ``close``.
        symbols: Symbols to include.
        lookback_days: Number of recent bars to use.
        annualize_factor: Multiplier to annualize (252 for daily data).

    Returns:
        Square annualized covariance DataFrame (symbols × symbols).
        Empty DataFrame if fewer than 2 symbols or insufficient data (< 3 bars).
    """
    if prices is None or prices.empty or len(symbols) < 2:
        return pd.DataFrame()
    required_cols = {"timestamp", "symbol", "close"}
    if not required_cols.issubset(prices.columns):
        return pd.DataFrame()

    rows = prices[prices["symbol"].isin(symbols)].copy()
    if rows.empty:
        return pd.DataFrame()

    rows = rows.sort_values(["symbol", "timestamp"])
    pivot = rows.pivot_table(
        index="timestamp",
        columns="symbol",
        values="close",
        aggfunc="last",
    )
    pivot = pivot.iloc[-lookback_days:] if len(pivot) > lookback_days else pivot
    returns = pivot.pct_change().dropna(how="all")

    valid_cols = [c for c in symbols if c in returns.columns]
    if len(valid_cols) < 2 or len(returns) < 3:
        return pd.DataFrame()

    cov = returns[valid_cols].cov() * annualize_factor
    return cov


def compute_symbol_vol_contributions(
    weights: dict[str, float],
    cov_matrix: pd.DataFrame,
) -> dict[str, float]:
    """Compute per-symbol marginal volatility contribution.

    Uses the standard marginal-contribution-to-risk formula:
        MCR_i = weight_i * (Sigma @ w)_i / portfolio_vol

    where Sigma is the covariance matrix and w is the weight vector.

    Args:
        weights: symbol -> weight mapping.
        cov_matrix: Annualized covariance DataFrame (symbols × symbols).

    Returns:
        symbol -> vol contribution dict (each value in annualized vol units).
        Empty dict if inputs are invalid or portfolio vol is near zero.
        Symbols absent from cov_matrix get contribution 0.0.
    """
    if not weights or cov_matrix is None or cov_matrix.empty:
        return {}

    symbols = [s for s in weights if s in cov_matrix.columns]
    if len(symbols) < 2:
        return {}

    w = np.array([weights[s] for s in symbols])
    sigma = cov_matrix.loc[symbols, symbols].values

    portfolio_var = float(w @ sigma @ w)
    if portfolio_var <= 0.0 or math.isnan(portfolio_var):
        return {}

    portfolio_vol = math.sqrt(portfolio_var)

    # Marginal contribution = w_i * (Sigma @ w)_i / vol
    marginal = (sigma @ w)
    contributions = {
        sym: float(w[i] * marginal[i] / portfolio_vol)
        for i, sym in enumerate(symbols)
    }
    return contributions


def compute_portfolio_vol(
    weights: dict[str, float],
    cov_matrix: pd.DataFrame,
) -> float:
    """Compute annualized portfolio volatility from weights and covariance matrix.

    Args:
        weights: symbol -> weight mapping.
        cov_matrix: Annualized covariance DataFrame.

    Returns:
        Portfolio annualized volatility (float). Returns float('nan') if
        insufficient data or portfolio var is non-positive.
    """
    if not weights or cov_matrix is None or cov_matrix.empty:
        return float("nan")

    symbols = [s for s in weights if s in cov_matrix.columns]
    if len(symbols) < 2:
        return float("nan")

    w = np.array([weights[s] for s in symbols])
    sigma = cov_matrix.loc[symbols, symbols].values
    portfolio_var = float(w @ sigma @ w)

    if portfolio_var <= 0.0 or math.isnan(portfolio_var):
        return float("nan")

    return math.sqrt(portfolio_var)


# ---------------------------------------------------------------------------
# Full attribution report
# ---------------------------------------------------------------------------


def compute_attribution_report(
    weights: dict[str, float],
    returns: dict[str, float],
    prices: pd.DataFrame,
    policy: dict[str, Any] | None = None,
    lookback_days: int = 60,
    annualize_factor: float = 252.0,
) -> dict[str, Any]:
    """Compute full attribution report: return and vol contributions per symbol.

    Args:
        weights: symbol -> weight mapping.
        returns: symbol -> period return mapping.
        prices: DataFrame with ``timestamp``, ``symbol``, ``close`` for covariance.
        policy: Optional policy dict. Reads from ``attribution`` section:
            - lookback_days (int, default 60)
            - annualize_factor (float, default 252.0)
        lookback_days: Fallback if not in policy.
        annualize_factor: Fallback if not in policy.

    Returns:
        Dict with keys:
            ``symbols``: list of symbols
            ``weights``: copy of input weights
            ``return_contributions``: symbol -> return contribution
            ``portfolio_return``: total portfolio return
            ``vol_contributions``: symbol -> vol contribution (nan if unavailable)
            ``portfolio_vol``: annualized portfolio vol (nan if unavailable)
            ``status``: "ok", "no_price_data", or "insufficient_data"
    """
    attr_cfg = ((policy or {}).get("attribution") or {})
    lb = int(attr_cfg.get("lookback_days", lookback_days) or lookback_days)
    af = float(attr_cfg.get("annualize_factor", annualize_factor) or annualize_factor)

    symbols = list(weights.keys()) if weights else []

    return_contribs = compute_symbol_return_contributions(weights or {}, returns or {})
    portfolio_return = sum(return_contribs.values()) if return_contribs else 0.0

    # Vol attribution requires price data
    if prices is None or prices.empty:
        return {
            "symbols": symbols,
            "weights": dict(weights or {}),
            "return_contributions": return_contribs,
            "portfolio_return": portfolio_return,
            "vol_contributions": {s: float("nan") for s in symbols},
            "portfolio_vol": float("nan"),
            "status": "no_price_data",
        }

    cov = compute_covariance_matrix(prices, symbols, lookback_days=lb, annualize_factor=af)

    if cov.empty:
        return {
            "symbols": symbols,
            "weights": dict(weights or {}),
            "return_contributions": return_contribs,
            "portfolio_return": portfolio_return,
            "vol_contributions": {s: float("nan") for s in symbols},
            "portfolio_vol": float("nan"),
            "status": "insufficient_data",
        }

    vol_contribs = compute_symbol_vol_contributions(weights, cov)
    portfolio_vol = compute_portfolio_vol(weights, cov)

    # Fill nan for symbols not in cov
    full_vol_contribs = {s: vol_contribs.get(s, float("nan")) for s in symbols}

    return {
        "symbols": symbols,
        "weights": dict(weights),
        "return_contributions": return_contribs,
        "portfolio_return": portfolio_return,
        "vol_contributions": full_vol_contribs,
        "portfolio_vol": portfolio_vol,
        "status": "ok",
    }


__all__ = [
    "compute_symbol_return_contributions",
    "compute_portfolio_return",
    "compute_covariance_matrix",
    "compute_symbol_vol_contributions",
    "compute_portfolio_vol",
    "compute_attribution_report",
]
