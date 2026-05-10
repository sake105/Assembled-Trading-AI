"""Position-Sizing-Modelle: Heat-Based, ATR, Risk-Budget, Volatility-Targeted.

Methoden
--------
- **Heat-Based (Larry Williams)**: % Equity bei Trade. f = heat / risk_per_share.
- **ATR-Based**: position_size = (capital × risk_pct) / (ATR × atr_mult).
- **Risk-Budget**: Σ_i risk_contrib_i == budget_i (siehe portfolio/risk_parity).
- **Volatility-Targeted**: notional = target_vol / asset_vol × capital.

Reference
---------
- Williams, L. (1979). *How I Made One Million Dollars Last Year Trading Commodities*.
- Wilder, J. (1978). *New Concepts in Technical Trading Systems* — ATR.
- Tharp, V. (2007). *Position Sizing*. Van Tharp Institute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def heat_based_sizing(
    equity: float,
    entry_price: float,
    stop_price: float,
    heat_pct: float = 0.02,
) -> dict:
    """Heat-Based Sizing.

    Args:
        equity: total account-equity.
        entry_price: planned entry.
        stop_price: stop-loss price.
        heat_pct: max % of equity at risk per trade.

    Returns:
        dict with shares, dollar_size, dollar_risk, ratio_of_equity.
    """
    if entry_price <= 0 or stop_price <= 0:
        return {"shares": 0, "dollar_size": 0, "dollar_risk": 0, "ratio_of_equity": 0}
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share == 0:
        return {"shares": 0, "dollar_size": 0, "dollar_risk": 0, "ratio_of_equity": 0}
    dollar_risk = equity * heat_pct
    shares = dollar_risk / risk_per_share
    dollar_size = shares * entry_price
    return {
        "shares": float(shares),
        "dollar_size": float(dollar_size),
        "dollar_risk": float(dollar_risk),
        "ratio_of_equity": float(dollar_size / equity) if equity > 0 else 0.0,
    }


def average_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Wilder's ATR (Average True Range)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder smoothing = EMA with α = 1/window
    return true_range.ewm(alpha=1 / window, adjust=False).mean()


def atr_position_size(
    equity: float,
    entry_price: float,
    atr: float,
    atr_multiplier: float = 2.0,
    risk_pct: float = 0.01,
) -> dict:
    """ATR-Based Sizing — Stop bei entry - atr_multiplier × ATR."""
    if atr <= 0 or entry_price <= 0:
        return {"shares": 0, "stop_distance": 0, "dollar_risk": 0}
    stop_distance = atr_multiplier * atr
    dollar_risk = equity * risk_pct
    shares = dollar_risk / stop_distance
    return {
        "shares": float(shares),
        "stop_distance": float(stop_distance),
        "dollar_risk": float(dollar_risk),
        "dollar_size": float(shares * entry_price),
    }


def volatility_targeted_size(
    capital: float,
    asset_vol_annualized: float,
    target_vol_annualized: float = 0.15,
    max_leverage: float = 2.0,
) -> dict:
    """Vol-Targeting: notional = capital × target_vol / asset_vol.

    Notional cap = capital × max_leverage.
    """
    if asset_vol_annualized <= 0:
        return {"notional": 0, "leverage": 0}
    leverage = target_vol_annualized / asset_vol_annualized
    leverage = min(leverage, max_leverage)
    return {
        "notional": float(capital * leverage),
        "leverage": float(leverage),
    }


def equal_risk_contribution_sizes(
    capital: float,
    asset_vols: pd.Series,
    target_total_vol: float = 0.15,
) -> pd.Series:
    """Diagonal ERC (no correlation): notional_i = capital × (target/N) / σ_i."""
    sigma = asset_vols.copy()
    sigma = sigma[sigma > 0]
    if sigma.empty:
        return pd.Series(dtype=float)
    n = len(sigma)
    per_asset_vol_budget = target_total_vol / np.sqrt(n)
    notionals = capital * per_asset_vol_budget / sigma
    return notionals


__all__ = [
    "heat_based_sizing",
    "average_true_range",
    "atr_position_size",
    "volatility_targeted_size",
    "equal_risk_contribution_sizes",
]
