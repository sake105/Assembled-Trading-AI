"""Dashboard Data Provider — Backend for Streamlit/Dash UI.

Provides structured data snapshots for dashboard rendering:
- Real-Time PnL Curve
- Current Positions + Exposure
- Risk Metrics (VaR, Drawdown, Exposure)
- Factor Performance Monitor
- Signal Strength Heatmap
- Trade Activity Log

Decoupled from any specific UI framework — returns dicts/DataFrames
that can be consumed by Streamlit, Dash, or REST API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DashboardSnapshot:
    """Complete dashboard state snapshot."""
    timestamp: str
    pnl_curve: dict[str, float]  # date_str -> cumulative PnL
    current_positions: list[dict[str, Any]]  # [{symbol, weight, pnl, sector}]
    risk_metrics: dict[str, float]  # {var_95, drawdown, vol, sharpe, ...}
    factor_performance: dict[str, float]  # {factor_name -> IC or return}
    signal_heatmap: dict[str, dict[str, float]]  # {symbol -> {signal_name -> strength}}
    trade_log: list[dict[str, Any]]  # recent trades
    exposure: dict[str, float]  # {gross, net, long, short}
    alerts: list[dict[str, str]]  # recent alerts [{level, message, ts}]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_pnl_curve(
    equity_series: pd.Series,
    initial_capital: float = 100000.0,
) -> dict[str, float]:
    """Build PnL curve from equity series.

    Args:
        equity_series: Daily equity values indexed by date.
        initial_capital: Starting capital.

    Returns:
        Dict of date_str -> cumulative PnL.
    """
    if equity_series.empty:
        return {}
    pnl = equity_series - initial_capital
    return {str(k): round(float(v), 2) for k, v in pnl.items()}


def build_position_table(
    weights: dict[str, float],
    prices: dict[str, float] | None = None,
    daily_pnl: dict[str, float] | None = None,
    sector_mapping: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build position table for dashboard.

    Args:
        weights: Symbol -> portfolio weight.
        prices: Symbol -> current price.
        daily_pnl: Symbol -> today's PnL.
        sector_mapping: Symbol -> sector.

    Returns:
        List of position dicts.
    """
    positions = []
    for sym, w in sorted(weights.items(), key=lambda x: -abs(x[1])):
        if abs(w) < 1e-8:
            continue
        positions.append({
            "symbol": sym,
            "weight": round(w, 4),
            "price": (prices or {}).get(sym, 0.0),
            "daily_pnl": round((daily_pnl or {}).get(sym, 0.0), 2),
            "sector": (sector_mapping or {}).get(sym, "Unknown"),
            "side": "long" if w > 0 else "short",
        })
    return positions


def compute_risk_snapshot(
    returns: pd.Series,
    lookback: int = 252,
) -> dict[str, float]:
    """Compute risk metrics snapshot.

    Args:
        returns: Daily portfolio returns.
        lookback: Number of days for rolling metrics.

    Returns:
        Dict of metric_name -> value.
    """
    if len(returns) < 2:
        return {}

    recent = returns.tail(lookback)
    cumret = (1 + recent).cumprod()
    peak = cumret.cummax()
    dd = (cumret - peak) / peak

    ann_ret = float(recent.mean()) * 252
    ann_vol = float(recent.std()) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-8 else 0.0

    return {
        "var_95": round(float(np.percentile(recent, 5)), 6),
        "var_99": round(float(np.percentile(recent, 1)), 6),
        "current_drawdown": round(float(dd.iloc[-1]), 6),
        "max_drawdown": round(float(dd.min()), 6),
        "annualized_return": round(ann_ret, 4),
        "annualized_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "win_rate": round(float((recent > 0).mean()), 4),
        "avg_win": round(float(recent[recent > 0].mean()), 6) if (recent > 0).any() else 0.0,
        "avg_loss": round(float(recent[recent < 0].mean()), 6) if (recent < 0).any() else 0.0,
    }


def compute_exposure(weights: dict[str, float]) -> dict[str, float]:
    """Compute exposure metrics from weights.

    Returns:
        Dict with gross, net, long, short exposure.
    """
    long_exp = sum(w for w in weights.values() if w > 0)
    short_exp = sum(abs(w) for w in weights.values() if w < 0)
    return {
        "gross": round(long_exp + short_exp, 4),
        "net": round(long_exp - short_exp, 4),
        "long": round(long_exp, 4),
        "short": round(short_exp, 4),
        "n_positions": sum(1 for w in weights.values() if abs(w) > 1e-6),
    }


def build_signal_heatmap(
    signals: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Build signal strength heatmap from signal DataFrame.

    Args:
        signals: DataFrame with symbols as index, signal names as columns.

    Returns:
        Nested dict: symbol -> signal_name -> strength.
    """
    if signals.empty:
        return {}
    result = {}
    for sym in signals.index:
        result[str(sym)] = {
            col: round(float(signals.loc[sym, col]), 4)
            for col in signals.columns
        }
    return result


__all__ = [
    "DashboardSnapshot",
    "build_pnl_curve",
    "build_position_table",
    "compute_risk_snapshot",
    "compute_exposure",
    "build_signal_heatmap",
]
