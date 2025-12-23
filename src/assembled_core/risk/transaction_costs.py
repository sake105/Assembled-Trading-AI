"""Transaction Cost Analysis (TCA) Module.

This module provides simple transaction cost analysis for backtest strategies,
including cost estimation, cost-adjusted returns, and performance metrics.

**Ziel:** Einfache TCA für Backtests mit Schätzung von Execution-Kosten und deren
Auswirkung auf Net-Returns und Performance-Metriken.

**Annahmen & Limitierungen:**
- Einfache, approximative Cost-Modelle (keine intraday-Microstructure-Analyse)
- Spread- und Slippage-Schätzungen basieren auf approximierten Faktoren (Vol, Liq)
- Round-Trip-Matching ist einfach (First-In-First-Out)
- Fokus auf Backtest-Analyse, nicht auf Live-Execution-Monitoring

**Basis-Ansatz:**
```
cost_per_trade = commission + spread/2 + slippage
```

- Commission: Fixe Gebühr pro Trade (z.B. 0.5 bps)
- Spread: Half-spread Approximation (aus Vol/Liq-Faktoren oder fix)
- Slippage: Market Impact Approximation (fix oder basierend auf Trade-Größe)

**Integration:**
- Nutzt bestehende CostModel-Parameter aus `costs.py`
- Verwendet Trades aus `backtest_engine.py` (BacktestResult.trades)
- Integriert mit `risk_metrics.py` für Net-Return-Metriken
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def estimate_per_trade_cost(
    trades: pd.DataFrame,
    method: str = "simple",
    commission_bps: float = 0.5,
    spread_bps: float | None = None,
    slippage_bps: float = 3.0,
    factor_panel_df: pd.DataFrame | None = None,
    price_panel_df: pd.DataFrame | None = None,
    **kwargs,
) -> pd.Series:
    """Estimate cost per trade.

    Computes transaction costs for each trade using a simple model:
    cost = commission + spread/2 + slippage

    Args:
        trades: DataFrame with trades (columns: timestamp, symbol, side, qty, price)
        method: Cost estimation method:
            - "simple": Fixed costs per trade (default)
            - "adaptive": Costs based on volatility/liquidity factors (future)
        commission_bps: Commission in basis points (default: 0.5)
        spread_bps: Spread in basis points. If None, estimated from factors or fixed default (5 bps)
        slippage_bps: Slippage in basis points (default: 3.0)
        factor_panel_df: Optional DataFrame with volatility/liquidity factors (rv_20, turnover_20d)
        price_panel_df: Optional DataFrame with prices and volume for adaptive estimation
        **kwargs: Additional parameters for future adaptive methods

    Returns:
        Series with estimated cost per trade (in absolute values, same index as trades)

    Raises:
        ValueError: If required columns are missing in trades DataFrame
    """
    # Validate required columns
    required_cols = ["timestamp", "symbol", "side", "qty", "price"]
    missing_cols = [col for col in required_cols if col not in trades.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in trades DataFrame: {missing_cols}"
        )

    # Compute notional value
    if "notional" in trades.columns:
        notional = trades["notional"].abs()
    else:
        notional = (trades["qty"] * trades["price"]).abs()

    # Commission cost
    if "commission" in trades.columns or "fee" in trades.columns:
        # Use existing commission/fee column if available
        commission_cost = trades.get(
            "commission", trades.get("fee", pd.Series(0.0, index=trades.index))
        )
        if isinstance(commission_cost, pd.Series):
            commission_cost = commission_cost.fillna(0.0)
    else:
        # Calculate commission from bps
        commission_cost = notional * (commission_bps / 10000.0)

    # Spread cost (half-spread approximation)
    if spread_bps is None:
        spread_bps = 5.0  # Default 5 bps

    # If adaptive method and factor_panel_df available, could estimate spread dynamically
    # For now, use fixed spread
    spread_cost = notional * (spread_bps / 2.0 / 10000.0)

    # Slippage cost
    slippage_cost = notional * (slippage_bps / 10000.0)

    # Total cost per trade
    total_cost = commission_cost + spread_cost + slippage_cost

    return total_cost


def compute_tca_for_trades(
    trades: pd.DataFrame,
    cost_per_trade: pd.Series | float | dict,
    *,
    price_col: str = "price",
    compute_pnl: bool = False,
) -> pd.DataFrame:
    """Compute TCA for all trades.

    Adds cost columns to trades DataFrame and optionally computes realized PnL.

    Args:
        trades: DataFrame with trades (columns: timestamp, symbol, side, qty, price)
        cost_per_trade: Cost per trade:
            - pd.Series: Costs per trade (index must match trades.index)
            - float: Constant cost for all trades
            - dict: Cost mapping (e.g., {symbol: cost} or {(symbol, side): cost})
        price_col: Name of price column (default: "price")
        compute_pnl: Whether to compute realized PnL (requires round-trip matching)

    Returns:
        DataFrame with additional columns:
        - cost_commission: Commission costs
        - cost_spread: Spread costs
        - cost_slippage: Slippage costs
        - cost_total: Total costs
        - realized_pnl_gross: Gross PnL (if compute_pnl=True)
        - realized_pnl_net: Net PnL (if compute_pnl=True)

    Raises:
        ValueError: If required columns are missing or cost_per_trade format is invalid
    """
    # Validate required columns
    required_cols = ["timestamp", "symbol", "side", "qty", price_col]
    missing_cols = [col for col in required_cols if col not in trades.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in trades DataFrame: {missing_cols}"
        )

    # Make a copy to avoid modifying original
    tca_trades = trades.copy()

    # Convert cost_per_trade to Series
    if isinstance(cost_per_trade, float):
        # Constant cost for all trades
        cost_series = pd.Series(cost_per_trade, index=trades.index)
    elif isinstance(cost_per_trade, dict):
        # Dict mapping - try to match by symbol or (symbol, side)
        cost_values = []
        for idx in trades.index:
            symbol = trades.loc[idx, "symbol"]
            side = trades.loc[idx, "side"]
            # Try (symbol, side) tuple first, then symbol, then default
            if (symbol, side) in cost_per_trade:
                cost_values.append(cost_per_trade[(symbol, side)])
            elif symbol in cost_per_trade:
                cost_values.append(cost_per_trade[symbol])
            else:
                raise ValueError(
                    f"No cost mapping found for symbol={symbol}, side={side}"
                )
        cost_series = pd.Series(cost_values, index=trades.index)
    elif isinstance(cost_per_trade, pd.Series):
        # Ensure index matches
        if not cost_per_trade.index.equals(trades.index):
            # Try to align by index
            cost_series = cost_per_trade.reindex(trades.index)
            if cost_series.isna().any():
                logger.warning("Some trades have missing costs - filling with 0.0")
                cost_series = cost_series.fillna(0.0)
        else:
            cost_series = cost_per_trade
    else:
        raise ValueError(
            f"Invalid cost_per_trade type: {type(cost_per_trade)}. Expected pd.Series, float, or dict."
        )

    # Add cost columns
    # For now, we put all costs into cost_total (can be split later if needed)
    tca_trades["cost_total"] = cost_series

    # Add placeholder columns for individual cost components (can be populated later)
    tca_trades["cost_commission"] = 0.0
    tca_trades["cost_spread"] = 0.0
    tca_trades["cost_slippage"] = 0.0

    # Compute notional
    if "notional" not in tca_trades.columns:
        tca_trades["notional"] = tca_trades["qty"] * tca_trades[price_col]

    # Compute PnL if requested
    if compute_pnl:
        # Simple PnL computation: requires matching entries/exits
        # For now, we'll compute gross_pnl per trade assuming we have entry/exit info
        # If not available, we'll leave as NaN
        gross_pnl = pd.Series(np.nan, index=trades.index)
        net_pnl = pd.Series(np.nan, index=trades.index)

        # If we have round-trip information, compute PnL
        # This is a simplified version - full round-trip matching would require more logic
        tca_trades["realized_pnl_gross"] = gross_pnl
        tca_trades["realized_pnl_net"] = net_pnl

        logger.info(
            "PnL computation requested but simplified - full round-trip matching not yet implemented"
        )

    return tca_trades


def summarize_tca(
    tca_trades: pd.DataFrame,
    freq: str = "D",
    equity_curve: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize TCA metrics over time.

    Aggregates transaction costs by period (daily, weekly, monthly).

    Args:
        tca_trades: DataFrame with cost-annotated trades (from compute_tca_for_trades)
        freq: Aggregation frequency ("D" = daily, "W" = weekly, "M" = monthly)
        equity_curve: Optional equity curve DataFrame (columns: timestamp, equity)
            for computing cost ratios

    Returns:
        DataFrame with aggregated metrics per period:
        - timestamp: Period timestamp
        - total_cost: Total transaction costs in period
        - n_trades: Number of trades in period
        - avg_cost_per_trade: Average cost per trade
        - total_gross_pnl: Total gross PnL (if available)
        - total_net_pnl: Total net PnL (if available)
        - cost_ratio: Cost / Gross Return (if equity_curve provided)

    Raises:
        ValueError: If required columns are missing in tca_trades
    """
    # Validate required columns
    if "timestamp" not in tca_trades.columns:
        raise ValueError("Missing required column 'timestamp' in tca_trades")
    if "cost_total" not in tca_trades.columns:
        raise ValueError("Missing required column 'cost_total' in tca_trades")

    # Ensure timestamp is datetime
    tca_trades = tca_trades.copy()
    if not pd.api.types.is_datetime64_any_dtype(tca_trades["timestamp"]):
        tca_trades["timestamp"] = pd.to_datetime(
            tca_trades["timestamp"], utc=True, errors="coerce"
        )

    # Set timestamp as index for resampling
    tca_trades_indexed = tca_trades.set_index("timestamp").sort_index()

    # Resample and aggregate
    agg_dict = {
        "cost_total": "sum",
    }

    # Add PnL columns if available
    if "realized_pnl_gross" in tca_trades.columns:
        agg_dict["realized_pnl_gross"] = "sum"
    if "realized_pnl_net" in tca_trades.columns:
        agg_dict["realized_pnl_net"] = "sum"

    # Group by frequency and aggregate
    resampled = tca_trades_indexed.resample(freq)
    summary = resampled.agg(agg_dict)

    # Count trades per period
    summary["n_trades"] = resampled.size()

    # Compute average cost per trade (robust to division by zero)
    summary["avg_cost_per_trade"] = summary["cost_total"] / summary["n_trades"].replace(
        0, np.nan
    )

    # Rename cost_total to total_cost for consistency
    summary = summary.rename(columns={"cost_total": "total_cost"})

    # Add total_gross_pnl and total_net_pnl if available
    if "realized_pnl_gross" in summary.columns:
        summary = summary.rename(columns={"realized_pnl_gross": "total_gross_pnl"})
    if "realized_pnl_net" in summary.columns:
        summary = summary.rename(columns={"realized_pnl_net": "total_net_pnl"})

    # Compute cost ratio if PnL available
    if "total_gross_pnl" in summary.columns:
        # Cost ratio = total_cost / |total_gross_pnl| (robust to division by zero)
        gross_pnl_abs = summary["total_gross_pnl"].abs().replace(0.0, np.nan)
        summary["cost_ratio"] = summary["total_cost"] / gross_pnl_abs

    # If equity_curve provided, compute cost ratio from returns
    if equity_curve is not None:
        if (
            "timestamp" not in equity_curve.columns
            or "equity" not in equity_curve.columns
        ):
            logger.warning(
                "equity_curve must have 'timestamp' and 'equity' columns. Skipping cost ratio calculation."
            )
        else:
            # Compute returns from equity curve
            equity_curve = equity_curve.copy()
            if not pd.api.types.is_datetime64_any_dtype(equity_curve["timestamp"]):
                equity_curve["timestamp"] = pd.to_datetime(
                    equity_curve["timestamp"], utc=True, errors="coerce"
                )
            equity_curve = equity_curve.set_index("timestamp").sort_index()
            equity_returns = equity_curve["equity"].pct_change().fillna(0.0)

            # Align with summary index
            equity_returns_resampled = equity_returns.resample(freq).sum()

            # Compute cost ratio = total_cost / |gross_return|
            returns_abs = equity_returns_resampled.abs().replace(0.0, np.nan)
            cost_ratio_from_returns = summary["total_cost"] / returns_abs

            # Use equity-based ratio if available, otherwise keep existing
            if "cost_ratio" not in summary.columns:
                summary["cost_ratio"] = cost_ratio_from_returns
            else:
                # Prefer PnL-based ratio, fallback to equity-based
                summary["cost_ratio"] = summary["cost_ratio"].fillna(
                    cost_ratio_from_returns
                )

    # Reset index to make timestamp a column
    summary = summary.reset_index()

    return summary


def compute_cost_adjusted_risk_metrics(
    returns: pd.Series,
    costs: pd.Series,
    freq: Literal["1d", "5min"] = "1d",
    risk_free_rate: float = 0.0,
) -> dict[str, float | None]:
    """Compute risk metrics with cost-adjusted returns.

    Calculates performance metrics using net returns (gross returns - costs).

    Args:
        returns: Series with gross returns (index = timestamp)
        costs: Series with daily costs (must align with returns index)
        freq: Frequency for annualization ("1d" or "5min")
        risk_free_rate: Risk-free rate (annualized, default: 0.0)

    Returns:
        Dictionary with net performance metrics:
        - net_mean_return_annualized: Annualized mean net return
        - net_vol_annualized: Annualized net return volatility
        - net_sharpe: Net Sharpe ratio
        - net_sortino: Net Sortino ratio
        - net_max_drawdown: Net maximum drawdown
        - cost_impact_sharpe: Sharpe difference (gross - net)
        - cost_impact_cagr: CAGR difference (gross - net)
        - total_cost: Total costs over period
        - cost_ratio: Total costs / Total gross return

    Raises:
        ValueError: If returns and costs indices don't align
    """
    # Validate inputs
    if len(returns) == 0:
        logger.warning("Empty returns Series - returning empty metrics")
        return {}

    if len(costs) == 0:
        logger.warning("Empty costs Series - treating as zero costs")
        costs = pd.Series(0.0, index=returns.index)

    # Align indices
    if not returns.index.equals(costs.index):
        # Try to align
        common_index = returns.index.intersection(costs.index)
        if len(common_index) == 0:
            raise ValueError("returns and costs indices have no common values")
        returns = returns.loc[common_index]
        costs = costs.loc[common_index]
        logger.warning(
            f"Aligned returns and costs to {len(common_index)} common periods"
        )

    # Handle edge case: single period
    if len(returns) == 1:
        logger.warning("Only one period available - metrics may be unreliable")
        net_return = returns.iloc[0] - costs.iloc[0]
        return {
            "net_mean_return_annualized": float(net_return)
            if not np.isnan(net_return)
            else None,
            "net_vol_annualized": None,
            "net_sharpe": None,
            "net_sortino": None,
            "net_max_drawdown": None,
            "cost_impact_sharpe": None,
            "cost_impact_cagr": None,
            "total_cost": float(costs.sum()),
            "cost_ratio": float(costs.sum() / returns.iloc[0])
            if returns.iloc[0] != 0
            else None,
            "n_periods": 1,
        }

    # Compute net returns
    net_returns = returns - costs

    # Import risk metrics functions
    from src.assembled_core.risk.risk_metrics import compute_basic_risk_metrics

    # Compute gross metrics
    gross_metrics = compute_basic_risk_metrics(
        returns, freq=freq, risk_free_rate=risk_free_rate
    )

    # Compute net metrics
    net_metrics = compute_basic_risk_metrics(
        net_returns, freq=freq, risk_free_rate=risk_free_rate
    )

    # Extract key metrics
    gross_sharpe = gross_metrics.get("sharpe")
    net_sharpe = net_metrics.get("sharpe")

    gross_cagr = gross_metrics.get("mean_return_annualized")  # Approximate CAGR
    net_cagr = net_metrics.get("mean_return_annualized")

    # Compute cost impact
    cost_impact_sharpe = (
        (gross_sharpe - net_sharpe)
        if (gross_sharpe is not None and net_sharpe is not None)
        else None
    )
    cost_impact_cagr = (
        (gross_cagr - net_cagr)
        if (gross_cagr is not None and net_cagr is not None)
        else None
    )

    # Total cost and cost ratio
    total_cost = float(costs.sum())
    total_gross_return = float(returns.sum())
    cost_ratio = (
        total_cost / abs(total_gross_return) if total_gross_return != 0 else None
    )

    # Build result dictionary
    result = {
        "net_mean_return_annualized": net_metrics.get("mean_return_annualized"),
        "net_vol_annualized": net_metrics.get("vol_annualized"),
        "net_sharpe": net_sharpe,
        "net_sortino": net_metrics.get("sortino"),
        "net_max_drawdown": net_metrics.get("max_drawdown"),
        "gross_sharpe": gross_sharpe,
        "gross_sortino": gross_metrics.get("sortino"),
        "gross_max_drawdown": gross_metrics.get("max_drawdown"),
        "cost_impact_sharpe": cost_impact_sharpe,
        "cost_impact_cagr": cost_impact_cagr,
        "total_cost": total_cost,
        "cost_ratio": cost_ratio,
        "n_periods": len(returns),
    }

    return result
