"""Legacy implementations of backtest functions for regression testing.

This module contains the original (pre-optimization) implementations of critical
backtest functions. These are used only for regression testing to ensure that
optimizations (vectorization, Numba) don't change the logic.

WARNING: These functions are for testing only. Do not use in production code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _legacy_update_equity_mark_to_market(
    timestamp: pd.Timestamp,
    cash: float,
    positions: dict[str, float],
    price_pivot: pd.DataFrame,
    symbols: list[str],
) -> float:
    """Legacy implementation: Update equity via mark-to-market (original loop-based).

    This is the original implementation before vectorization.
    Used only for regression testing.

    Args:
        timestamp: Current timestamp
        cash: Current cash balance
        positions: Dictionary mapping symbol -> quantity (current positions)
        price_pivot: Pivoted DataFrame with timestamp index and symbol columns (close prices)
        symbols: List of all symbols in the portfolio

    Returns:
        Equity value (cash + mark-to-market value of positions)
    """
    if timestamp not in price_pivot.index:
        return cash  # Falls Lücke in Preisdaten

    mtm = 0.0
    row = price_pivot.loc[timestamp]

    # Original loop-based implementation
    for symbol in symbols:
        price = float(row.get(symbol, np.nan))
        if not np.isnan(price):
            position_qty = positions.get(symbol, 0.0)
            mtm += position_qty * price

    equity = cash + mtm
    return equity


def _legacy_simulate_equity(
    prices: pd.DataFrame, orders: pd.DataFrame, start_capital: float
) -> pd.DataFrame:
    """Legacy implementation: Simulate equity curve (original implementation).

    This is the original implementation before optimizations.
    Used only for regression testing.

    Args:
        prices: DataFrame with columns: timestamp, symbol, close
        orders: DataFrame with columns: timestamp, symbol, side, qty, price
        start_capital: Starting capital

    Returns:
        DataFrame with columns: timestamp, equity
        Sorted by timestamp
    """
    # Timeline & Price pivot
    prices = prices.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    timeline = prices["timestamp"].sort_values().drop_duplicates().to_list()
    symbols = sorted(prices["symbol"].unique().tolist())
    px = prices.pivot(index="timestamp", columns="symbol", values="close").sort_index()

    # Orders nach Timestamp gruppieren
    orders = orders.sort_values("timestamp").reset_index(drop=True)
    orders_by_ts = {}
    if len(orders):
        for ts, group in orders.groupby("timestamp"):
            orders_by_ts[ts] = group

    # Simulationszustand
    cash = float(start_capital)
    pos = {s: 0.0 for s in symbols}
    equity_series = []

    # Per-timestamp loop: execute orders and update equity
    # Original implementation using legacy mark-to-market
    for ts in timeline:
        # Execute orders at this timestamp (original loop-based)
        if ts in orders_by_ts:
            for _, row in orders_by_ts[ts].iterrows():
                sym = row.get("symbol", "")
                side = row.get("side", "")
                qty = float(row.get("qty", 0.0))
                price = float(row.get("price", np.nan))

                if not sym or np.isnan(price) or qty == 0.0:
                    continue

                if side == "BUY":
                    cash -= qty * price
                    pos[sym] = pos.get(sym, 0.0) + qty
                elif side == "SELL":
                    cash += qty * price
                    pos[sym] = pos.get(sym, 0.0) - qty

        # Mark-to-market (legacy loop-based)
        equity = _legacy_update_equity_mark_to_market(ts, cash, pos, px, symbols)
        equity_series.append((ts, float(equity)))

    eq = pd.DataFrame(equity_series, columns=["timestamp", "equity"])
    # Sanitizing
    s = pd.Series(eq["equity"].values, index=eq["timestamp"])
    s = s.replace([np.inf, -np.inf], np.nan).ffill().fillna(start_capital)
    eq["equity"] = s.values
    return eq

