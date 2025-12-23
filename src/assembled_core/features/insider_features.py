"""Insider trading features module (Phase 6 Skeleton).

This module provides functions to compute features from insider trading events.
Currently provides skeleton implementations with simple aggregation logic.

Zukünftige Integration:
- Erweiterte Features: Insider buy/sell ratios, role-weighted signals, etc.
- Time-weighted features (recent trades more important)
- Cross-symbol correlations
"""

from __future__ import annotations

import pandas as pd


def add_insider_features(prices: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Add insider trading features to price DataFrame.

    Computes features like:
    - insider_net_buy_20d: Net shares bought (positive) or sold (negative) in last 20 days
    - insider_trade_count_20d: Number of insider trades in last 20 days
    - insider_net_buy_60d: Net shares bought/sold in last 60 days
    - insider_trade_count_60d: Number of insider trades in last 60 days

    Args:
        prices: DataFrame with columns: timestamp (UTC), symbol, close (and optionally other price columns)
        events: DataFrame from insider_ingest.load_insider_sample() with columns:
            timestamp, symbol, trades_count, net_shares, role

    Returns:
        Copy of prices DataFrame with additional columns:
        - insider_net_buy_20d: Net shares bought in last 20 days
        - insider_trade_count_20d: Trade count in last 20 days
        - insider_net_buy_60d: Net shares bought in last 60 days
        - insider_trade_count_60d: Trade count in last 60 days

    Raises:
        KeyError: If required columns are missing in prices or events
    """
    # Validate inputs
    required_price_cols = ["timestamp", "symbol", "close"]
    for col in required_price_cols:
        if col not in prices.columns:
            raise KeyError(f"Required column '{col}' not found in prices DataFrame")

    required_event_cols = ["timestamp", "symbol", "net_shares", "trades_count"]
    for col in required_event_cols:
        if col not in events.columns:
            raise KeyError(f"Required column '{col}' not found in events DataFrame")

    result = prices.copy()

    # Ensure timestamps are datetime
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)

    # Initialize feature columns
    result["insider_net_buy_20d"] = 0.0
    result["insider_trade_count_20d"] = 0
    result["insider_net_buy_60d"] = 0.0
    result["insider_trade_count_60d"] = 0

    # Group by symbol for efficient processing
    for symbol in result["symbol"].unique():
        symbol_mask = result["symbol"] == symbol
        symbol_prices = result[symbol_mask].copy()

        # Get events for this symbol
        symbol_events = events[events["symbol"] == symbol].copy()

        if symbol_events.empty:
            continue

        # For each price row, compute features based on events in rolling windows
        for idx in symbol_prices.index:
            price_time = symbol_prices.loc[idx, "timestamp"]

            # 20-day window
            window_20d = symbol_events[
                (symbol_events["timestamp"] <= price_time)
                & (symbol_events["timestamp"] > price_time - pd.Timedelta(days=20))
            ]
            result.loc[idx, "insider_net_buy_20d"] = window_20d["net_shares"].sum()
            result.loc[idx, "insider_trade_count_20d"] = len(window_20d)

            # 60-day window
            window_60d = symbol_events[
                (symbol_events["timestamp"] <= price_time)
                & (symbol_events["timestamp"] > price_time - pd.Timedelta(days=60))
            ]
            result.loc[idx, "insider_net_buy_60d"] = window_60d["net_shares"].sum()
            result.loc[idx, "insider_trade_count_60d"] = len(window_60d)

    return result
