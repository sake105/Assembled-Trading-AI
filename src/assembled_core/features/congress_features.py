"""Congressional trading features module (Phase 6 Skeleton).

This module provides functions to compute features from congressional trading events.
Currently provides skeleton implementations with simple aggregation logic.

Zukünftige Integration:
- Party-weighted features (D vs R)
- Politician influence scores
- Sector-level aggregations
"""
from __future__ import annotations

import pandas as pd


def add_congress_features(
    prices: pd.DataFrame,
    events: pd.DataFrame
) -> pd.DataFrame:
    """Add congressional trading features to price DataFrame.
    
    Computes features like:
    - congress_trade_count_60d: Number of congressional trades in last 60 days
    - congress_total_amount_60d: Total trade amount in USD in last 60 days
    - congress_trade_count_90d: Number of trades in last 90 days
    - congress_total_amount_90d: Total trade amount in last 90 days
    
    Args:
        prices: DataFrame with columns: timestamp (UTC), symbol, close (and optionally other price columns)
        events: DataFrame from congress_trades_ingest.load_congress_sample() with columns:
            timestamp, symbol, politician, party, amount
    
    Returns:
        Copy of prices DataFrame with additional columns:
        - congress_trade_count_60d: Trade count in last 60 days
        - congress_total_amount_60d: Total amount in last 60 days
        - congress_trade_count_90d: Trade count in last 90 days
        - congress_total_amount_90d: Total amount in last 90 days
    
    Raises:
        KeyError: If required columns are missing in prices or events
    """
    # Validate inputs
    required_price_cols = ["timestamp", "symbol", "close"]
    for col in required_price_cols:
        if col not in prices.columns:
            raise KeyError(f"Required column '{col}' not found in prices DataFrame")
    
    required_event_cols = ["timestamp", "symbol", "amount"]
    for col in required_event_cols:
        if col not in events.columns:
            raise KeyError(f"Required column '{col}' not found in events DataFrame")
    
    result = prices.copy()
    
    # Ensure timestamps are datetime
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
    
    # Initialize feature columns
    result["congress_trade_count_60d"] = 0
    result["congress_total_amount_60d"] = 0.0
    result["congress_trade_count_90d"] = 0
    result["congress_total_amount_90d"] = 0.0
    
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
            
            # 60-day window
            window_60d = symbol_events[
                (symbol_events["timestamp"] <= price_time) &
                (symbol_events["timestamp"] > price_time - pd.Timedelta(days=60))
            ]
            result.loc[idx, "congress_trade_count_60d"] = len(window_60d)
            result.loc[idx, "congress_total_amount_60d"] = window_60d["amount"].sum()
            
            # 90-day window
            window_90d = symbol_events[
                (symbol_events["timestamp"] <= price_time) &
                (symbol_events["timestamp"] > price_time - pd.Timedelta(days=90))
            ]
            result.loc[idx, "congress_trade_count_90d"] = len(window_90d)
            result.loc[idx, "congress_total_amount_90d"] = window_90d["amount"].sum()
    
    return result

