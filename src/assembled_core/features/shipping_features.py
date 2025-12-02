"""Shipping routes features module (Phase 6 Skeleton).

This module provides functions to compute features from shipping route and congestion data.
Currently provides skeleton implementations with simple aggregation logic.

Zukünftige Integration:
- Route-specific features (China routes vs Europe routes)
- Port congestion correlations with stock prices
- Supply chain disruption indicators
"""
from __future__ import annotations

import pandas as pd


def add_shipping_features(
    prices: pd.DataFrame,
    events: pd.DataFrame
) -> pd.DataFrame:
    """Add shipping route features to price DataFrame.
    
    Computes features like:
    - shipping_congestion_score: Average congestion score for routes related to this symbol
    - shipping_ships_count: Total number of ships on related routes
    - shipping_congestion_score_7d: 7-day average congestion score
    - shipping_ships_count_7d: 7-day average ship count
    
    Args:
        prices: DataFrame with columns: timestamp (UTC), symbol, close (and optionally other price columns)
        events: DataFrame from shipping_routes_ingest.load_shipping_sample() with columns:
            timestamp, route_id, port_from, port_to, symbol, ships, congestion_score
    
    Returns:
        Copy of prices DataFrame with additional columns:
        - shipping_congestion_score: Current congestion score (or NaN if no recent data)
        - shipping_ships_count: Current ship count (or NaN if no recent data)
        - shipping_congestion_score_7d: 7-day average congestion score
        - shipping_ships_count_7d: 7-day average ship count
    
    Raises:
        KeyError: If required columns are missing in prices or events
    """
    # Validate inputs
    required_price_cols = ["timestamp", "symbol", "close"]
    for col in required_price_cols:
        if col not in prices.columns:
            raise KeyError(f"Required column '{col}' not found in prices DataFrame")
    
    required_event_cols = ["timestamp", "symbol", "congestion_score", "ships"]
    for col in required_event_cols:
        if col not in events.columns:
            raise KeyError(f"Required column '{col}' not found in events DataFrame")
    
    result = prices.copy()
    
    # Ensure timestamps are datetime
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
    
    # Initialize feature columns
    result["shipping_congestion_score"] = pd.NA
    result["shipping_ships_count"] = pd.NA
    result["shipping_congestion_score_7d"] = pd.NA
    result["shipping_ships_count_7d"] = pd.NA
    
    # Group by symbol for efficient processing
    for symbol in result["symbol"].unique():
        symbol_mask = result["symbol"] == symbol
        symbol_prices = result[symbol_mask].copy()
        
        # Get events for this symbol
        symbol_events = events[events["symbol"] == symbol].copy()
        
        if symbol_events.empty:
            continue
        
        # For each price row, compute features based on events
        for idx in symbol_prices.index:
            price_time = symbol_prices.loc[idx, "timestamp"]
            
            # Most recent event (within 1 day)
            recent = symbol_events[
                (symbol_events["timestamp"] <= price_time) &
                (symbol_events["timestamp"] > price_time - pd.Timedelta(days=1))
            ]
            if not recent.empty:
                # Use most recent
                latest = recent.iloc[-1]
                result.loc[idx, "shipping_congestion_score"] = latest["congestion_score"]
                result.loc[idx, "shipping_ships_count"] = latest["ships"]
            
            # 7-day window for averages
            window_7d = symbol_events[
                (symbol_events["timestamp"] <= price_time) &
                (symbol_events["timestamp"] > price_time - pd.Timedelta(days=7))
            ]
            if not window_7d.empty:
                result.loc[idx, "shipping_congestion_score_7d"] = window_7d["congestion_score"].mean()
                result.loc[idx, "shipping_ships_count_7d"] = window_7d["ships"].mean()
    
    return result

