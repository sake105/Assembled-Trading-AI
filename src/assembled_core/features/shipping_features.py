"""Shipping routes features module (Phase 6 Skeleton).

This module provides functions to compute features from shipping route and congestion data.
Currently provides skeleton implementations with simple aggregation logic.

Zukünftige Integration:
- Route-specific features (China routes vs Europe routes)
- Port congestion correlations with stock prices
- Supply chain disruption indicators
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_shipping_features(prices: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
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

    # Pre-group events by symbol to avoid O(N*M) per-symbol filter
    _events_by_sym = {sym: grp for sym, grp in events.groupby("symbol", sort=False)}

    td1_ns = int(pd.Timedelta(days=1).value)
    td7_ns = int(pd.Timedelta(days=7).value)

    # Group by symbol for efficient processing
    for symbol, symbol_prices in result.groupby("symbol", sort=False):
        symbol_events = _events_by_sym.get(symbol, pd.DataFrame())

        if symbol_events.empty:
            continue

        # Sort events by timestamp to enable searchsorted
        symbol_events = symbol_events.sort_values("timestamp")

        ev_time_ns = symbol_events["timestamp"].values.astype("int64")
        ev_congestion = symbol_events["congestion_score"].values.astype(float)
        ev_ships = symbol_events["ships"].values.astype(float)

        price_ts_ns = symbol_prices["timestamp"].values.astype("int64")
        n_prices = len(price_ts_ns)

        cong_1d = np.full(n_prices, np.nan)
        ships_1d = np.full(n_prices, np.nan)
        cong_7d = np.full(n_prices, np.nan)
        ships_7d = np.full(n_prices, np.nan)

        for i, pt_ns in enumerate(price_ts_ns):
            # searchsorted gives O(log N) window bounds on sorted events
            hi = int(np.searchsorted(ev_time_ns, pt_ns, side="right"))

            lo1 = int(np.searchsorted(ev_time_ns, pt_ns - td1_ns + 1))
            if lo1 < hi:
                cong_1d[i] = ev_congestion[hi - 1]
                ships_1d[i] = ev_ships[hi - 1]

            lo7 = int(np.searchsorted(ev_time_ns, pt_ns - td7_ns + 1))
            if lo7 < hi:
                cong_7d[i] = ev_congestion[lo7:hi].mean()
                ships_7d[i] = ev_ships[lo7:hi].mean()

        result.loc[symbol_prices.index, "shipping_congestion_score"] = cong_1d
        result.loc[symbol_prices.index, "shipping_ships_count"] = ships_1d
        result.loc[symbol_prices.index, "shipping_congestion_score_7d"] = cong_7d
        result.loc[symbol_prices.index, "shipping_ships_count_7d"] = ships_7d

    return result
