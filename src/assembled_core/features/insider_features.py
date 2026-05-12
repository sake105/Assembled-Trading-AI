"""Insider trading features module (Phase 6 Skeleton).

This module provides functions to compute features from insider trading events.
Currently provides skeleton implementations with simple aggregation logic.

B2 Integration: This module now supports PIT-safe filtering using disclosure_date.
Events are filtered to only include those disclosed by as_of, preventing look-ahead bias.

Zukünftige Integration:
- Erweiterte Features: Insider buy/sell ratios, role-weighted signals, etc.
- Time-weighted features (recent trades more important)
- Cross-symbol correlations
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.assembled_core.data.latency import (
    apply_source_latency,
    ensure_event_schema,
    filter_events_as_of,
)


def add_insider_features(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    disclosure_latency_days: int = 2,  # audit C4-082 — data.source_latencies.INSIDER_DAYS
) -> pd.DataFrame:
    """Add insider trading features to price DataFrame (PIT-safe).

    Computes features like:
    - insider_net_buy_20d: Net shares bought (positive) or sold (negative) in last 20 days
    - insider_trade_count_20d: Number of insider trades in last 20 days
    - insider_net_buy_60d: Net shares bought/sold in last 60 days
    - insider_trade_count_60d: Number of insider trades in last 60 days

    B2 PIT Safety:
    - If as_of is provided, only events with disclosure_date <= as_of are used
    - If disclosure_date is missing, it is derived from timestamp + disclosure_latency_days
    - This ensures features are "blind" to events not yet disclosed

    Args:
        prices: DataFrame with columns: timestamp (UTC), symbol, close (and optionally other price columns)
        events: DataFrame from insider_ingest.load_insider_sample() with columns:
            timestamp, symbol, trades_count, net_shares, role
            Optional: event_date, disclosure_date (if missing, derived from timestamp)
        as_of: Optional point-in-time cutoff (pd.Timestamp, UTC)
            Only events with disclosure_date <= as_of are used. If None, all events are used.
        disclosure_latency_days: Number of days between event_date and disclosure_date (default: 2)
            Used if disclosure_date is missing (typical for Form 4 filings: T+2)

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

    result = prices.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)

    # Ensure event schema (timestamp, symbol required)
    events = ensure_event_schema(
        events, required_cols=["timestamp", "symbol"], strict=False
    )

    # Ensure events have disclosure_date (derive from timestamp if missing)
    if "disclosure_date" not in events.columns:
        events = apply_source_latency(
            events,
            days=disclosure_latency_days,
            event_date_col="event_date",
            timestamp_col="timestamp",
        )

    # Apply PIT-safe filtering if as_of is provided
    if as_of is not None:
        # Determine max as_of from prices (if as_of not provided, use max price timestamp)
        if isinstance(as_of, pd.Timestamp):
            events = filter_events_as_of(
                events, as_of, disclosure_col="disclosure_date"
            )
        else:
            # If as_of is per-price-row, we filter per row (handled in loop below)
            pass

    # For per-row filtering (if as_of varies per price row), filter in the loop
    # For now, assume as_of is a single timestamp (or None)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)

    # Ensure required event columns exist (with defaults if needed)
    if "net_shares" not in events.columns:
        events["net_shares"] = 0.0
    if "trades_count" not in events.columns:
        events["trades_count"] = 1

    # Initialize feature columns
    result["insider_net_buy_20d"] = 0.0
    result["insider_trade_count_20d"] = 0
    result["insider_net_buy_60d"] = 0.0
    result["insider_trade_count_60d"] = 0

    # Pre-group events by symbol to avoid O(N*M) per-symbol filter
    _events_by_sym = {sym: grp for sym, grp in events.groupby("symbol", sort=False)}

    td20_ns = int(pd.Timedelta(days=20).value)
    td60_ns = int(pd.Timedelta(days=60).value)

    # Group by symbol for efficient processing
    for symbol, symbol_prices in result.groupby("symbol", sort=False):
        # Get events for this symbol
        symbol_events = _events_by_sym.get(symbol, pd.DataFrame())
        if symbol_events.empty:
            continue

        # Determine window time column once (constant per symbol group)
        window_time_col = (
            "event_date" if "event_date" in symbol_events.columns else "timestamp"
        )
        has_pit = "disclosure_date" in symbol_events.columns
        has_ns = "net_shares" in symbol_events.columns

        # Pre-extract event arrays in int64 ns for fast comparison
        ev_time_ns = symbol_events[window_time_col].values.astype("int64")
        ev_ns_vals = (
            symbol_events["net_shares"].values
            if has_ns
            else np.zeros(len(symbol_events))
        )
        if has_pit:
            ev_disclose_ns = symbol_events["disclosure_date"].values.astype("int64")
        else:
            ev_disclose_ns = None

        price_ts_ns = symbol_prices["timestamp"].values.astype("int64")
        n_prices = len(price_ts_ns)

        nb20 = np.zeros(n_prices)
        tc20 = np.zeros(n_prices, dtype=np.int64)
        nb60 = np.zeros(n_prices)
        tc60 = np.zeros(n_prices, dtype=np.int64)

        for i, pt_ns in enumerate(price_ts_ns):
            # PIT: disclosure_date.normalize() <= price_time.normalize() in int64 ns
            if ev_disclose_ns is not None:
                # Normalize to day boundary (truncate to days in ns)
                _ns_per_day = 86_400_000_000_000
                pt_day_ns = (pt_ns // _ns_per_day) * _ns_per_day
                ev_day_ns = (ev_disclose_ns // _ns_per_day) * _ns_per_day
                pit_mask = ev_day_ns <= pt_day_ns
            else:
                pit_mask = np.ones(len(ev_time_ns), dtype=bool)

            # 20-day and 60-day windows combined with PIT
            w20 = pit_mask & (ev_time_ns <= pt_ns) & (ev_time_ns > pt_ns - td20_ns)
            w60 = pit_mask & (ev_time_ns <= pt_ns) & (ev_time_ns > pt_ns - td60_ns)

            nb20[i] = ev_ns_vals[w20].sum()
            tc20[i] = int(w20.sum())
            nb60[i] = ev_ns_vals[w60].sum()
            tc60[i] = int(w60.sum())

        # Bulk-assign results for this symbol (4 assignments vs 4 × N_prices)
        result.loc[symbol_prices.index, "insider_net_buy_20d"] = nb20
        result.loc[symbol_prices.index, "insider_trade_count_20d"] = tc20
        result.loc[symbol_prices.index, "insider_net_buy_60d"] = nb60
        result.loc[symbol_prices.index, "insider_trade_count_60d"] = tc60

    return result
