"""Event-based signal rules module (Phase 6).

This module provides signal generation based on event data:
- Insider trading activity (net buys/sells)
- Shipping route congestion

Signal logic:
- LONG (+1): Strong insider net buy + low shipping congestion
- SHORT (-1): Strong insider net sell + high shipping congestion
- FLAT (0): Otherwise

Zukünftige Integration:
- Additional event sources (Congress trades, News sentiment)
- Configurable thresholds and weights
- Multi-event signal combination
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.assembled_core.logging_utils import get_logger

logger = get_logger("assembled_core.signals.event")


def generate_event_signals(
    prices_with_features: pd.DataFrame,
    *,
    insider_weight: float = 1.0,
    shipping_weight: float = 1.0,
    insider_net_buy_threshold: float = 1000.0,
    insider_net_sell_threshold: float = -1000.0,
    shipping_congestion_low_threshold: float = 30.0,
    shipping_congestion_high_threshold: float = 70.0,
) -> pd.DataFrame:
    """Generate event-based signals from insider trading and shipping data.

    Signal logic:
    - LONG (+1): insider_net_buy_20d > threshold AND shipping_congestion_score_7d < low_threshold
    - SHORT (-1): insider_net_buy_20d < threshold AND shipping_congestion_score_7d > high_threshold
    - FLAT (0): Otherwise

    Args:
        prices_with_features: DataFrame with columns:
            - timestamp, symbol, close (required)
            - insider_net_buy_20d, insider_trade_count_20d (from insider_features)
            - shipping_congestion_score_7d, shipping_ships_count_7d (from shipping_features)
        insider_weight: Weight for insider signals (default: 1.0)
        shipping_weight: Weight for shipping signals (default: 1.0)
        insider_net_buy_threshold: Minimum net buy for LONG signal (default: 1000.0)
        insider_net_sell_threshold: Maximum net buy (negative = sell) for SHORT signal (default: -1000.0)
        shipping_congestion_low_threshold: Maximum congestion for LONG signal (default: 30.0)
        shipping_congestion_high_threshold: Minimum congestion for SHORT signal (default: 70.0)

    Returns:
        DataFrame with columns: timestamp, symbol, direction, score
        direction: "LONG", "FLAT", or "SHORT"
        score: Signal strength (0.0 to 1.0), based on feature magnitudes
        Sorted by symbol, then timestamp

    Raises:
        KeyError: If required columns are missing
        ValueError: If weights are invalid
    """
    # Validate weights
    if insider_weight < 0 or shipping_weight < 0:
        raise ValueError(
            f"weights must be non-negative, got insider_weight={insider_weight}, shipping_weight={shipping_weight}"
        )

    df = prices_with_features.copy()

    # Ensure required columns
    required = ["timestamp", "symbol", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}. Available: {list(df.columns)}"
        )

    # Check for required feature columns
    required_features = ["insider_net_buy_20d", "shipping_congestion_score_7d"]
    missing_features = [c for c in required_features if c not in df.columns]
    if missing_features:
        raise KeyError(
            f"Missing required feature columns: {missing_features}. "
            f"Available: {list(df.columns)}. "
            f"Please run add_insider_features() and add_shipping_features() first."
        )

    # Sort by symbol and timestamp
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Fill NaN values with 0 for feature columns (no events = neutral)
    # Convert to numeric first, then fill NaN to avoid FutureWarning about downcasting
    if "insider_net_buy_20d" in df.columns:
        df["insider_net_buy_20d"] = pd.to_numeric(
            df["insider_net_buy_20d"], errors="coerce"
        ).fillna(0.0)
    if "shipping_congestion_score_7d" in df.columns:
        df["shipping_congestion_score_7d"] = pd.to_numeric(
            df["shipping_congestion_score_7d"], errors="coerce"
        ).fillna(50.0)  # Neutral congestion

    # Compute signal components
    # Insider component: positive for net buy, negative for net sell
    insider_signal = np.where(
        df["insider_net_buy_20d"] > insider_net_buy_threshold,
        1.0,  # Strong buy
        np.where(
            df["insider_net_buy_20d"] < insider_net_sell_threshold,
            -1.0,  # Strong sell
            0.0,  # Neutral
        ),
    )

    # Shipping component: positive for low congestion (good), negative for high congestion (bad)
    # Low congestion = bullish, High congestion = bearish
    shipping_signal = np.where(
        df["shipping_congestion_score_7d"] < shipping_congestion_low_threshold,
        1.0,  # Low congestion (bullish)
        np.where(
            df["shipping_congestion_score_7d"] > shipping_congestion_high_threshold,
            -1.0,  # High congestion (bearish)
            0.0,  # Neutral
        ),
    )

    # Combine signals with weights
    # For LONG: insider positive (buy) AND shipping positive (low congestion)
    # For SHORT: insider negative (sell) AND shipping negative (high congestion)
    combined_signal = (insider_weight * insider_signal) + (
        shipping_weight * shipping_signal
    )

    # Generate final signal direction
    # LONG: combined > 0.5 (insider buy + low congestion)
    # SHORT: combined < -0.5 (insider sell + high congestion)
    # FLAT: otherwise
    df["direction"] = np.where(
        combined_signal > 0.5, "LONG", np.where(combined_signal < -0.5, "SHORT", "FLAT")
    )

    # Compute signal score (0.0 to 1.0)
    # Score based on absolute magnitude of combined signal
    abs_combined = np.abs(combined_signal)
    df["score"] = np.clip(abs_combined / 2.0, 0.0, 1.0)  # Normalize to [0, 1]

    # Only set score for non-FLAT signals
    df["score"] = np.where(df["direction"] != "FLAT", df["score"], 0.0)

    # Select output columns
    result = df[["timestamp", "symbol", "direction", "score"]].copy()

    logger.debug(
        f"Generated event signals: {len(result)} rows, "
        f"LONG: {(result['direction'] == 'LONG').sum()}, "
        f"FLAT: {(result['direction'] == 'FLAT').sum()}, "
        f"SHORT: {(result['direction'] == 'SHORT').sum()}"
    )

    return result
