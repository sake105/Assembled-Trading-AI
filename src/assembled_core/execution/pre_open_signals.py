"""Pre-Open Signal Generation (M20 Task 20.5).

Generates signals before market open (9:20 AM EST) for opening auction orders.
Uses overnight data, pre-market activity, and global market signals.

Edge: +10-25 bps vs. next regular open price
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import time as dt_time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PreOpenSignal:
    """Pre-open trading signal for a single symbol."""
    symbol: str
    direction: str          # "BUY", "SELL", "NEUTRAL"
    strength: float         # 0-1 signal strength
    expected_open_move: float  # Expected open vs previous close (%)
    confidence: float       # 0-1 confidence in the signal
    components: dict        # Contributing signal components


@dataclass
class PreOpenConfig:
    """Configuration for pre-open signal generation."""
    signal_time: dt_time = dt_time(9, 20)  # 9:20 AM EST
    min_strength: float = 0.3
    overnight_weight: float = 0.3
    global_weight: float = 0.3
    premarket_weight: float = 0.2
    news_weight: float = 0.2


def compute_overnight_gap_signal(
    prev_close: float,
    premarket_price: float | None = None,
    futures_return: float = 0.0,
) -> tuple[float, float]:
    """Compute overnight gap signal.

    Args:
        prev_close: Previous day's close price.
        premarket_price: Pre-market indication price (if available).
        futures_return: Overnight futures return (e.g., ES futures).

    Returns:
        (signal, confidence) tuple. Signal in [-1, 1].
    """
    if premarket_price is not None and prev_close > 0:
        gap_pct = (premarket_price / prev_close - 1.0) * 100
        # Mean-reversion for large gaps, momentum for small gaps
        if abs(gap_pct) > 2.0:
            # Large gap → mean reversion
            signal = -np.tanh(gap_pct / 3.0)
            confidence = min(abs(gap_pct) / 5.0, 0.9)
        else:
            # Small gap → momentum continuation
            signal = np.tanh(gap_pct / 1.0)
            confidence = 0.5
    elif futures_return != 0:
        signal = np.tanh(futures_return * 100 / 1.5)
        confidence = 0.4
    else:
        signal = 0.0
        confidence = 0.0

    return float(signal), float(confidence)


def compute_global_market_signal(
    europe_return: float = 0.0,
    asia_return: float = 0.0,
    vix_change: float = 0.0,
    dollar_change: float = 0.0,
) -> tuple[float, float]:
    """Compute global market context signal for US open.

    Args:
        europe_return: European market return (e.g., STOXX 600).
        asia_return: Asian market return (e.g., Nikkei/Hang Seng).
        vix_change: VIX change from previous close (%).
        dollar_change: USD index change (%).

    Returns:
        (signal, confidence) tuple.
    """
    # Positive global markets → positive US signal
    global_momentum = europe_return * 0.5 + asia_return * 0.3

    # VIX spike → bearish
    vix_signal = -np.tanh(vix_change / 10.0) * 0.3

    # Dollar strength → slight bearish for US equities
    dollar_signal = -np.tanh(dollar_change * 100 / 1.0) * 0.2

    combined = np.tanh(global_momentum * 100 + vix_signal + dollar_signal)
    confidence = min(0.3 + abs(combined) * 0.4, 0.8)

    return float(combined), float(confidence)


def generate_pre_open_signals(
    symbols: list[str],
    prev_closes: dict[str, float],
    premarket_prices: dict[str, float] | None = None,
    futures_return: float = 0.0,
    europe_return: float = 0.0,
    asia_return: float = 0.0,
    vix_change: float = 0.0,
    news_scores: dict[str, float] | None = None,
    config: PreOpenConfig | None = None,
) -> list[PreOpenSignal]:
    """Generate pre-open signals for all symbols.

    Args:
        symbols: List of ticker symbols.
        prev_closes: {symbol: previous close price}.
        premarket_prices: {symbol: pre-market price} (optional).
        futures_return: Overnight futures return.
        europe_return: European market return.
        asia_return: Asian market return.
        vix_change: VIX change.
        news_scores: {symbol: sentiment score} from overnight news.
        config: Signal configuration.

    Returns:
        List of PreOpenSignal for each symbol.
    """
    cfg = config or PreOpenConfig()
    premarket_prices = premarket_prices or {}
    news_scores = news_scores or {}

    # Global signal (same for all stocks)
    global_sig, global_conf = compute_global_market_signal(
        europe_return, asia_return, vix_change
    )

    signals = []
    for sym in symbols:
        prev_close = prev_closes.get(sym, 0.0)
        if prev_close <= 0:
            continue

        components = {}

        # Overnight gap
        premarket = premarket_prices.get(sym)
        overnight_sig, overnight_conf = compute_overnight_gap_signal(
            prev_close, premarket, futures_return
        )
        components["overnight"] = overnight_sig

        # Global
        components["global"] = global_sig

        # Premarket volume/price (if available)
        premarket_sig = 0.0
        if premarket is not None:
            premarket_sig = np.tanh((premarket / prev_close - 1.0) * 50)
        components["premarket"] = premarket_sig

        # News sentiment
        news_sig = news_scores.get(sym, 0.0)
        components["news"] = news_sig

        # Weighted combination
        combined = (
            cfg.overnight_weight * overnight_sig +
            cfg.global_weight * global_sig +
            cfg.premarket_weight * premarket_sig +
            cfg.news_weight * news_sig
        )
        combined = float(np.clip(combined, -1, 1))

        # Confidence
        confidence = float(np.clip(
            overnight_conf * 0.4 + global_conf * 0.3 + (0.3 if premarket else 0.1),
            0, 1
        ))

        strength = abs(combined)
        if strength < cfg.min_strength:
            direction = "NEUTRAL"
        elif combined > 0:
            direction = "BUY"
        else:
            direction = "SELL"

        expected_move = 0.0
        if premarket is not None:
            expected_move = (premarket / prev_close - 1.0) * 100

        signals.append(PreOpenSignal(
            symbol=sym,
            direction=direction,
            strength=round(strength, 4),
            expected_open_move=round(expected_move, 4),
            confidence=round(confidence, 4),
            components=components,
        ))

    # Sort by strength descending
    signals.sort(key=lambda s: s.strength, reverse=True)

    n_active = sum(1 for s in signals if s.direction != "NEUTRAL")
    logger.info("[PreOpen] Generated %d signals (%d active)", len(signals), n_active)

    return signals


__all__ = [
    "PreOpenSignal",
    "PreOpenConfig",
    "compute_overnight_gap_signal",
    "compute_global_market_signal",
    "generate_pre_open_signals",
]
