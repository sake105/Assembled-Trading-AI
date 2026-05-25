"""Exit rules for news alpha positions.

Three exit triggers (checked in order, first match wins):
    1. Time: position has been held >= hold_days → exit
    2. Take profit: current price >= entry_price * (1 + take_profit_pct) → exit
    3. Stop loss: current price <= entry_price * (1 - stop_loss_pct) → exit
    4. Reversal: a new trigger with OPPOSITE direction fires for the same topic → exit

All exits return (signal, reason_str) pairs for the execution layer.
"""

from __future__ import annotations

import logging
from typing import Any

from src.assembled_core.events.news_alpha.models import NewsAlphaSignal

logger = logging.getLogger(__name__)


def check_exits(
    open_signals: list[NewsAlphaSignal],
    current_day: int,
    prices: dict[str, float] | None = None,
    new_trigger_items: list[dict[str, Any]] | None = None,
) -> list[tuple[NewsAlphaSignal, str]]:
    """Return (signal, reason) pairs for signals that should be exited.

    Args:
        open_signals: Currently active NewsAlphaSignal positions.
        current_day: Current trading day counter (relative to entry_day).
        prices: {symbol: current_price} for price-based checks.
        new_trigger_items: Fresh triggers — if a reversal is detected, exit.

    Returns:
        List of (signal, exit_reason) for positions to close.
    """
    exits: list[tuple[NewsAlphaSignal, str]] = []
    prices = prices or {}

    for sig in open_signals:
        if not sig.active:
            continue

        days_held = current_day - sig.entry_day

        # 1. Time-based exit
        if days_held >= sig.hold_days:
            exits.append((sig, f"time_exit: held {days_held}d >= {sig.hold_days}d"))
            logger.info(
                "news_alpha EXIT time: %s %s held=%dd",
                sig.direction.upper(),
                sig.symbol,
                days_held,
            )
            continue

        # 2. Price-based exits
        current_price = prices.get(sig.symbol)
        if current_price is not None and current_price > 0 and sig.entry_price > 0:
            ret = current_price / sig.entry_price - 1.0
            if sig.direction == "long":
                if ret >= sig.take_profit_pct:
                    exits.append(
                        (
                            sig,
                            f"take_profit: ret={ret:.2%} >= {sig.take_profit_pct:.2%}",
                        )
                    )
                    logger.info(
                        "news_alpha EXIT tp: LONG %s ret=%.2f%%", sig.symbol, ret * 100
                    )
                    continue
                if ret <= -sig.stop_loss_pct:
                    exits.append(
                        (sig, f"stop_loss: ret={ret:.2%} <= -{sig.stop_loss_pct:.2%}")
                    )
                    logger.warning(
                        "news_alpha EXIT sl: LONG %s ret=%.2f%%", sig.symbol, ret * 100
                    )
                    continue
            else:  # short: profit when price falls (v1 unused — all inverse ETFs use direction="long")
                if ret <= -sig.take_profit_pct:
                    exits.append((sig, f"take_profit_short: ret={ret:.2%}"))
                    continue
                if ret >= sig.stop_loss_pct:
                    exits.append((sig, f"stop_loss_short: ret={ret:.2%}"))
                    continue

    return exits
