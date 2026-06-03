"""Exit rules for news alpha positions.

Three exit triggers (checked in order, first match wins):
    1. Time: position has been held >= hold_days → exit
    2. Take profit: current price >= entry_price * (1 + take_profit_pct) → exit
    3. Stop loss: current price <= entry_price * (1 - stop_loss_pct) → exit
    4. Reversal: a new trigger with OPPOSITE direction fires for the same topic → exit

The reversal exit (trigger 4) is LIVE-ACTIVE in the intraday paper path: it is
consumed via run_news_alpha_pipeline (events/news_alpha/pipeline.py) →
_execute_exits in the intraday runner. It is EXIT-only — it closes already-held
positions (submits a sell for the held qty, skips when qty <= 0) and never opens
a position. In practice it only fires for the central_bank hike↔cut pair, the
only bidirectional theme in the routing table.

All exits return (signal, reason_str) pairs for the execution layer.
"""

from __future__ import annotations

import logging
from typing import Any

from src.assembled_core.events.news_alpha.asset_router import (
    get_route,
    split_central_bank_topic,
)
from src.assembled_core.events.news_alpha.models import NewsAlphaSignal

logger = logging.getLogger(__name__)


def _resolve_theme_and_sign(item: dict[str, Any]) -> tuple[str, int] | None:
    """Resolve a fresh trigger item to a (theme, direction_sign) pair.

    Mirrors signal_generator/asset_router routing exactly:
        raw_topic = str(item['topic'])
        topic_id  = split_central_bank_topic(item) if raw_topic == 'central_bank'
                    else raw_topic
        route     = get_route(topic_id)

    ``theme`` collapses the only opposite-direction pair the routing table
    contains (central_bank_hike vs central_bank_cut) onto the shared underlying
    theme ``central_bank`` so a hike-then-cut reversal is detectable; every
    other topic_id is its own theme. ``sign`` is +1 for the "primary/long"
    directional intent and -1 for the opposite intent. Within a single theme,
    two routes with opposite signs are a reversal.

    Returns None when the item has no route (cannot be a reversal trigger).
    """
    raw_topic = str(item.get("topic", ""))
    topic_id = (
        split_central_bank_topic(item) if raw_topic == "central_bank" else raw_topic
    )
    if topic_id is None or get_route(topic_id) is None:
        return None
    # Only central_bank has a genuine opposite pair in the routing table.
    # The opposite-sign reversal logic therefore currently only fires for the
    # central_bank hike↔cut pair, because every non-central_bank topic_id maps
    # to sign +1 in ROUTING_TABLE (so no opposite-sign pair exists). If a future
    # ROUTING_TABLE entry introduces another genuinely bidirectional theme, this
    # helper must be updated so reversal fires for it too.
    if topic_id == "central_bank_hike":
        return ("central_bank", +1)
    if topic_id == "central_bank_cut":
        return ("central_bank", -1)
    # All other routes are single-direction: theme == topic_id, sign +1. They
    # can never reverse against another route (no opposite-sign route exists),
    # which keeps non-central-bank reversal inert by construction.
    return (str(topic_id), +1)


def _sig_theme_and_sign(sig: NewsAlphaSignal) -> tuple[str, int]:
    """(theme, sign) for an OPEN signal, keyed on its stored topic_id."""
    if sig.topic_id == "central_bank_hike":
        return ("central_bank", +1)
    if sig.topic_id == "central_bank_cut":
        return ("central_bank", -1)
    return (str(sig.topic_id), +1)


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
    exited_ids: set[int] = set()

    for sig in open_signals:
        if not sig.active:
            continue

        days_held = current_day - sig.entry_day

        # 1. Time-based exit
        if days_held >= sig.hold_days:
            exits.append((sig, f"time_exit: held {days_held}d >= {sig.hold_days}d"))
            exited_ids.add(id(sig))
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
                    exited_ids.add(id(sig))
                    logger.info(
                        "news_alpha EXIT tp: LONG %s ret=%.2f%%", sig.symbol, ret * 100
                    )
                    continue
                if ret <= -sig.stop_loss_pct:
                    exits.append(
                        (sig, f"stop_loss: ret={ret:.2%} <= -{sig.stop_loss_pct:.2%}")
                    )
                    exited_ids.add(id(sig))
                    logger.warning(
                        "news_alpha EXIT sl: LONG %s ret=%.2f%%", sig.symbol, ret * 100
                    )
                    continue
            else:  # short: profit when price falls (v1 unused — all inverse ETFs use direction="long")
                if ret <= -sig.take_profit_pct:
                    exits.append((sig, f"take_profit_short: ret={ret:.2%}"))
                    exited_ids.add(id(sig))
                    continue
                if ret >= sig.stop_loss_pct:
                    exits.append((sig, f"stop_loss_short: ret={ret:.2%}"))
                    exited_ids.add(id(sig))
                    continue

    # 4. Reversal exit — a fresh trigger resolving to the SAME theme but the
    # OPPOSITE directional intent as an open position closes that position.
    # Inert unless new_trigger_items is a non-empty list (default None → skip),
    # so callers that do not pass fresh triggers are unaffected. Runs only on
    # positions not already exited by checks 1-3. First match wins per signal.
    if isinstance(new_trigger_items, list) and new_trigger_items:
        fresh: list[tuple[str, int]] = []
        for item in new_trigger_items:
            if not isinstance(item, dict):
                continue
            resolved = _resolve_theme_and_sign(item)
            if resolved is not None:
                fresh.append(resolved)

        if fresh:
            for sig in open_signals:
                if not sig.active or id(sig) in exited_ids:
                    continue
                sig_theme, sig_sign = _sig_theme_and_sign(sig)
                for theme, sign in fresh:
                    if theme == sig_theme and sign == -sig_sign:
                        reason = f"reversal: opposite trigger topic={sig.topic_id}"
                        exits.append((sig, reason))
                        exited_ids.add(id(sig))
                        logger.info(
                            "news_alpha EXIT reversal: %s %s topic=%s "
                            "(opposite fresh trigger, theme=%s)",
                            sig.direction.upper(),
                            sig.symbol,
                            sig.topic_id,
                            sig_theme,
                        )
                        break

    return exits
