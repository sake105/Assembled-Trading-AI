"""News Alpha pipeline — event-driven directional trading.

This is the intended Crisis Alpha: when a high-impact news event fires,
immediately open directional positions in the specific affected assets.

Example flow:
    RSS/API detects "Strait of Hormuz blockade"
    → trigger_scoring classifies as shipping_disruption, severity=3
    → asset_router maps to Long XLE (+ UCO if leverage allowed)
    → signal_generator creates NewsAlphaSignal with weight/hold_days
    → execution layer enters position at next available price

This module runs ALONGSIDE the existing crisis_alpha defensive overlay
(events/crisis_alpha/). The two are separate:
    - crisis_alpha: slow MDD-reduction basket (weeks, defensive ETFs)
    - news_alpha: fast directional alpha (days, event-specific assets)

Integration point: run_news_alpha_pipeline() is called from _tc_sizing.py
after crisis_alpha, with its result added to effective target weights.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.assembled_core.events.news_alpha.exit_rules import check_exits
from src.assembled_core.events.news_alpha.models import NewsAlphaResult, NewsAlphaSignal
from src.assembled_core.events.news_alpha.signal_generator import (
    generate_signals,
    signals_to_weights,
)

logger = logging.getLogger(__name__)


def run_news_alpha_pipeline(
    trigger_items: list[dict[str, Any]],
    open_signals: list[NewsAlphaSignal] | None = None,
    current_day: int = 0,
    prices: dict[str, float] | None = None,
    policy: dict | None = None,
    *,
    shadow_only: bool = True,
    timestamp_utc: datetime | None = None,
) -> NewsAlphaResult:
    """Run one news-alpha evaluation cycle.

    Args:
        trigger_items: Fresh trigger dicts from the news pipeline.
            Each must have 'severity' (int 1-3) and 'topic' (str).
        open_signals: Currently held NewsAlphaSignal positions (for exit checks).
        current_day: Monotonically increasing day counter for time-exit tracking.
        prices: {symbol: current_price} for stop-loss / take-profit evaluation.
        policy: Policy dict. Reads from policy["news_alpha"].
        shadow_only: If True, signals are generated but NOT meant to be executed.
        timestamp_utc: Reference timestamp (default: now).

    Returns:
        NewsAlphaResult with:
            signals:          new signals generated this cycle
            target_weights:   {symbol: weight} for new entries (populated even when shadow_only=True)
            positions_to_exit: [(signal, reason)] for open positions to close
            shadow_only:      forwarded flag — CALLER MUST CHECK before applying target_weights
            errors:           non-fatal warnings

    IMPORTANT: shadow_only is a caller-enforced contract, not an execution barrier.
    target_weights is always populated so the caller can log/monitor regardless of mode.
    At the wiring point in _tc_sizing.py, the caller MUST gate on `result.shadow_only is False`.

    NOTE: target_weights covers only NEW signals generated this cycle.
    open_signals weights are not included in the gross cap calculation.
    The downstream sizer must re-enforce the global gross cap across all positions.
    """
    p = policy or {}
    cfg = p.get("news_alpha", {})
    enabled = cfg.get("enabled", False)

    ts = timestamp_utc or datetime.now(timezone.utc)
    ts_str = ts.isoformat()

    if not enabled:
        logger.debug("news_alpha: disabled — skipping")
        return NewsAlphaResult(timestamp_utc=ts_str, shadow_only=shadow_only)

    errors: list[str] = []
    open_sigs = open_signals or []

    # --- Step 1: Check exits for open positions ---
    exits = []
    try:
        exits = check_exits(
            open_sigs, current_day, prices=prices, new_trigger_items=trigger_items
        )
        for sig, reason in exits:
            sig.active = False
            logger.info(
                "[NEWS_ALPHA] EXIT %s %s: %s", sig.direction.upper(), sig.symbol, reason
            )
    except Exception as exc:
        err = f"exit_check failed: {exc}"
        errors.append(err)
        logger.exception("[NEWS_ALPHA] %s", err)

    # --- Step 2: Generate new signals from fresh triggers ---
    new_signals: list[NewsAlphaSignal] = []
    try:
        new_signals = generate_signals(trigger_items, policy=policy, signal_utc=ts_str)
        for sig in new_signals:
            sig.entry_day = current_day
    except Exception as exc:
        err = f"signal_generation failed: {exc}"
        errors.append(err)
        logger.exception("[NEWS_ALPHA] %s", err)

    # --- Step 3: Build target weights from new signals ---
    target_weights: dict[str, float] = {}
    try:
        if new_signals:
            target_weights = signals_to_weights(new_signals, policy=policy)
    except Exception as exc:
        err = f"weight_generation failed: {exc}"
        errors.append(err)
        logger.exception("[NEWS_ALPHA] %s", err)

    if shadow_only and (new_signals or exits):
        logger.info(
            "[NEWS_ALPHA] shadow_only=True — %d new signals, %d exits NOT executed | "
            "targets: %s",
            len(new_signals),
            len(exits),
            {s: f"{w:+.3f}" for s, w in target_weights.items()},
        )
    elif new_signals:
        logger.info(
            "[NEWS_ALPHA] %d signals | targets: %s",
            len(new_signals),
            {s: f"{w:+.3f}" for s, w in target_weights.items()},
        )

    return NewsAlphaResult(
        timestamp_utc=ts_str,
        signals=new_signals,
        target_weights=target_weights,
        positions_to_exit=exits,
        shadow_only=shadow_only,
        errors=errors,
    )
