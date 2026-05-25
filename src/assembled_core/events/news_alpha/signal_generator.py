"""Convert classified news trigger items into NewsAlphaSignal objects.

Input:  list of trigger dicts from trigger_scoring.score_triggers()
        each dict has: severity, topic, source, [optional: details, confidence]

Output: list of NewsAlphaSignal ready for position sizing

Sizing logic:
    base_weight = policy.news_alpha.base_weight  (default 0.08 per signal)
    scaled_weight = base_weight * size_multiplier * (severity / 2.0)
    capped at policy.news_alpha.max_single_weight  (default 0.20)
    total gross capped at policy.news_alpha.max_gross_exposure  (default 0.40)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from src.assembled_core.events.news_alpha.asset_router import (
    get_route,
    split_central_bank_topic,
)
from src.assembled_core.events.news_alpha.models import NewsAlphaSignal

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "base_weight": 0.08,
    "max_single_weight": 0.20,
    "max_gross_exposure": 0.40,
    "min_severity": 2,
    "leverage_etfs_allowed": False,
    "stop_loss_pct": 0.08,
    "take_profit_pct": 0.15,
}


def _get_cfg(policy: dict | None) -> dict:
    p = policy or {}
    return {**_DEFAULTS, **p.get("news_alpha", {})}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_signals(
    trigger_items: list[dict[str, Any]],
    policy: dict | None = None,
    signal_utc: str | None = None,
) -> list[NewsAlphaSignal]:
    """Convert trigger items to directional trade signals.

    Args:
        trigger_items: From trigger_scoring.score_triggers() or news pipeline.
            Each item must have 'severity' (int) and 'topic' (str).
        policy: Policy dict. Reads from policy["news_alpha"].
        signal_utc: ISO UTC timestamp for the signal (default: now).

    Returns:
        List of NewsAlphaSignal, one per (event, symbol) pair.
        Empty list if no triggers meet severity threshold.
    """
    cfg = _get_cfg(policy)
    ts = signal_utc or _now_utc()
    min_sev = int(cfg["min_severity"])
    base_w = float(cfg["base_weight"])
    max_single = float(cfg["max_single_weight"])
    leverage_ok = bool(cfg["leverage_etfs_allowed"])
    sl_pct = float(cfg["stop_loss_pct"])
    tp_pct = float(cfg["take_profit_pct"])

    signals: list[NewsAlphaSignal] = []

    for item in trigger_items:
        severity = int(item.get("severity", 0))
        if severity < min_sev:
            continue

        raw_topic = str(item.get("topic", ""))
        source = str(item.get("source", ""))
        seen_symbols: set[str] = (
            set()
        )  # per-event dedup: same symbol can't appear twice in one event

        # Refine central_bank into hike vs cut
        topic_id = (
            split_central_bank_topic(item) if raw_topic == "central_bank" else raw_topic
        )

        route = get_route(topic_id)
        if route is None:
            logger.debug("news_alpha: no route for topic=%s — skipped", topic_id)
            continue

        if severity < route.get("min_severity", 2):
            continue

        size_mult = float(route.get("size_multiplier", 1.0))
        hold_days = int(route.get("hold_days", 5))
        rationale = route.get("rationale", "")
        event_id = str(item.get("event_id", str(uuid.uuid4())))

        raw_w = min(base_w * size_mult * (severity / 2.0), max_single)

        # Long signals
        long_etfs = list(route.get("long_etfs", []))
        if leverage_ok:
            long_etfs += [
                e for e in route.get("long_etfs_2x", []) if e not in long_etfs
            ]

        for sym in long_etfs:
            if sym in seen_symbols:
                continue
            seen_symbols.add(sym)
            is_2x = sym in route.get("long_etfs_2x", [])
            signals.append(
                NewsAlphaSignal(
                    event_id=event_id,
                    topic_id=topic_id,
                    trigger_type=str(route.get("trigger_type", "")),
                    source=source,
                    symbol=sym,
                    direction="long",
                    is_2x=is_2x,
                    raw_weight=raw_w,
                    severity=severity,
                    signal_utc=ts,
                    hold_days=hold_days,
                    stop_loss_pct=sl_pct,
                    take_profit_pct=tp_pct,
                    rationale=rationale,
                    active=True,
                )
            )
            logger.info(
                "news_alpha: LONG %s | topic=%s sev=%d w=%.3f hold=%dd | %s",
                sym,
                topic_id,
                severity,
                raw_w,
                hold_days,
                rationale,
            )

        # Inverse ETF hedges (1x, always allowed) — direction="long" because we BUY the inverse ETF
        for sym in route.get("inverse_etfs", []):
            if sym in seen_symbols:
                continue
            seen_symbols.add(sym)
            signals.append(
                NewsAlphaSignal(
                    event_id=event_id,
                    topic_id=topic_id,
                    trigger_type=str(route.get("trigger_type", "")),
                    source=source,
                    symbol=sym,
                    direction="long",
                    is_2x=False,
                    raw_weight=raw_w * 0.5,
                    severity=severity,
                    signal_utc=ts,
                    hold_days=hold_days,
                    stop_loss_pct=sl_pct,
                    take_profit_pct=tp_pct,
                    rationale=f"HEDGE via inverse ETF: {rationale}",
                    active=True,
                )
            )
            logger.info(
                "news_alpha: HEDGE(inverse) LONG %s | topic=%s sev=%d | %s",
                sym,
                topic_id,
                severity,
                rationale,
            )

        # 2x leveraged inverse ETFs — only if leverage_etfs_allowed (same gate as long_etfs_2x)
        if leverage_ok:
            for sym in route.get("inverse_etfs_2x", []):
                if sym in seen_symbols:
                    continue
                seen_symbols.add(sym)
                signals.append(
                    NewsAlphaSignal(
                        event_id=event_id,
                        topic_id=topic_id,
                        trigger_type=str(route.get("trigger_type", "")),
                        source=source,
                        symbol=sym,
                        direction="long",
                        is_2x=True,
                        raw_weight=raw_w * 0.5,
                        severity=severity,
                        signal_utc=ts,
                        hold_days=hold_days,
                        stop_loss_pct=sl_pct,
                        take_profit_pct=tp_pct,
                        rationale=f"HEDGE via 2x inverse ETF: {rationale}",
                        active=True,
                    )
                )
                logger.info(
                    "news_alpha: HEDGE(2x-inverse) LONG %s | topic=%s sev=%d | %s",
                    sym,
                    topic_id,
                    severity,
                    rationale,
                )

    return signals


def signals_to_weights(
    signals: list[NewsAlphaSignal],
    policy: dict | None = None,
) -> dict[str, float]:
    """Convert signal list to target weight dict, respecting gross cap.

    Longs are positive weights, shorts are negative weights.
    Total |weight| capped at max_gross_exposure.
    """
    cfg = _get_cfg(policy)
    max_gross = float(cfg["max_gross_exposure"])

    weights: dict[str, float] = {}
    for sig in signals:
        if not sig.active:
            continue
        signed = sig.raw_weight if sig.direction == "long" else -sig.raw_weight
        if sig.symbol in weights:
            existing = weights[sig.symbol]
            if (signed > 0) != (existing > 0):
                # Direction conflict across events for same symbol — keep existing, log warning
                logger.warning(
                    "news_alpha: direction conflict for %s (existing=%.3f new=%.3f) — keeping existing",
                    sig.symbol,
                    existing,
                    signed,
                )
                continue
            if abs(signed) > abs(existing):
                weights[sig.symbol] = signed
        else:
            weights[sig.symbol] = signed

    # Scale down if gross exceeds cap
    gross = sum(abs(w) for w in weights.values())
    if gross > max_gross and gross > 0:
        scale = max_gross / gross
        weights = {s: w * scale for s, w in weights.items()}
        logger.debug(
            "news_alpha: gross %.3f > cap %.3f — scaled by %.3f",
            gross,
            max_gross,
            scale,
        )

    return weights
