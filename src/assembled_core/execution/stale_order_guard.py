"""Stale open order guard for restarts and cycle initialization.

On restart, cancels open orders older than a configurable threshold
so that the system doesn't submit duplicate orders on top of stale ones.

Usage:
    from src.assembled_core.execution.stale_order_guard import cancel_stale_orders

    cancel_stale_orders(broker_client, max_age_minutes=5)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_MAX_AGE_MINUTES = 5


def cancel_stale_orders(
    broker_client: Any,
    max_age_minutes: int = DEFAULT_MAX_AGE_MINUTES,
    dry_run: bool = False,
) -> dict:
    """Cancel open orders older than max_age_minutes.

    Args:
        broker_client: Alpaca (or compatible) client with get_orders() / cancel_order().
        max_age_minutes: Orders submitted more than this many minutes ago are stale.
        dry_run: If True, log what would be cancelled but don't call cancel.

    Returns:
        Dict with 'cancelled', 'skipped', 'errors' counts.
    """
    cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=max_age_minutes)
    cancelled = 0
    skipped = 0
    errors = 0

    try:
        open_orders = broker_client.get_orders(status="open")
    except Exception as exc:
        log.error("[stale-orders] Failed to fetch open orders: %s", exc)
        return {"cancelled": 0, "skipped": 0, "errors": 1}

    for order in open_orders:
        submitted_at = _parse_timestamp(getattr(order, "submitted_at", None))
        if submitted_at is None or submitted_at >= cutoff:
            skipped += 1
            continue

        age_min = (datetime.now(tz=timezone.utc) - submitted_at).total_seconds() / 60
        order_id = getattr(order, "id", str(order))
        symbol = getattr(order, "symbol", "?")

        if dry_run:
            log.info(
                "[stale-orders] [DRY-RUN] Would cancel %s (%s, age=%.1f min)",
                order_id,
                symbol,
                age_min,
            )
            cancelled += 1
            continue

        try:
            broker_client.cancel_order(order_id)
            log.info(
                "[stale-orders] Cancelled %s (%s, age=%.1f min)",
                order_id,
                symbol,
                age_min,
            )
            cancelled += 1
        except Exception as exc:
            log.error("[stale-orders] Failed to cancel %s: %s", order_id, exc)
            errors += 1

    log.info(
        "[stale-orders] Done: cancelled=%d skipped=%d errors=%d",
        cancelled,
        skipped,
        errors,
    )
    return {"cancelled": cancelled, "skipped": skipped, "errors": errors}


def _parse_timestamp(ts: Any) -> datetime | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
    try:
        import datetime as dt_mod

        parsed = dt_mod.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    except Exception:
        return None
