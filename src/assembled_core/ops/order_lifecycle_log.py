"""Order Lifecycle Log — per-event append-only JSONL audit trail.

One entry per state transition (SUBMITTED / ROUTED / PARTIAL_FILL / FILLED /
REJECTED / CANCELLED).  Complements the fill-level trade_journal.jsonl with
event-level granularity required by GO_LIVE_CHECKLIST C1.

Schema per entry (JSONL):
    order_id        str          unique order identifier
    timestamp_utc   str          ISO-8601 UTC
    event_type      str          SUBMITTED | ROUTED | PARTIAL_FILL |
                                 FILLED | REJECTED | CANCELLED
    symbol          str
    side            str          BUY | SELL
    qty             float
    price           float|null   fill price when available
    reason          str|null     rejection / cancellation reason
    strategy        str
    actor           str          who triggered the transition
    run_id          str          run / date identifier
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_LIFECYCLE_LOG_PATH = Path("output/journal/order_lifecycle.jsonl")

TERMINAL_EVENTS: frozenset[str] = frozenset({"FILLED", "REJECTED", "CANCELLED"})


def append_lifecycle_event(
    event_type: str,
    order_id: str,
    symbol: str,
    side: str,
    qty: float,
    *,
    price: float | None = None,
    reason: str | None = None,
    strategy: str = "",
    actor: str = "pipeline",
    run_id: str = "",
    log_path: Path | str | None = None,
) -> None:
    """Append a single lifecycle event to the order lifecycle log.

    Non-blocking: exceptions are logged at WARNING and swallowed so that a
    disk failure never interrupts the trading path.

    Args:
        event_type: One of SUBMITTED / ROUTED / PARTIAL_FILL / FILLED /
                    REJECTED / CANCELLED.
        order_id: Unique order identifier.
        symbol: Ticker symbol.
        side: BUY or SELL.
        qty: Order quantity (shares).
        price: Fill price (set for FILLED / PARTIAL_FILL; None otherwise).
        reason: Human-readable reason for REJECTED / CANCELLED events.
        strategy: Strategy name that generated the order.
        actor: Component that triggered this transition.
        run_id: Run / date identifier (e.g. "2026-05-28").
        log_path: Override the default log path.
    """
    jpath = Path(log_path) if log_path else DEFAULT_LIFECYCLE_LOG_PATH
    try:
        jpath.parent.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {
            "order_id": order_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "symbol": symbol,
            "side": str(side).upper(),
            "qty": float(qty),
            "price": float(price) if price is not None else None,
            "reason": reason,
            "strategy": strategy,
            "actor": actor,
            "run_id": run_id,
        }
        with open(jpath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as exc:
        logger.warning(
            "[OrderLifecycleLog] failed to write %s event: %s", event_type, exc
        )


def find_open_orders(log_path: Path | str | None = None) -> list[str]:
    """Return order_ids that have no terminal event in the lifecycle log.

    An order is considered "open" if it has at least one entry but none of
    its entries have event_type in TERMINAL_EVENTS (FILLED / REJECTED /
    CANCELLED).

    Args:
        log_path: Path to the lifecycle log JSONL file.

    Returns:
        Sorted list of order_ids without a terminal event.
        Returns [] if the log does not exist or cannot be read.
    """
    jpath = Path(log_path) if log_path else DEFAULT_LIFECYCLE_LOG_PATH
    if not jpath.exists():
        return []

    seen: dict[str, bool] = {}  # order_id -> has_terminal
    try:
        for line in jpath.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            oid = entry.get("order_id", "")
            if not oid:
                continue
            if oid not in seen:
                seen[oid] = False
            if entry.get("event_type", "") in TERMINAL_EVENTS:
                seen[oid] = True
    except Exception as exc:
        logger.warning("[OrderLifecycleLog] find_open_orders read error: %s", exc)
        return []

    return sorted(oid for oid, terminal in seen.items() if not terminal)
