"""Intent store for M4 execution workers — idempotency and audit trail.

Records hard operational actions (STOP, KILL, FLATTEN, RECONCILE) as
append-only JSONL entries. Idempotency keys prevent duplicate hard actions
from firing on worker re-runs.

Design:
- Append-only JSONL file (one record per line).
- Each record has: action, idempotency_key, timestamp_utc, metadata.
- Idempotency is keyed on a caller-supplied string (e.g. action::date).
- Single writer assumed per store file (no distributed locking).

Usage:
    from src.assembled_core.execution.intent_store import (
        make_daily_key,
        has_intent,
        record_intent,
        load_intents,
    )

    key = make_daily_key("STOP")
    if not has_intent(key):
        record_intent("STOP", key, metadata={"reason": "manual"})
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Allowed intent actions
IntentAction = Literal[
    "STOP", "KILL", "FLATTEN", "RECONCILE", "ORDER_SUBMIT", "ORDER_COMPLETE"
]

# Default store location (relative to project root; workers may override)
_DEFAULT_STORE_PATH = Path("output") / "ops" / "intent_store.jsonl"


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------


def _sha256_prefix(text: str, length: int = 16) -> str:
    """Return a short hex prefix of the SHA-256 hash of `text`."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def make_daily_key(action: str, date_str: str | None = None) -> str:
    """Return a stable idempotency key: one per action per UTC calendar day.

    Args:
        action: Action name (e.g. "STOP", "KILL").
        date_str: ISO date string "YYYY-MM-DD". Defaults to today UTC.

    Returns:
        16-char hex key that is deterministic for (action, date).

    Example:
        >>> make_daily_key("STOP", "2026-03-30")
        '...'  # stable hex string
    """
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return _sha256_prefix(f"{action}::{date_str}")


def make_run_key(action: str, run_id: str) -> str:
    """Return a stable idempotency key scoped to a specific run_id.

    Useful when multiple reconcile or stop actions may occur on the same day
    but should be recorded separately per run.
    """
    return _sha256_prefix(f"{action}::{run_id}")


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def _store_path_resolved(store_path: Path | str | None) -> Path:
    return Path(store_path) if store_path is not None else _DEFAULT_STORE_PATH


def load_intents(store_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load all intent records from the store.

    Returns an empty list if the store file does not exist.
    Malformed lines are skipped with a warning.

    Args:
        store_path: Path to the JSONL intent store. Uses default if None.

    Returns:
        List of intent record dicts (chronological, as written).
    """
    path = _store_path_resolved(store_path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(
                    "[WARN] intent_store: skipping malformed line %d in %s", i, path
                )
    return records


def has_intent(
    idempotency_key: str,
    store_path: Path | str | None = None,
) -> bool:
    """Return True if a record with this idempotency key exists in the store.

    Args:
        idempotency_key: Key to check.
        store_path: Path to the JSONL intent store. Uses default if None.

    Returns:
        True if found, False otherwise.
    """
    intents = load_intents(store_path)
    return any(r.get("idempotency_key") == idempotency_key for r in intents)


def record_intent(
    action: IntentAction,
    idempotency_key: str,
    *,
    metadata: dict[str, Any] | None = None,
    store_path: Path | str | None = None,
) -> dict[str, Any]:
    """Append a new intent record to the store and return it.

    Does not check for duplicates — call ``has_intent`` first if idempotency
    is needed.

    Args:
        action: One of "STOP", "KILL", "FLATTEN", "RECONCILE".
        idempotency_key: Caller-supplied stable key (see make_daily_key).
        metadata: Optional dict of additional context (reason, paths, etc.).
        store_path: Path to the JSONL intent store. Uses default if None.

    Returns:
        The recorded dict entry (same data that was written).
    """
    path = _store_path_resolved(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc).isoformat()
    record: dict[str, Any] = {
        "action": action,
        "idempotency_key": idempotency_key,
        "timestamp_utc": now_utc,
        "metadata": metadata or {},
    }

    line = json.dumps(record, ensure_ascii=True) + "\n"
    # K1: flush + fsync so a crash between write() and kernel flush cannot lose an
    # ORDER_SUBMIT intent. The extra syscall is a per-record cost; this path is
    # not in a tight loop.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError as exc:
            # Some environments (e.g. certain mocked/virtual filesystems) do not
            # support fsync on text-mode handles — best-effort is acceptable
            # there. But on a real production filesystem an fsync failure
            # silently downgrades the K1 durability contract documented at the
            # top of this module. Surface it at WARNING with the action and
            # idempotency key so a post-mortem can identify the at-risk intent.
            logger.warning(
                "[INTENT] fsync failed for action=%s key=%s path=%s: %s",
                action, idempotency_key, path, exc,
            )

    logger.info(
        "[INTENT] recorded action=%s key=%s store=%s", action, idempotency_key, path
    )
    return record


def filter_intents_by_action(
    action: IntentAction,
    store_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return all records for a specific action type, chronologically.

    Args:
        action: Action type to filter on.
        store_path: Path to the JSONL intent store. Uses default if None.

    Returns:
        Filtered list of intent record dicts.
    """
    return [r for r in load_intents(store_path) if r.get("action") == action]


# ---------------------------------------------------------------------------
# Broker order intent helpers (for crash recovery)
# ---------------------------------------------------------------------------


def make_order_key(
    symbol: str,
    side: str,
    qty: float,
    nonce: str | None = None,
) -> str:
    """Create a unique key for a broker order intent.

    Each call generates a unique key by default (using UTC timestamp with
    microsecond precision as nonce). This prevents key collision when the
    same symbol+side+qty is submitted multiple times in one day.

    Args:
        symbol: Ticker symbol.
        side: "buy" or "sell".
        qty: Order quantity.
        nonce: Explicit nonce (default: current UTC timestamp ISO format).

    Returns:
        16-char hex idempotency key.
    """
    if nonce is None:
        # Same-microsecond submits (e.g. parallel workers, fast retries) used
        # to collide on the ISO timestamp alone and produce duplicate
        # idempotency keys — defeating the whole point. Append random bytes
        # so collision probability is negligible regardless of clock
        # resolution.
        nonce = f"{datetime.now(timezone.utc).isoformat()}::{os.urandom(8).hex()}"
    return _sha256_prefix(f"ORDER::{symbol}::{side}::{qty}::{nonce}")


def record_order_submit(
    symbol: str,
    side: str,
    qty: float,
    broker_order_id: str = "",
    *,
    nonce: str | None = None,
    store_path: Path | str | None = None,
) -> dict[str, Any]:
    """Record an order submission intent (before or after API call).

    Used for crash recovery: on restart, find ORDER_SUBMIT without
    matching ORDER_COMPLETE and reconcile against broker.

    The nonce ensures each submit gets a unique key even if the same
    symbol+side+qty is submitted multiple times in one day.
    The returned dict contains 'idempotency_key' which MUST be passed
    to record_order_complete for correct pairing.
    """
    key = make_order_key(symbol, side, qty, nonce=nonce)
    return record_intent(
        "ORDER_SUBMIT",
        key,
        metadata={
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "broker_order_id": broker_order_id,
        },
        store_path=store_path,
    )


def record_order_complete(
    symbol: str,
    side: str,
    qty: float,
    filled_qty: float = 0.0,
    filled_price: float | None = None,
    status: str = "filled",
    *,
    intent_key: str | None = None,
    store_path: Path | str | None = None,
) -> dict[str, Any]:
    """Record an order completion (fill, cancel, reject).

    Args:
        intent_key: The idempotency_key from the matching ORDER_SUBMIT record.
            If provided, ensures correct pairing with the submit record.
            If None, generates a new key (legacy behavior, not recommended).
    """
    if intent_key is not None:
        key = intent_key
    else:
        key = make_order_key(symbol, side, qty)
    return record_intent(
        "ORDER_COMPLETE",
        key,
        metadata={
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "filled_qty": filled_qty,
            "filled_price": filled_price,
            "status": status,
        },
        store_path=store_path,
    )


def find_pending_order_intents(
    store_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Find ORDER_SUBMIT intents without matching ORDER_COMPLETE.

    These represent orders that may have been sent to the broker but
    whose fills were never recorded — typically due to a crash.

    Returns:
        List of ORDER_SUBMIT records without matching completions.
    """
    intents = load_intents(store_path)
    submitted_keys: dict[str, dict[str, Any]] = {}
    completed_keys: set[str] = set()

    for r in intents:
        key = r.get("idempotency_key", "")
        action = r.get("action", "")
        if action == "ORDER_SUBMIT":
            submitted_keys[key] = r
        elif action == "ORDER_COMPLETE":
            completed_keys.add(key)

    return [r for k, r in submitted_keys.items() if k not in completed_keys]
