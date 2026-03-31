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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Allowed intent actions
IntentAction = Literal["STOP", "KILL", "FLATTEN", "RECONCILE"]

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
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)

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
