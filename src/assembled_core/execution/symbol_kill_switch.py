"""Per-symbol kill switch (Sprint 4 / Plan C27).

Sidecar helper that blocks trading for a specific set of symbols while the
global kill switch remains untouched. Typical use cases:

  * Symbol is delisted, halted, or under a corporate-action freeze
  * Data-quality incident isolated to a single ticker
  * Risk committee manually disables a name while the rest of the portfolio
    keeps trading

This module deliberately keeps its own JSON state file and does NOT mutate
``execution/kill_switch.py``. Callers opt in by running orders through
:func:`filter_orders_by_symbol_blocks` (or the policy wrapper below).

State file layout::

    {
      "blocked": {
        "AAPL": {"reason": "earnings blackout", "blocked_at": "2026-..."},
        "XYZ":  {"reason": "delisting",         "blocked_at": "2026-..."}
      }
    }
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_STATE_PATH = Path("output") / "state" / "symbol_kill_switch.json"

# Per-file locks to prevent concurrent read-modify-write corruption.
# Keyed by resolved path so independent state files get independent locks.
_FILE_LOCKS: dict[str, threading.Lock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


def _get_lock(path: Path) -> threading.Lock:
    """Return a per-path threading lock (created on first access)."""
    key = str(path.resolve())
    with _FILE_LOCKS_GUARD:
        if key not in _FILE_LOCKS:
            _FILE_LOCKS[key] = threading.Lock()
        return _FILE_LOCKS[key]


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"blocked": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - tolerate corrupt state, start clean
        logger.warning("[symbol_kill_switch] could not read %s: %s", path, exc)
        return {"blocked": {}}
    if not isinstance(raw, dict) or "blocked" not in raw:
        return {"blocked": {}}
    if not isinstance(raw["blocked"], dict):
        raw["blocked"] = {}
    return raw


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def block_symbol(
    symbol: str,
    reason: str,
    *,
    state_path: str | Path | None = None,
) -> dict[str, Any]:
    """Block ``symbol`` with a free-form reason. Idempotent.

    Thread-safe: uses a per-file lock so concurrent block/unblock calls on
    the same state file do not corrupt each other's writes.
    """
    path = Path(state_path or _DEFAULT_STATE_PATH)
    lock = _get_lock(path)
    with lock:
        state = _read_state(path)
        state["blocked"][symbol] = {
            "reason": reason,
            "blocked_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_state(path, state)
    logger.info("[symbol_kill_switch] BLOCK %s: %s", symbol, reason)
    return state["blocked"][symbol]


def unblock_symbol(
    symbol: str,
    *,
    state_path: str | Path | None = None,
) -> bool:
    """Remove ``symbol`` from the block list. Returns True if removed.

    Thread-safe: uses a per-file lock.
    """
    path = Path(state_path or _DEFAULT_STATE_PATH)
    lock = _get_lock(path)
    with lock:
        state = _read_state(path)
        if symbol in state["blocked"]:
            del state["blocked"][symbol]
            _write_state(path, state)
            logger.info("[symbol_kill_switch] UNBLOCK %s", symbol)
            return True
    return False


def list_blocked_symbols(
    *,
    state_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the current ``{symbol: {reason, blocked_at}}`` mapping."""
    path = Path(state_path or _DEFAULT_STATE_PATH)
    state = _read_state(path)
    return dict(state.get("blocked", {}))


def is_symbol_blocked(
    symbol: str,
    *,
    state_path: str | Path | None = None,
) -> bool:
    """True if ``symbol`` is currently blocked."""
    return symbol in list_blocked_symbols(state_path=state_path)


def filter_orders_by_symbol_blocks(
    orders: pd.DataFrame,
    *,
    state_path: str | Path | None = None,
    symbol_col: str = "symbol",
) -> tuple[pd.DataFrame, list[str]]:
    """Drop rows whose ``symbol`` is in the current block list.

    Returns ``(filtered_orders, reasons)``. Never mutates input.
    """
    if orders is None or orders.empty or symbol_col not in orders.columns:
        return (orders.copy() if orders is not None else pd.DataFrame()), []

    blocked = list_blocked_symbols(state_path=state_path)
    if not blocked:
        return orders.copy(), []

    mask_blocked = orders[symbol_col].astype(str).isin(blocked.keys())
    if not bool(mask_blocked.any()):
        return orders.copy(), []

    reasons: list[str] = []
    for _, row in orders.loc[mask_blocked].iterrows():
        sym = str(row[symbol_col])
        info = blocked.get(sym, {})
        reasons.append(
            f"symbol_kill_switch: {sym} rejected — reason="
            f"{info.get('reason', 'unknown')}"
        )

    filtered = orders.loc[~mask_blocked].copy()
    return filtered, reasons


def filter_orders_from_policy(
    orders: pd.DataFrame,
    policy: dict[str, Any],
    *,
    state_path: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Read ``policy['symbol_kill_switch']`` and apply. Disabled is a no-op."""
    cfg = (policy or {}).get("symbol_kill_switch") or {}
    if not cfg.get("enabled", False):
        return (orders.copy() if orders is not None else pd.DataFrame()), []
    return filter_orders_by_symbol_blocks(orders, state_path=state_path)


__all__ = [
    "block_symbol",
    "filter_orders_by_symbol_blocks",
    "filter_orders_from_policy",
    "is_symbol_blocked",
    "list_blocked_symbols",
    "unblock_symbol",
]
