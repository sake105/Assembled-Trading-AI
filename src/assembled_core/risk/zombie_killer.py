"""Time stop / zombie killer: flag positions held too long with insufficient gain.

A "zombie" position is one that satisfies both:
  1. Held longer than ``max_hold_days``.
  2. Has not achieved the minimum gain threshold (``min_gain_pct``).

These positions are flagged for exit; the caller decides the execution path.
This module is stateless — no side effects, deterministic per call.

M6-T05: implement time stop / zombie killer rules.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _parse_utc(ts_str: str) -> datetime | None:
    """Parse ISO UTC string to aware datetime. Returns None on failure."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _hold_hours(entry_ts: str, now_utc: datetime) -> float:
    """Hours elapsed since entry_ts. Returns -1.0 if entry_ts is unparseable."""
    entry_dt = _parse_utc(entry_ts)
    if entry_dt is None:
        return -1.0
    delta = now_utc - entry_dt
    return delta.total_seconds() / 3600.0


def _position_return(position: dict[str, Any]) -> float | None:
    """Compute position return from entry_price and current_price.

    Returns None if either price is absent or entry_price <= 0.
    Handles long and short sides.
    """
    entry_price = position.get("entry_price")
    current_price = position.get("current_price")
    if entry_price is None or current_price is None:
        return None
    try:
        ep = float(entry_price)
        cp = float(current_price)
    except (TypeError, ValueError):
        return None
    if ep <= 0:
        return None

    side = str(position.get("side", "long")).lower()
    if side in ("short", "sell"):
        return (ep / cp) - 1.0 if cp > 0 else None
    # long / buy / unknown: standard long return
    return (cp / ep) - 1.0


def check_zombie_position(
    position: dict[str, Any],
    now_utc: datetime,
    max_hold_days: float = 5.0,
    min_gain_pct: float = 0.005,
) -> tuple[bool, str]:
    """Check if a single position is a zombie (held too long, insufficient gain).

    Args:
        position: Dict with at least ``entry_ts`` (ISO UTC string), ``symbol``.
                  Optionally: ``entry_price``, ``current_price``, ``side``.
        now_utc: Current UTC datetime (must be timezone-aware).
        max_hold_days: Days held before the time-stop check kicks in (default 5).
        min_gain_pct: Minimum gain to avoid the zombie flag (default 0.5%).

    Returns:
        (is_zombie, reason). reason is empty string if not a zombie.
    """
    symbol = position.get("symbol", "UNKNOWN")
    entry_ts = str(position.get("entry_ts", ""))

    hours_held = _hold_hours(entry_ts, now_utc)
    if hours_held < 0:
        # Unparseable entry_ts — safe default: do not flag
        return False, ""

    max_hold_hours = max_hold_days * 24.0
    if hours_held < max_hold_hours:
        return False, ""

    # Past hold limit — check gain
    ret = _position_return(position)
    if ret is None:
        # No price data; flag conservatively when hold limit is exceeded
        reason = (
            f"zombie_killer: {symbol} held {hours_held:.1f}h "
            f"(limit={max_hold_hours:.1f}h), no price data for gain check"
        )
        return True, reason

    if ret < min_gain_pct:
        reason = (
            f"zombie_killer: {symbol} held {hours_held:.1f}h "
            f"(limit={max_hold_hours:.1f}h), gain={ret:.3%} < min={min_gain_pct:.3%}"
        )
        return True, reason

    return False, ""


def get_zombie_positions(
    positions: list[dict[str, Any]],
    now_utc: datetime,
    policy: dict[str, Any],
) -> list[tuple[dict[str, Any], str]]:
    """Scan all open positions and return (position, reason) for zombies.

    Args:
        positions: List of open position dicts.
        now_utc: Current UTC datetime.
        policy: Policy dict. Reads from ``zombie_killer`` section:
            - enabled (bool, default True)
            - max_hold_days (float, default 5.0)
            - min_gain_pct (float, default 0.005)

    Returns:
        List of (position, reason) tuples for flagged zombies.
        Empty list if disabled or no zombies found.
    """
    zk = (policy or {}).get("zombie_killer") or {}
    if not zk.get("enabled", False):
        return []

    max_hold_days = float(zk.get("max_hold_days", 5.0) or 5.0)
    min_gain_pct = float(zk.get("min_gain_pct", 0.005) or 0.005)

    result: list[tuple[dict[str, Any], str]] = []
    for pos in positions or []:
        is_zombie, reason = check_zombie_position(
            pos,
            now_utc,
            max_hold_days=max_hold_days,
            min_gain_pct=min_gain_pct,
        )
        if is_zombie:
            result.append((pos, reason))
    return result


__all__ = ["check_zombie_position", "get_zombie_positions"]
