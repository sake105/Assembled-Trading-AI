"""Crisis-Alpha exit rules and deactivation triggers — M5.

Exit rules determine when individual crisis positions should be closed.
Deactivation triggers determine when the sub-portfolio should flatten
entirely (transition to COOLDOWN or PAUSE).

Exit rule types (v1):
    time_stop:    Close position if held longer than max_hold_hours.
    break_even:   Move stop to entry price once position gains break_even_pct.
    no_overnight: Flag positions that must be closed before market close.
                  (Caller is responsible for acting on this flag — this module
                  only detects and flags; it does not submit orders.)

Deactivation triggers (checked against current state and context):
    state_not_active: State machine is not ACTIVE → flatten all.
    daily_loss:       Daily loss limit breached → flatten all (PAUSE state).
    health_error:     Health is ERROR → flatten all → COOLDOWN.

All functions are pure (no I/O) for testability.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.assembled_core.events.crisis_alpha.context import CrisisAlphaContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------


def _get(d: dict, *keys, default=None):
    node = d
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
        if node is None:
            return default
    return node


# ---------------------------------------------------------------------------
# Individual position exit checks
# ---------------------------------------------------------------------------


def check_time_stop(
    position: dict[str, Any],
    now_utc: datetime,
    max_hold_hours: float = 8.0,
) -> tuple[bool, str]:
    """Return (should_exit, reason) based on maximum hold time.

    Args:
        position: Dict with at least ``entry_ts`` (ISO string or datetime).
        now_utc: Current UTC datetime.
        max_hold_hours: Maximum allowed hold duration in hours.

    Returns:
        (True, reason) if position should be closed, (False, "OK") otherwise.
    """
    entry_ts_raw = position.get("entry_ts")
    if entry_ts_raw is None:
        return True, "time_stop: no entry_ts recorded — close for safety"

    try:
        if isinstance(entry_ts_raw, datetime):
            entry_ts = entry_ts_raw
        else:
            entry_ts = datetime.fromisoformat(str(entry_ts_raw))
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.replace(tzinfo=timezone.utc)
    except ValueError:
        return (
            True,
            f"time_stop: could not parse entry_ts='{entry_ts_raw}' — close for safety",
        )

    hold_hours = (now_utc - entry_ts).total_seconds() / 3600
    if hold_hours >= max_hold_hours:
        return True, f"time_stop: held {hold_hours:.1f}h >= max {max_hold_hours:.1f}h"

    return False, f"time_stop: OK (held {hold_hours:.1f}h)"


def check_break_even(
    position: dict[str, Any],
    current_price: float,
    break_even_pct: float = 0.005,
) -> tuple[bool, str]:
    """Return (at_break_even, reason). True if position has gained >= break_even_pct.

    When True, caller should set a stop at entry price (or close if already below).

    Args:
        position: Dict with ``entry_price``, ``side`` ("long" / "short").
        current_price: Current market price.
        break_even_pct: Fractional gain threshold to trigger break-even (default 0.5%).

    Returns:
        (True, reason) if break-even level has been reached.
    """
    entry_price = float(position.get("entry_price", 0.0))
    side = str(position.get("side", "long")).lower()

    if entry_price <= 0:
        return False, "break_even: no valid entry_price"

    if side == "long":
        gain_pct = (current_price - entry_price) / entry_price
    else:
        gain_pct = (entry_price - current_price) / entry_price

    if gain_pct >= break_even_pct:
        return (
            True,
            f"break_even: gain={gain_pct:.4f} >= {break_even_pct:.4f} — stop at entry",
        )
    return False, f"break_even: gain={gain_pct:.4f} < {break_even_pct:.4f}"


def check_no_overnight(
    position: dict[str, Any],
    now_utc: datetime,
    market_close_hour_utc: int = 21,  # 21:00 UTC ≈ 17:00 ET (approx)
) -> tuple[bool, str]:
    """Return (must_close_today, reason). True if position must not be held overnight.

    Flags positions to be closed before market close.  Caller must act on this.

    Args:
        position: Position dict (must_close_today flag or no-overnight tag).
        now_utc: Current UTC datetime.
        market_close_hour_utc: Hour (UTC) at which market closes (approximate).

    Returns:
        (True, reason) if within the closing window, (False, "OK") otherwise.
    """
    if now_utc.hour >= market_close_hour_utc:
        return (
            True,
            f"no_overnight: current hour {now_utc.hour}h UTC >= close {market_close_hour_utc}h UTC — must close",
        )
    return False, "no_overnight: OK (before market close)"


# ---------------------------------------------------------------------------
# Portfolio-level deactivation check
# ---------------------------------------------------------------------------


def check_deactivation_triggers(
    ctx: CrisisAlphaContext,
    current_crisis_state: str,
) -> tuple[bool, str]:
    """Return (should_flatten_all, reason).

    Returns True if ALL crisis positions should be closed immediately.
    This is separate from the state machine — it is a pre-trade check.

    Triggers:
    - State machine is not ACTIVE → flatten all (already deactivated)
    - Daily loss limit breached → flatten all
    - Health ERROR → flatten all

    Args:
        ctx: Current CrisisAlphaContext.
        current_crisis_state: Current state string (e.g. "WATCH", "ACTIVE").

    Returns:
        (True, reason) if all positions should be closed, (False, "OK") otherwise.
    """
    if current_crisis_state != "ACTIVE":
        return (
            True,
            f"deactivation: state={current_crisis_state} (not ACTIVE) — flatten all",
        )

    if ctx.daily_loss_breached():
        return (
            True,
            f"deactivation: daily loss breached pnl={ctx.daily_pnl:.4f} — flatten all",
        )

    if not ctx.health_ok:
        return True, "deactivation: health ERROR — flatten all"

    return False, "deactivation: OK (state=ACTIVE, health=OK, daily loss within limit)"


# ---------------------------------------------------------------------------
# Batch exit check: apply exit rules to all open positions
# ---------------------------------------------------------------------------


def get_positions_to_exit(
    open_positions: list[dict[str, Any]],
    now_utc: datetime,
    policy: dict | None = None,
    prices: dict[str, float] | None = None,
) -> list[tuple[dict[str, Any], str]]:
    """Return list of (position, reason) for positions that should be exited.

    Applies time_stop, break_even, and no_overnight checks to each position.

    Args:
        open_positions: List of position dicts from CrisisAlphaContext.
        now_utc: Current UTC datetime.
        policy: Policy dict (reads crisis_alpha.exit.*).
        prices: Optional dict of {symbol: current_price} for break-even check.

    Returns:
        List of (position, exit_reason) for positions to be closed.
    """
    policy = policy or {}
    prices = prices or {}
    cfg = _get(policy, "crisis_alpha", "exit", default={})

    max_hold_hours: float = float(_get(cfg, "max_hold_hours", default=8.0))
    break_even_pct: float = float(_get(cfg, "break_even_pct", default=0.005))
    market_close_hour: int = int(_get(cfg, "market_close_hour_utc", default=21))
    no_overnight_enabled: bool = bool(_get(cfg, "no_overnight", default=True))

    to_exit: list[tuple[dict[str, Any], str]] = []

    for pos in open_positions:
        symbol = pos.get("symbol", "?")

        # Time stop
        ts_exit, ts_reason = check_time_stop(pos, now_utc, max_hold_hours)
        if ts_exit:
            to_exit.append((pos, ts_reason))
            logger.info("[CRISIS_EXIT] %s: %s", symbol, ts_reason)
            continue

        # Break-even (only if price available)
        current_price = prices.get(symbol)
        if current_price is not None:
            be_exit, be_reason = check_break_even(pos, current_price, break_even_pct)
            if be_exit:
                to_exit.append((pos, be_reason))
                logger.info("[CRISIS_EXIT] %s: %s", symbol, be_reason)
                continue

        # No-overnight
        if no_overnight_enabled:
            no_ov_exit, no_ov_reason = check_no_overnight(
                pos, now_utc, market_close_hour
            )
            if no_ov_exit:
                to_exit.append((pos, no_ov_reason))
                logger.info("[CRISIS_EXIT] %s: %s", symbol, no_ov_reason)

    return to_exit
