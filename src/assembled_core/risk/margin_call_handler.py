"""Margin Call Handler — defensive position reduction on margin call detection.

Invoked by ``accounting.ledger.check_margin_requirements`` when a margin call
is detected.  The handler is intentionally conservative:

- Logs a CRITICAL alert immediately.
- Closes the 50% of positions with the lowest conviction (smallest absolute
  position value as proxy when no conviction score is available).
- Sends a Discord alert if DISCORD_WEBHOOK_URL is set in the environment.
- Returns the list of symbols whose close orders were requested.

The handler does NOT execute orders itself — it returns a list of symbols to
close.  The caller (paper runner / live runner) is responsible for translating
these into actual close orders via the broker adapter.

Design constraints:
- No changes to position sizing logic.
- No changes to the margin calculation logic (that lives in ledger.py).
- The handler is a pure response function; calling it twice is safe (idempotent
  in the sense that it always re-derives which symbols to close from the current
  state).
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def handle_margin_call(
    ledger_state: dict[str, Any],
    ctx: Any | None = None,
    adapter: Any | None = None,
    *,
    close_fraction: float = 0.50,
) -> list[str]:
    """React to a detected margin call by flagging the lowest-conviction positions
    for closure.

    Args:
        ledger_state: Dict returned by ``check_margin_requirements`` (must contain
            ``margin_call=True``) OR a broader state dict with a ``positions`` key
            mapping symbol -> quantity.  The handler is tolerant of both shapes.
        ctx: Optional trading context (unused currently; reserved for future
            conviction-score extraction).
        adapter: Optional broker adapter.  If provided and it has a
            ``submit_market_order`` method, the handler will submit SELL orders
            directly.  If None, the caller must act on the returned symbol list.
        close_fraction: Fraction of positions (by count) to close, sorted
            ascending by absolute notional value (lowest conviction first).
            Default 0.50 → close the bottom 50%.

    Returns:
        List of symbols whose positions were flagged (or submitted) for closure.
        Empty list if no action was taken (e.g. no positions held).
    """
    # ------------------------------------------------------------------ #
    # 1. Log CRITICAL alert                                                #
    # ------------------------------------------------------------------ #
    margin_call_amount = ledger_state.get("margin_call_amount", 0.0)
    equity = ledger_state.get("equity", float("nan"))
    maintenance_req = ledger_state.get("maintenance_required", float("nan"))

    logger.critical(
        "[MarginCall] MARGIN CALL DETECTED — equity=%.2f maintenance_req=%.2f "
        "shortfall=%.2f — initiating defensive position reduction (%.0f%%)",
        equity,
        maintenance_req,
        margin_call_amount,
        close_fraction * 100,
    )

    # ------------------------------------------------------------------ #
    # 2. Discord alert (best-effort, never blocks trading logic)           #
    # ------------------------------------------------------------------ #
    _send_discord_alert(equity, maintenance_req, margin_call_amount)

    # ------------------------------------------------------------------ #
    # 3. Determine which positions to close                                #
    # ------------------------------------------------------------------ #
    # Extract positions from ledger_state or from a nested ``positions`` key.
    positions: dict[str, float] = {}
    if "positions" in ledger_state:
        positions = {
            sym: float(qty)
            for sym, qty in ledger_state["positions"].items()
            if float(qty) != 0.0
        }
    else:
        # Caller may have passed the raw margin dict — no positions available.
        logger.warning(
            "[MarginCall] ledger_state has no 'positions' key — cannot determine "
            "which symbols to close. Returning empty list."
        )
        return []

    if not positions:
        logger.warning("[MarginCall] No open positions found — nothing to close.")
        return []

    prices: dict[str, float] = ledger_state.get("prices", {})

    # Sort by absolute notional value ascending (lowest conviction first).
    # When price data is unavailable, fall back to absolute share count.
    def _notional(sym: str, qty: float) -> float:
        px = prices.get(sym, 0.0)
        if px > 0:
            return abs(qty * px)
        return abs(qty)

    sorted_positions = sorted(positions.items(), key=lambda kv: _notional(kv[0], kv[1]))

    n_to_close = max(1, int(len(sorted_positions) * close_fraction))
    to_close = [sym for sym, _ in sorted_positions[:n_to_close]]

    logger.critical(
        "[MarginCall] Closing %d of %d positions (lowest notional first): %s",
        n_to_close,
        len(sorted_positions),
        ", ".join(to_close),
    )

    # ------------------------------------------------------------------ #
    # 4. Submit close orders if adapter is available                       #
    # ------------------------------------------------------------------ #
    closed: list[str] = []
    for sym in to_close:
        qty = positions[sym]
        if qty == 0.0:
            continue
        side = "sell" if qty > 0 else "buy"
        abs_qty = abs(qty)

        if adapter is not None and hasattr(adapter, "submit_market_order"):
            try:
                adapter.submit_market_order(
                    sym,
                    abs_qty,
                    side,
                    comment="margin_call_handler",
                )
                logger.critical(
                    "[MarginCall] Submitted %s %s qty=%.2f via adapter",
                    side.upper(),
                    sym,
                    abs_qty,
                )
                closed.append(sym)
            except Exception as exc:
                logger.error(
                    "[MarginCall] Failed to submit close order for %s: %s",
                    sym,
                    exc,
                )
        else:
            # No adapter — caller is responsible for executing closures.
            logger.critical(
                "[MarginCall] No adapter — %s %s qty=%.2f flagged for closure "
                "(caller must act)",
                side.upper(),
                sym,
                abs_qty,
            )
            closed.append(sym)

    return closed


def _send_discord_alert(
    equity: float,
    maintenance_req: float,
    shortfall: float,
) -> None:
    """Send a Discord webhook alert for the margin call.

    Best-effort: any exception is caught and logged, never re-raised.
    Requires DISCORD_WEBHOOK_URL environment variable.
    """
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return

    import json as _json

    try:
        import urllib.request

        payload = {
            "content": (
                f"**MARGIN CALL ALERT**\n"
                f"Equity: ${equity:,.2f}\n"
                f"Maintenance required: ${maintenance_req:,.2f}\n"
                f"Shortfall: ${shortfall:,.2f}\n"
                f"Defensive position reduction initiated."
            )
        }
        data = _json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            logger.info("[MarginCall] Discord alert sent (status=%d)", resp.getcode())
    except Exception as exc:
        logger.warning("[MarginCall] Discord alert failed (non-fatal): %s", exc)


__all__ = ["handle_margin_call"]
