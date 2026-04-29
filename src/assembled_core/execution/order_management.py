"""Order management helpers.

From 33_EXECUTION_ORDERMANAGEMENT.md §33.2–§33.12.

Covers: idempotent submit, partial-fill policy, position reconciliation,
and the execution cost model for backtest-vs-live parity.
Async broker helpers (ExitManager.check_exits, eod_reconciliation) are
stubs that require a live broker client to function.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

import numpy as np

from assembled_core.execution.idempotency import (
    build_client_order_id,
    is_duplicate_error,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Idempotent submit wrapper
# ---------------------------------------------------------------------------

def submit_with_idempotency(client: Any, intent: Any) -> Tuple[str, Any, Optional[str]]:
    """Submit an order via alpaca client, handling duplicate client_order_id gracefully.

    Args:
        client: Alpaca broker client with ``submit_order`` and
                ``get_order_by_client_order_id`` methods.
        intent: Order intent with attributes: signal_id, intent_hash, symbol,
                qty, side, order_type, tif, limit_price.

    Returns:
        (status, order_or_None, error_or_None) where status is one of:
        'submitted', 'already_submitted', 'rejected', 'error'.
    """
    coid = build_client_order_id(intent.signal_id, intent.intent_hash)
    try:
        resp = client.submit_order(
            symbol=intent.symbol,
            qty=intent.qty,
            side=intent.side,
            type=intent.order_type,
            time_in_force=intent.tif,
            client_order_id=coid,
            limit_price=getattr(intent, "limit_price", None),
        )
        return ("submitted", resp, None)
    except Exception as e:
        msg = str(e).lower()
        if is_duplicate_error(msg):
            try:
                existing = client.get_order_by_client_order_id(coid)
                return ("already_submitted", existing, None)
            except Exception:
                return ("error", None, f"duplicate but fetch failed: {e}")
        elif "insufficient" in msg:
            return ("rejected", None, "insufficient_buying_power")
        elif "wash" in msg:
            return ("rejected", None, "wash_sale_block")
        elif "pattern day trading" in msg or "403" in msg:
            return ("rejected", None, "pdt_protection")
        else:
            logger.error("submit_order failed for %s: %s", getattr(intent, "symbol", "?"), e)
            return ("error", None, str(e))


# ---------------------------------------------------------------------------
# Partial-fill policy
# ---------------------------------------------------------------------------

class PartialFillPolicy:
    """Policy constants for handling partial fills.

    After CANCEL_AFTER_SECONDS, if fill_ratio < MIN_FILL_RATIO the order
    is cancelled. ON_CANCEL='keep' retains the partial position; 'liquidate'
    immediately closes it at market.
    """

    CANCEL_AFTER_SECONDS: int = 120
    MIN_FILL_RATIO: float = 0.5
    ON_CANCEL: str = "keep"  # or "liquidate"

    @classmethod
    def classify(
        cls,
        submitted_at: datetime,
        qty: float,
        filled_qty: float,
        now: datetime | None = None,
    ) -> str:
        """Classify the current fill state.

        Returns: 'complete' | 'wait' | 'partial_accepted' | 'partial_failed'
        """
        if now is None:
            now = datetime.now(timezone.utc)
        fill_ratio = filled_qty / qty if qty else 0.0
        if fill_ratio >= 1.0:
            return "complete"
        elapsed = (now - submitted_at).total_seconds()
        if elapsed < cls.CANCEL_AFTER_SECONDS:
            return "wait"
        if fill_ratio >= cls.MIN_FILL_RATIO:
            return "partial_accepted"
        return "partial_failed"


# ---------------------------------------------------------------------------
# Position reconciliation
# ---------------------------------------------------------------------------

MIN_TRADE_SIZE: float = 1.0  # minimum notional delta to generate an order


def position_reconcile_before_signal(
    symbol: str,
    target_size: float,
    current_size: float,
    min_trade_size: float = MIN_TRADE_SIZE,
    min_rebalance_pct: float = 0.10,
) -> Optional[dict]:
    """Compute delta order needed to align current position to target.

    Returns None if the delta is too small to warrant a trade.
    Returns a dict with 'symbol' and 'delta' otherwise.
    """
    delta = target_size - current_size
    if abs(delta) < min_trade_size:
        return None
    if target_size != 0 and abs(delta / target_size) < min_rebalance_pct:
        return None
    return {"symbol": symbol, "delta": delta}


# ---------------------------------------------------------------------------
# Execution cost model (backtest use)
# ---------------------------------------------------------------------------

@dataclass
class BarSnapshot:
    """Minimal bar snapshot for cost estimation."""

    close: float
    realized_vol_20d: float = 0.02
    adv: float = 50_000_000.0  # average daily volume in USD


class ExecutionCostModel:
    """Conservative fill-price estimator for backtesting.

    Estimates spread + slippage + regulatory fees so backtest fills are
    realistically worse than mid-price.
    """

    def estimate_fill(
        self,
        side: str,
        symbol: str,
        qty: float,
        bar: BarSnapshot,
    ) -> float:
        """Return estimated fill price relative to bar.close.

        Args:
            side: 'buy' or 'sell'.
            symbol: Ticker (unused currently; reserved for per-ticker lookup).
            qty: Number of shares.
            bar: Price/volume/vol snapshot for the bar.

        Returns:
            Estimated fill price in dollars.
        """
        spread_bps = self._get_spread(symbol, bar)
        participation_pct = (qty * bar.close) / max(bar.adv, 1.0)
        slippage_bps = 10.0 * np.sqrt(participation_pct)
        fees_bps = 0.2 if side == "sell" else 0.0

        total_bps = spread_bps / 2 + slippage_bps + fees_bps
        direction = 1 if side == "buy" else -1
        return bar.close * (1 + direction * total_bps / 10_000)

    @staticmethod
    def _get_spread(symbol: str, bar: BarSnapshot) -> float:
        """Estimate half-spread in bps from ADV and realized volatility."""
        adv = bar.adv
        vol = bar.realized_vol_20d
        if adv > 100_000_000:
            base = 1.0
        elif adv > 10_000_000:
            base = 3.0
        else:
            base = 10.0
        return base * (1 + 5 * vol)


# ---------------------------------------------------------------------------
# Rejection handler (33.4 — Rejection-Routing)
# ---------------------------------------------------------------------------

REJECTION_ACTIONS: dict[str, str] = {
    "insufficient_buying_power": "pause_symbol_30min",
    "wash_sale_block": "mark_wash_sale",
    "short_not_available": "cache_unshortable_24h",
    "pdt_restriction": "alert_pdt_counter_bug",
    "position_halted": "wait_60s_then_re_evaluate",
    "market_closed": "resubmit_with_extended_hours_or_discard",
    "invalid_symbol": "remove_from_universe",
    "exceeds_price_bands": "adjust_limit_or_convert_to_market",
}


def handle_rejection(reason: str, symbol: str) -> dict:
    """Route order rejection to the appropriate side-effect action.

    This is a synchronous routing helper — actual side effects (DB writes,
    Slack alerts, cache updates) are delegated to the caller.

    Args:
        reason: Rejection reason string (matches REJECTION_ACTIONS keys).
        symbol: Ticker symbol of the rejected order.

    Returns:
        Dict with 'reason', 'symbol', and 'action' keys describing what
        the caller should do.
    """
    action = REJECTION_ACTIONS.get(reason, "log_and_skip")
    logger.warning("Order rejected for %s: %s → action=%s", symbol, reason, action)
    return {"reason": reason, "symbol": symbol, "action": action}


# ---------------------------------------------------------------------------
# Wash-sale precheck helper (33.4)
# ---------------------------------------------------------------------------

def has_recent_loss_close(
    symbol: str,
    closed_positions: list[dict],
    days: int = 30,
    reference_date: Optional[datetime] = None,
) -> bool:
    """Return True if there is a loss-closing trade within the wash-sale window.

    This is a synchronous helper operating on an in-memory list of closed
    positions (no DB dependency). The async DB version is left to the caller.

    Args:
        symbol: Ticker to check.
        closed_positions: List of dicts with keys: symbol, closed_at (datetime),
                          realized_pnl (float).
        days: Wash-sale lookback window in calendar days (default 30).
        reference_date: Reference datetime for "now" (UTC). Defaults to utcnow.

    Returns:
        True if any loss-close for this symbol exists within the window.
    """
    if reference_date is None:
        reference_date = datetime.now(timezone.utc)
    cutoff = reference_date.replace(tzinfo=timezone.utc) if reference_date.tzinfo is None else reference_date
    from datetime import timedelta
    cutoff = reference_date - timedelta(days=days)

    for row in closed_positions:
        if row.get("symbol") != symbol:
            continue
        closed_at = row.get("closed_at")
        if closed_at is None:
            continue
        if closed_at.tzinfo is None:
            closed_at = closed_at.replace(tzinfo=timezone.utc)
        if closed_at >= cutoff and float(row.get("realized_pnl", 0.0)) < 0:
            return True
    return False


# ---------------------------------------------------------------------------
# Position/cash reconciliation helpers (33.5)
# ---------------------------------------------------------------------------

RECONCILE_SCHEDULE: dict[str, int] = {
    "fast": 30,       # seconds — during trading hours
    "normal": 300,    # 5 min — trading hours, idle
    "slow": 3600,     # 1 hour — outside trading hours
}


def reconcile_positions(
    broker_positions: list[dict],
    internal_positions: list[dict],
) -> list[dict]:
    """Compute position drift between broker state and internal state.

    Broker is authoritative. Returns a list of drift records for the caller
    to act on (alert, correct, log).

    Args:
        broker_positions: List of dicts with 'symbol' and 'qty'.
        internal_positions: List of dicts with 'symbol' and 'qty'.

    Returns:
        List of drift dicts: {symbol, broker_qty, internal_qty, delta}.
        Empty list if no drift.
    """
    broker_map = {p["symbol"]: float(p.get("qty", 0)) for p in broker_positions}
    internal_map = {p["symbol"]: float(p.get("qty", 0)) for p in internal_positions}
    all_symbols = set(broker_map) | set(internal_map)

    drifts = []
    for sym in sorted(all_symbols):
        b_qty = broker_map.get(sym, 0.0)
        i_qty = internal_map.get(sym, 0.0)
        if abs(b_qty - i_qty) > 1e-6:
            drifts.append({
                "symbol": sym,
                "broker_qty": b_qty,
                "internal_qty": i_qty,
                "delta": b_qty - i_qty,
            })
    if drifts:
        logger.warning("Position drift detected: %d symbols", len(drifts))
    return drifts


def reconcile_cash(
    broker_cash: float,
    internal_cash: float,
    tolerance_usd: float = 1.0,
) -> Optional[dict]:
    """Detect cash drift between broker and internal ledger.

    Args:
        broker_cash: Cash balance reported by broker.
        internal_cash: Cash balance from internal ledger.
        tolerance_usd: Minimum drift magnitude to report.

    Returns:
        Drift dict {broker_cash, internal_cash, delta} if drift > tolerance,
        else None.
    """
    delta = broker_cash - internal_cash
    if abs(delta) > tolerance_usd:
        logger.warning("Cash drift detected: broker=%.2f internal=%.2f delta=%.2f",
                       broker_cash, internal_cash, delta)
        return {"broker_cash": broker_cash, "internal_cash": internal_cash, "delta": delta}
    return None


# ---------------------------------------------------------------------------
# ExitManager (33.11 — Stop/PT/Vertical-Barrier exit logic)
# ---------------------------------------------------------------------------

@dataclass
class PositionRecord:
    """Minimal position record for exit evaluation."""

    symbol: str
    qty: float
    avg_entry_price: float
    stop_price: Optional[float] = None
    profit_target_price: Optional[float] = None
    max_holding_days: Optional[int] = None
    opened_at: Optional[datetime] = None
    side: str = "long"  # 'long' or 'short'


@dataclass
class ExitSignal:
    """Exit decision emitted by ExitManager."""

    symbol: str
    exit_reason: str  # 'stop_hit' | 'pt_hit' | 'vertical_barrier' | 'regime_change'
    current_price: float
    side: str


class ExitManager:
    """Pure-logic exit evaluator for Stop-Loss, Profit-Target, and Vertical Barrier.

    Designed to be called from either backtest replay or live paper loop.
    No async/DB dependency — caller passes current prices and positions.
    """

    def check_exits(
        self,
        positions: list[PositionRecord],
        current_prices: dict[str, float],
        reference_dt: Optional[datetime] = None,
    ) -> list[ExitSignal]:
        """Evaluate all positions and return any that should exit.

        Args:
            positions: Open position records.
            current_prices: {symbol: current_price} mapping.
            reference_dt: Reference datetime for vertical-barrier check (UTC).

        Returns:
            List of ExitSignal for positions that should be closed.
        """
        if reference_dt is None:
            reference_dt = datetime.now(timezone.utc)

        signals: list[ExitSignal] = []
        for p in positions:
            price = current_prices.get(p.symbol)
            if price is None:
                continue

            if p.stop_price is not None and self._stop_hit(p, price):
                signals.append(ExitSignal(p.symbol, "stop_hit", price, p.side))
                continue

            if p.profit_target_price is not None and self._pt_hit(p, price):
                signals.append(ExitSignal(p.symbol, "pt_hit", price, p.side))
                continue

            if p.max_holding_days is not None and p.opened_at is not None:
                opened = p.opened_at if p.opened_at.tzinfo else p.opened_at.replace(tzinfo=timezone.utc)
                days_held = (reference_dt - opened).days
                if days_held >= p.max_holding_days:
                    signals.append(ExitSignal(p.symbol, "vertical_barrier", price, p.side))

        return signals

    @staticmethod
    def _stop_hit(position: PositionRecord, current_price: float) -> bool:
        if position.side == "long":
            return current_price <= position.stop_price  # type: ignore[operator]
        return current_price >= position.stop_price  # type: ignore[operator]

    @staticmethod
    def _pt_hit(position: PositionRecord, current_price: float) -> bool:
        if position.side == "long":
            return current_price >= position.profit_target_price  # type: ignore[operator]
        return current_price <= position.profit_target_price  # type: ignore[operator]


# ---------------------------------------------------------------------------
# OrderStatusStream stub (33.9 — WS + polling architecture description)
# ---------------------------------------------------------------------------

class OrderStatusStream:
    """Order status stream stub — architecture from 33.9.

    In a live/paper environment this wraps:
      - Primary WebSocket (Alpaca trade-updates stream)
      - Polling fallback every 30 s for the last 24 h of orders
      - Reconciliation loop every 300 s

    This synchronous stub exposes the interface for testing and mocking.
    Actual async start() requires an alpaca client and async runtime.
    """

    POLL_INTERVAL_SECONDS: int = 30
    RECONCILE_INTERVAL_SECONDS: int = 300

    def __init__(self, alpaca_client: Any = None) -> None:
        self._client = alpaca_client
        self._running: bool = False

    def is_running(self) -> bool:
        return self._running

    def apply_event(self, event: dict) -> Optional[str]:
        """Process a single order-update event dict.

        Args:
            event: Dict with at least 'event' (str) and 'order' (dict) keys.

        Returns:
            Normalized status string or None if event type unrecognized.
        """
        event_type = event.get("event", "")
        order = event.get("order", {})
        symbol = order.get("symbol", "?")
        status = order.get("status", "")
        logger.debug("OrderStatusStream.apply_event: %s %s %s", event_type, symbol, status)
        known = {"fill", "partial_fill", "canceled", "expired", "replaced", "rejected"}
        return event_type if event_type in known else None


__all__ = [
    "submit_with_idempotency",
    "PartialFillPolicy",
    "position_reconcile_before_signal",
    "BarSnapshot",
    "ExecutionCostModel",
    "MIN_TRADE_SIZE",
    "REJECTION_ACTIONS",
    "handle_rejection",
    "has_recent_loss_close",
    "RECONCILE_SCHEDULE",
    "reconcile_positions",
    "reconcile_cash",
    "PositionRecord",
    "ExitSignal",
    "ExitManager",
    "OrderStatusStream",
]
