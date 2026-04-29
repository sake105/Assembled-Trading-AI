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


__all__ = [
    "submit_with_idempotency",
    "PartialFillPolicy",
    "position_reconcile_before_signal",
    "BarSnapshot",
    "ExecutionCostModel",
    "MIN_TRADE_SIZE",
]
