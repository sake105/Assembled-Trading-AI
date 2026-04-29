"""Pre-order risk gate.

From 41_PDT_REGEL_INTRADAY_MARGIN.md §4.1.

Checks whether an order may be submitted before sending to the broker.
Currently enforces PDT rules; extensible to margin, kill-switch, etc.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional

from assembled_core.execution.pdt_tracker import PDTTracker
from assembled_core.execution.round_trip_detector import RoundTripDetector

logger = logging.getLogger(__name__)


class OrderDecision(Enum):
    ALLOWED = "allowed"
    BLOCKED_PDT = "blocked_pdt"
    BLOCKED_MARGIN = "blocked_margin"
    BLOCKED_KILL_SWITCH = "blocked_kill_switch"


@dataclass
class GateResult:
    decision: OrderDecision
    reason: str
    suggested_action: Optional[str] = None


class OrderGate:
    """Pre-order checks before broker API submission.

    Two-layer defense: this gate + broker's built-in protection.
    """

    def __init__(self, pdt_tracker: PDTTracker, rt_detector: RoundTripDetector) -> None:
        self.pdt_tracker = pdt_tracker
        self.rt_detector = rt_detector

    def check_order(self, ticker: str, side: str, qty: int) -> GateResult:
        if self.pdt_tracker.enabled:
            if self._would_be_day_trade(ticker, side):
                if self.pdt_tracker.would_violate_pdt():
                    days = self.pdt_tracker.days_until_pdt_reset()
                    count = self.pdt_tracker.count_recent_day_trades()
                    equity = self.pdt_tracker.account_equity
                    return GateResult(
                        decision=OrderDecision.BLOCKED_PDT,
                        reason=(
                            f"Would be 4th day trade in 5 business days. "
                            f"Count: {count}/3. Account equity ${equity:,.0f} < $25k."
                        ),
                        suggested_action=(
                            f"Wait {days} business days, "
                            f"OR hold position overnight (swing trade), "
                            f"OR skip this signal."
                        ),
                    )
        return GateResult(decision=OrderDecision.ALLOWED, reason="all checks passed")

    def _would_be_day_trade(self, ticker: str, side: str) -> bool:
        if ticker not in self.rt_detector.open_positions:
            return False
        open_ts, _, _, open_side = self.rt_detector.open_positions[ticker]
        if open_ts.date() != date.today():
            return False
        return (open_side == "long" and side == "sell") or (
            open_side == "short" and side == "buy"
        )


__all__ = ["OrderDecision", "GateResult", "OrderGate"]
