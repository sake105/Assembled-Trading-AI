"""Order-to-Trade Ratio (OTR) Monitor — MiFID II compliance.

Monitors the ratio of orders submitted vs. orders filled to detect
potential market manipulation patterns or system anomalies.

MiFID II requires firms to maintain reasonable OTR levels.
High OTR (many orders, few fills) may indicate:
- Spoofing/layering
- Quote stuffing
- System malfunction
- Excessive order amendments

Typical threshold: OTR > 4:1 triggers review, > 10:1 triggers alert.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class OTRAlertLevel(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    BREACH = "breach"


@dataclass
class OTRSnapshot:
    """Point-in-time OTR measurement."""
    timestamp: str
    orders_submitted: int
    orders_filled: int
    orders_cancelled: int
    otr_ratio: float
    alert_level: str
    symbols_flagged: list[str] = field(default_factory=list)


class OTRMonitor:
    """Monitors Order-to-Trade ratio for MiFID II compliance.

    Args:
        warning_threshold: OTR above this triggers warning (default: 4.0).
        critical_threshold: OTR above this triggers critical (default: 8.0).
        breach_threshold: OTR above this triggers breach (default: 12.0).
        window_minutes: Rolling window for OTR calculation (default: 60).
    """

    def __init__(
        self,
        warning_threshold: float = 4.0,
        critical_threshold: float = 8.0,
        breach_threshold: float = 12.0,
        window_minutes: int = 60,
    ) -> None:
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.breach_threshold = breach_threshold
        self.window_minutes = window_minutes

        self._orders: list[dict] = []  # {symbol, ts, type: submit/fill/cancel}
        self._snapshots: list[OTRSnapshot] = []

    def record_order(self, symbol: str, order_type: str = "submit") -> None:
        """Record an order event.

        Args:
            symbol: Trading symbol.
            order_type: One of "submit", "fill", "cancel".
        """
        self._orders.append({
            "symbol": symbol,
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": order_type,
        })

    def compute_otr(self, symbol: str | None = None) -> OTRSnapshot:
        """Compute current OTR ratio.

        Args:
            symbol: If provided, compute per-symbol OTR.
                If None, compute aggregate.

        Returns:
            OTRSnapshot with current metrics.
        """
        orders = self._orders
        if symbol:
            orders = [o for o in orders if o["symbol"] == symbol]

        n_submit = sum(1 for o in orders if o["type"] == "submit")
        n_fill = sum(1 for o in orders if o["type"] == "fill")
        n_cancel = sum(1 for o in orders if o["type"] == "cancel")

        if n_fill == 0:
            ratio = float(n_submit) if n_submit > 0 else 0.0
        else:
            ratio = n_submit / n_fill

        # Determine alert level
        if ratio >= self.breach_threshold:
            level = OTRAlertLevel.BREACH
        elif ratio >= self.critical_threshold:
            level = OTRAlertLevel.CRITICAL
        elif ratio >= self.warning_threshold:
            level = OTRAlertLevel.WARNING
        else:
            level = OTRAlertLevel.NORMAL

        # Find per-symbol flagged
        flagged = []
        if symbol is None:
            sym_set = set(o["symbol"] for o in self._orders)
            for s in sym_set:
                s_submit = sum(1 for o in self._orders if o["symbol"] == s and o["type"] == "submit")
                s_fill = sum(1 for o in self._orders if o["symbol"] == s and o["type"] == "fill")
                s_ratio = s_submit / max(s_fill, 1)
                if s_ratio >= self.warning_threshold:
                    flagged.append(s)

        snapshot = OTRSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            orders_submitted=n_submit,
            orders_filled=n_fill,
            orders_cancelled=n_cancel,
            otr_ratio=round(ratio, 2),
            alert_level=level.value,
            symbols_flagged=flagged,
        )
        self._snapshots.append(snapshot)

        if level != OTRAlertLevel.NORMAL:
            logger.warning(
                "[OTR] Alert %s: ratio=%.1f (submit=%d, fill=%d)",
                level.value, ratio, n_submit, n_fill,
            )

        return snapshot

    def reset(self) -> None:
        """Reset order counters (e.g., start of new trading day)."""
        self._orders.clear()

    @property
    def history(self) -> list[OTRSnapshot]:
        return list(self._snapshots)


__all__ = [
    "OTRAlertLevel",
    "OTRSnapshot",
    "OTRMonitor",
]
