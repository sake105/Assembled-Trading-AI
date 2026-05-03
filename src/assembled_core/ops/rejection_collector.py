"""Thread-safe rejection reason counter for Prometheus export.

NOTE: This singleton-based collector is currently NOT wired into the production
path. The trading cycle writes rejection counts into result.meta["rejection_counts"]
(accumulated in check_risk()) for Phase 11 KPI export instead. This module is
retained for potential future use in multi-process or long-running contexts
where a persistent singleton accumulator across trading bars is needed.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class RejectionCollector:
    """Accumulate order rejection counts keyed by rejection reason.

    Thread-safe. Call record() or record_fills() to accumulate, then
    snapshot() to read (and optionally reset) counters for export.
    """

    _counts: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, reason: str) -> None:
        with self._lock:
            self._counts[reason] = self._counts.get(reason, 0) + 1

    def record_fills(self, fills_df: "pd.DataFrame") -> None:
        """Bulk-record rejections from a fills DataFrame.

        Counts rows where status == 'rejected' grouped by reject_reason column.
        If reject_reason column is absent, falls back to reason 'UNKNOWN'.
        """
        if fills_df is None or fills_df.empty:
            return
        if "status" not in fills_df.columns:
            return
        rejected = fills_df[fills_df["status"] == "rejected"]
        if rejected.empty:
            return
        reason_col = "reject_reason" if "reject_reason" in rejected.columns else None
        if reason_col:
            reasons = rejected[reason_col].fillna("UNKNOWN").astype(str).tolist()
        else:
            reasons = ["UNKNOWN"] * len(rejected)
        counts_map: dict[str, int] = {}
        for r in reasons:
            counts_map[r] = counts_map.get(r, 0) + 1
        with self._lock:
            for r, c in counts_map.items():
                self._counts[r] = self._counts.get(r, 0) + c

    def record_blocked_reasons(self, reasons: list[str]) -> None:
        """Record a list of block reasons (e.g. from PreTradeResult.blocked_reasons)."""
        with self._lock:
            for r in reasons:
                self._counts[r] = self._counts.get(r, 0) + 1

    def snapshot(self, *, reset: bool = False) -> dict[str, int]:
        """Return current counts dict; optionally reset all counters."""
        with self._lock:
            result = dict(self._counts)
            if reset:
                self._counts.clear()
            return result

    def total(self) -> int:
        with self._lock:
            return sum(self._counts.values())

    def __len__(self) -> int:
        return self.total()


__all__ = ["RejectionCollector"]
