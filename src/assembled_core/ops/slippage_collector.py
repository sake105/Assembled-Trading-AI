"""Thread-safe slippage observation accumulator for Prometheus histogram export."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class SlippageCollector:
    """Accumulate fill slippage (basis points) during a trading cycle run.

    Thread-safe: multiple fill threads may call record() concurrently.
    Slippage = (fill_price - mid_price) / mid_price * 10_000.
    Positive = BUY filled above mid (cost); negative = SELL filled below mid (cost).
    """

    _obs: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, slippage_bps: float) -> None:
        with self._lock:
            self._obs.append(float(slippage_bps))

    def record_fills(self, fills_df: "pd.DataFrame") -> None:
        """Bulk-record slippage from a fills DataFrame.

        Requires columns: fill_price, mid_price, status.
        Rows where status not in {filled, partial} are skipped.
        Rows where mid_price == 0 are skipped (division guard).
        """
        if fills_df is None or fills_df.empty:
            return
        needed = {"fill_price", "mid_price", "status"}
        if not needed.issubset(fills_df.columns):
            return
        filled = fills_df[fills_df["status"].isin({"filled", "partial"})]
        if filled.empty:
            return
        mid = filled["mid_price"]
        safe_mid = mid.where(mid != 0, other=float("nan"))
        bps = ((filled["fill_price"] - mid) / safe_mid * 10_000).dropna()
        with self._lock:
            self._obs.extend(bps.tolist())

    def snapshot(self, *, reset: bool = False) -> list[float]:
        """Return current observation list; optionally reset the accumulator."""
        with self._lock:
            result = list(self._obs)
            if reset:
                self._obs.clear()
            return result

    def __len__(self) -> int:
        with self._lock:
            return len(self._obs)


__all__ = ["SlippageCollector"]
