"""News-Replay harness for backtesting (X5).

Synchronises archived news/trigger snapshots with price data for PIT-correct
backtesting. Relies on PITStore (X1 full) for snapshot iteration.

Usage:
    replayer = NewsReplayer(pit_store, price_panel_df)
    for step in replayer.replay("news", "triggers", start="2024-01-01", end="2024-12-31"):
        # step.timestamp: current backtest date
        # step.triggers: triggers visible as of this date (PIT-correct)
        # step.prices: prices up to this date
        do_something(step)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    import pandas as pd
    from src.assembled_core.intel.pit_store import PITStore

logger = logging.getLogger(__name__)


@dataclass
class ReplayStep:
    """A single step in the news replay sequence."""
    timestamp: datetime
    run_id: str
    triggers: dict | list | None
    prices: "pd.DataFrame | None" = field(default=None, repr=False)


def _parse_dt(s: str | datetime) -> datetime:
    if isinstance(s, datetime):
        dt = s
    else:
        dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class NewsReplayer:
    """PIT-correct news replay for backtesting.

    Iterates archived snapshots in chronological order, optionally joined
    to a price panel DataFrame so that each step carries the prices visible
    at that point in time.
    """

    def __init__(
        self,
        pit_store: "PITStore",
        prices: "pd.DataFrame | None" = None,
        *,
        price_timestamp_col: str = "timestamp",
    ) -> None:
        self._store = pit_store
        self._prices = prices
        self._ts_col = price_timestamp_col

    def replay(
        self,
        source: str,
        artifact_type: str,
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> Iterator[ReplayStep]:
        """Yield ReplayStep objects in chronological order.

        Args:
            source: PITStore source name (e.g. ``"news"``).
            artifact_type: Artifact type to replay (e.g. ``"triggers"``).
            start: Optional start datetime (inclusive). Steps before this are skipped.
            end: Optional end datetime (inclusive). Steps after this are skipped.
        """
        start_dt = _parse_dt(start) if start else None
        end_dt = _parse_dt(end) if end else None

        for run_id, data in self._store.iter_chronological(source, artifact_type):
            manifest = self._store.manifest(source, run_id)
            entry = manifest.get(artifact_type, {})
            try:
                step_dt = _parse_dt(entry.get("archived_utc", "1970-01-01T00:00:00+00:00"))
            except Exception:
                logger.debug("[SKIP] NewsReplayer: unparseable archived_utc for run_id=%s", run_id)
                continue

            if start_dt and step_dt < start_dt:
                continue
            if end_dt and step_dt > end_dt:
                break

            prices_slice = self._slice_prices(step_dt)

            yield ReplayStep(
                timestamp=step_dt,
                run_id=run_id,
                triggers=data,
                prices=prices_slice,
            )

    def _slice_prices(self, as_of: datetime) -> "pd.DataFrame | None":
        if self._prices is None:
            return None
        try:
            import pandas as pd
            ts = pd.to_datetime(self._prices[self._ts_col], utc=True)
            return self._prices[ts <= as_of].copy()
        except Exception as exc:
            logger.debug("[WARN] NewsReplayer._slice_prices: %s", exc)
            return None


__all__ = ["NewsReplayer", "ReplayStep"]
