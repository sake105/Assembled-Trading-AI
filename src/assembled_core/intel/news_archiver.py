"""JSONL event archiver for news replay and backtesting (Point 29).

Persists NewsEvent objects to date-partitioned JSONL files so they can be
replayed PIT-correctly in backtests.

Usage:
    archiver = NewsArchiver(base_dir="data/intel/archive")
    archiver.append(events)        # write to today's JSONL
    archiver.append(events, date)  # write to a specific date partition

Replay:
    for event in archiver.iter_events(start="2024-01-01", end="2024-12-31"):
        process(event)
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


class NewsArchiver:
    """Date-partitioned JSONL archive for NewsEvent objects.

    Layout: <base_dir>/<YYYY-MM>/<YYYY-MM-DD>.jsonl
    Each line is one JSON-serialised NewsEvent (model_dump()).
    """

    def __init__(self, base_dir: str | Path = "data/intel/archive") -> None:
        self._base = Path(base_dir)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(
        self,
        events: list,
        partition_date: date | datetime | None = None,
    ) -> int:
        """Append events to the appropriate JSONL partition.

        Args:
            events: List of NewsEvent objects (must support .model_dump()).
            partition_date: Date for partitioning. Defaults to today UTC.

        Returns:
            Number of events written.
        """
        if not events:
            return 0

        if partition_date is None:
            partition_date = datetime.now(tz=timezone.utc).date()
        elif isinstance(partition_date, datetime):
            partition_date = partition_date.date()

        path = self._partition_path(partition_date)
        path.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        with open(path, "a", encoding="utf-8") as fh:
            for evt in events:
                try:
                    if hasattr(evt, "model_dump"):
                        record = evt.model_dump(mode="json")
                    elif hasattr(evt, "__dict__"):
                        record = {k: v for k, v in vars(evt).items() if not k.startswith("_")}
                    else:
                        continue
                    fh.write(json.dumps(record, default=str) + "\n")
                    written += 1
                except Exception as exc:
                    logger.debug("[SKIP] NewsArchiver.append: %s", exc)

        logger.debug("[OK] NewsArchiver: wrote %d events to %s", written, path)
        return written

    # ------------------------------------------------------------------
    # Read / replay
    # ------------------------------------------------------------------

    def iter_events(
        self,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> Iterator[dict]:
        """Iterate raw event dicts in chronological order.

        Args:
            start: Inclusive start date (YYYY-MM-DD string or date object).
            end: Inclusive end date (YYYY-MM-DD string or date object).

        Yields:
            Raw dicts (as archived — caller reconstructs NewsEvent if needed).
        """
        start_d = _parse_date(start)
        end_d = _parse_date(end)

        for path in sorted(self._base.rglob("*.jsonl")):
            try:
                file_date = _date_from_path(path)
            except ValueError:
                continue
            if start_d and file_date < start_d:
                continue
            if end_d and file_date > end_d:
                break
            yield from _read_jsonl(path)

    def list_partitions(self) -> list[date]:
        """Return sorted list of available partition dates."""
        dates: list[date] = []
        for path in self._base.rglob("*.jsonl"):
            try:
                dates.append(_date_from_path(path))
            except ValueError:
                continue
        return sorted(dates)

    def count_events(
        self,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> int:
        """Count total archived events in the date range."""
        return sum(1 for _ in self.iter_events(start=start, end=end))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _partition_path(self, d: date) -> Path:
        month_dir = self._base / d.strftime("%Y-%m")
        return month_dir / f"{d.isoformat()}.jsonl"


def _parse_date(val: str | date | None) -> date | None:
    if val is None:
        return None
    if isinstance(val, date):
        return val
    return date.fromisoformat(str(val)[:10])


def _date_from_path(path: Path) -> date:
    stem = path.stem  # e.g. "2024-01-15"
    return date.fromisoformat(stem)


def _read_jsonl(path: Path) -> Iterator[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        logger.warning("[WARN] NewsArchiver: could not read %s: %s", path, exc)


__all__ = ["NewsArchiver"]
