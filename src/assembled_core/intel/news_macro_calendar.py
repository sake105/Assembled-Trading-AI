"""Macro calendar hooks for the news engine.

Holds a lightweight in-memory schedule of scheduled macro events (FOMC, CPI,
NFP, ECB, BOJ, earnings-heavy days) and answers proximity queries used by the
intel cycle to boost urgency / gate trading around events.

No external dependencies. The calendar can be loaded from JSON or populated
programmatically. If no entries are loaded, all proximity queries return None.

Usage:
    cal = MacroCalendar()
    cal.add(MacroEvent("FOMC_DEC_2026", "fomc", datetime(2026, 12, 16, 18, 0, tzinfo=utc)))
    if cal.is_blackout("fomc", now=datetime.now(tz=utc), window_min=30):
        ...
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# Default pre/post windows (minutes) per event class.
_DEFAULT_WINDOWS: dict[str, tuple[int, int]] = {
    "fomc": (60, 120),
    "ecb": (60, 120),
    "boj": (60, 120),
    "cpi": (30, 30),
    "nfp": (30, 30),
    "ppi": (30, 30),
    "retail_sales": (15, 15),
    "earnings": (0, 15),
    "gdp": (30, 30),
    "pmi": (15, 15),
}


@dataclass
class MacroEvent:
    event_id: str
    kind: str          # "fomc", "cpi", "nfp", "earnings", ...
    ts: datetime       # scheduled timestamp (UTC)
    importance: int = 3   # 1 (low) – 5 (top)
    tickers: list[str] = field(default_factory=list)  # relevant tickers (for earnings)
    note: str = ""


@dataclass
class Proximity:
    event: MacroEvent
    minutes_to_event: float   # negative if event already passed
    within_blackout: bool


class MacroCalendar:
    """In-memory macro calendar with proximity queries."""

    def __init__(self) -> None:
        self._events: list[MacroEvent] = []

    # ---- loading -------------------------------------------------
    def add(self, event: MacroEvent) -> None:
        if event.ts.tzinfo is None:
            event = MacroEvent(
                event_id=event.event_id, kind=event.kind,
                ts=event.ts.replace(tzinfo=timezone.utc),
                importance=event.importance, tickers=list(event.tickers),
                note=event.note,
            )
        self._events.append(event)

    def load_json(self, path: str | Path) -> int:
        """Load events from a JSON list. Returns the number of events added."""
        p = Path(path)
        if not p.exists():
            return 0
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[WARN] MacroCalendar.load_json: %s", exc)
            return 0
        count = 0
        for entry in raw or []:
            try:
                ts = datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))
                self.add(MacroEvent(
                    event_id=entry["event_id"],
                    kind=entry.get("kind", "other").lower(),
                    ts=ts,
                    importance=int(entry.get("importance", 3)),
                    tickers=list(entry.get("tickers", [])),
                    note=entry.get("note", ""),
                ))
                count += 1
            except Exception as exc:
                logger.debug("[SKIP] MacroCalendar entry: %s", exc)
        return count

    # ---- queries -------------------------------------------------
    def next_event(
        self,
        now: datetime | None = None,
        kind: str | None = None,
    ) -> MacroEvent | None:
        if now is None:
            now = datetime.now(tz=timezone.utc)
        future = [e for e in self._events if e.ts >= now]
        if kind:
            kind = kind.lower()
            future = [e for e in future if e.kind == kind]
        if not future:
            return None
        return min(future, key=lambda e: e.ts)

    def proximity(
        self,
        kind: str,
        now: datetime | None = None,
        *,
        pre_min: int | None = None,
        post_min: int | None = None,
    ) -> Proximity | None:
        """Return the nearest event of `kind` and its blackout state."""
        if now is None:
            now = datetime.now(tz=timezone.utc)
        kind = (kind or "").lower()
        evts = [e for e in self._events if e.kind == kind]
        if not evts:
            return None
        nearest = min(evts, key=lambda e: abs((e.ts - now).total_seconds()))
        pre, post = _DEFAULT_WINDOWS.get(kind, (30, 30))
        if pre_min is not None:
            pre = pre_min
        if post_min is not None:
            post = post_min
        minutes = (nearest.ts - now).total_seconds() / 60.0
        in_blackout = -post <= minutes <= pre
        return Proximity(event=nearest, minutes_to_event=minutes, within_blackout=in_blackout)

    def is_blackout(
        self,
        kind: str,
        now: datetime | None = None,
        window_min: int | None = None,
    ) -> bool:
        """True if now is within the symmetric blackout window of the nearest event."""
        prox = self.proximity(
            kind, now=now,
            pre_min=window_min, post_min=window_min,
        )
        return bool(prox and prox.within_blackout)

    def upcoming(
        self,
        now: datetime | None = None,
        horizon_hours: float = 24.0,
    ) -> list[MacroEvent]:
        """Events occurring within the next `horizon_hours`, sorted chronologically."""
        if now is None:
            now = datetime.now(tz=timezone.utc)
        cutoff = now + timedelta(hours=horizon_hours)
        return sorted(
            (e for e in self._events if now <= e.ts <= cutoff),
            key=lambda e: e.ts,
        )

    def clear(self) -> None:
        self._events.clear()

    @property
    def size(self) -> int:
        return len(self._events)


__all__ = ["MacroCalendar", "MacroEvent", "Proximity"]
