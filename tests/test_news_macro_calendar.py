"""Tests for MacroCalendar."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.assembled_core.intel.news_macro_calendar import MacroCalendar, MacroEvent


@pytest.mark.phase12
class TestMacroCalendar:
    def test_empty_next(self):
        cal = MacroCalendar()
        assert cal.next_event() is None
        assert cal.proximity("fomc") is None
        assert cal.is_blackout("fomc") is False

    def test_add_and_next(self):
        cal = MacroCalendar()
        now = datetime.now(tz=timezone.utc)
        cal.add(MacroEvent("FOMC1", "fomc", now + timedelta(hours=2)))
        cal.add(MacroEvent("CPI1", "cpi", now + timedelta(hours=1)))
        nxt = cal.next_event(now=now)
        assert nxt is not None and nxt.event_id == "CPI1"
        nxt_fomc = cal.next_event(now=now, kind="fomc")
        assert nxt_fomc.event_id == "FOMC1"

    def test_proximity_and_blackout(self):
        cal = MacroCalendar()
        now = datetime.now(tz=timezone.utc)
        cal.add(MacroEvent("FOMC1", "fomc", now + timedelta(minutes=30)))
        prox = cal.proximity("fomc", now=now)
        assert prox is not None
        assert 29.0 < prox.minutes_to_event < 31.0
        # default pre window for fomc is 60 min, post 120 min -> blackout
        assert prox.within_blackout is True
        assert cal.is_blackout("fomc", now=now) is True

    def test_naive_ts_auto_tz(self):
        cal = MacroCalendar()
        naive = datetime(2026, 12, 31, 18, 0)   # no tzinfo
        cal.add(MacroEvent("E", "cpi", naive))
        assert cal._events[0].ts.tzinfo is timezone.utc

    def test_upcoming_window(self):
        cal = MacroCalendar()
        now = datetime.now(tz=timezone.utc)
        cal.add(MacroEvent("A", "cpi", now + timedelta(hours=1)))
        cal.add(MacroEvent("B", "nfp", now + timedelta(hours=48)))
        up = cal.upcoming(now=now, horizon_hours=24)
        assert [e.event_id for e in up] == ["A"]

    def test_load_json(self, tmp_path):
        now = datetime.now(tz=timezone.utc)
        data = [
            {
                "event_id": "CPI_MAY",
                "kind": "cpi",
                "ts": (now + timedelta(days=1)).isoformat(),
                "importance": 4,
            },
            {"malformed": "skip"},
        ]
        p = tmp_path / "cal.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        cal = MacroCalendar()
        assert cal.load_json(p) == 1
        assert cal.size == 1

    def test_load_missing_file(self, tmp_path):
        cal = MacroCalendar()
        assert cal.load_json(tmp_path / "nope.json") == 0

    def test_not_in_blackout_when_far(self):
        cal = MacroCalendar()
        now = datetime.now(tz=timezone.utc)
        cal.add(MacroEvent("CPI1", "cpi", now + timedelta(hours=5)))
        assert cal.is_blackout("cpi", now=now) is False
