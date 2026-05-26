"""Smoke tests for scripts/run_news_alpha_intraday.py helper functions.

Covers: _headline_to_topic_id, _severity_float_to_int,
_events_to_triggers, _save_state/_load_state atomic round-trip.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the runner module — scripts/ is not a package, add once to sys.path
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import run_news_alpha_intraday as runner  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    title: str,
    severity: float = 5.0,
    content_hash: str = "abc123",
    url: str = "http://example.com/news",
    source_id: str = "reuters",
    urgency: float = 0.0,
) -> types.SimpleNamespace:
    """Minimal RSS event object compatible with _events_to_triggers."""
    source_tier = types.SimpleNamespace(value="T1")
    return types.SimpleNamespace(
        title=title,
        severity=severity,
        content_hash=content_hash,
        url=url,
        source_id=source_id,
        source_tier=source_tier,
        geo_tags=["US"],
        urgency=urgency,
    )


# ---------------------------------------------------------------------------
# _headline_to_topic_id
# ---------------------------------------------------------------------------


class TestHeadlineToTopicId:
    def test_shipping_disruption_hormuz(self):
        assert (
            runner._headline_to_topic_id("Tensions rising in Strait of Hormuz")
            == "shipping_disruption"
        )

    def test_energy_crisis_opec(self):
        assert (
            runner._headline_to_topic_id("OPEC cuts production by 1M barrels")
            == "energy_crisis"
        )

    def test_taiwan_strait(self):
        assert (
            runner._headline_to_topic_id("China increases military exercises Taiwan")
            == "taiwan_strait"
        )

    def test_nuclear_risk(self):
        assert (
            runner._headline_to_topic_id("North Korea nuclear weapon test imminent")
            == "nuclear_risk"
        )

    def test_market_crash(self):
        assert (
            runner._headline_to_topic_id(
                "NYSE circuit breaker triggered after flash crash"
            )
            == "market_crash"
        )

    def test_geopolitical_conflict(self):
        assert (
            runner._headline_to_topic_id("Russian troops advance on frontline")
            == "geopolitical_conflict"
        )

    def test_central_bank(self):
        assert (
            runner._headline_to_topic_id("Fed raises rates by 50 basis point emergency")
            == "central_bank"
        )

    def test_no_match_returns_none(self):
        assert runner._headline_to_topic_id("Apple releases new iPhone model") is None

    def test_case_insensitive(self):
        assert (
            runner._headline_to_topic_id("HORMUZ STRAIT TENSIONS ESCALATE")
            == "shipping_disruption"
        )

    def test_empty_string_returns_none(self):
        assert runner._headline_to_topic_id("") is None


# ---------------------------------------------------------------------------
# _severity_float_to_int
# ---------------------------------------------------------------------------


class TestSeverityFloatToInt:
    def test_high(self):
        assert runner._severity_float_to_int(8.5) == 3

    def test_medium(self):
        assert runner._severity_float_to_int(5.0) == 2

    def test_low(self):
        assert runner._severity_float_to_int(2.0) == 1

    def test_boundary_7_is_high(self):
        assert runner._severity_float_to_int(7.0) == 3

    def test_boundary_4_is_medium(self):
        assert runner._severity_float_to_int(4.0) == 2

    def test_just_below_4_is_low(self):
        assert runner._severity_float_to_int(3.99) == 1

    def test_zero_is_low(self):
        assert runner._severity_float_to_int(0.0) == 1

    def test_ten_is_high(self):
        assert runner._severity_float_to_int(10.0) == 3


# ---------------------------------------------------------------------------
# _events_to_triggers
# ---------------------------------------------------------------------------


class TestEventsToTriggers:
    def test_returns_trigger_for_matching_event(self):
        ev = _make_event("Hormuz strait tanker seized by forces", severity=6.0)
        triggers, new_ids = runner._events_to_triggers([ev], set(), min_severity=1)
        assert len(triggers) == 1
        assert triggers[0]["topic"] == "shipping_disruption"
        assert new_ids == ["abc123"]

    def test_skips_already_seen_event(self):
        ev = _make_event(
            "Hormuz strait tanker seized", severity=6.0, content_hash="xyz"
        )
        triggers, new_ids = runner._events_to_triggers([ev], {"xyz"}, min_severity=1)
        assert triggers == []
        assert new_ids == []

    def test_no_topic_match_not_added(self):
        ev = _make_event("Apple announces record earnings", severity=8.0)
        triggers, new_ids = runner._events_to_triggers([ev], set(), min_severity=1)
        assert triggers == []
        assert new_ids == []

    def test_no_topic_match_not_marked_seen(self):
        """Non-matching events must NOT be added to seen_ids so they can re-evaluate."""
        ev = _make_event("Weather forecast for New York", severity=9.0)
        _, new_ids = runner._events_to_triggers([ev], set(), min_severity=1)
        assert new_ids == []

    def test_min_severity_filters_low(self):
        # severity=4.0 → int_sev=2 < min_severity=3 → filtered
        ev = _make_event("Hormuz tanker seized forces", severity=4.0)
        triggers, _ = runner._events_to_triggers([ev], set(), min_severity=3)
        assert triggers == []

    def test_urgency_floor_boosts_severity(self):
        # urgency > 0.5 floors sev_float to max(3.0, 7.0)=7.0 → int=3
        ev = _make_event("Hormuz strait tanker seized", severity=3.0, urgency=1.0)
        triggers, new_ids = runner._events_to_triggers([ev], set(), min_severity=3)
        assert len(triggers) == 1
        assert triggers[0]["severity"] == 3

    def test_trigger_contains_required_fields(self):
        ev = _make_event("OPEC cuts production sharply", severity=5.0)
        triggers, _ = runner._events_to_triggers([ev], set(), min_severity=1)
        t = triggers[0]
        assert "severity" in t
        assert "topic" in t
        assert "source" in t
        assert "details" in t
        assert "event_id" in t

    def test_details_contains_headline(self):
        title = "Houthi missile attack on oil tanker in Red Sea"
        ev = _make_event(title, severity=7.0)
        triggers, _ = runner._events_to_triggers([ev], set(), min_severity=1)
        assert triggers[0]["details"] == title

    def test_multiple_events_deduped_by_seen(self):
        ev1 = _make_event("Hormuz tanker seized", severity=6.0, content_hash="h1")
        ev2 = _make_event("OPEC cuts oil output", severity=6.0, content_hash="h2")
        # h1 already seen
        triggers, new_ids = runner._events_to_triggers(
            [ev1, ev2], {"h1"}, min_severity=1
        )
        assert len(triggers) == 1
        assert triggers[0]["topic"] == "energy_crisis"
        assert new_ids == ["h2"]

    def test_empty_events_list_returns_empty(self):
        triggers, new_ids = runner._events_to_triggers([], set(), min_severity=1)
        assert triggers == []
        assert new_ids == []


# ---------------------------------------------------------------------------
# seen_ids trim — insertion-order determinism (F-002 regression)
# ---------------------------------------------------------------------------


class TestSeenIdsTrim:
    """Regression guard: _update_seen_ids() must retain the MOST RECENTLY added IDs."""

    def test_trim_retains_recent_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Patch _MAX_SEEN_IDS to a small value to trigger trim cheaply.
        monkeypatch.setattr(runner, "_MAX_SEEN_IDS", 4)

        # Start with 4 IDs already in state (at capacity).
        ids_list: list[str] = ["old1", "old2", "old3", "old4"]
        ids_set: set[str] = set(ids_list)

        # Add one new ID — pushes len to 5 > _MAX_SEEN_IDS=4.
        ids_list, ids_set = runner._update_seen_ids(ids_list, ids_set, ["new1"])

        # The most recently added ID MUST be retained; trim target = 4//2 = 2.
        assert "new1" in ids_set, "most recent ID dropped after trim"
        assert len(ids_list) == 2

    def test_no_trim_below_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runner, "_MAX_SEEN_IDS", 10)
        ids_list = ["a", "b", "c"]
        ids_set = set(ids_list)
        ids_list, ids_set = runner._update_seen_ids(ids_list, ids_set, ["d"])
        assert "d" in ids_set
        assert len(ids_list) == 4  # no trim

    def test_duplicate_not_added_twice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runner, "_MAX_SEEN_IDS", 10)
        ids_list = ["a", "b"]
        ids_set = set(ids_list)
        ids_list, ids_set = runner._update_seen_ids(ids_list, ids_set, ["a"])
        assert len(ids_list) == 2  # "a" already in set, not appended again

    def test_list_and_set_stay_in_sync_after_trim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runner, "_MAX_SEEN_IDS", 4)
        ids_list = ["x1", "x2", "x3", "x4"]
        ids_set = set(ids_list)
        ids_list, ids_set = runner._update_seen_ids(ids_list, ids_set, ["x5"])
        assert set(ids_list) == ids_set, "list and set diverged after trim"


# ---------------------------------------------------------------------------
# _save_state / _load_state  (atomic round-trip)
# ---------------------------------------------------------------------------


class TestStatePersistence:
    def test_round_trip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runner, "_STATE_FILE", tmp_path / "state.json")
        state = {
            "open_signals": [],
            "seen_event_ids": ["id1", "id2"],
            "day_counter": 3,
            "last_date": "2026-05-26",
        }
        runner._save_state(state)
        loaded = runner._load_state()
        assert loaded == state

    def test_tmp_file_removed_after_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state_file = tmp_path / "state.json"
        monkeypatch.setattr(runner, "_STATE_FILE", state_file)
        runner._save_state(
            {
                "open_signals": [],
                "seen_event_ids": [],
                "day_counter": 0,
                "last_date": "",
            }
        )
        assert not state_file.with_suffix(".tmp").exists(), (
            ".tmp must be removed after atomic rename"
        )

    def test_state_file_is_valid_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json

        state_file = tmp_path / "state.json"
        monkeypatch.setattr(runner, "_STATE_FILE", state_file)
        runner._save_state(
            {
                "open_signals": [],
                "seen_event_ids": ["x"],
                "day_counter": 1,
                "last_date": "2026-05-26",
            }
        )
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["day_counter"] == 1

    def test_load_missing_file_returns_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runner, "_STATE_FILE", tmp_path / "nonexistent.json")
        state = runner._load_state()
        assert state["open_signals"] == []
        assert state["day_counter"] == 0
        assert "seen_event_ids" in state

    def test_load_corrupt_json_returns_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state_file = tmp_path / "state.json"
        monkeypatch.setattr(runner, "_STATE_FILE", state_file)
        state_file.write_text("{corrupt json", encoding="utf-8")
        state = runner._load_state()
        assert state["open_signals"] == []

    def test_save_creates_parent_directories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        nested = tmp_path / "sub" / "dir" / "state.json"
        monkeypatch.setattr(runner, "_STATE_FILE", nested)
        runner._save_state(
            {
                "open_signals": [],
                "seen_event_ids": [],
                "day_counter": 0,
                "last_date": "",
            }
        )
        assert nested.exists()

    def test_overwrite_updates_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runner, "_STATE_FILE", tmp_path / "state.json")
        runner._save_state(
            {
                "open_signals": [],
                "seen_event_ids": [],
                "day_counter": 0,
                "last_date": "",
            }
        )
        runner._save_state(
            {
                "open_signals": [],
                "seen_event_ids": ["new"],
                "day_counter": 5,
                "last_date": "2026-05-26",
            }
        )
        loaded = runner._load_state()
        assert loaded["day_counter"] == 5
        assert loaded["seen_event_ids"] == ["new"]
