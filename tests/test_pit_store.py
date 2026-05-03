"""Tests for PITStore (X1 full)."""

from __future__ import annotations

import json
import pytest

from src.assembled_core.intel.pit_store import PITStore

# news_replay archived — TestNewsReplayer skipped
pytest.importorskip("src.assembled_core.intel.news_replay")
from src.assembled_core.intel.news_replay import NewsReplayer, ReplayStep  # noqa: E402


@pytest.mark.phase12
class TestPITStore:
    def test_archive_and_load(self, tmp_path):
        store = PITStore(tmp_path / "pit")
        data = {"triggers": [{"type": "GEO_CONFLICT", "geo": "RU"}]}
        dest = store.archive("news", "run_001", "triggers", data)
        assert dest.exists()
        loaded = store.load("news", "run_001", "triggers")
        assert loaded == data

    def test_archive_overwrite_false(self, tmp_path):
        store = PITStore(tmp_path / "pit")
        store.archive("news", "run_001", "triggers", {"v": 1})
        store.archive("news", "run_001", "triggers", {"v": 2}, overwrite=False)
        loaded = store.load("news", "run_001", "triggers")
        assert loaded["v"] == 1  # original preserved

    def test_archive_overwrite_true(self, tmp_path):
        store = PITStore(tmp_path / "pit")
        store.archive("news", "run_001", "triggers", {"v": 1})
        store.archive("news", "run_001", "triggers", {"v": 2}, overwrite=True)
        assert store.load("news", "run_001", "triggers")["v"] == 2

    def test_load_missing_returns_none(self, tmp_path):
        store = PITStore(tmp_path / "pit")
        assert store.load("news", "run_999", "triggers") is None

    def test_list_run_ids(self, tmp_path):
        store = PITStore(tmp_path / "pit")
        store.archive("news", "run_001", "triggers", {})
        store.archive("news", "run_002", "triggers", {})
        assert store.list_run_ids("news") == ["run_001", "run_002"]

    def test_manifest_tracks_artifact_types(self, tmp_path):
        store = PITStore(tmp_path / "pit")
        store.archive("news", "run_001", "triggers", {"t": 1})
        store.archive("news", "run_001", "clusters", {"c": 2})
        manifest = store.manifest("news", "run_001")
        assert "triggers" in manifest
        assert "clusters" in manifest

    def test_latest_returns_most_recent(self, tmp_path):
        store = PITStore(tmp_path / "pit")
        store.archive("news", "run_001", "triggers", {"v": 1})
        store.archive("news", "run_002", "triggers", {"v": 2})
        latest = store.latest("news", "triggers")
        assert latest is not None

    def test_multiple_sources_isolated(self, tmp_path):
        store = PITStore(tmp_path / "pit")
        store.archive("news", "run_001", "triggers", {"src": "news"})
        store.archive("disclosures", "run_001", "triggers", {"src": "disc"})
        assert store.load("news", "run_001", "triggers")["src"] == "news"
        assert store.load("disclosures", "run_001", "triggers")["src"] == "disc"

    def test_load_as_of(self, tmp_path):
        from datetime import datetime, timezone
        store = PITStore(tmp_path / "pit")
        t1 = datetime(2024, 1, 10, tzinfo=timezone.utc)
        t2 = datetime(2024, 1, 20, tzinfo=timezone.utc)
        store.archive("news", "run_001", "triggers", {"v": 1}, archived_utc=t1)
        store.archive("news", "run_002", "triggers", {"v": 2}, archived_utc=t2)
        # as_of before t2 → should return run_001's data
        result = store.load_as_of("news", "triggers", as_of="2024-01-15T00:00:00+00:00")
        assert result == {"v": 1}

    def test_load_as_of_returns_none_if_all_after(self, tmp_path):
        from datetime import datetime, timezone
        store = PITStore(tmp_path / "pit")
        t1 = datetime(2024, 6, 1, tzinfo=timezone.utc)
        store.archive("news", "run_001", "triggers", {"v": 1}, archived_utc=t1)
        result = store.load_as_of("news", "triggers", as_of="2024-01-01T00:00:00+00:00")
        assert result is None

    def test_archive_file(self, tmp_path):
        store = PITStore(tmp_path / "pit")
        src_file = tmp_path / "triggers.json"
        src_file.write_text(json.dumps({"data": "hello"}), encoding="utf-8")
        dest = store.archive_file("news", "run_001", "triggers", src_file)
        assert dest is not None
        assert store.load("news", "run_001", "triggers") == {"data": "hello"}

    def test_iter_chronological(self, tmp_path):
        from datetime import datetime, timezone
        store = PITStore(tmp_path / "pit")
        t1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2024, 1, 2, tzinfo=timezone.utc)
        store.archive("news", "run_001", "triggers", {"v": 1}, archived_utc=t1)
        store.archive("news", "run_002", "triggers", {"v": 2}, archived_utc=t2)
        steps = list(store.iter_chronological("news", "triggers"))
        assert len(steps) == 2
        assert steps[0][1]["v"] == 1
        assert steps[1][1]["v"] == 2


@pytest.mark.phase12
class TestNewsReplayer:
    def _build_store(self, tmp_path):
        from datetime import datetime
        store = PITStore(tmp_path / "pit")
        for i, day in enumerate(["2024-01-10", "2024-01-20", "2024-01-30"]):
            dt = datetime.fromisoformat(day + "T12:00:00+00:00")
            store.archive("news", f"run_{i:03d}", "triggers", {"day": day}, archived_utc=dt)
        return store

    def test_replay_yields_steps(self, tmp_path):
        store = self._build_store(tmp_path)
        replayer = NewsReplayer(store)
        steps = list(replayer.replay("news", "triggers"))
        assert len(steps) == 3
        assert all(isinstance(s, ReplayStep) for s in steps)

    def test_replay_start_filter(self, tmp_path):
        store = self._build_store(tmp_path)
        replayer = NewsReplayer(store)
        steps = list(replayer.replay("news", "triggers", start="2024-01-15"))
        assert len(steps) == 2

    def test_replay_end_filter(self, tmp_path):
        store = self._build_store(tmp_path)
        replayer = NewsReplayer(store)
        steps = list(replayer.replay("news", "triggers", end="2024-01-15"))
        assert len(steps) == 1

    def test_replay_with_prices(self, tmp_path):
        import pandas as pd
        store = self._build_store(tmp_path)
        prices = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC"),
            "close": range(30),
        })
        replayer = NewsReplayer(store, prices)
        steps = list(replayer.replay("news", "triggers"))
        for step in steps:
            assert step.prices is not None
            assert not step.prices.empty
