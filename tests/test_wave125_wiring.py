"""Tests for wave-125 module wiring into trading_cycle.py.

Covers:
  Step 8.69 — events.news.dedupe (dedupe_events)
  Step 8.70 — events.news.dedupe_store (DedupeStoreSQLite)
  Step 8.71 — events.news.emit (emit_json_artifact)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.assembled_core.events.news.dedupe import dedupe_events
from src.assembled_core.events.news.dedupe_store import DedupeStoreSQLite
from src.assembled_core.events.news.emit import emit_json_artifact


# ---------------------------------------------------------------------------
# events.news.dedupe (Step 8.69)
# ---------------------------------------------------------------------------

def test_dedupe_events_importable():
    assert dedupe_events is not None


def test_dedupe_events_empty():
    result = dedupe_events([])
    assert isinstance(result, list)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# events.news.dedupe_store (Step 8.70)
# ---------------------------------------------------------------------------

def test_dedupe_store_sqlite_importable():
    assert DedupeStoreSQLite is not None


def test_dedupe_store_creates_with_temp_db():
    import gc
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "dedupe.db"
        store = DedupeStoreSQLite(db_path)
        assert store.path == db_path
        assert db_path.exists()
        del store
        gc.collect()


# ---------------------------------------------------------------------------
# events.news.emit (Step 8.71)
# ---------------------------------------------------------------------------

def test_emit_json_artifact_news_importable():
    assert emit_json_artifact is not None


def test_emit_json_artifact_news_writes_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "news_artifact.json"
        emit_json_artifact({"event": "test"}, path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data == {"event": "test"}
