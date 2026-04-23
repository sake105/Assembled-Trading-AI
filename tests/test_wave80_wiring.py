"""Tests for wave-80 module wiring into trading_cycle.py.

Covers:
  Step 8.93 — intel.news_alerts (AlertEngine / NewsAlert)
  Step 8.94 — intel.news_archive (NewsArchiveReader / NewsArchiveWriter)
  Step 8.95 — intel.news_archiver (NewsArchiver)
"""

from __future__ import annotations

import pytest

from src.assembled_core.intel.news_alerts import AlertEngine, NewsAlert
from src.assembled_core.intel.news_archive import NewsArchiveReader, NewsArchiveWriter
from src.assembled_core.intel.news_archiver import NewsArchiver


# ---------------------------------------------------------------------------
# news_alerts (Step 8.93)
# ---------------------------------------------------------------------------

def test_alert_engine_creates():
    ae = AlertEngine(include_default_log_handler=False)
    assert isinstance(ae, AlertEngine)


def test_alert_engine_evaluate_empty():
    ae = AlertEngine(include_default_log_handler=False)
    result = ae.evaluate([])
    assert isinstance(result, list)
    assert len(result) == 0


def test_alert_engine_counters_start_zero():
    ae = AlertEngine(include_default_log_handler=False)
    assert ae.dropped_dedup == 0
    assert ae.dropped_rate == 0


def test_alert_engine_add_handler():
    ae = AlertEngine(include_default_log_handler=False)
    ae.add_handler(lambda a: None)
    # should not raise


def test_news_alert_importable():
    assert NewsAlert is not None


# ---------------------------------------------------------------------------
# news_archive (Step 8.94)
# ---------------------------------------------------------------------------

def test_news_archive_reader_creates(tmp_path):
    reader = NewsArchiveReader(tmp_path / "test.jsonl")
    assert isinstance(reader, NewsArchiveReader)


def test_news_archive_reader_nonexistent_is_falsy(tmp_path):
    reader = NewsArchiveReader(tmp_path / "missing.jsonl")
    assert not bool(reader)


def test_news_archive_reader_iter_empty(tmp_path):
    reader = NewsArchiveReader(tmp_path / "missing.jsonl")
    events = list(reader.iter_events())
    assert isinstance(events, list)
    assert len(events) == 0


def test_news_archive_writer_creates(tmp_path):
    writer = NewsArchiveWriter(tmp_path / "test.jsonl", fsync=False)
    assert isinstance(writer, NewsArchiveWriter)


def test_news_archive_writer_append_empty(tmp_path):
    writer = NewsArchiveWriter(tmp_path / "test.jsonl", fsync=False)
    written = writer.append([])
    assert written == 0


# ---------------------------------------------------------------------------
# news_archiver (Step 8.95)
# ---------------------------------------------------------------------------

def test_news_archiver_creates(tmp_path):
    archiver = NewsArchiver(base_dir=tmp_path)
    assert isinstance(archiver, NewsArchiver)


def test_news_archiver_append_empty(tmp_path):
    archiver = NewsArchiver(base_dir=tmp_path)
    written = archiver.append([])
    assert written == 0


def test_news_archiver_base_dir_set(tmp_path):
    from pathlib import Path
    archiver = NewsArchiver(base_dir=tmp_path)
    assert archiver._base == Path(tmp_path)
