"""Tests for NewsArchiveWriter / NewsArchiveReader."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from src.assembled_core.intel.models import NewsEvent, SourceTier
from src.assembled_core.intel.news_archive import NewsArchiveReader, NewsArchiveWriter


def _evt(eid: str, title: str, ts: datetime | None = None) -> NewsEvent:
    ts = ts or datetime.now(tz=timezone.utc)
    return NewsEvent(
        event_id=eid,
        source_id="reuters",
        source_tier=SourceTier.T1,
        title=title,
        url=f"https://example.com/{eid}",
        published_at=ts,
        ingested_at=ts,
        content_hash=hashlib.sha256((title + eid).encode()).hexdigest()[:16],
    )


@pytest.mark.fast
class TestNewsArchive:
    def test_missing_file_reader(self, tmp_path):
        r = NewsArchiveReader(tmp_path / "nope.jsonl")
        assert bool(r) is False
        assert list(r.iter_events()) == []
        assert r.count() == 0

    def test_roundtrip(self, tmp_path):
        p = tmp_path / "n.jsonl"
        evts = [_evt("e1", "first"), _evt("e2", "second")]
        with NewsArchiveWriter(p) as w:
            assert w.append(evts) == 2
        r = NewsArchiveReader(p)
        out = list(r.iter_events())
        assert len(out) == 2
        assert out[0].title == "first"

    def test_since_until_filter(self, tmp_path):
        p = tmp_path / "n.jsonl"
        now = datetime.now(tz=timezone.utc)
        evts = [
            _evt("e1", "a", ts=now - timedelta(hours=2)),
            _evt("e2", "b", ts=now - timedelta(hours=1)),
            _evt("e3", "c", ts=now),
        ]
        with NewsArchiveWriter(p) as w:
            w.append(evts)
        r = NewsArchiveReader(p)
        subset = list(r.iter_events(since=now - timedelta(minutes=90)))
        assert [e.title for e in subset] == ["b", "c"]
        upper = list(r.iter_events(until=now - timedelta(minutes=30)))
        assert [e.title for e in upper] == ["a", "b"]

    def test_max_events(self, tmp_path):
        p = tmp_path / "n.jsonl"
        with NewsArchiveWriter(p) as w:
            w.append([_evt(f"e{i}", str(i)) for i in range(10)])
        r = NewsArchiveReader(p)
        assert len(list(r.iter_events(max_events=3))) == 3

    def test_count(self, tmp_path):
        p = tmp_path / "n.jsonl"
        with NewsArchiveWriter(p) as w:
            w.append([_evt("e1", "x")])
            w.append([_evt("e2", "y")])
        assert NewsArchiveReader(p).count() == 2

    def test_malformed_line_skipped(self, tmp_path):
        p = tmp_path / "n.jsonl"
        with NewsArchiveWriter(p) as w:
            w.append([_evt("e1", "ok")])
        # Corrupt the file: prepend a junk line
        content = p.read_text(encoding="utf-8")
        p.write_text("not-json\n" + content, encoding="utf-8")
        out = list(NewsArchiveReader(p).iter_events())
        assert len(out) == 1
        assert out[0].title == "ok"
