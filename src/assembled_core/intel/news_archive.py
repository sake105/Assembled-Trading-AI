"""JSONL archive for NewsEvents — raw append + chronological replay.

Separate from `news_replay.NewsReplayer` (which uses the PITStore). This
archive is a simple line-delimited JSON log intended for:

* lightweight persistence of every ingested NewsEvent
* offline replay in tests / research without a PITStore bring-up
* forensic inspection after the fact

Usage:
    writer = NewsArchiveWriter("output/news/2026-04-20.jsonl")
    writer.append(events)
    writer.close()

    reader = NewsArchiveReader("output/news/2026-04-20.jsonl")
    for evt in reader.iter_events(max_events=1000):
        ...

Only fields on the NewsEvent pydantic model survive the roundtrip.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from src.assembled_core.intel.models import NewsEvent

logger = logging.getLogger(__name__)

# H5: schema marker prefixed as a comment-like first JSON object so old
# readers either see the sentinel and skip it, or see it as a record with
# only `_schema_version` / `_header` keys and ignore via _deserialise.
_ARCHIVE_SCHEMA_VERSION = 1
_ARCHIVE_HEADER_MARKER = "__news_archive_header__"


def _serialise(evt: NewsEvent) -> dict[str, Any]:
    try:
        return evt.model_dump(mode="json")
    except Exception as exc:
        logger.debug("[SKIP] archive _serialise: %s", exc)
        return {}


def _deserialise(raw: dict[str, Any]) -> NewsEvent | None:
    try:
        return NewsEvent.model_validate(raw)
    except Exception as exc:
        logger.debug("[SKIP] archive _deserialise: %s", exc)
        return None


class NewsArchiveWriter:
    """Append-only JSONL writer for NewsEvents."""

    def __init__(self, path: str | Path, *, fsync: bool = True) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self._path.exists() or self._path.stat().st_size == 0
        self._fh = self._path.open("a", encoding="utf-8")
        self._fsync = bool(fsync)
        if is_new:
            header = {
                _ARCHIVE_HEADER_MARKER: True,
                "_schema_version": _ARCHIVE_SCHEMA_VERSION,
            }
            self._fh.write(json.dumps(header, ensure_ascii=False) + "\n")
            self._fh.flush()

    def append(self, events: list[NewsEvent]) -> int:
        written = 0
        for evt in events or []:
            data = _serialise(evt)
            if not data:
                continue
            # H5: write each event as a single line; flush+fsync so a crash
            # cannot leave a half-written record visible to the reader.
            self._fh.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
            written += 1
        self._fh.flush()
        if self._fsync:
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                pass
        return written

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "NewsArchiveWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class NewsArchiveReader:
    """Streaming JSONL reader."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def __bool__(self) -> bool:
        return self._path.exists()

    def iter_events(
        self,
        *,
        max_events: int | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> Iterator[NewsEvent]:
        if not self._path.exists():
            return
        count = 0
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except Exception as exc:
                    logger.debug("[SKIP] archive line: %s", exc)
                    continue
                # H5: skip schema header marker lines.
                if isinstance(raw, dict) and raw.get(_ARCHIVE_HEADER_MARKER):
                    continue
                evt = _deserialise(raw)
                if evt is None:
                    continue
                pub = evt.published_at
                if since is not None and pub < since:
                    continue
                if until is not None and pub > until:
                    continue
                yield evt
                count += 1
                if max_events is not None and count >= max_events:
                    return

    def count(self) -> int:
        if not self._path.exists():
            return 0
        n = 0
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                # H5: don't count schema header rows as events.
                try:
                    raw = json.loads(s)
                    if isinstance(raw, dict) and raw.get(_ARCHIVE_HEADER_MARKER):
                        continue
                except Exception:
                    continue
                n += 1
        return n


__all__ = ["NewsArchiveWriter", "NewsArchiveReader"]
