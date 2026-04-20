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
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from src.assembled_core.intel.models import NewsEvent

logger = logging.getLogger(__name__)


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

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")

    def append(self, events: list[NewsEvent]) -> int:
        written = 0
        for evt in events or []:
            data = _serialise(evt)
            if not data:
                continue
            self._fh.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
            written += 1
        self._fh.flush()
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
        with self._path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())


__all__ = ["NewsArchiveWriter", "NewsArchiveReader"]
