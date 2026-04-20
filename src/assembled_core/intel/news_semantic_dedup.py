"""Semantic deduplication (gated).

Two backends, selected at construction:

* `enabled=False` (default) — uses a cheap lexical fallback: overlap of
  bag-of-words on the title. No external dependencies. Useful as a safety
  net when sentence-transformers is unavailable (CI, lean installs).

* `enabled=True` — attempts to import `sentence_transformers`. If unavailable,
  raises ImportError. Uses cosine similarity on the title embedding.

The API is deliberately narrow: `is_duplicate(event, threshold) -> bool`.
State is retained per instance; callers can `prune(now, retention_hours)`.

Never enable by default in this repo — the import cost is non-trivial and
not all CI images carry torch.
"""

from __future__ import annotations

import logging
import math
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]+")
_STOP = frozenset({
    "the", "a", "an", "of", "to", "in", "on", "for", "with",
    "and", "or", "but", "is", "are", "was", "were", "be", "by",
    "as", "at", "from", "that", "this", "it", "its",
})


@dataclass
class _Entry:
    ts: datetime
    title: str
    tokens: frozenset[str]
    embedding: list[float] | None = None


def _tokenise(title: str) -> frozenset[str]:
    return frozenset(
        w.lower() for w in _WORD_RE.findall(title or "")
        if w.lower() not in _STOP and len(w) > 2
    )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _cosine(u: list[float], v: list[float]) -> float:
    if not u or not v or len(u) != len(v):
        return 0.0
    dot = sum(ui * vi for ui, vi in zip(u, v))
    nu = math.sqrt(sum(ui * ui for ui in u))
    nv = math.sqrt(sum(vi * vi for vi in v))
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (nu * nv)


class SemanticDedup:
    """Gated semantic dedup with lexical fallback."""

    def __init__(
        self,
        enabled: bool = False,
        model_name: str = "all-MiniLM-L6-v2",
        retention_hours: float = 12.0,
        max_entries: int = 5000,
    ) -> None:
        self._enabled = enabled
        self._retention = timedelta(hours=retention_hours)
        self._entries: deque[_Entry] = deque(maxlen=max_entries)
        self._model = None
        if enabled:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(model_name)
            except Exception as exc:  # pragma: no cover — depends on env
                logger.warning(
                    "[WARN] SemanticDedup: sentence_transformers unavailable (%s); "
                    "falling back to lexical",
                    exc,
                )
                self._model = None

    @property
    def backend(self) -> str:
        return "semantic" if self._model is not None else "lexical"

    def _embed(self, title: str) -> list[float] | None:
        if self._model is None:
            return None
        try:
            vec = self._model.encode(title, convert_to_numpy=False)
            return [float(x) for x in vec]
        except Exception as exc:  # pragma: no cover
            logger.debug("[SKIP] semantic embed failed: %s", exc)
            return None

    def is_duplicate(
        self,
        event,
        threshold: float = 0.85,
        now: datetime | None = None,
    ) -> bool:
        """Return True if a recent similar title is in the buffer, then record."""
        if now is None:
            now = datetime.now(tz=timezone.utc)
        title = getattr(event, "title", "") or ""
        if not title.strip():
            return False

        tokens = _tokenise(title)
        emb = self._embed(title)

        # compare against existing entries after pruning
        cutoff = now - self._retention
        while self._entries and self._entries[0].ts < cutoff:
            self._entries.popleft()

        for entry in self._entries:
            if emb is not None and entry.embedding is not None:
                score = _cosine(emb, entry.embedding)
            else:
                score = _jaccard(tokens, entry.tokens)
            if score >= threshold:
                return True

        self._entries.append(_Entry(ts=now, title=title, tokens=tokens, embedding=emb))
        return False

    def prune(self, now: datetime | None = None) -> int:
        if now is None:
            now = datetime.now(tz=timezone.utc)
        cutoff = now - self._retention
        dropped = 0
        while self._entries and self._entries[0].ts < cutoff:
            self._entries.popleft()
            dropped += 1
        return dropped

    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()


__all__ = ["SemanticDedup"]
