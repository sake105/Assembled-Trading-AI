"""LLM-RAG news reasoning pipeline.

Retrieval-Augmented Generation for news impact analysis:
  1. Embed incoming news headlines with sentence-transformers.
  2. Store/retrieve from Qdrant vector DB.
  3. Retrieve top-k similar historical events with their outcomes.
  4. Pass retrieved context + new event to Claude API for structured reasoning.

Graceful degradation:
  - Missing sentence-transformers → TF-IDF fallback embedder.
  - Missing Qdrant → in-memory cosine-similarity store.
  - Missing Anthropic SDK → return embedding-only similarity scores without LLM reasoning.

Usage::

    from assembled_core.intel.news_rag import NewsRAG

    rag = NewsRAG()
    rag.ingest(headline="Fed raises rates 50bps", ticker="SPY", date="2024-06-12",
               outcome_return=0.012)

    result = rag.query("Federal Reserve surprises with emergency cut")
    print(result.reasoning)
    print(result.predicted_direction)  # "bullish" / "bearish" / "neutral"
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---- optional dependency detection ----

_SENTENCE_TRANSFORMERS = False
_QDRANT = False
_ANTHROPIC = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import]

    _SENTENCE_TRANSFORMERS = True
except ImportError:
    pass

try:
    from qdrant_client import QdrantClient  # type: ignore[import]
    from qdrant_client.models import (  # type: ignore[import]
        Distance,
        PointStruct,
        VectorParams,
    )

    _QDRANT = True
except ImportError:
    pass

try:
    import anthropic  # type: ignore[import]  # noqa: F401

    _ANTHROPIC = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class NewsRecord:
    """A stored news event with its embedding and observed market outcome."""

    record_id: str
    headline: str
    ticker: str
    date: str
    outcome_return: float  # observed next-day return after this news
    embedding: list[float] = field(default_factory=list)


@dataclass
class RAGResult:
    """Result of a RAG query for a new news headline."""

    query_headline: str
    retrieved: list[NewsRecord]  # top-k similar historical events
    similarity_scores: list[float]
    predicted_direction: str  # "bullish" / "bearish" / "neutral"
    confidence: float  # 0-1
    reasoning: str  # LLM-generated or rule-based
    backend: str  # "llm" / "embedding_only"


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class _TFIDFEmbedder:
    """Minimal TF-IDF fallback embedder when sentence-transformers is absent."""

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim
        self._vocab: dict[str, int] = {}

    def _tokenise(self, text: str) -> list[str]:
        return text.lower().split()

    def _hash_token(self, tok: str) -> int:
        return (
            int(hashlib.md5(tok.encode(), usedforsecurity=False).hexdigest(), 16)
            % self._dim
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), self._dim), dtype=float)
        for i, text in enumerate(texts):
            for tok in self._tokenise(text):
                result[i, self._hash_token(tok)] += 1.0
            norm = float(np.linalg.norm(result[i]))
            if norm > 1e-9:
                result[i] /= norm
        return result


# ---------------------------------------------------------------------------
# In-memory vector store
# ---------------------------------------------------------------------------


class _MemoryVectorStore:
    """Cosine-similarity in-memory store for when Qdrant is unavailable."""

    def __init__(self) -> None:
        self._records: list[NewsRecord] = []
        self._embeddings: list[np.ndarray] = []

    def add(self, record: NewsRecord, embedding: np.ndarray) -> None:
        self._records.append(record)
        self._embeddings.append(embedding / max(float(np.linalg.norm(embedding)), 1e-9))

    def search(
        self, query_emb: np.ndarray, top_k: int = 5
    ) -> list[tuple[NewsRecord, float]]:
        if not self._records:
            return []
        q = query_emb / max(float(np.linalg.norm(query_emb)), 1e-9)
        sims = [float(np.dot(q, e)) for e in self._embeddings]
        ranked = sorted(zip(sims, self._records), key=lambda x: -x[0])
        return [(r, s) for s, r in ranked[:top_k]]

    def __len__(self) -> int:
        return len(self._records)


# ---------------------------------------------------------------------------
# NewsRAG
# ---------------------------------------------------------------------------


class NewsRAG:
    """Retrieval-Augmented Generation for news impact analysis.

    Args:
        model_name: sentence-transformers model (default: all-MiniLM-L6-v2).
        qdrant_host: Qdrant server host (None = in-memory).
        qdrant_port: Qdrant port.
        collection_name: Qdrant collection name.
        anthropic_model: Claude model to use for reasoning.
        top_k: Number of similar events to retrieve.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        qdrant_host: str | None = None,
        qdrant_port: int = 6333,
        collection_name: str = "news_events",
        anthropic_model: str = "claude-haiku-4-5-20251001",
        top_k: int = 5,
    ) -> None:
        self._top_k = top_k
        self._anthropic_model = anthropic_model
        self._collection = collection_name

        # Embedder
        if _SENTENCE_TRANSFORMERS:
            try:
                self._embedder = SentenceTransformer(model_name)
                self._embed_dim = self._embedder.get_sentence_embedding_dimension()
                self._embed_backend = "sentence_transformers"
            except Exception as exc:
                logger.warning(
                    "[NewsRAG] SentenceTransformer failed (%s), using TF-IDF", exc
                )
                self._embedder = _TFIDFEmbedder()  # type: ignore[assignment]
                self._embed_dim = 128
                self._embed_backend = "tfidf"
        else:
            self._embedder = _TFIDFEmbedder()  # type: ignore[assignment]
            self._embed_dim = 128
            self._embed_backend = "tfidf"

        # Vector store
        # B4-IN-08 HIGH fix (R5): use get-or-create pattern instead of
        # recreate_collection (which DROPS existing data on every restart).
        # The docstring promises "RAG retrieval over historical events" — we
        # need to PRESERVE the historical corpus across process restarts.
        # Use bootstrap() method for explicit opt-in collection recreation.
        self._qdrant_client: Any = None
        if _QDRANT and qdrant_host:
            try:
                self._qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
                # Check if collection exists; create only if missing.
                try:
                    existing = self._qdrant_client.get_collection(
                        collection_name=collection_name
                    )
                    # Validate dimension matches; if not, log warning but preserve data
                    cfg_dim = getattr(getattr(existing, "config", None), "params", None)
                    if cfg_dim is not None and hasattr(cfg_dim, "vectors"):
                        existing_dim = getattr(cfg_dim.vectors, "size", None)
                        if existing_dim and existing_dim != self._embed_dim:
                            logger.warning(
                                "[NewsRAG] Qdrant collection %s exists with dim=%d "
                                "but embedder uses dim=%d. Preserving collection — "
                                "call bootstrap() explicitly to recreate.",
                                collection_name,
                                existing_dim,
                                self._embed_dim,
                            )
                    logger.info(
                        "[NewsRAG] Qdrant collection %s already exists, preserving historical corpus",
                        collection_name,
                    )
                except Exception:
                    # Collection missing → create it
                    self._qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=self._embed_dim, distance=Distance.COSINE
                        ),
                    )
                    logger.info(
                        "[NewsRAG] Created new Qdrant collection %s (dim=%d)",
                        collection_name,
                        self._embed_dim,
                    )
                self._store_backend = "qdrant"
                self._collection_name = collection_name
            except Exception as exc:
                logger.warning(
                    "[NewsRAG] Qdrant unavailable (%s), using in-memory store", exc
                )
                self._qdrant_client = None
                self._store_backend = "memory"
        else:
            self._store_backend = "memory"

        self._mem_store = _MemoryVectorStore()
        self._qdrant_point_counter = 0

        # Anthropic client
        self._anthropic_client: Any = None
        if _ANTHROPIC:
            try:
                import anthropic

                self._anthropic_client = anthropic.Anthropic()
            except Exception:
                pass

    def bootstrap(self, *, force: bool = False) -> None:
        """Explicitly (re)create the Qdrant collection. DESTRUCTIVE.

        B4-IN-08 R5: previously the constructor called recreate_collection
        which DROPPED data on every restart. The constructor now uses a
        get-or-create pattern. Use bootstrap(force=True) explicitly when
        you want to wipe and reinitialize — e.g. during a schema migration
        or embedder-dim change.

        Args:
            force: Required to actually recreate. Default False is a safety
                net — bootstrap() without force does nothing.
        """
        if not force:
            logger.warning(
                "[NewsRAG] bootstrap() called without force=True — no-op. "
                "Pass force=True to recreate and wipe the collection."
            )
            return
        if self._qdrant_client is None:
            logger.warning("[NewsRAG] bootstrap: no Qdrant client, nothing to recreate")
            return
        collection_name = getattr(self, "_collection_name", "news_rag")
        logger.warning(
            "[NewsRAG] bootstrap(force=True): RECREATING collection %s — historical data will be LOST",
            collection_name,
        )
        self._qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self._embed_dim, distance=Distance.COSINE),
        )

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._embed_backend == "sentence_transformers":
            return self._embedder.encode(texts, show_progress_bar=False)  # type: ignore[union-attr]
        return self._embedder.encode(texts)  # type: ignore[union-attr]

    def ingest(
        self,
        headline: str,
        ticker: str,
        date: str,
        outcome_return: float,
        record_id: str | None = None,
    ) -> None:
        """Store a news event with its observed market outcome.

        Args:
            headline: News headline text.
            ticker: Primary affected ticker.
            date: Event date (ISO string, e.g. "2024-06-12").
            outcome_return: Observed next-day return (decimal).
            record_id: Unique ID; auto-generated from content if None.
        """
        if record_id is None:
            record_id = hashlib.md5(
                f"{date}:{ticker}:{headline}".encode(), usedforsecurity=False
            ).hexdigest()[:16]

        emb = self._embed([headline])[0]

        record = NewsRecord(
            record_id=record_id,
            headline=headline,
            ticker=ticker,
            date=date,
            outcome_return=outcome_return,
            embedding=emb.tolist(),
        )

        if self._qdrant_client is not None:
            try:
                payload = {
                    "headline": headline,
                    "ticker": ticker,
                    "date": date,
                    "outcome_return": outcome_return,
                    "record_id": record_id,
                }
                self._qdrant_point_counter += 1
                self._qdrant_client.upsert(
                    collection_name=self._collection,
                    points=[
                        PointStruct(
                            id=self._qdrant_point_counter,
                            vector=emb.tolist(),
                            payload=payload,
                        )
                    ],
                )
            except Exception as exc:
                logger.debug("[NewsRAG] Qdrant upsert failed: %s", exc)
                self._mem_store.add(record, emb)
        else:
            self._mem_store.add(record, emb)

    def query(self, headline: str, ticker: str = "") -> RAGResult:
        """Find similar historical news events and reason about likely impact.

        Args:
            headline: New (unseen) news headline to analyse.
            ticker: Optional ticker context.

        Returns:
            RAGResult with retrieved events, predicted direction, and reasoning.
        """
        emb = self._embed([headline])[0]

        # Retrieve similar events
        if self._qdrant_client is not None:
            retrieved, scores = self._qdrant_search(emb)
        else:
            raw = self._mem_store.search(emb, top_k=self._top_k)
            retrieved = [r for r, _ in raw]
            scores = [s for _, s in raw]

        # Direction prediction from retrieved outcomes
        if retrieved:
            avg_outcome = float(np.mean([r.outcome_return for r in retrieved]))
            if avg_outcome > 0.002:
                direction = "bullish"
                confidence = min(0.9, 0.5 + abs(avg_outcome) * 20)
            elif avg_outcome < -0.002:
                direction = "bearish"
                confidence = min(0.9, 0.5 + abs(avg_outcome) * 20)
            else:
                direction = "neutral"
                confidence = 0.4
        else:
            direction, confidence = "neutral", 0.3

        # LLM reasoning
        reasoning = ""
        backend = "embedding_only"
        if self._anthropic_client is not None and retrieved:
            reasoning = self._llm_reason(headline, ticker, retrieved, scores)
            backend = "llm"
        elif retrieved:
            examples = "; ".join(
                f"'{r.headline}' → {r.outcome_return:+.2%}" for r in retrieved[:3]
            )
            reasoning = (
                f"Top-{len(retrieved)} similar events: {examples}. "
                f"Average outcome: {avg_outcome:+.3%}. "
                f"Predicted direction: {direction} (confidence {confidence:.0%})."
            )

        return RAGResult(
            query_headline=headline,
            retrieved=retrieved,
            similarity_scores=[round(s, 4) for s in scores],
            predicted_direction=direction,
            confidence=round(confidence, 3),
            reasoning=reasoning,
            backend=backend,
        )

    def _qdrant_search(self, emb: np.ndarray) -> tuple[list[NewsRecord], list[float]]:
        try:
            hits = self._qdrant_client.search(
                collection_name=self._collection,
                query_vector=emb.tolist(),
                limit=self._top_k,
            )
            records = []
            scores = []
            for hit in hits:
                p = hit.payload
                records.append(
                    NewsRecord(
                        record_id=p.get("record_id", ""),
                        headline=p.get("headline", ""),
                        ticker=p.get("ticker", ""),
                        date=p.get("date", ""),
                        outcome_return=float(p.get("outcome_return", 0.0)),
                    )
                )
                scores.append(float(hit.score))
            return records, scores
        except Exception as exc:
            logger.debug("[NewsRAG] Qdrant search failed: %s", exc)
            return [], []

    def _llm_reason(
        self,
        headline: str,
        ticker: str,
        retrieved: list[NewsRecord],
        scores: list[float],
    ) -> str:
        """Ask Claude to reason about this news given historical precedents."""
        context_lines = [
            f'- [{r.date}] "{r.headline}" (similarity {s:.2f}) → outcome: {r.outcome_return:+.2%}'
            for r, s in zip(retrieved, scores)
        ]
        context = "\n".join(context_lines)

        prompt = (
            f"You are a quantitative analyst. A new news headline has arrived:\n\n"
            f'Headline: "{headline}"\n'
            f"Ticker: {ticker or 'unspecified'}\n\n"
            f"Most similar historical events (retrieved by embedding similarity):\n{context}\n\n"
            f"Based on the historical precedents, briefly assess:\n"
            f"1. Likely market direction (bullish/bearish/neutral) and confidence.\n"
            f"2. Key risk factors to watch.\n"
            f"Keep response under 150 words."
        )

        try:
            msg = self._anthropic_client.messages.create(
                model=self._anthropic_model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as exc:
            logger.debug("[NewsRAG] LLM call failed: %s", exc)
            return ""

    @property
    def n_stored(self) -> int:
        """Number of stored news events."""
        if self._qdrant_client is not None:
            try:
                info = self._qdrant_client.get_collection(self._collection)
                return int(info.points_count or 0)
            except Exception as _exc:
                logger.debug("[NewsRAG] get_collection failed: %s", _exc)
        return len(self._mem_store)
