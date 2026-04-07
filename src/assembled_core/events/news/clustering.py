from __future__ import annotations

from typing import Dict, List

from .models import NewsEvent
from .tfidf import build_tfidf_vectors, cosine_sparse, tokenize


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _same_day(e1: NewsEvent, e2: NewsEvent) -> bool:
    return (e1.published_utc or "")[:10] == (e2.published_utc or "")[:10]


def _overlap(a: List[str], b: List[str]) -> bool:
    if not a or not b:
        return False
    sa = set(a)
    for x in b:
        if x in sa:
            return True
    return False


def build_clusters(events: List[NewsEvent], cfg: Dict) -> List[Dict]:
    """Build lightweight deterministic clusters from NewsEvents."""
    n = len(events)
    if n == 0:
        return []

    uf = UnionFind(n)
    id_to_idx: Dict[str, int] = {e.event_id: i for i, e in enumerate(events)}

    # (Step 0) Union via near_duplicate_of links
    for i, ev in enumerate(events):
        raw = ev.raw or {}
        target_id = raw.get("near_duplicate_of")
        if not target_id:
            continue
        j = id_to_idx.get(str(target_id))
        if j is not None:
            uf.union(i, j)

    # (Step 1) TF-IDF + cosine similarity unions
    if (
        bool(cfg.get("enabled", True))
        and cfg.get("algorithm", "tfidf_cosine") == "tfidf_cosine"
    ):
        similarity_threshold = float(cfg.get("similarity_threshold", 0.45))
        require_overlap = bool(cfg.get("require_overlap", True))
        same_day_only = bool(cfg.get("same_day_only", True))
        max_checks = int(cfg.get("max_pair_checks", 2000) or 0)
        if similarity_threshold > 0.0 and max_checks > 0:
            texts = [f"{e.title or ''} {e.summary or ''}".strip() for e in events]
            vectors = build_tfidf_vectors(texts)
            checks = 0
            for i in range(n):
                ei = events[i]
                vi = vectors[i]
                if not vi:
                    continue
                for j in range(i + 1, n):
                    if checks >= max_checks:
                        break
                    checks += 1
                    ej = events[j]
                    vj = vectors[j]
                    if not vj:
                        continue
                    if same_day_only and not _same_day(ei, ej):
                        continue
                    if require_overlap and not (
                        _overlap(ei.countries, ej.countries)
                        or _overlap(ei.entities, ej.entities)
                    ):
                        continue
                    sim = cosine_sparse(vi, vj)
                    if sim >= similarity_threshold:
                        uf.union(i, j)

    # Form groups
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)

    generated_utc = str(cfg.get("generated_utc") or "")
    min_cluster_size = int(cfg.get("min_cluster_size", 3) or 3)
    top_phrases_k = int(cfg.get("top_phrases_k", 8) or 0)
    top_entities_k = int(cfg.get("top_entities_k", 8) or 0)

    clusters: List[Dict] = []
    for indices in groups.values():
        if len(indices) < min_cluster_size:
            continue
        # Representative: oldest published_utc, then event_id
        rep_idx = min(
            indices,
            key=lambda k: (
                events[k].published_utc or "",
                events[k].event_id or "",
            ),
        )
        rep = events[rep_idx]
        event_ids = sorted(events[k].event_id for k in indices)

        countries_set = set()
        entities_set = set()
        for k in indices:
            countries_set.update(events[k].countries or [])
            entities_set.update(events[k].entities or [])

        countries = sorted(countries_set)
        entities = sorted(entities_set)

        # Top entities (including optional ISO2 countries)
        ent_counts: Dict[str, int] = {}
        for k in indices:
            for ent in events[k].entities or []:
                ent_counts[ent] = ent_counts.get(ent, 0) + 1
            for c in events[k].countries or []:
                ent_counts[c] = ent_counts.get(c, 0) + 1
        sorted_ents = sorted(
            ent_counts.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        top_entities = (
            [name for name, _ in sorted_ents[:top_entities_k]]
            if top_entities_k > 0
            else []
        )

        # Top phrases (2- and 3-grams) from group text
        phrase_counts: Dict[str, int] = {}
        for k in indices:
            text = f"{events[k].title or ''} {events[k].summary or ''}"
            toks = tokenize(text)
            for ngram_len in (2, 3):
                if len(toks) < ngram_len:
                    continue
                for i2 in range(len(toks) - ngram_len + 1):
                    phrase = " ".join(toks[i2 : i2 + ngram_len])
                    phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        # Optional min_count >= 2 filter
        phrase_items = [(p, c) for p, c in phrase_counts.items() if c >= 2]
        phrase_items.sort(key=lambda kv: (-kv[1], kv[0]))
        top_phrases = (
            [p for p, _ in phrase_items[:top_phrases_k]] if top_phrases_k > 0 else []
        )

        # Deterministic sample titles (up to 3)
        ordered = sorted(
            indices,
            key=lambda k: (
                events[k].published_utc or "",
                events[k].event_id or "",
            ),
        )
        sample_titles: List[str] = []
        for k in ordered[:3]:
            sample_titles.append(events[k].title)

        clusters.append(
            {
                "cluster_id": f"clu_{rep.event_id}",
                "generated_utc": generated_utc or rep.fetched_utc,
                "event_ids": event_ids,
                "representative_event_id": rep.event_id,
                "countries": countries,
                "entities": entities,
                "top_entities": top_entities,
                "top_phrases": top_phrases,
                "sample_titles": sample_titles,
            }
        )

    # Deterministic ordering of clusters
    clusters.sort(key=lambda c: c["cluster_id"])
    return clusters


# ---------------------------------------------------------------------------
# FinBERT Sentiment Integration (Plan 4.4)
# ---------------------------------------------------------------------------

import logging

_logger = logging.getLogger(__name__)

try:
    from transformers import pipeline as _hf_pipeline  # type: ignore

    _FINBERT_AVAILABLE = True
except ImportError:
    _FINBERT_AVAILABLE = False
    _hf_pipeline = None


def _load_finbert():
    """Lazy-load FinBERT sentiment pipeline."""
    if not _FINBERT_AVAILABLE:
        return None
    try:
        return _hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
    except Exception as exc:
        _logger.warning("[FinBERT] Failed to load model: %s", exc)
        return None


_finbert_pipeline = None


def score_cluster_sentiment(
    sample_titles: List[str],
    max_texts: int = 5,
) -> float:
    """Score cluster sentiment using FinBERT.

    Args:
        sample_titles: Representative titles from the cluster.
        max_texts: Max texts to score (FinBERT latency budget).

    Returns:
        Median sentiment score in [-1, 1].
        Returns 0.0 if FinBERT unavailable.
    """
    global _finbert_pipeline

    if not _FINBERT_AVAILABLE:
        return 0.0

    texts = [t for t in sample_titles[:max_texts] if t and len(t.strip()) > 10]
    if not texts:
        return 0.0

    if _finbert_pipeline is None:
        _finbert_pipeline = _load_finbert()
    if _finbert_pipeline is None:
        return 0.0

    try:
        results = _finbert_pipeline(texts)
        scores = []
        for r in results:
            label = r["label"].lower()
            score = r["score"]
            if label == "positive":
                scores.append(score)
            elif label == "negative":
                scores.append(-score)
            else:
                scores.append(0.0)

        if not scores:
            return 0.0
        scores.sort()
        n = len(scores)
        return round(scores[n // 2], 4)  # median
    except Exception as exc:
        _logger.warning("[FinBERT] Scoring failed: %s", exc)
        return 0.0


def enrich_clusters_with_sentiment(
    clusters: List[Dict],
    escalation_boost: float = 0.20,
    deescalation_dampen: float = 0.20,
) -> List[Dict]:
    """Add sentiment scores to clusters and adjust magnitude.

    - sentiment < -0.7 → escalation amplifier (+boost magnitude)
    - sentiment > 0.3 → de-escalation dampener (-dampen magnitude)

    Args:
        clusters: List of cluster dicts from build_clusters().
        escalation_boost: Magnitude increase for very negative sentiment.
        deescalation_dampen: Magnitude decrease for positive sentiment.

    Returns:
        Clusters with added ``sentiment_score`` and ``magnitude_adjustment``.
    """
    for cl in clusters:
        titles = cl.get("sample_titles", [])
        sentiment = score_cluster_sentiment(titles)
        cl["sentiment_score"] = sentiment

        if sentiment < -0.7:
            cl["magnitude_adjustment"] = escalation_boost
        elif sentiment > 0.3:
            cl["magnitude_adjustment"] = -deescalation_dampen
        else:
            cl["magnitude_adjustment"] = 0.0

    return clusters


__all__ = ["UnionFind", "build_clusters", "score_cluster_sentiment", "enrich_clusters_with_sentiment"]
