"""News-Event-Clustering via TF-IDF + Agglomerative-Clustering.

Anwendung
---------
Aus einem Tag mit 1000 News → 30-50 distinct "Events". Jedes Event ist eine
Cluster ähnlicher Headlines. Nutzen:
- Deduplication beyond SimHash (semantically related, not just text-similar)
- Event-Impact-Aggregation (mehrere Headlines → 1 Event)
- Material-Event-Detection (Cluster mit unusual size = wichtige News)

Implementation
--------------
TF-IDF + Cosine-Distance + Agglomerative-Clustering (sklearn falls verfügbar).
Fallback: SimHash-basiertes Clustering ohne sklearn.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def cluster_news_tfidf(
    headlines: list[str],
    n_clusters: int | None = None,
    distance_threshold: float = 0.5,
) -> list[int]:
    """Agglomerative-Clustering via TF-IDF.

    Args:
        headlines: list of strings.
        n_clusters: target number; if None use distance_threshold.
        distance_threshold: max-cluster-cohesion-distance.

    Returns:
        List of cluster-labels (one per headline).
    """
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_distances
    except ImportError:
        return _fallback_cluster_via_simhash(headlines)

    if not headlines or len(headlines) < 2:
        return [0] * len(headlines)

    valid_headlines = [
        h if isinstance(h, str) and h.strip() else "EMPTY" for h in headlines
    ]
    vec = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
    try:
        X = vec.fit_transform(valid_headlines)
    except ValueError:
        return [0] * len(headlines)
    dist = cosine_distances(X)

    if n_clusters is not None:
        model = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="average"
        )
    else:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage="average",
        )
    try:
        labels = model.fit_predict(dist)
        return labels.tolist()
    except Exception as e:  # noqa: BLE001
        logger.warning("clustering failed: %s", e)
        return _fallback_cluster_via_simhash(headlines)


def _fallback_cluster_via_simhash(headlines: list[str]) -> list[int]:
    """SimHash-based fallback when sklearn missing."""
    import hashlib
    import re

    token_re = re.compile(r"\w+")

    def simhash(text: str) -> int:
        if not isinstance(text, str):
            return 0
        v = [0] * 64
        for tok in token_re.findall(text.lower()):
            h = int(hashlib.sha1(tok.encode("utf-8")).hexdigest(), 16) & (
                (1 << 64) - 1
            )  # noqa: S324
            for i in range(64):
                if (h >> i) & 1:
                    v[i] += 1
                else:
                    v[i] -= 1
        out = 0
        for i, val in enumerate(v):
            if val > 0:
                out |= 1 << i
        return out

    hashes = [simhash(h) for h in headlines]
    labels = [0] * len(headlines)
    cluster_centers: list[int] = []
    next_id = 0
    for i, h in enumerate(hashes):
        assigned = False
        for j, c in enumerate(cluster_centers):
            hd = bin(h ^ c).count("1")
            if hd <= 6:  # tight cluster
                labels[i] = j
                assigned = True
                break
        if not assigned:
            cluster_centers.append(h)
            labels[i] = next_id
            next_id += 1
    return labels


def event_clusters_per_day(
    news_df: pd.DataFrame,
    text_col: str = "headline",
    date_col: str = "date",
    distance_threshold: float = 0.5,
) -> pd.DataFrame:
    """Cluster news per day, add cluster-id column.

    Returns:
        DataFrame with added ``event_cluster_id`` (per-day cluster).
    """
    if news_df.empty:
        return news_df
    df = news_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.normalize()
    out_rows = []
    for d, group in df.groupby(date_col):
        headlines = group[text_col].fillna("").tolist()
        labels = cluster_news_tfidf(headlines, distance_threshold=distance_threshold)
        sub = group.copy()
        sub["event_cluster_id"] = [f"{d.date()}_{lbl}" for lbl in labels]
        out_rows.append(sub)
    return pd.concat(out_rows, ignore_index=True)


def event_size_distribution(
    clustered_news: pd.DataFrame, cluster_col: str = "event_cluster_id"
) -> pd.DataFrame:
    """Distribution of cluster-sizes per day.

    Anomalously large clusters = major news-events worth attention.
    """
    if clustered_news.empty:
        return pd.DataFrame()
    counts = clustered_news.groupby(cluster_col).size()
    return counts.sort_values(ascending=False).rename("n_articles").reset_index()


__all__ = [
    "cluster_news_tfidf",
    "event_clusters_per_day",
    "event_size_distribution",
]
