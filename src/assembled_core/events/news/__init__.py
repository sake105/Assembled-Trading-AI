"""News v1 pipeline (MVP, free & robust).

This package implements a minimal but robust NEWS v1 pipeline:

- Fetch from RSS (and optionally GDELT)
- Normalize into a clean NewsEvent schema
- Dedupe events
- Compute NewsHealth
- Emit JSON artifacts under output/intel/news/
"""

from __future__ import annotations

from .baseline import compute_version_hash, update_baseline
from .burst import compute_bursts_for_window
from .clustering import (
    build_clusters,
    enrich_clusters_with_sentiment,
    score_cluster_sentiment,
)
from .dedupe import dedupe_events
from .dedupe_store import DedupeStoreSQLite
from .emit import emit_json_artifact
from .entities import extract_countries, extract_entities
from .evidence import summarize_cluster_evidence
from .fetch_gdelt import fetch_gdelt_events, fetch_gdelt_multi_domain
from .fetch_rss import fetch_rss_feed
from .fingerprint import hamming_distance, simhash64
from .health import compute_health
from .models import NewsEvent, NewsHealth
from .normalize import canonicalize_url, normalize_raw_item, now_utc_iso, sanitize_text
from .pipeline import run_news_pipeline
from .sources import NewsSource, load_news_params, load_sources_registry
from .state import load_fetch_state, save_fetch_state
from .tfidf import build_tfidf_vectors, cosine_sparse, tokenize
from .trigger_scoring import score_triggers

__all__ = [
    "DedupeStoreSQLite",
    "NewsEvent",
    "NewsHealth",
    "NewsSource",
    "build_clusters",
    "build_tfidf_vectors",
    "canonicalize_url",
    "compute_bursts_for_window",
    "compute_health",
    "compute_version_hash",
    "cosine_sparse",
    "dedupe_events",
    "emit_json_artifact",
    "enrich_clusters_with_sentiment",
    "extract_countries",
    "extract_entities",
    "fetch_gdelt_events",
    "fetch_gdelt_multi_domain",
    "fetch_rss_feed",
    "hamming_distance",
    "load_fetch_state",
    "load_news_params",
    "load_sources_registry",
    "normalize_raw_item",
    "now_utc_iso",
    "run_news_pipeline",
    "sanitize_text",
    "save_fetch_state",
    "score_cluster_sentiment",
    "score_triggers",
    "simhash64",
    "summarize_cluster_evidence",
    "tokenize",
    "update_baseline",
]
