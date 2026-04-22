"""News data modules: entity linking + persistent store.

Previously contained only a docstring; wiring added 2026-04-22 so that
``data.news.entity_linking`` and ``data.news.store`` are discoverable.
"""

from __future__ import annotations

from src.assembled_core.data.news.entity_linking import link_news_to_symbols
from src.assembled_core.data.news.store import (
    list_news_partitions,
    load_news,
    load_news_parquet,
    news_partition_path,
    store_news_parquet,
)

__all__ = [
    "link_news_to_symbols",
    "news_partition_path",
    "store_news_parquet",
    "load_news_parquet",
    "list_news_partitions",
    "load_news",
]
