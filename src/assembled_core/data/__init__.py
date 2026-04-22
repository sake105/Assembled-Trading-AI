"""Data ingestion, storage, quality, versioning, streaming and alt-data modules.

Public surface aggregates previously orphaned modules (wired 2026-04-22).
"""

from __future__ import annotations

from src.assembled_core.data.cost_model_policy import (
    compute_adv_usd,
    compute_cost_drag_per_period,
    estimate_rebalance_cost_fraction,
    get_effective_cost_params,
    load_cost_tiers,
)
from src.assembled_core.data.data_versioning import (
    compute_data_hash,
    create_lineage_record,
)
from src.assembled_core.data.freshness_monitor import (
    FreshnessMonitor,
    SourceFreshness,
    detect_stale_features,
)
from src.assembled_core.data.quality_checks import (
    QualityCheckResult,
    check_panel_quality,
    check_price_quality,
)
from src.assembled_core.data.realism_meta import (
    build_realism_label,
    build_realism_label_from_policy,
)
from src.assembled_core.data.synthetic_generator import (
    generate_crisis_returns,
    generate_normal_returns,
)
from src.assembled_core.data.news_ingest import (
    load_news_sample,
    normalize_news,
)
from src.assembled_core.data.panel_store import (  # noqa: F401
    load_price_panel_parquet,
    panel_exists,
    panel_path,
    store_price_panel_parquet,
)
from src.assembled_core.data.news.contract import (  # noqa: F401
    filter_news_pit,
    normalize_news_events,
)
from src.assembled_core.data import macro as macro  # noqa: F401
from src.assembled_core.data import shipping as shipping  # noqa: F401

__all__ = [
    "compute_adv_usd",
    "compute_cost_drag_per_period",
    "estimate_rebalance_cost_fraction",
    "get_effective_cost_params",
    "load_cost_tiers",
    "compute_data_hash",
    "create_lineage_record",
    "FreshnessMonitor",
    "SourceFreshness",
    "detect_stale_features",
    "QualityCheckResult",
    "check_panel_quality",
    "check_price_quality",
    "build_realism_label",
    "build_realism_label_from_policy",
    "generate_crisis_returns",
    "generate_normal_returns",
    "load_news_sample",
    "normalize_news",
]
