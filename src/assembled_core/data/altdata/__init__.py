"""Alt-data modules (insider trades, Congress trades, satellite, patents, social).

Public API aggregates parsers and feature builders for previously
orphaned alt-data modules (wired 2026-04-22).
"""

from __future__ import annotations

from src.assembled_core.data.altdata.house_ptr_parser import (
    HousePTRTransaction,
    filter_stock_transactions,
    parse_house_ptr_csv,
    to_altdata_events,
)
from src.assembled_core.data.altdata.patent_features import (
    PatentConfig,
    compute_patent_features,
)
from src.assembled_core.data.altdata.satellite_features import (  # noqa: F401
    SatelliteConfig,
    compute_nightlight_features,
    process_parking_lot_data,
    process_shipping_data,
)
from src.assembled_core.data.altdata.social_sentiment import (
    SentimentConfig,
    SymbolSentiment,
    add_sentiment_momentum,
    aggregate_daily_sentiment,
    compute_crowd_consensus,
)
from src.assembled_core.data.altdata.web_scraping import (
    WebScrapingConfig,
    compute_app_rating_features,
    compute_job_posting_features,
    compute_website_traffic_features,
)

__all__ = [
    "HousePTRTransaction",
    "parse_house_ptr_csv",
    "filter_stock_transactions",
    "to_altdata_events",
    "PatentConfig",
    "compute_patent_features",
    "SentimentConfig",
    "SymbolSentiment",
    "aggregate_daily_sentiment",
    "add_sentiment_momentum",
    "compute_crowd_consensus",
    "WebScrapingConfig",
    "compute_job_posting_features",
    "compute_app_rating_features",
    "compute_website_traffic_features",
]
