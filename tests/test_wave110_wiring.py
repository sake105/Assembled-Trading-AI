"""Tests for wave-110 module wiring into trading_cycle.py.

Covers:
  Step 2.79 — data.altdata.patent_features (PatentConfig / compute_patent_features)
  Step 2.80 — data.altdata.satellite_features (SatelliteConfig / process_parking_lot_data)
  Step 2.81 — data.altdata.social_sentiment (SentimentConfig / aggregate_daily_sentiment)
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.data.altdata.patent_features import PatentConfig, compute_patent_features
from src.assembled_core.data.altdata.satellite_features import SatelliteConfig, process_parking_lot_data
from src.assembled_core.data.altdata.social_sentiment import SentimentConfig, aggregate_daily_sentiment


# ---------------------------------------------------------------------------
# data.altdata.patent_features (Step 2.79)
# ---------------------------------------------------------------------------

def test_patent_config_creates():
    cfg = PatentConfig()
    assert isinstance(cfg, PatentConfig)


def test_patent_config_defaults():
    cfg = PatentConfig()
    assert cfg.lookback_months == 12
    assert cfg.citation_decay_years == 3.0


def test_compute_patent_features_importable():
    assert compute_patent_features is not None


def test_compute_patent_features_empty_df():
    result = compute_patent_features(pd.DataFrame(), as_of="2024-06-01")
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# data.altdata.satellite_features (Step 2.80)
# ---------------------------------------------------------------------------

def test_satellite_config_creates():
    cfg = SatelliteConfig()
    assert isinstance(cfg, SatelliteConfig)


def test_satellite_config_defaults():
    cfg = SatelliteConfig()
    assert cfg.processing_lag_days == 2
    assert cfg.trend_window_weeks == 4


def test_process_parking_lot_data_importable():
    assert process_parking_lot_data is not None


def test_process_parking_lot_data_empty_df():
    result = process_parking_lot_data(pd.DataFrame(), as_of="2024-06-01")
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# data.altdata.social_sentiment (Step 2.81)
# ---------------------------------------------------------------------------

def test_sentiment_config_creates():
    cfg = SentimentConfig()
    assert isinstance(cfg, SentimentConfig)


def test_sentiment_config_defaults():
    cfg = SentimentConfig()
    assert cfg.min_mentions == 5
    assert cfg.momentum_window == 5


def test_aggregate_daily_sentiment_importable():
    assert aggregate_daily_sentiment is not None


def test_aggregate_daily_sentiment_empty_df():
    result = aggregate_daily_sentiment(pd.DataFrame())
    assert isinstance(result, pd.DataFrame)
