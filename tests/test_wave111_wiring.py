"""Tests for wave-111 module wiring into trading_cycle.py.

Covers:
  Step 2.82 — data.altdata.web_scraping (WebScrapingConfig / compute_job_posting_features)
  Step 2.83 — data.calendar (calendar_mode / is_trading_day_safe)
  Step 2.84 — data.data_source (get_price_data_source)
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.data.altdata.web_scraping import WebScrapingConfig, compute_job_posting_features
from src.assembled_core.data.calendar import calendar_mode, is_trading_day_safe
from src.assembled_core.data.data_source import get_price_data_source


# ---------------------------------------------------------------------------
# data.altdata.web_scraping (Step 2.82)
# ---------------------------------------------------------------------------

def test_web_scraping_config_creates():
    cfg = WebScrapingConfig()
    assert isinstance(cfg, WebScrapingConfig)


def test_web_scraping_config_defaults():
    cfg = WebScrapingConfig()
    assert cfg.trend_window_days == 30
    assert cfg.processing_lag_days == 1


def test_compute_job_posting_features_importable():
    assert compute_job_posting_features is not None


def test_compute_job_posting_features_empty_df():
    result = compute_job_posting_features(pd.DataFrame(), as_of="2024-06-01")
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# data.calendar (Step 2.83)
# ---------------------------------------------------------------------------

def test_calendar_mode_returns_string():
    result = calendar_mode()
    assert isinstance(result, str)


def test_calendar_mode_valid_value():
    result = calendar_mode()
    assert result in ("nyse", "fallback", "simple")


def test_is_trading_day_safe_returns_bool():
    result = is_trading_day_safe("2024-06-03")  # Monday
    assert isinstance(result, bool)


def test_is_trading_day_safe_weekend():
    result = is_trading_day_safe("2024-06-01")  # Saturday
    assert result is False


# ---------------------------------------------------------------------------
# data.data_source (Step 2.84)
# ---------------------------------------------------------------------------

def test_get_price_data_source_importable():
    assert get_price_data_source is not None


def test_get_price_data_source_external_forbidden():
    from src.assembled_core.config import Settings
    settings = Settings()
    with pytest.raises(ValueError, match="forbidden"):
        get_price_data_source(settings, data_source="yahoo", allow_external_fetch=False)
