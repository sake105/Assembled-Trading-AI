"""Tests for wave-117 module wiring into trading_cycle.py.

Covers:
  Step 2.100 — data.sources.worldbank_source (fetch_worldbank_indicator)
  Step 2.101 — data.sources.yfinance_source (fetch_prices_yfinance)
  Step 5.60  — data.streaming.minute_bar_aggregator (MinuteBarAggregator)
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.data.sources.worldbank_source import fetch_worldbank_indicator
from src.assembled_core.data.sources.yfinance_source import fetch_prices_yfinance
from src.assembled_core.data.streaming.minute_bar_aggregator import MinuteBarAggregator, AggregatedBar


# ---------------------------------------------------------------------------
# data.sources.worldbank_source (Step 2.100)
# ---------------------------------------------------------------------------

def test_fetch_worldbank_indicator_importable():
    assert fetch_worldbank_indicator is not None


# ---------------------------------------------------------------------------
# data.sources.yfinance_source (Step 2.101)
# ---------------------------------------------------------------------------

def test_fetch_prices_yfinance_importable():
    assert fetch_prices_yfinance is not None


def test_fetch_prices_yfinance_empty_symbols():
    result = fetch_prices_yfinance([], "2024-01-01", "2024-06-01")
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# data.streaming.minute_bar_aggregator (Step 5.60)
# ---------------------------------------------------------------------------

def test_minute_bar_aggregator_creates():
    agg = MinuteBarAggregator()
    assert isinstance(agg, MinuteBarAggregator)


def test_minute_bar_aggregator_default_history():
    agg = MinuteBarAggregator()
    assert agg.max_history == 390


def test_minute_bar_aggregator_custom_history():
    agg = MinuteBarAggregator(max_history_minutes=60)
    assert agg.max_history == 60


def test_minute_bar_aggregator_flush_empty():
    agg = MinuteBarAggregator()
    bars = agg.flush_completed_bars()
    assert isinstance(bars, list)
    assert len(bars) == 0


def test_aggregated_bar_importable():
    assert AggregatedBar is not None
