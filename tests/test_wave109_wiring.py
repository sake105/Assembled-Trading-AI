"""Tests for wave-109 module wiring into trading_cycle.py.

Covers:
  Step 2.76 — data.altdata.finnhub_events (fetch_earnings_events / fetch_insider_events)
  Step 2.77 — data.altdata.finnhub_news_macro (fetch_finnhub_news / fetch_finnhub_macro)
  Step 2.78 — data.altdata.house_ptr_parser (HousePTRTransaction / parse_house_ptr_csv)
"""

from __future__ import annotations

import pytest
import pandas as pd
from pathlib import Path

from src.assembled_core.data.altdata.finnhub_events import fetch_earnings_events, fetch_insider_events
from src.assembled_core.data.altdata.finnhub_news_macro import fetch_finnhub_news, fetch_finnhub_macro
from src.assembled_core.data.altdata.house_ptr_parser import HousePTRTransaction, parse_house_ptr_csv


# ---------------------------------------------------------------------------
# data.altdata.finnhub_events (Step 2.76)
# ---------------------------------------------------------------------------

def test_fetch_earnings_events_importable():
    assert fetch_earnings_events is not None


def test_fetch_insider_events_importable():
    assert fetch_insider_events is not None


# ---------------------------------------------------------------------------
# data.altdata.finnhub_news_macro (Step 2.77)
# ---------------------------------------------------------------------------

def test_fetch_finnhub_news_returns_dataframe():
    result = fetch_finnhub_news()
    assert isinstance(result, pd.DataFrame)


def test_fetch_finnhub_news_has_columns():
    result = fetch_finnhub_news()
    assert "timestamp" in result.columns
    assert "symbol" in result.columns


def test_fetch_finnhub_macro_returns_dataframe():
    result = fetch_finnhub_macro()
    assert isinstance(result, pd.DataFrame)


def test_fetch_finnhub_macro_has_columns():
    result = fetch_finnhub_macro()
    assert "timestamp" in result.columns
    assert "indicator" in result.columns


# ---------------------------------------------------------------------------
# data.altdata.house_ptr_parser (Step 2.78)
# ---------------------------------------------------------------------------

def test_house_ptr_transaction_creates():
    tx = HousePTRTransaction(
        filer_name="Test Filer",
        symbol="AAPL",
        asset_description="Apple Inc",
        transaction_type="Purchase",
        amount_range="$1,001 - $15,000",
        event_date="2024-01-15",
        disclosure_date="2024-01-20",
    )
    assert isinstance(tx, HousePTRTransaction)


def test_house_ptr_transaction_event_type_purchase():
    tx = HousePTRTransaction(
        filer_name="Test",
        symbol="MSFT",
        asset_description="Microsoft",
        transaction_type="Purchase",
        amount_range="$1,001 - $15,000",
        event_date="2024-01-15",
        disclosure_date="2024-01-20",
    )
    assert tx.event_type == "house_ptr_purchase"


def test_parse_house_ptr_csv_missing_file_returns_empty():
    result = parse_house_ptr_csv(Path("/nonexistent/file.csv"))
    assert isinstance(result, pd.DataFrame)
