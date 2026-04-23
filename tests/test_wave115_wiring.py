"""Tests for wave-115 module wiring into trading_cycle.py.

Covers:
  Step 2.94 — data.sources.bls_source (fetch_bls_series)
  Step 2.95 — data.sources.cboe_source (CBOESource)
  Step 2.96 — data.sources.edgar_source (fetch_insider_trades)
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.data.sources.bls_source import fetch_bls_series
from src.assembled_core.data.sources.cboe_source import CBOESource
from src.assembled_core.data.sources.edgar_source import fetch_insider_trades


# ---------------------------------------------------------------------------
# data.sources.bls_source (Step 2.94)
# ---------------------------------------------------------------------------

def test_fetch_bls_series_importable():
    assert fetch_bls_series is not None


def test_fetch_bls_series_empty_list():
    result = fetch_bls_series([], start_year=2024, end_year=2024)
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# data.sources.cboe_source (Step 2.95)
# ---------------------------------------------------------------------------

def test_cboe_source_creates():
    source = CBOESource()
    assert isinstance(source, CBOESource)


def test_cboe_source_no_api_key():
    source = CBOESource()
    assert source.fred_api_key is None


def test_cboe_source_with_api_key():
    source = CBOESource(fred_api_key="test_key")
    assert source.fred_api_key == "test_key"


# ---------------------------------------------------------------------------
# data.sources.edgar_source (Step 2.96)
# ---------------------------------------------------------------------------

def test_fetch_insider_trades_importable():
    assert fetch_insider_trades is not None
