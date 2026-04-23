"""Tests for wave-99 module wiring into trading_cycle.py.

Covers:
  Step 2.72 — data.insider_ingest (load_insider_sample / normalize_insider)
  Step 2.73 — data.prices_ingest (validate_price_data / load_eod_prices)
  Step 2.74 — data.shipping_routes_ingest (load_shipping_sample / normalize_shipping)
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.data.insider_ingest import load_insider_sample, normalize_insider
from src.assembled_core.data.prices_ingest import validate_price_data, load_eod_prices
from src.assembled_core.data.shipping_routes_ingest import load_shipping_sample, normalize_shipping


# ---------------------------------------------------------------------------
# insider_ingest (Step 2.72)
# ---------------------------------------------------------------------------

def test_load_insider_sample_returns_dataframe():
    df = load_insider_sample()
    assert isinstance(df, pd.DataFrame)


def test_load_insider_sample_has_rows():
    df = load_insider_sample()
    assert len(df) > 0


def test_normalize_insider_importable():
    assert normalize_insider is not None


def test_normalize_insider_raises_on_empty():
    with pytest.raises((KeyError, ValueError)):
        normalize_insider(pd.DataFrame())


# ---------------------------------------------------------------------------
# prices_ingest (Step 2.73)
# ---------------------------------------------------------------------------

def test_validate_price_data_returns_dict():
    df = pd.DataFrame(columns=["symbol", "timestamp", "close"])
    result = validate_price_data(df)
    assert isinstance(result, dict)


def test_load_eod_prices_importable():
    assert load_eod_prices is not None


# ---------------------------------------------------------------------------
# shipping_routes_ingest (Step 2.74)
# ---------------------------------------------------------------------------

def test_load_shipping_sample_returns_dataframe():
    df = load_shipping_sample()
    assert isinstance(df, pd.DataFrame)


def test_load_shipping_sample_has_rows():
    df = load_shipping_sample()
    assert len(df) > 0


def test_normalize_shipping_importable():
    assert normalize_shipping is not None


def test_normalize_shipping_raises_on_empty():
    with pytest.raises((KeyError, ValueError)):
        normalize_shipping(pd.DataFrame())
