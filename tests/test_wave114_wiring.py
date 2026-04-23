"""Tests for wave-114 module wiring into trading_cycle.py.

Covers:
  Step 2.91 — data.shipping.contract (normalize_shipping_releases)
  Step 2.92 — data.snapshot (compute_price_panel_snapshot_id)
  Step 2.93 — data.sources.alphavantage_source (fetch_prices_alphavantage)
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.data.shipping.contract import normalize_shipping_releases
from src.assembled_core.data.snapshot import compute_price_panel_snapshot_id
from src.assembled_core.data.sources.alphavantage_source import fetch_prices_alphavantage


# ---------------------------------------------------------------------------
# data.shipping.contract (Step 2.91)
# ---------------------------------------------------------------------------

def test_normalize_shipping_releases_importable():
    assert normalize_shipping_releases is not None


def test_normalize_shipping_releases_missing_cols_raises():
    with pytest.raises(ValueError):
        normalize_shipping_releases(pd.DataFrame({"foo": [1]}))


def test_normalize_shipping_releases_empty_valid_df():
    cols = ["series_id", "release_ts", "value", "available_ts"]
    result = normalize_shipping_releases(pd.DataFrame(columns=cols))
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# data.snapshot (Step 2.92)
# ---------------------------------------------------------------------------

def test_compute_price_panel_snapshot_id_importable():
    assert compute_price_panel_snapshot_id is not None


def test_compute_price_panel_snapshot_id_empty_df():
    df = pd.DataFrame(columns=["timestamp", "symbol", "close"])
    result = compute_price_panel_snapshot_id(df)
    assert isinstance(result, str)
    assert len(result) == 64


def test_compute_price_panel_snapshot_id_missing_cols_raises():
    with pytest.raises(ValueError):
        compute_price_panel_snapshot_id(pd.DataFrame())


def test_compute_price_panel_snapshot_id_deterministic():
    df = pd.DataFrame(columns=["timestamp", "symbol", "close"])
    r1 = compute_price_panel_snapshot_id(df)
    r2 = compute_price_panel_snapshot_id(df)
    assert r1 == r2


# ---------------------------------------------------------------------------
# data.sources.alphavantage_source (Step 2.93)
# ---------------------------------------------------------------------------

def test_fetch_prices_alphavantage_importable():
    assert fetch_prices_alphavantage is not None


def test_fetch_prices_alphavantage_empty_symbols():
    result = fetch_prices_alphavantage([], "2024-01-01", "2024-06-01")
    assert isinstance(result, pd.DataFrame)
