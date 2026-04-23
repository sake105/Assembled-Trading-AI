"""Tests for wave-112 module wiring into trading_cycle.py.

Covers:
  Step 2.85 — data.ledger_store (LedgerStore)
  Step 2.86 — data.macro.contract (normalize_macro_releases)
  Step 2.87 — data.news.contract (normalize_news_events)
"""

from __future__ import annotations

import pytest
import pandas as pd
import tempfile
from pathlib import Path

from src.assembled_core.data.ledger_store import LedgerStore
from src.assembled_core.data.macro.contract import normalize_macro_releases
from src.assembled_core.data.news.contract import normalize_news_events


# ---------------------------------------------------------------------------
# data.ledger_store (Step 2.85)
# ---------------------------------------------------------------------------

def test_ledger_store_importable():
    assert LedgerStore is not None


def test_ledger_store_creates_with_temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_ledger.db"
        store = LedgerStore(db_path=db_path, initial_cash=50_000.0)
        assert store.db_path == db_path


def test_ledger_store_initial_cash():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = LedgerStore(db_path=db_path, initial_cash=75_000.0)
        assert store._initial_cash == 75_000.0


# ---------------------------------------------------------------------------
# data.macro.contract (Step 2.86)
# ---------------------------------------------------------------------------

def test_normalize_macro_releases_importable():
    assert normalize_macro_releases is not None


def test_normalize_macro_releases_missing_cols_raises():
    with pytest.raises(ValueError):
        normalize_macro_releases(pd.DataFrame({"foo": [1]}))


def test_normalize_macro_releases_empty_valid_df():
    cols = ["series_id", "release_ts", "value", "available_ts"]
    result = normalize_macro_releases(pd.DataFrame(columns=cols))
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# data.news.contract (Step 2.87)
# ---------------------------------------------------------------------------

def test_normalize_news_events_importable():
    assert normalize_news_events is not None


def test_normalize_news_events_missing_cols_raises():
    with pytest.raises(ValueError):
        normalize_news_events(pd.DataFrame({"foo": [1]}))


def test_normalize_news_events_missing_identifier_raises():
    with pytest.raises(ValueError, match="identifier"):
        normalize_news_events(pd.DataFrame({"publish_ts": ["2024-01-01"], "source": ["reuters"]}))
