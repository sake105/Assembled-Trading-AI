"""Tests für yfinance_cache_loader."""

from __future__ import annotations

import pandas as pd
import pytest

from erweiterung.altdata.yfinance_cache_loader import (
    list_cached_symbols,
    load_symbol_parquet,
    load_universe_panel,
    panel_coverage_report,
)

CACHE_DIR = "data/cache/yfinance"


def _cache_available() -> bool:
    from pathlib import Path

    return Path(CACHE_DIR).exists() and any(Path(CACHE_DIR).glob("*.parquet"))


pytestmark = pytest.mark.skipif(
    not _cache_available(), reason="yfinance cache not present"
)


def test_list_cached_symbols_returns_sorted():
    syms = list_cached_symbols(CACHE_DIR)
    assert len(syms) > 0
    assert syms == sorted(syms)
    assert all(isinstance(s, str) for s in syms)


def test_load_symbol_aapl_ok():
    df = load_symbol_parquet(CACHE_DIR, "AAPL")
    assert "date" in df.columns
    assert "close" in df.columns
    assert "symbol" in df.columns
    assert (df["symbol"] == "AAPL").all()
    assert len(df) > 100


def test_load_symbol_unknown_raises():
    with pytest.raises(FileNotFoundError):
        load_symbol_parquet(CACHE_DIR, "DEFINITELY_NOT_A_TICKER_XYZ")


def test_load_universe_panel_basic():
    panel = load_universe_panel(
        CACHE_DIR, ["AAPL", "MSFT", "NVDA"], require_min_rows=100
    )
    assert set(panel["symbol"].unique()) == {"AAPL", "MSFT", "NVDA"}
    assert "return" in panel.columns
    assert panel.sort_values(["symbol", "date"]).equals(panel)


def test_load_universe_panel_skips_missing():
    panel = load_universe_panel(
        CACHE_DIR,
        ["AAPL", "DEFINITELY_NOT_A_TICKER_XYZ"],
        skip_missing=True,
        require_min_rows=100,
    )
    assert set(panel["symbol"].unique()) == {"AAPL"}
    assert "DEFINITELY_NOT_A_TICKER_XYZ" in panel.attrs.get("skipped_symbols", [])


def test_panel_coverage_report():
    panel = load_universe_panel(CACHE_DIR, ["AAPL", "MSFT"], require_min_rows=100)
    rep = panel_coverage_report(panel)
    assert set(rep.columns) == {
        "symbol",
        "n_rows",
        "date_min",
        "date_max",
        "nan_close_pct",
    }
    assert len(rep) == 2
    assert (rep["n_rows"] > 0).all()


def test_date_range_filter():
    df = load_symbol_parquet(CACHE_DIR, "AAPL", start="2023-01-01", end="2023-12-31")
    if not df.empty:
        assert df["date"].min() >= pd.Timestamp("2023-01-01", tz="UTC")
        assert df["date"].max() <= pd.Timestamp("2023-12-31", tz="UTC")
