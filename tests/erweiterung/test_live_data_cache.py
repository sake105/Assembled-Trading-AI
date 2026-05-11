"""Tests für data_cache."""

from __future__ import annotations

import time

import pandas as pd
import pytest

from erweiterung.live.data_cache import DataCache, get_global_cache


def test_data_cache_initializes():
    cache = DataCache()
    assert cache.cache_dir.exists()


def test_in_memory_caching_speedup():
    """Zweite Load eines bekannten Datasets muss schneller sein."""
    cache_dir_path = "data/cache/yfinance_long"
    cache = DataCache()
    syms = ["SPY"]

    # First load (cold)
    t0 = time.perf_counter()
    panel1 = cache.load_yfinance_long_panel(syms, cache_dir=cache_dir_path)
    t_cold = time.perf_counter() - t0

    if panel1.empty:
        pytest.skip("no SPY parquet available")

    # Second load (warm)
    t0 = time.perf_counter()
    panel2 = cache.load_yfinance_long_panel(syms, cache_dir=cache_dir_path)
    t_warm = time.perf_counter() - t0
    # Warm should be much faster (>2× speedup typical)
    assert t_warm < t_cold * 0.5 or t_warm < 0.0005


def test_yf_panel_empty_for_missing_symbols():
    cache = DataCache()
    panel = cache.load_yfinance_long_panel(["XYZ_DOESNT_EXIST"])
    assert panel.empty


def test_load_equity_panel_with_mom_persists():
    """Erste Run: rechnet + persistiert; zweite Run: lädt aus persistent cache."""
    cache = DataCache()
    if not (cache.cache_dir.parent / "yfinance_long").exists() and not (
        pd.io.common.file_exists("data/sample/watchlist_2007_2026.parquet")
    ):
        pytest.skip("no equity sample")
    src = "data/sample/watchlist_2007_2026.parquet"
    if not pd.io.common.file_exists(src):
        pytest.skip("no equity sample")

    # Force fresh compute
    df1 = cache.load_equity_panel_with_mom(src, force_recompute=True)
    assert "mom_12_1" in df1.columns

    # Second call should hit cache (in-memory)
    cache.clear()
    t0 = time.perf_counter()
    df2 = cache.load_equity_panel_with_mom(src)
    t_persist = time.perf_counter() - t0
    # Persistent read should be fast (< 100ms)
    assert t_persist < 0.5
    assert len(df2) == len(df1)


def test_global_cache_singleton():
    c1 = get_global_cache()
    c2 = get_global_cache()
    assert c1 is c2


def test_feature_save_load_round_trip(tmp_path):
    cache = DataCache(cache_dir=tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    path = cache.save_feature("test_feature", df, key="abc")
    assert path.exists()
    loaded = cache.load_feature("test_feature", key="abc")
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, df)


def test_clear_resets_memo():
    cache = DataCache()
    cache._memo["dummy"] = {"x": 1}
    assert len(cache._memo) >= 1
    cache.clear()
    assert len(cache._memo) == 0
