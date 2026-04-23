"""Tests for wave-69 module wiring into trading_cycle.py.

Covers:
  Step 2.44 — data.synthetic_generator (generate_crisis_returns / generate_normal_returns)
  Step 2.45 — data.resample (resample_to_weekly / resample_to_monthly)
  Step 2.46 — data.panel_store (panel_exists / panel_path)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.assembled_core.data.synthetic_generator import (
    generate_crisis_returns,
    generate_normal_returns,
)
from src.assembled_core.data.resample import resample_to_weekly, resample_to_monthly
from src.assembled_core.data.panel_store import panel_exists, panel_path


# ---------------------------------------------------------------------------
# synthetic_generator (Step 2.44)
# ---------------------------------------------------------------------------

def test_generate_normal_returns_shape():
    df = generate_normal_returns(n_days=30, n_assets=3, seed=0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 30
    assert df.shape[1] == 3


def test_generate_normal_returns_seed_deterministic():
    df1 = generate_normal_returns(n_days=10, n_assets=2, seed=42)
    df2 = generate_normal_returns(n_days=10, n_assets=2, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_crisis_returns_shape():
    df = generate_crisis_returns(n_assets=5, seed=1)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 5


def test_generate_crisis_returns_template():
    df = generate_crisis_returns(template="2008_gfc", n_assets=3, seed=0)
    assert isinstance(df, pd.DataFrame)


def test_generate_crisis_returns_unknown_template():
    with pytest.raises((ValueError, KeyError)):
        generate_crisis_returns(template="nonexistent_crisis")


# ---------------------------------------------------------------------------
# resample (Step 2.45)
# ---------------------------------------------------------------------------

def _make_daily_panel():
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    return pd.DataFrame({
        "symbol": "AAPL",
        "timestamp": idx,
        "open": np.random.uniform(140, 160, 60),
        "high": np.random.uniform(150, 170, 60),
        "low": np.random.uniform(130, 150, 60),
        "close": np.random.uniform(140, 160, 60),
        "volume": np.random.uniform(1e6, 2e6, 60),
    })


def test_resample_to_weekly_returns_df():
    daily = _make_daily_panel()
    weekly = resample_to_weekly(daily)
    assert isinstance(weekly, pd.DataFrame)


def test_resample_to_weekly_fewer_rows():
    daily = _make_daily_panel()
    weekly = resample_to_weekly(daily)
    assert len(weekly) < len(daily)


def test_resample_to_monthly_returns_df():
    daily = _make_daily_panel()
    monthly = resample_to_monthly(daily)
    assert isinstance(monthly, pd.DataFrame)


def test_resample_to_monthly_fewer_rows():
    daily = _make_daily_panel()
    monthly = resample_to_monthly(daily)
    assert len(monthly) < len(daily)


# ---------------------------------------------------------------------------
# panel_store (Step 2.46)
# ---------------------------------------------------------------------------

def test_panel_exists_returns_bool():
    result = panel_exists("__test_nonexistent__")
    assert isinstance(result, bool)


def test_panel_exists_nonexistent_is_false():
    result = panel_exists("__this_panel_does_not_exist_12345__")
    assert result is False


def test_panel_path_returns_path():
    from pathlib import Path
    p = panel_path("test_panel")
    assert isinstance(p, (str, Path))
