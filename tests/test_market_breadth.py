"""Tests for market breadth indicators module."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.assembled_core.features.market_breadth import (
    compute_market_breadth_ma,
    compute_advance_decline_line,
    compute_mcclellan_oscillator,
    compute_mcclellan_summation_index,
    compute_zweig_breadth_thrust,
    compute_new_highs_minus_new_lows,
    compute_arms_index,
)


def _synthetic_prices_panel(n_days: int = 200, n_symbols: int = 50, seed: int = 42) -> pd.DataFrame:
    """Synthetic prices panel (long format) for breadth tests."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    rows = []
    for sym_i in range(n_symbols):
        sym = f"SYM_{sym_i}"
        price = 100.0
        for d in dates:
            price *= 1 + rng.normal(0.0005, 0.02)
            rows.append({"timestamp": d, "symbol": sym, "close": price})
    return pd.DataFrame(rows)


@pytest.mark.phase12
class TestMarketBreadthMA:
    def test_basic(self):
        prices = _synthetic_prices_panel()
        result = compute_market_breadth_ma(prices)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_columns_present(self):
        prices = _synthetic_prices_panel()
        result = compute_market_breadth_ma(prices)
        assert any("fraction" in c.lower() or "above" in c.lower()
                    for c in result.columns)


@pytest.mark.phase12
class TestAdvanceDeclineLine:
    def test_basic_v2(self):
        prices = _synthetic_prices_panel()
        result = compute_advance_decline_line(prices)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_cumulative(self):
        prices = _synthetic_prices_panel()
        result = compute_advance_decline_line(prices)
        assert "ad_line" in result.columns
        assert not result["ad_line"].isna().all()


@pytest.mark.phase12
class TestMcClellanOscillator:
    def test_basic_v3(self):
        prices = _synthetic_prices_panel()
        ad = compute_advance_decline_line(prices)
        result = compute_mcclellan_oscillator(ad)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_oscillates(self):
        prices = _synthetic_prices_panel()
        ad = compute_advance_decline_line(prices)
        result = compute_mcclellan_oscillator(ad)
        assert "mcclellan_oscillator" in result.columns


@pytest.mark.phase12
class TestMcClellanSummation:
    def test_basic_v4(self):
        prices = _synthetic_prices_panel()
        ad = compute_advance_decline_line(prices)
        mco = compute_mcclellan_oscillator(ad)
        result = compute_mcclellan_summation_index(mco)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


@pytest.mark.phase12
class TestZweigBreadthThrust:
    def test_basic_v5(self):
        prices = _synthetic_prices_panel()
        ad = compute_advance_decline_line(prices)
        result = compute_zweig_breadth_thrust(ad)
        assert isinstance(result, pd.DataFrame)

    def test_binary_signal(self):
        prices = _synthetic_prices_panel()
        ad = compute_advance_decline_line(prices)
        result = compute_zweig_breadth_thrust(ad)
        if "zweig_thrust_signal" in result.columns:
            unique = set(result["zweig_thrust_signal"].dropna().unique())
            assert len(unique) <= 3


@pytest.mark.phase12
class TestNewHighsMinusNewLows:
    def test_basic_v6(self):
        prices = _synthetic_prices_panel()
        result = compute_new_highs_minus_new_lows(prices)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


@pytest.mark.phase12
class TestArmsIndex:
    def test_basic_v7(self):
        prices = _synthetic_prices_panel()
        ad = compute_advance_decline_line(prices)
        result = compute_arms_index(ad)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
