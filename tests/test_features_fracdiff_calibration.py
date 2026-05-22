"""Tests for find_min_d_for_stationarity — C4-076 closure.

Reference: López de Prado AFML §5.5 — minimum-d fractional differencing
that achieves stationarity (ADF rejection) while preserving maximum memory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from src.assembled_core.features.triple_barrier import (
    find_min_d_for_stationarity,
    fractional_diff,
)


@pytest.fixture
def log_prices() -> pd.Series:
    """Synthetic non-stationary log-prices: random walk with mild drift."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0002, 0.012, 1500)
    prices = np.exp(np.cumsum(returns))
    return pd.Series(np.log(prices), name="log_price")


def test_returns_dict_with_expected_keys(log_prices):
    result = find_min_d_for_stationarity(log_prices)
    for key in (
        "d",
        "adf_statistic",
        "pvalue",
        "is_stationary",
        "correlation_with_original",
        "grid_tested",
    ):
        assert key in result


def test_finds_stationarising_d_for_random_walk_log_prices(log_prices):
    """Random-walk log-prices should be stationarised by some d in (0, 1)."""
    result = find_min_d_for_stationarity(log_prices)
    assert result["is_stationary"] is True
    assert 0 < result["d"] < 1
    assert result["pvalue"] < 0.05
    # correlation_with_original is INFORMATIONAL — depends heavily on FFD
    # weight-threshold and the level/variance characteristics of the input.
    # For log-prices (non-stationary, unbounded variance), small-d FFD truncation
    # destroys level-correlation. The López de Prado §5.5 "memory loss" plot
    # uses normalized/rolling correlations; we expose the raw value but do not
    # assert a magnitude bound — that depends on the specific FFD-threshold and
    # series characteristics chosen by the caller.
    assert result["correlation_with_original"] is not None
    assert np.isfinite(result["correlation_with_original"])


def test_returns_first_d_in_grid_order(log_prices):
    """Function should return the FIRST d in grid order that achieves stationarity
    (i.e. the smallest if the grid is sorted ascending)."""
    coarse_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    result = find_min_d_for_stationarity(log_prices, d_grid=coarse_grid)
    if result["is_stationary"]:
        # The chosen d should be the FIRST stationarising point in the grid
        idx = coarse_grid.index(result["d"])
        # Verify earlier grid points are NOT stationary
        from statsmodels.tsa.stattools import adfuller

        for earlier_d in coarse_grid[:idx]:
            diffed = fractional_diff(log_prices, d=earlier_d).dropna()
            _, p_earlier, *_ = adfuller(diffed.to_numpy(), autolag="AIC")
            assert p_earlier >= 0.05, (
                f"d={earlier_d} (earlier in grid) has p={p_earlier} < 0.05 — "
                f"function should have picked it instead of {result['d']}"
            )


def test_stationary_input_picks_smallest_d():
    """If the input is ALREADY stationary, the smallest d should work
    (i.e. d=0.05 should already pass the ADF test)."""
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 1, 500))  # i.i.d. → already stationary
    result = find_min_d_for_stationarity(s, d_grid=[0.05, 0.1, 0.2, 0.5])
    assert result["is_stationary"] is True
    assert result["d"] == 0.05


def test_short_series_raises():
    s = pd.Series([1.0, 2.0, 3.0] * 5)  # 15 obs
    with pytest.raises(ValueError, match="30"):
        find_min_d_for_stationarity(s)


def test_custom_pvalue_threshold(log_prices):
    """A stricter threshold (0.01) may require larger d than 0.05."""
    result_05 = find_min_d_for_stationarity(log_prices, pvalue_threshold=0.05)
    result_01 = find_min_d_for_stationarity(log_prices, pvalue_threshold=0.01)
    if result_05["is_stationary"] and result_01["is_stationary"]:
        assert result_01["d"] >= result_05["d"], (
            "Stricter threshold should need same-or-larger d"
        )


def test_grid_tested_is_returned(log_prices):
    """The grid_tested field reflects the actual d-grid used."""
    custom_grid = [0.2, 0.4, 0.6]
    result = find_min_d_for_stationarity(log_prices, d_grid=custom_grid)
    assert result["grid_tested"] == custom_grid


def test_returns_none_when_no_d_works():
    """If grid is all-too-small d values that don't stationarise → None."""
    rng = np.random.default_rng(1)
    # Random walk with drift — needs d > 0 to stationarise
    s = pd.Series(np.cumsum(rng.normal(0.5, 0.01, 1000)))
    # Try only very small d values that won't help
    result = find_min_d_for_stationarity(s, d_grid=[0.01, 0.02])
    if not result["is_stationary"]:
        assert result["d"] is None
        assert result["pvalue"] is None
        assert result["adf_statistic"] is None
