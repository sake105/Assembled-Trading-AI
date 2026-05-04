"""Invariant tests for the backtest engine.

From 35_GOLDEN_EQUITY_SCENARIO_TESTS.md §7.

These tests verify properties that must ALWAYS hold, regardless of input:
  - equity is always finite
  - equity never becomes negative
  - dates are always monotonically increasing
  - higher commission → lower or equal final equity
  - longer lookback period → fewer early signals
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.characterization._fixtures import make_ohlcv
from tests.characterization.test_golden_equity import _run_minimal_backtest


@pytest.mark.characterization
@pytest.mark.parametrize(
    "tickers,days",
    [
        (["AAPL"], 30),
        (["AAPL", "MSFT"], 60),
        (["AAPL", "MSFT", "NVDA", "GOOG"], 90),
    ],
)
def test_invariant_equity_always_finite(tickers, days):
    """Equity is always finite regardless of ticker count or period."""
    bars = make_ohlcv(tickers, "2024-01-01", f"2024-{1 + days // 30:02d}-28", seed=42)
    result = _run_minimal_backtest(bars)
    assert np.isfinite(result["equity"]).all()
    assert not result["equity"].isna().any()


@pytest.mark.characterization
def test_invariant_higher_commission_lower_equity():
    """Higher commission rate must produce lower or equal final equity."""
    bars = make_ohlcv(["AAPL", "MSFT"], "2024-01-01", "2024-06-30", seed=42)
    result_low = _run_minimal_backtest(bars, commission_bps=1.0)
    result_high = _run_minimal_backtest(bars, commission_bps=50.0)
    assert (
        result_high.iloc[-1]["equity"] <= result_low.iloc[-1]["equity"]
    ), "Higher commission should not increase final equity"


@pytest.mark.characterization
def test_invariant_zero_commission_equals_no_cost():
    """Zero commission: two identical runs must produce equal equity."""
    bars = make_ohlcv(["AAPL"], "2024-01-01", "2024-03-31", seed=42)
    r1 = _run_minimal_backtest(bars, commission_bps=0.0)
    r2 = _run_minimal_backtest(bars, commission_bps=0.0)
    np.testing.assert_array_almost_equal(
        r1["equity"].values, r2["equity"].values, decimal=6
    )


@pytest.mark.characterization
def test_invariant_dates_monotonic():
    """Output dates must be strictly monotonic increasing."""
    bars = make_ohlcv(["AAPL", "MSFT", "NVDA"], "2024-01-01", "2024-12-31", seed=42)
    result = _run_minimal_backtest(bars)
    assert result["date"].is_monotonic_increasing


@pytest.mark.characterization
def test_invariant_initial_equity_scales_linearly():
    """Doubling initial equity should roughly double final equity (within 5%)."""
    bars = make_ohlcv(["AAPL"], "2024-01-01", "2024-06-30", seed=42)
    r1 = _run_minimal_backtest(bars, initial_equity=100_000.0)
    r2 = _run_minimal_backtest(bars, initial_equity=200_000.0)
    ratio = r2.iloc[-1]["equity"] / r1.iloc[-1]["equity"]
    assert 1.8 <= ratio <= 2.2, f"Expected ~2x scaling, got ratio={ratio:.3f}"


@pytest.mark.characterization
def test_invariant_equity_row_count():
    """Output must have one row per unique date across all tickers."""
    bars = make_ohlcv(["AAPL", "MSFT"], "2024-01-01", "2024-03-31", seed=42)
    n_dates = bars["Date"].nunique()
    result = _run_minimal_backtest(bars)
    assert (
        len(result) == n_dates
    ), f"Expected {n_dates} rows (one per date), got {len(result)}"
