"""Test that weekly rebalance produces fewer trades than daily, same equity timeline."""

from __future__ import annotations

import pytest

import pandas as pd

from src.assembled_core.qa.backtest_engine import run_portfolio_backtest


def _synthetic_prices() -> pd.DataFrame:
    """Synthetic OHLCV, 3 symbols, ~60 bars (quick)."""
    dates = pd.date_range(start="2021-01-01", end="2021-03-31", freq="B", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOG"]
    rows = []
    for sym in symbols:
        base = 100.0
        for i, ts in enumerate(dates):
            c = base + i * 0.1
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "open": c * 0.99,
                    "high": c * 1.01,
                    "low": c * 0.98,
                    "close": c,
                    "volume": 1e6,
                }
            )
    return (
        pd.DataFrame(rows).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    )


def _dummy_signal(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices[["timestamp", "symbol"]].copy()
    out["direction"] = "LONG"
    out["score"] = 1.0
    return out


def _dummy_sizing(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
    out = (
        signals[["timestamp", "symbol"]].copy()
        if "timestamp" in signals.columns
        else signals.copy()
    )
    out["target_weight"] = 0.25
    out["target_qty"] = capital * 0.25
    return out


@pytest.mark.unit
def test_rebalance_weekly_reduces_trades():
    """Weekly rebalance has fewer filled trades than daily; equity timeline length unchanged."""
    prices = _synthetic_prices()
    result_daily = run_portfolio_backtest(
        prices=prices,
        signal_fn=_dummy_signal,
        position_sizing_fn=_dummy_sizing,
        start_capital=10000.0,
        include_costs=True,
        strict_session_gate=False,
        include_trades=True,
        rebalance_schedule="daily",
    )
    result_weekly = run_portfolio_backtest(
        prices=prices,
        signal_fn=_dummy_signal,
        position_sizing_fn=_dummy_sizing,
        start_capital=10000.0,
        include_costs=True,
        strict_session_gate=False,
        include_trades=True,
        rebalance_schedule="weekly",
    )
    # Same equity timeline (one row per bar)
    assert len(result_daily.equity) == len(
        result_weekly.equity
    ), "equity DataFrame length must be equal (same timeline)"
    # Weekly should have fewer filled trades
    filled_daily = (
        (result_daily.trades["fill_qty"].fillna(0).astype(float) > 0).sum()
        if result_daily.trades is not None and not result_daily.trades.empty
        else 0
    )
    filled_weekly = (
        (result_weekly.trades["fill_qty"].fillna(0).astype(float) > 0).sum()
        if result_weekly.trades is not None and not result_weekly.trades.empty
        else 0
    )
    assert (
        filled_weekly < filled_daily
    ), f"weekly rebalance should have fewer fills: weekly={filled_weekly}, daily={filled_daily}"
