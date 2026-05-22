"""Test that equity_curve is MTM (cash + positions value), not cash-only."""

from __future__ import annotations

import pytest

import numpy as np
import pandas as pd

from src.assembled_core.qa.backtest_engine import run_portfolio_backtest


def _synthetic_prices_three_symbols_upward() -> pd.DataFrame:
    """Synthetic OHLCV for 3 symbols, ~120 bars, clear upward trend."""
    dates = pd.date_range(start="2021-01-01", end="2021-06-30", freq="D", tz="UTC")
    dates = dates[dates.weekday < 5]
    n_bars = len(dates)
    symbols = ["AAPL", "MSFT", "GOOG"]
    rows = []
    for sym in symbols:
        base = 100.0 if sym == "AAPL" else 150.0 if sym == "MSFT" else 200.0
        # Upward trend: +0.3% per day on average
        trend = np.linspace(0, 0.35 * n_bars * 0.003, n_bars)
        noise = np.random.RandomState(42).normal(0, 0.01, n_bars)
        close = base * np.exp(trend + np.cumsum(noise))
        for i, ts in enumerate(dates):
            c = close[i]
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "open": c * 0.99,
                    "high": c * 1.02,
                    "low": c * 0.98,
                    "close": c,
                    "volume": 1_000_000,
                }
            )
    return (
        pd.DataFrame(rows).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    )


def _dummy_signal_fn(prices: pd.DataFrame) -> pd.DataFrame:
    """Long-only signal so we get BUYs and positions. API: timestamp, symbol, direction, score."""
    out = prices[["timestamp", "symbol"]].copy()
    out["direction"] = "LONG"
    out["score"] = 1.0
    return out


def _dummy_position_sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
    """Allocate ~1/4 of capital per symbol (target_qty = notional for order generation)."""
    out = (
        signals[["timestamp", "symbol"]].copy()
        if "timestamp" in signals.columns
        else signals.copy()
    )
    out["target_weight"] = 0.25
    out["target_qty"] = capital * 0.25  # notional per symbol
    return out


@pytest.mark.unit
def test_equity_curve_is_mtm():
    """Equity curve = cash + MTM positions; cash_curve is cash-only; with trend and fills, equity_max > start_capital."""
    prices = _synthetic_prices_three_symbols_upward()
    start_capital = 10000.0

    result = run_portfolio_backtest(
        prices=prices,
        signal_fn=_dummy_signal_fn,
        position_sizing_fn=_dummy_position_sizing_fn,
        start_capital=start_capital,
        include_costs=True,
        commission_bps=5.0,
        spread_w=0.25,
        impact_w=0.25,
        strict_session_gate=False,
        include_trades=True,
    )

    equity_df = result.equity
    assert len(equity_df) > 50, "equity_curve should have > 50 rows"

    # MTM equity: with upward trend and filled buys, equity max should exceed start capital
    equity_max = equity_df["equity"].max()
    assert equity_max > start_capital, (
        f"With upward trend and fills, equity_max ({equity_max}) should be > start_capital ({start_capital})"
    )

    # Cash column present (for cash_curve CSV)
    assert "cash" in equity_df.columns, "equity DataFrame should include cash column"

    # Equity and cash should differ when there are positions (MTM = cash + positions_value)
    same = np.isclose(
        equity_df["equity"].values, equity_df["cash"].values, rtol=1e-9, atol=1e-9
    )
    assert not np.all(same), (
        "equity and cash series should differ when there are positions"
    )
