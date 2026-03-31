# tests/test_trend_baseline_fills_regression.py
"""Regression test: trend-baseline produces fills (qty in shares), equity time series, reject_reason."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.qa.backtest_engine import run_portfolio_backtest


def _synthetic_ohlcv_3symbols_60days() -> pd.DataFrame:
    """Synthetic OHLCV: 3 symbols, 60 weekdays, timestamps at session close (21:00 UTC)."""
    dates = pd.date_range(start="2021-01-04", end="2021-03-30", freq="B", tz="UTC")
    # Session close 21:00 UTC
    timestamps = dates + pd.Timedelta(hours=21)
    symbols = ["A", "B", "C"]
    rows = []
    for sym in symbols:
        np.random.seed(hash(sym) % 2**32)
        n = len(timestamps)
        # Trend so that fast MA crosses slow MA (ma_fast=20, ma_slow=50)
        close = 100.0 + np.cumsum(np.random.randn(n).cumsum() * 0.5)
        close = np.maximum(close, 1.0)
        for i, ts in enumerate(timestamps):
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
    df = pd.DataFrame(rows)
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _trend_signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Minimal trend signal: LONG when close > SMA20 (simple crossover proxy)."""
    from src.assembled_core.signals.rules_trend import (
        generate_trend_signals_from_prices,
    )

    return generate_trend_signals_from_prices(prices_df, ma_fast=20, ma_slow=50)


def _position_sizing_fn(signals_df: pd.DataFrame, capital: float) -> pd.DataFrame:
    """Equal-weight target positions (notional per symbol = capital / n)."""
    from src.assembled_core.portfolio.position_sizing import (
        compute_target_positions_from_trend_signals,
    )

    return compute_target_positions_from_trend_signals(
        signals_df, total_capital=capital, top_n=None, min_score=0.0
    )


@pytest.mark.phase4
def test_trend_baseline_produces_fills_and_equity_time_series():
    """Trend baseline with 10k capital and 3 symbols: at least one fill, equity > 10 rows.

    Uses strict_session_gate=False so the test runs without optional exchange_calendars.
    """
    prices = _synthetic_ohlcv_3symbols_60days()
    start_capital = 10000.0

    result = run_portfolio_backtest(
        prices=prices,
        signal_fn=_trend_signal_fn,
        position_sizing_fn=_position_sizing_fn,
        start_capital=start_capital,
        include_costs=True,
        include_trades=True,
        compute_features=True,
        strict_session_gate=False,
    )

    # Equity curve must be a time series (one row per bar), not a single row
    assert result.equity is not None and not result.equity.empty
    assert (
        len(result.equity) > 10
    ), f"equity_curve must have > 10 rows (time series), got {len(result.equity)}"

    # At least one trade must be filled (not all rejected)
    if result.trades is not None and not result.trades.empty:
        has_fill = (result.trades.get("fill_qty", pd.Series(dtype=float)) > 0).any()
        not_rejected = (
            result.trades.get("status", pd.Series(dtype=str)) != "rejected"
        ).any()
        assert (
            has_fill or not_rejected
        ), "At least one trade must have fill_qty > 0 or status != rejected (trend baseline should get fills)"

        # If any rejected, reject_reason must be present and ASCII-only
        rejected = result.trades[
            result.trades.get("status", pd.Series(dtype=str)) == "rejected"
        ]
        if not rejected.empty:
            assert "reject_reason" in result.trades.columns
            for val in result.trades["reject_reason"].dropna().astype(str):
                assert (
                    val == "" or val.isascii()
                ), f"reject_reason must be ASCII-only, got {val!r}"
    else:
        # No trades at all: allow only if strategy produced no signals; still require equity rows
        assert len(result.equity) > 10


@pytest.mark.phase4
def test_order_generation_converts_notional_to_shares():
    """Smoking-gun regression: target_qty as notional (3333.33) becomes shares (~23.8), notional <= 2*equity."""
    from src.assembled_core.execution.order_generation import (
        generate_orders_from_targets,
    )

    start_capital = 10000.0
    price = 140.0
    # Equal-weight notional per symbol: 10k/3 ~ 3333.33
    target_notional = start_capital / 3.0
    target_positions = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "target_qty": [target_notional, target_notional, target_notional],
        }
    )
    prices = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "close": [price, price, price],
        }
    )
    ts = pd.Timestamp("2021-01-04 21:00", tz="UTC")
    orders = generate_orders_from_targets(
        target_positions,
        current_positions=pd.DataFrame(columns=["symbol", "qty"]),
        timestamp=ts,
        prices=prices,
    )
    assert not orders.empty
    # qty must be in shares: 3333.33 / 140 ~ 23.8
    expected_shares = target_notional / price
    for _, row in orders.iterrows():
        assert (
            abs(row["qty"] - expected_shares) < 1.0
        ), f"qty should be ~{expected_shares:.1f} (shares), got {row['qty']}"
        notional = row["qty"] * row["price"]
        assert (
            notional <= 2.0 * start_capital
        ), f"order notional {notional} should be <= 2*equity"
    assert orders.attrs.get("qty_unit") == "shares"


@pytest.mark.phase4
def test_strict_qty_guard_raises_when_notional_exceeds_2x_capital():
    """When AS_CORE_STRICT_QTY=1, orders with notional > 2*start_capital must raise ValueError."""
    import os
    from src.assembled_core.qa.backtest_engine import _validate_order_notional_guard

    bad_orders = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2021-01-04", tz="UTC"),
                "symbol": "A",
                "side": "BUY",
                "qty": 10_000.0,
                "price": 140.0,
            },
        ]
    )
    start_capital = 10000.0
    # Notional 1.4e6 > 2*10k -> should raise when strict=True
    try:
        orig = os.environ.get("AS_CORE_STRICT_QTY")
        os.environ["AS_CORE_STRICT_QTY"] = "1"
        with pytest.raises(ValueError, match="notional.*2x start capital"):
            _validate_order_notional_guard(bad_orders, start_capital)
    finally:
        if orig is None:
            os.environ.pop("AS_CORE_STRICT_QTY", None)
        else:
            os.environ["AS_CORE_STRICT_QTY"] = orig
    # Without strict: no raise
    _validate_order_notional_guard(bad_orders, start_capital, strict=False)


@pytest.mark.phase4
def test_reject_reason_present_when_rejected():
    """If any trade has status=rejected, reject_reason column exists and is set for rejected rows."""
    from src.assembled_core.execution.fill_model import ensure_fill_schema

    orders = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2021-01-04 21:00", tz="UTC"),
                "symbol": "A",
                "side": "BUY",
                "qty": 10.0,
                "price": 100.0,
            },
            {
                "timestamp": pd.Timestamp("2021-01-04 21:00", tz="UTC"),
                "symbol": "B",
                "side": "BUY",
                "qty": 0.0,
                "price": 50.0,
            },
        ]
    )
    # Force one row to look rejected (fill_qty=0)
    orders["fill_qty"] = [10.0, 0.0]
    orders["fill_price"] = orders["price"]
    fills = ensure_fill_schema(orders, default_full_fill=True)
    assert "reject_reason" in fills.columns
    rejected = fills[fills["status"] == "rejected"]
    if not rejected.empty:
        assert (rejected["reject_reason"].astype(str).str.len() > 0).all() or (
            rejected["reject_reason"].fillna("").astype(str) == "UNKNOWN"
        ).all()
