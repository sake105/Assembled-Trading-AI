"""A8: trading_cycle_v2 book_fills applies cost annotation in backtest/paper modes."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.mark.characterization
@pytest.mark.fast
def test_backtest_mode_adds_cost_columns(tmp_path):
    """book_fills in mode=backtest annotates orders_filtered with cost columns."""
    from src.assembled_core.pipeline.trading_cycle_shared import (
        TradingContext,
        TradingCycleResult,
    )
    from src.assembled_core.pipeline.trading_cycle_v2 import book_fills

    orders = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02")],
            "symbol": ["AAPL"],
            "side": ["buy"],
            "qty": [10.0],
            "price": [150.0],
        }
    )

    prices = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02")],
            "symbol": ["AAPL"],
            "close": [150.0],
            "open": [149.0],
            "high": [151.0],
            "low": [148.0],
            "volume": [1_000_000.0],
        }
    )
    ctx = TradingContext(
        as_of=pd.Timestamp("2024-01-02"),
        prices=prices,
        mode="backtest",
        write_outputs=False,
        output_dir=tmp_path,
    )
    result = TradingCycleResult(
        status="ok", orders=orders, orders_filtered=orders.copy()
    )
    out = book_fills(result, ctx)

    assert (
        "total_cost_cash" in out.orders_filtered.columns
    ), "backtest mode must add cost columns via add_cost_columns_to_trades"
    assert "commission_cash" in out.orders_filtered.columns


@pytest.mark.characterization
@pytest.mark.fast
def test_live_mode_does_not_add_cost_columns(tmp_path):
    """book_fills in mode=live must NOT apply cost annotation."""
    from src.assembled_core.pipeline.trading_cycle_shared import (
        TradingContext,
        TradingCycleResult,
    )
    from src.assembled_core.pipeline.trading_cycle_v2 import book_fills

    orders = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02")],
            "symbol": ["AAPL"],
            "side": ["buy"],
            "qty": [10.0],
            "price": [150.0],
        }
    )

    prices = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02")],
            "symbol": ["AAPL"],
            "close": [150.0],
            "open": [149.0],
            "high": [151.0],
            "low": [148.0],
            "volume": [1_000_000.0],
        }
    )
    ctx = TradingContext(
        as_of=pd.Timestamp("2024-01-02"),
        prices=prices,
        mode="live",
        write_outputs=False,
        output_dir=tmp_path,
    )
    result = TradingCycleResult(
        status="ok", orders=orders, orders_filtered=orders.copy()
    )
    out = book_fills(result, ctx)

    assert (
        "total_cost_cash" not in out.orders_filtered.columns
    ), "live mode must NOT add cost columns"


@pytest.mark.fast
def test_add_cost_columns_wired_in_v2_source():
    """trading_cycle_v2.py must reference add_cost_columns_to_trades (A8 wiring check)."""
    import inspect
    import src.assembled_core.pipeline.trading_cycle_v2 as mod

    src_text = inspect.getsource(mod)
    assert (
        "add_cost_columns_to_trades" in src_text
    ), "trading_cycle_v2 must wire add_cost_columns_to_trades in book_fills"
