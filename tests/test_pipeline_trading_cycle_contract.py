"""Contract tests for unified trading cycle API (B1).

These tests verify the API contract (structure, types, docstrings) without
requiring full implementation. They ensure:
- TradingContext can be instantiated with required fields
- TradingCycleResult can be instantiated
- run_trading_cycle() validates inputs and returns proper structure
- Hook points are callable and return expected types
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.assembled_core.pipeline.trading_cycle import (
    TradingContext,
    TradingCycleResult,
    run_trading_cycle,
)


def test_trading_context_creation_minimal() -> None:
    """Test creating TradingContext with minimal required fields."""
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "close": [100.0] * 10,
        }
    )

    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": df["timestamp"].unique(),
                "symbol": ["AAPL"],
                "direction": ["LONG"],
                "score": [0.5],
            }
        )

    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "target_weight": [0.1],
                "target_qty": [10.0],
            }
        )

    ctx = TradingContext(
        prices=prices,
        signal_fn=signal_fn,
        position_sizing_fn=sizing_fn,
    )

    assert ctx.prices is not None
    assert ctx.signal_fn is not None
    assert ctx.position_sizing_fn is not None
    assert ctx.as_of is None
    assert ctx.capital == 10000.0  # default


def test_trading_context_creation_full() -> None:
    """Test creating TradingContext with all fields."""
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "close": [100.0] * 10,
        }
    )

    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": df["timestamp"].unique(),
                "symbol": ["AAPL"],
                "direction": ["LONG"],
                "score": [0.5],
            }
        )

    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "target_weight": [0.1],
                "target_qty": [10.0],
            }
        )

    ctx = TradingContext(
        prices=prices,
        as_of=pd.Timestamp("2025-01-10", tz="UTC"),
        freq="1d",
        universe=["AAPL", "MSFT"],
        use_factor_store=True,
        factor_store_root=Path("data/factors"),
        factor_group="core_ta",
        signal_fn=signal_fn,
        signal_config={"ma_fast": 20, "ma_slow": 50},
        position_sizing_fn=sizing_fn,
        capital=50000.0,
        current_positions=pd.DataFrame({"symbol": ["MSFT"], "qty": [5.0]}),
        order_timestamp=pd.Timestamp("2025-01-10", tz="UTC"),
        enable_risk_controls=True,
        output_dir=Path("output/test"),
        output_format="safe_csv",
        write_outputs=True,
        run_id="test_run_001",
        strategy_name="test_strategy",
    )

    assert ctx.as_of == pd.Timestamp("2025-01-10", tz="UTC")
    assert ctx.freq == "1d"
    assert ctx.universe == ["AAPL", "MSFT"]
    assert ctx.use_factor_store is True
    assert ctx.run_id == "test_run_001"


def test_trading_cycle_result_creation() -> None:
    """Test creating TradingCycleResult with all fields."""
    prices = pd.DataFrame({"symbol": ["AAPL"], "close": [100.0]})
    signals = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2025-01-01", tz="UTC")],
            "symbol": ["AAPL"],
            "direction": ["LONG"],
            "score": [0.5],
        }
    )

    result = TradingCycleResult(
        prices_filtered=prices,
        prices_with_features=prices,
        signals=signals,
        target_positions=pd.DataFrame(
            {"symbol": ["AAPL"], "target_weight": [0.1], "target_qty": [10.0]}
        ),
        orders=pd.DataFrame(
            {"symbol": ["AAPL"], "side": ["BUY"], "qty": [10.0], "price": [100.0]}
        ),
        orders_filtered=pd.DataFrame(
            {"symbol": ["AAPL"], "side": ["BUY"], "qty": [10.0], "price": [100.0]}
        ),
        run_id="test_run_001",
        status="success",
        meta={"test": "value"},
        output_paths={"safe_csv": Path("output/test.csv")},
    )

    assert result.status == "success"
    assert result.run_id == "test_run_001"
    assert len(result.signals) == 1
    assert len(result.orders) == 1


def test_run_trading_cycle_validation_missing_prices() -> None:
    """Test run_trading_cycle() validation: missing prices."""

    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()

    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return pd.DataFrame()

    ctx = TradingContext(
        prices=pd.DataFrame(),  # Empty
        signal_fn=signal_fn,
        position_sizing_fn=sizing_fn,
    )

    result = run_trading_cycle(ctx)

    assert result.status == "error"
    assert "prices DataFrame is None or empty" in result.error_message


def test_run_trading_cycle_validation_missing_columns() -> None:
    """Test run_trading_cycle() validation: missing required columns."""
    prices = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            # Missing "timestamp" and "close"
        }
    )

    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()

    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return pd.DataFrame()

    ctx = TradingContext(
        prices=prices,
        signal_fn=signal_fn,
        position_sizing_fn=sizing_fn,
    )

    result = run_trading_cycle(ctx)

    assert result.status == "error"
    assert "Missing required price columns" in result.error_message


def test_run_trading_cycle_validation_missing_signal_fn() -> None:
    """Test run_trading_cycle() validation: missing signal_fn."""
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "close": [100.0] * 10,
        }
    )

    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return pd.DataFrame()

    ctx = TradingContext(
        prices=prices,
        signal_fn=None,  # Missing
        position_sizing_fn=sizing_fn,
    )

    result = run_trading_cycle(ctx)

    assert result.status == "error"
    assert "signal_fn is required" in result.error_message


def test_run_trading_cycle_validation_missing_position_sizing_fn() -> None:
    """Test run_trading_cycle() validation: missing position_sizing_fn."""
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "close": [100.0] * 10,
        }
    )

    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()

    ctx = TradingContext(
        prices=prices,
        signal_fn=signal_fn,
        position_sizing_fn=None,  # Missing
    )

    result = run_trading_cycle(ctx)

    assert result.status == "error"
    assert "position_sizing_fn is required" in result.error_message


def test_run_trading_cycle_signal_validation_missing_columns() -> None:
    """Test run_trading_cycle() validation: signals missing required columns."""
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "close": [100.0] * 10,
        }
    )

    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        # Missing "direction" column
        return pd.DataFrame(
            {
                "timestamp": df["timestamp"].unique(),
                "symbol": ["AAPL"],
                "score": [0.5],
            }
        )

    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "target_weight": [0.1],
                "target_qty": [10.0],
            }
        )

    ctx = TradingContext(
        prices=prices,
        signal_fn=signal_fn,
        position_sizing_fn=sizing_fn,
    )

    result = run_trading_cycle(ctx)

    assert result.status == "error"
    assert "signals missing required columns" in result.error_message


def test_run_trading_cycle_success_skeleton() -> None:
    """Test run_trading_cycle() skeleton execution (success path)."""
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "close": [100.0] * 10,
        }
    )

    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": df["timestamp"].unique(),
                "symbol": ["AAPL"],
                "direction": ["LONG"],
                "score": [0.5],
            }
        )

    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "target_weight": [0.1],
                "target_qty": [10.0],
            }
        )

    ctx = TradingContext(
        prices=prices,
        signal_fn=signal_fn,
        position_sizing_fn=sizing_fn,
        run_id="test_run_001",
    )

    result = run_trading_cycle(ctx)

    assert result.status == "success"
    assert result.run_id == "test_run_001"
    # EOD mode (default) returns last row per symbol — 1 row for single-symbol input
    assert len(result.prices_filtered) == 1
    assert len(result.signals) == 1
    assert len(result.target_positions) == 1
    assert result.error_message is None


def test_run_trading_cycle_with_logger() -> None:
    """Test run_trading_cycle() with custom logger."""
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "close": [100.0] * 10,
        }
    )

    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": df["timestamp"].unique(),
                "symbol": ["AAPL"],
                "direction": ["LONG"],
                "score": [0.5],
            }
        )

    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "target_weight": [0.1],
                "target_qty": [10.0],
            }
        )

    custom_logger = logging.getLogger("test_trading_cycle")

    ctx = TradingContext(
        prices=prices,
        signal_fn=signal_fn,
        position_sizing_fn=sizing_fn,
        logger=custom_logger,
    )

    result = run_trading_cycle(ctx)

    assert result.status == "success"
    assert ctx.logger is custom_logger
