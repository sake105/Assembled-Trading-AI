"""Smoke tests for writing broker snapshot from backtest (Sprint 13).

Tests that --write-broker-snapshot flag creates snapshot files.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.qa.backtest_engine import run_portfolio_backtest


def create_synthetic_prices() -> pd.DataFrame:
    """Create synthetic price data for testing."""
    base_time = datetime(2025, 1, 15, 10, 0, 0)
    data = []
    for i in range(100):
        ts = pd.Timestamp(base_time, tz="UTC") + pd.Timedelta(days=i)
        data.append(
            {
                "timestamp": ts,
                "symbol": "AAPL",
                "close": 100.0 + i * 0.1,
                "open": 100.0 + i * 0.1 - 0.05,
                "high": 100.0 + i * 0.1 + 0.1,
                "low": 100.0 + i * 0.1 - 0.1,
                "volume": 1000000.0,
            }
        )
    return pd.DataFrame(data)


def test_backtest_write_broker_snapshot_smoke(tmp_path: Path):
    """Test that backtest with write_broker_snapshot=True creates snapshot files."""
    prices = create_synthetic_prices()
    run_id = "test_write_snapshot"
    output_dir = tmp_path

    # Simple signal function (trend-based)
    def signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
        from src.assembled_core.signals.rules_trend import (
            generate_trend_signals_from_prices,
        )

        return generate_trend_signals_from_prices(prices_df, ma_fast=20, ma_slow=50)

    # Simple position sizing function
    def position_sizing_fn(signals_df: pd.DataFrame, capital: float) -> pd.DataFrame:
        from src.assembled_core.portfolio.position_sizing import (
            compute_target_positions_from_trend_signals,
        )

        return compute_target_positions_from_trend_signals(
            signals_df, total_capital=capital
        )

    # Run backtest with write_broker_snapshot=True
    result = run_portfolio_backtest(
        prices=prices,
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        start_capital=10000.0,
        include_costs=True,
        include_trades=True,
        include_ledger=True,
        run_id=run_id,
        output_dir=output_dir,
        write_broker_snapshot=True,  # Enable snapshot writing
        broker_snapshot_policy="prefer",
    )

    # Verify result has broker_snapshot_path in meta
    assert result.meta is not None, "Result should have meta dict"
    assert (
        "broker_snapshot_path" in result.meta
    ), "Meta should contain broker_snapshot_path"

    broker_snapshot_path = result.meta.get("broker_snapshot_path")
    assert broker_snapshot_path is not None, "broker_snapshot_path should not be None"

    # Verify snapshot directory exists
    snapshot_dir = output_dir / broker_snapshot_path
    assert (
        snapshot_dir.exists()
    ), f"Broker snapshot directory should exist: {snapshot_dir}"

    # Verify snapshot JSON file exists (find by pattern)
    snapshot_files = list(snapshot_dir.glob("snapshot_*.json"))
    assert (
        len(snapshot_files) > 0
    ), f"At least one snapshot JSON file should exist in {snapshot_dir}"

    # Load and verify schema
    snapshot_file = snapshot_files[0]
    with snapshot_file.open("r", encoding="utf-8") as f:
        snapshot = json.load(f)

    assert "as_of_date" in snapshot
    assert "cash" in snapshot
    assert "positions" in snapshot
    assert isinstance(snapshot["positions"], list)

    # Verify positions are normalized/sorted
    if len(snapshot["positions"]) > 0:
        positions = snapshot["positions"]
        # Check that positions have required fields
        for pos in positions:
            assert "symbol" in pos
            assert "qty" in pos

        # Verify sorting (symbols should be in ascending order)
        symbols = [pos["symbol"] for pos in positions]
        assert symbols == sorted(symbols), "Positions should be sorted by symbol"
