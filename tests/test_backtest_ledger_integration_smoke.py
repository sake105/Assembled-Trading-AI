"""Smoke tests for backtest ledger integration (Sprint 13).

Tests that backtest runs generate ledger packs and reconciliation reports.
"""

from __future__ import annotations

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


def test_backtest_ledger_integration_smoke(tmp_path: Path) -> None:
    """Test that backtest generates ledger pack and reconciliation report."""
    prices = create_synthetic_prices()
    run_id = "test_backtest_ledger_smoke"
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

    # Run backtest with ledger enabled
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
    )

    # Verify result has ledger info in meta
    assert result.meta is not None, "Result should have meta dict"
    assert "ledger_pack_path" in result.meta, "Meta should contain ledger_pack_path"
    assert "reconcile_report_path" in result.meta, (
        "Meta should contain reconcile_report_path"
    )
    assert "reconciliation_ok" in result.meta, "Meta should contain reconciliation_ok"

    # Verify ledger pack path exists
    ledger_pack_path = result.meta.get("ledger_pack_path")
    assert ledger_pack_path is not None, "ledger_pack_path should not be None"
    ledger_dir = output_dir / ledger_pack_path
    assert ledger_dir.exists(), f"Ledger directory should exist: {ledger_dir}"

    # Verify ledger events file exists
    ledger_events_file = ledger_dir / "ledger_events.parquet"
    assert ledger_events_file.exists(), (
        f"Ledger events file should exist: {ledger_events_file}"
    )

    # Verify reconciliation report exists
    reconcile_report_path = result.meta.get("reconcile_report_path")
    if reconcile_report_path:
        report_file = output_dir / reconcile_report_path
        assert report_file.exists(), (
            f"Reconciliation report should exist: {report_file}"
        )

    # Verify reconciliation_ok is True (or at least present)
    reconciliation_ok = result.meta.get("reconciliation_ok")
    assert reconciliation_ok is not None, "reconciliation_ok should be present"
    # In paper mode, reconciliation should pass (ledger and broker are same source)
    assert reconciliation_ok is True, "reconciliation_ok should be True in paper mode"


def test_backtest_ledger_disabled(tmp_path: Path) -> None:
    """Test that backtest skips ledger when include_ledger=False."""
    prices = create_synthetic_prices()

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

    # Run backtest with ledger disabled
    result = run_portfolio_backtest(
        prices=prices,
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        start_capital=10000.0,
        include_costs=True,
        include_trades=True,
        include_ledger=False,  # Disabled
        run_id=None,
        output_dir=None,
    )

    # Verify result does not have ledger info (or has None values)
    if result.meta:
        assert (
            "ledger_pack_path" not in result.meta
            or result.meta.get("ledger_pack_path") is None
        )
        assert (
            "reconcile_report_path" not in result.meta
            or result.meta.get("reconcile_report_path") is None
        )


def test_backtest_ledger_deterministic_event_ids(tmp_path: Path) -> None:
    """Test that same inputs produce same event IDs (determinism)."""
    prices = create_synthetic_prices()
    run_id = "test_deterministic"
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

    # Run backtest twice with same inputs
    result1 = run_portfolio_backtest(
        prices=prices.copy(),
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        start_capital=10000.0,
        include_costs=True,
        include_trades=True,
        include_ledger=True,
        run_id=run_id,
        output_dir=output_dir / "run1",
    )

    result2 = run_portfolio_backtest(
        prices=prices.copy(),
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        start_capital=10000.0,
        include_costs=True,
        include_trades=True,
        include_ledger=True,
        run_id=run_id,
        output_dir=output_dir / "run2",
    )

    # Load ledger events from both runs
    from src.assembled_core.accounting.ledger_store import load_ledger_events_parquet

    ledger_path1 = result1.meta.get("ledger_pack_path")
    ledger_path2 = result2.meta.get("ledger_pack_path")

    if ledger_path1 and ledger_path2:
        events1 = load_ledger_events_parquet(output_dir / "run1", run_id)
        events2 = load_ledger_events_parquet(output_dir / "run2", run_id)

        # Verify event IDs are identical (deterministic)
        assert len(events1) == len(events2), (
            "Both runs should have same number of events"
        )
        if len(events1) > 0:
            event_ids1 = set(events1["event_id"].unique())
            event_ids2 = set(events2["event_id"].unique())
            assert event_ids1 == event_ids2, (
                "Event IDs should be identical for same inputs"
            )
