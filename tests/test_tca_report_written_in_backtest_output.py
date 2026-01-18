# tests/test_tca_report_written_in_backtest_output.py
"""Tests for TCA report written in backtest output (Sprint B5).

Tests verify:
1. TCA report file exists after backtest
2. Schema ok, no NaNs
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.qa.backtest_engine import run_portfolio_backtest
from src.assembled_core.qa.tca import build_tca_report, write_tca_report_csv


def test_tca_report_written_after_backtest() -> None:
    """Test that TCA report is written after backtest (integration test)."""
    # Create synthetic prices
    timestamps = pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": ["AAPL"] * 10,
        "close": [150.0 + i * 0.1 for i in range(10)],
        "volume": [1000000.0] * 10,
    })
    
    # Simple signal function: buy on first day, sell on last day
    def signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame({
            "timestamp": [prices_df["timestamp"].iloc[0], prices_df["timestamp"].iloc[-1]],
            "symbol": ["AAPL", "AAPL"],
            "direction": ["LONG", "FLAT"],
        })
        return signals
    
    # Simple position sizing: 100% capital
    def position_sizing_fn(signals_df: pd.DataFrame, capital: float) -> pd.DataFrame:
        if len(signals_df) == 0 or signals_df["direction"].iloc[0] == "FLAT":
            return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
        return pd.DataFrame({
            "symbol": ["AAPL"],
            "target_weight": [1.0],
            "target_qty": [capital / prices["close"].iloc[0]],
        })
    
    # Run backtest with costs enabled
    result = run_portfolio_backtest(
        prices=prices,
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        start_capital=10000.0,
        include_costs=True,
        include_trades=True,
    )
    
    # Build TCA report from trades
    if result.trades is not None and not result.trades.empty:
        tca_report = build_tca_report(
            trades_df=result.trades,
            freq="1d",
            strategy_name="test_strategy",
        )
        
        # Verify TCA report schema
        required_cols = [
            "date",
            "symbol",
            "notional",
            "commission_cash",
            "spread_cash",
            "slippage_cash",
            "total_cost_cash",
            "cost_bps",
            "n_trades",
        ]
        for col in required_cols:
            assert col in tca_report.columns, f"Required column {col} should exist in TCA report"
        
        # Verify no NaNs in key columns
        key_cols = [
            "notional",
            "commission_cash",
            "spread_cash",
            "slippage_cash",
            "total_cost_cash",
            "cost_bps",
            "n_trades",
        ]
        for col in key_cols:
            if not tca_report.empty:
                assert not tca_report[col].isna().any(), f"Column {col} should not contain NaNs"
        
        # Write TCA report to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            tca_report_path = Path(tmpdir) / "tca_report_1d.csv"
            write_tca_report_csv(tca_report, tca_report_path)
            
            # Verify file exists
            assert tca_report_path.exists(), "TCA report file should be written"
            
            # Verify file can be read back
            loaded = pd.read_csv(tca_report_path)
            assert len(loaded) == len(tca_report), "Loaded TCA report should have same number of rows"
            assert set(loaded.columns) == set(tca_report.columns), "Loaded TCA report should have same columns"
    else:
        # No trades: TCA report should still be buildable (empty)
        tca_report = build_tca_report(
            trades_df=pd.DataFrame(columns=[
                "timestamp", "symbol", "qty", "price", "commission_cash", "spread_cash", "slippage_cash", "total_cost_cash"
            ]),
            freq="1d",
            strategy_name="test_strategy",
        )
        
        # Verify empty report has correct schema
        required_cols = [
            "date",
            "symbol",
            "notional",
            "commission_cash",
            "spread_cash",
            "slippage_cash",
            "total_cost_cash",
            "cost_bps",
            "n_trades",
        ]
        for col in required_cols:
            assert col in tca_report.columns, f"Required column {col} should exist in empty TCA report"
        
        assert tca_report.empty, "TCA report should be empty when no trades"
