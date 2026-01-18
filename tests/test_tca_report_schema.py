# tests/test_tca_report_schema.py
"""Tests for TCA Report Schema (Sprint B4).

Tests verify:
1. Required columns exist
2. No NaNs in key columns
3. Deterministic ordering
4. Cost totals match sum of components
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.qa.tca import build_tca_report, write_tca_report_csv, write_tca_report_md


def test_tca_report_required_cols_exist() -> None:
    """Test that TCA report has all required columns."""
    # Create synthetic trades with cost columns
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "MSFT", "AAPL", "GOOGL", "MSFT"],
        "qty": [100.0, 50.0, 200.0, 150.0, 75.0],
        "price": [150.0, 200.0, 152.0, 100.0, 201.0],
        "commission_cash": [1.5, 1.0, 3.0, 1.5, 1.5],
        "spread_cash": [2.25, 1.5, 4.5, 2.25, 1.5],
        "slippage_cash": [1.0, 0.5, 2.0, 1.0, 0.75],
        "total_cost_cash": [4.75, 3.0, 9.5, 4.75, 3.75],
    })

    tca_report = build_tca_report(trades, freq="1d")

    # Verify required columns exist
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


def test_tca_report_no_nans_in_key_cols() -> None:
    """Test that TCA report has no NaNs in key columns."""
    # Create synthetic trades with cost columns
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "qty": [100.0, 50.0, 200.0],
        "price": [150.0, 200.0, 100.0],
        "commission_cash": [1.5, 1.0, 3.0],
        "spread_cash": [2.25, 1.5, 4.5],
        "slippage_cash": [1.0, 0.5, 2.0],
        "total_cost_cash": [4.75, 3.0, 9.5],
    })

    tca_report = build_tca_report(trades, freq="1d")

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
        assert not tca_report[col].isna().any(), f"Column {col} should not contain NaNs"


def test_tca_report_deterministic_ordering() -> None:
    """Test that TCA report is deterministically sorted."""
    # Create synthetic trades with cost columns (multiple days, multiple symbols)
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=6, freq="1d", tz="UTC"),
        "symbol": ["MSFT", "AAPL", "GOOGL", "AAPL", "MSFT", "GOOGL"],
        "qty": [100.0, 50.0, 200.0, 150.0, 75.0, 100.0],
        "price": [200.0, 150.0, 100.0, 152.0, 201.0, 101.0],
        "commission_cash": [1.0, 1.5, 3.0, 2.25, 1.5, 2.0],
        "spread_cash": [1.5, 2.25, 4.5, 3.375, 2.25, 3.0],
        "slippage_cash": [0.5, 1.0, 2.0, 1.5, 0.75, 1.0],
        "total_cost_cash": [3.0, 4.75, 9.5, 7.125, 4.5, 6.0],
    })

    tca_report = build_tca_report(trades, freq="1d")

    # Verify deterministic sorting (by date, symbol)
    assert tca_report.equals(
        tca_report.sort_values(["date", "symbol"], ignore_index=True)
    ), "TCA report should be sorted by (date, symbol)"


def test_tca_report_cost_totals_match_components() -> None:
    """Test that cost totals match sum of components."""
    # Create synthetic trades with cost columns
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=4, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"],
        "qty": [100.0, 200.0, 50.0, 75.0],
        "price": [150.0, 152.0, 200.0, 201.0],
        "commission_cash": [1.5, 3.0, 1.0, 1.5],
        "spread_cash": [2.25, 4.5, 1.5, 2.25],
        "slippage_cash": [1.0, 2.0, 0.5, 0.75],
        "total_cost_cash": [4.75, 9.5, 3.0, 4.5],
    })

    tca_report = build_tca_report(trades, freq="1d")

    # Verify cost totals match sum of components
    expected_total = (
        tca_report["commission_cash"]
        + tca_report["spread_cash"]
        + tca_report["slippage_cash"]
    )
    np.testing.assert_array_almost_equal(
        tca_report["total_cost_cash"].values,
        expected_total.values,
        decimal=10,
        err_msg="total_cost_cash should equal sum of commission_cash + spread_cash + slippage_cash"
    )


def test_tca_report_cost_bps_calculation() -> None:
    """Test that cost_bps is calculated correctly."""
    # Create synthetic trades with known costs
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=2, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "MSFT"],
        "qty": [100.0, 50.0],
        "price": [150.0, 200.0],
        "commission_cash": [1.5, 1.0],
        "spread_cash": [2.25, 1.5],
        "slippage_cash": [1.0, 0.5],
        "total_cost_cash": [4.75, 3.0],
    })

    tca_report = build_tca_report(trades, freq="1d")

    # Verify cost_bps calculation: total_cost_cash / notional * 10000
    # AAPL: notional = 100 * 150 = 15000, total_cost = 4.75, cost_bps = 4.75 / 15000 * 10000 = 3.167
    # MSFT: notional = 50 * 200 = 10000, total_cost = 3.0, cost_bps = 3.0 / 10000 * 10000 = 3.0
    expected_bps = (tca_report["total_cost_cash"] / tca_report["notional"]) * 10000.0
    np.testing.assert_array_almost_equal(
        tca_report["cost_bps"].values,
        expected_bps.values,
        decimal=2,
        err_msg="cost_bps should equal (total_cost_cash / notional) * 10000"
    )


def test_tca_report_empty_trades() -> None:
    """Test that empty trades DataFrame returns stable empty report with schema."""
    trades = pd.DataFrame(columns=[
        "timestamp",
        "symbol",
        "qty",
        "price",
        "commission_cash",
        "spread_cash",
        "slippage_cash",
        "total_cost_cash",
    ])

    tca_report = build_tca_report(trades, freq="1d")

    # Verify schema: required columns
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
        assert col in tca_report.columns, f"Column {col} should exist in empty TCA report"

    # Verify empty
    assert tca_report.empty, "Result should be empty"

    # Verify no NaNs (even in empty DataFrame)
    for col in required_cols:
        if col not in ["date", "symbol"]:  # Skip non-numeric columns
            assert not tca_report[col].isna().any(), f"Column {col} should not contain NaNs"


def test_tca_report_aggregation_per_day_symbol() -> None:
    """Test that aggregation per day+symbol works correctly."""
    # Create trades: same symbol, same day (should aggregate)
    trades = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2024-01-01 16:00:00", tz="UTC"),
            pd.Timestamp("2024-01-01 16:00:00", tz="UTC"),
            pd.Timestamp("2024-01-02 16:00:00", tz="UTC"),
        ],
        "symbol": ["AAPL", "AAPL", "AAPL"],
        "qty": [100.0, 200.0, 150.0],
        "price": [150.0, 152.0, 153.0],
        "commission_cash": [1.5, 3.0, 2.25],
        "spread_cash": [2.25, 4.5, 3.375],
        "slippage_cash": [1.0, 2.0, 1.5],
        "total_cost_cash": [4.75, 9.5, 7.125],
    })

    tca_report = build_tca_report(trades, freq="1d")

    # Should have 2 rows (2 days, same symbol)
    assert len(tca_report) == 2, "Should have 2 rows (2 days)"

    # First day: 2 trades aggregated
    day1 = tca_report[tca_report["date"] == pd.Timestamp("2024-01-01").date()].iloc[0]
    assert day1["n_trades"] == 2, "Day 1 should have 2 trades"
    assert day1["notional"] == (100.0 * 150.0 + 200.0 * 152.0), "Day 1 notional should be sum"
    assert day1["total_cost_cash"] == (4.75 + 9.5), "Day 1 total_cost should be sum"

    # Second day: 1 trade
    day2 = tca_report[tca_report["date"] == pd.Timestamp("2024-01-02").date()].iloc[0]
    assert day2["n_trades"] == 1, "Day 2 should have 1 trade"


def test_write_tca_report_csv() -> None:
    """Test that write_tca_report_csv writes CSV file correctly."""
    import tempfile

    # Create synthetic TCA report
    tca_report = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-02").date()],
        "symbol": ["AAPL", "MSFT"],
        "notional": [15000.0, 10000.0],
        "commission_cash": [1.5, 1.0],
        "spread_cash": [2.25, 1.5],
        "slippage_cash": [1.0, 0.5],
        "total_cost_cash": [4.75, 3.0],
        "cost_bps": [3.17, 3.0],
        "n_trades": [1, 1],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "tca_report.csv"
        written_path = write_tca_report_csv(tca_report, output_path)

        # Verify file exists
        assert written_path.exists(), "CSV file should be written"

        # Verify file can be read back
        loaded = pd.read_csv(written_path)
        assert len(loaded) == len(tca_report), "Loaded CSV should have same number of rows"
        assert set(loaded.columns) == set(tca_report.columns), "Loaded CSV should have same columns"


def test_write_tca_report_md() -> None:
    """Test that write_tca_report_md writes Markdown file correctly."""
    import tempfile

    # Create synthetic TCA report
    tca_report = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-02").date()],
        "symbol": ["AAPL", "MSFT"],
        "notional": [15000.0, 10000.0],
        "commission_cash": [1.5, 1.0],
        "spread_cash": [2.25, 1.5],
        "slippage_cash": [1.0, 0.5],
        "total_cost_cash": [4.75, 3.0],
        "cost_bps": [3.17, 3.0],
        "n_trades": [1, 1],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "tca_report.md"
        written_path = write_tca_report_md(tca_report, output_path, strategy_name="test_strategy")

        # Verify file exists
        assert written_path.exists(), "Markdown file should be written"

        # Verify file content
        content = written_path.read_text(encoding="utf-8")
        assert "Transaction Cost Analysis" in content, "Markdown should contain title"
        assert "test_strategy" in content, "Markdown should contain strategy name"
        assert "AAPL" in content, "Markdown should contain symbol data"
        assert "MSFT" in content, "Markdown should contain symbol data"
