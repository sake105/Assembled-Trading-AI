"""Mini E2E Tests for Paper Track.

Tests paper track with synthetic prices over 3-5 days, checking outputs and determinism.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.paper.paper_track import (
    PaperTrackConfig,
    run_paper_day,
    save_paper_state,
    write_paper_day_outputs,
)

pytestmark = pytest.mark.advanced


@pytest.fixture
def synthetic_prices_5days(tmp_path: Path) -> pd.DataFrame:
    """Create synthetic price data for 5 days (3 symbols, 5 days)."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    dates = pd.date_range("2025-01-06", periods=5, freq="D", tz="UTC")

    # Create price data with some trend
    data = []
    for sym in symbols:
        base_price = 100.0 if sym == "AAPL" else (200.0 if sym == "MSFT" else 150.0)
        for i, date in enumerate(dates):
            # Small random walk with drift
            price = base_price + (i * 0.5) + np.random.randn() * 0.5
            data.append(
                {
                    "timestamp": date,
                    "symbol": sym,
                    "open": price - 0.1,
                    "high": price + 0.3,
                    "low": price - 0.2,
                    "close": price,
                    "volume": 1000000.0 + i * 10000.0,
                }
            )

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@pytest.fixture
def synthetic_price_file(synthetic_prices_5days: pd.DataFrame, tmp_path: Path) -> Path:
    """Save synthetic prices to parquet file and return path."""
    # Save to parquet in a format that load_eod_prices_for_universe expects
    # We need to simulate the directory structure
    price_dir = tmp_path / "data" / "raw"
    price_dir.mkdir(parents=True, exist_ok=True)

    # Save as parquet (one file for all symbols, as expected by some loaders)
    price_file = price_dir / "synthetic_prices.parquet"
    synthetic_prices_5days.to_parquet(price_file, index=False)

    return price_file


@pytest.fixture
def paper_track_config(tmp_path: Path, synthetic_price_file: Path) -> PaperTrackConfig:
    """Create a PaperTrackConfig for testing."""
    # Create universe file
    universe_file = tmp_path / "universe.txt"
    universe_file.write_text("AAPL\nMSFT\nGOOGL\n", encoding="utf-8")

    output_root = tmp_path / "output" / "paper_track" / "test_strategy"

    return PaperTrackConfig(
        strategy_name="test_strategy",
        strategy_type="trend_baseline",
        strategy_params={
            "ma_fast": 2,  # Short windows for small dataset
            "ma_slow": 3,
            "top_n": 2,  # Top 2 symbols
            "min_score": 0.0,
        },
        universe_file=str(universe_file),
        freq="1d",
        seed_capital=100000.0,
        commission_bps=0.5,
        spread_w=0.25,
        impact_w=0.5,
        output_root=str(output_root),
        random_seed=42,  # Fixed seed for determinism
        enable_pit_checks=False,  # Disable for synthetic data
    )


@pytest.mark.advanced
def test_paper_track_mini_e2e_5days(
    paper_track_config: PaperTrackConfig,
    synthetic_prices_5days: pd.DataFrame,
    tmp_path: Path,
):
    """Mini E2E test: Run paper track for 5 days with synthetic prices."""
    # Monkey-patch load_eod_prices_for_universe to return our synthetic data
    import src.assembled_core.paper.paper_track as paper_module

    def mock_load_prices(universe_file, freq):
        return synthetic_prices_5days.copy()

    # Temporarily replace the function
    original_load = paper_module.load_eod_prices_for_universe
    paper_module.load_eod_prices_for_universe = mock_load_prices

    try:
        output_root = Path(paper_track_config.output_root)
        state_path = output_root / "state" / "state.json"

        results = []
        dates = synthetic_prices_5days["timestamp"].unique()[:5]  # 5 days

        # Run for each day
        for i, date in enumerate(sorted(dates)):
            result = run_paper_day(
                config=paper_track_config,
                as_of=pd.Timestamp(date),
                state_path=state_path,
            )
            results.append(result)

            # Write outputs
            write_paper_day_outputs(result, output_root)

            # Save state
            save_paper_state(result.state_after, state_path)

            # Verify outputs exist
            run_date_str = result.date.strftime("%Y%m%d")
            run_dir = output_root / "runs" / run_date_str

            assert run_dir.exists(), f"Run directory should exist: {run_dir}"
            assert (run_dir / "equity_snapshot.json").exists(), (
                "equity_snapshot.json should exist"
            )
            assert (run_dir / "positions.csv").exists(), "positions.csv should exist"
            assert (run_dir / "daily_summary.json").exists(), (
                "daily_summary.json should exist"
            )
            assert (run_dir / "daily_summary.md").exists(), (
                "daily_summary.md should exist"
            )

            # Verify state is updated
            assert result.state_after.last_run_date == pd.Timestamp(date), (
                "State should be updated with run date"
            )
            assert result.status == "success", f"Run should succeed for day {i + 1}"

        # Verify state file exists
        assert state_path.exists(), "State file should exist after runs"

        # Verify aggregated outputs (if any)
        # Note: Equity curve aggregation might be done separately

        # Check that equity changes over time (should not be flat)
        equity_values = [r.state_after.equity for r in results]
        assert len(set(equity_values)) > 1, "Equity should change over time"

        # Check that total_trades increases
        trade_counts = [r.state_after.total_trades for r in results]
        assert all(
            trade_counts[i] >= trade_counts[i - 1] for i in range(1, len(trade_counts))
        ), "Total trades should be non-decreasing"

    finally:
        # Restore original function
        paper_module.load_eod_prices_for_universe = original_load


@pytest.mark.advanced
def test_paper_track_determinism(
    paper_track_config: PaperTrackConfig,
    synthetic_prices_5days: pd.DataFrame,
    tmp_path: Path,
):
    """Test that paper track produces deterministic results with fixed seed."""
    import src.assembled_core.paper.paper_track as paper_module

    def mock_load_prices(universe_file, freq):
        return synthetic_prices_5days.copy()

    original_load = paper_module.load_eod_prices_for_universe
    paper_module.load_eod_prices_for_universe = mock_load_prices

    try:
        # Run first time
        output_root1 = tmp_path / "run1" / "output" / "paper_track" / "test_strategy"
        config1 = PaperTrackConfig(
            **{
                **paper_track_config.__dict__,
                "output_root": str(output_root1),
                "random_seed": 42,
            }
        )

        state_path1 = output_root1 / "state" / "state.json"
        date = pd.Timestamp("2025-01-06", tz="UTC")

        result1 = run_paper_day(config1, date, state_path1)
        write_paper_day_outputs(result1, output_root1)
        save_paper_state(result1.state_after, state_path1)

        # Run second time with same seed
        output_root2 = tmp_path / "run2" / "output" / "paper_track" / "test_strategy"
        config2 = PaperTrackConfig(
            **{
                **paper_track_config.__dict__,
                "output_root": str(output_root2),
                "random_seed": 42,
            }
        )

        state_path2 = output_root2 / "state" / "state.json"

        result2 = run_paper_day(config2, date, state_path2)
        write_paper_day_outputs(result2, output_root2)
        save_paper_state(result2.state_after, state_path2)

        # Compare results (should be identical)
        assert result1.state_after.equity == result2.state_after.equity, (
            f"Equity should be identical: {result1.state_after.equity} vs {result2.state_after.equity}"
        )
        assert result1.state_after.cash == result2.state_after.cash, (
            "Cash should be identical"
        )
        assert result1.daily_pnl == result2.daily_pnl, "Daily PnL should be identical"
        assert result1.trades_count == result2.trades_count, (
            "Trades count should be identical"
        )

        # Compare orders (should be identical)
        pd.testing.assert_frame_equal(
            result1.orders.reset_index(drop=True),
            result2.orders.reset_index(drop=True),
            check_dtype=False,  # Allow minor dtype differences
        )

        # Compare positions (should be identical)
        pd.testing.assert_frame_equal(
            result1.state_after.positions.sort_values("symbol").reset_index(drop=True),
            result2.state_after.positions.sort_values("symbol").reset_index(drop=True),
            check_dtype=False,
        )

    finally:
        paper_module.load_eod_prices_for_universe = original_load


@pytest.mark.advanced
def test_paper_track_outputs_format(
    paper_track_config: PaperTrackConfig,
    synthetic_prices_5days: pd.DataFrame,
    tmp_path: Path,
):
    """Test that paper track outputs have correct format and content."""
    import src.assembled_core.paper.paper_track as paper_module

    def mock_load_prices(universe_file, freq):
        return synthetic_prices_5days.copy()

    original_load = paper_module.load_eod_prices_for_universe
    paper_module.load_eod_prices_for_universe = mock_load_prices

    try:
        output_root = Path(paper_track_config.output_root)
        state_path = output_root / "state" / "state.json"
        date = pd.Timestamp("2025-01-06", tz="UTC")

        result = run_paper_day(paper_track_config, date, state_path)
        write_paper_day_outputs(result, output_root)

        run_date_str = result.date.strftime("%Y%m%d")
        run_dir = output_root / "runs" / run_date_str

        # Check equity_snapshot.json
        with open(run_dir / "equity_snapshot.json", "r", encoding="utf-8") as f:
            equity_snapshot = json.load(f)

        assert "timestamp" in equity_snapshot
        assert "equity" in equity_snapshot
        assert "cash" in equity_snapshot
        assert "positions_value" in equity_snapshot
        assert equity_snapshot["equity"] > 0, "Equity should be positive"

        # Check daily_summary.json
        with open(run_dir / "daily_summary.json", "r", encoding="utf-8") as f:
            daily_summary = json.load(f)

        assert "date" in daily_summary
        assert "equity" in daily_summary
        assert "daily_return_pct" in daily_summary
        assert "trades_count" in daily_summary
        assert daily_summary["trades_count"] >= 0, "Trades count should be non-negative"

        # Check positions.csv
        positions_df = pd.read_csv(run_dir / "positions.csv")
        assert "symbol" in positions_df.columns
        assert "qty" in positions_df.columns

        # Check daily_summary.md (should be readable)
        summary_md = (run_dir / "daily_summary.md").read_text(encoding="utf-8")
        assert "Daily Summary" in summary_md
        assert "Equity:" in summary_md

    finally:
        paper_module.load_eod_prices_for_universe = original_load
