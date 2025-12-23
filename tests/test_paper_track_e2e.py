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
from src.assembled_core.utils.random_state import set_global_seed

pytestmark = pytest.mark.advanced


@pytest.fixture
def synthetic_prices_5days(tmp_path: Path) -> pd.DataFrame:
    """Create synthetic price data for 5 days (3 symbols, 5 days) with deterministic variation.
    
    Uses fixed seed for determinism. Prices have drift + noise for realistic variation.
    """
    # Set seed for deterministic price generation
    set_global_seed(42)
    
    symbols = ["AAPL", "MSFT", "GOOGL"]
    dates = pd.date_range("2025-01-06", periods=5, freq="D", tz="UTC")

    # Create price data with guaranteed variation (drift + noise)
    data = []
    for sym_idx, sym in enumerate(symbols):
        # Different base prices per symbol
        base_price = 100.0 + (sym_idx * 50.0)  # AAPL=100, MSFT=150, GOOGL=200
        
        # Symbol-specific drift (different direction per symbol)
        drift_per_day = 0.8 + (sym_idx * 0.3)  # AAPL=0.8, MSFT=1.1, GOOGL=1.4
        
        for i, date in enumerate(dates):
            # Cumulative drift + noise (deterministic via seeded RNG)
            drift_component = i * drift_per_day
            noise_component = np.random.randn() * 2.0  # 2% std dev
            
            price = base_price + drift_component + noise_component
            # Ensure price is positive
            price = max(price, 1.0)
            
            data.append(
                {
                    "timestamp": date,
                    "symbol": sym,
                    "open": price * 0.998,  # Open slightly below close
                    "high": price * 1.015,  # High ~1.5% above close
                    "low": price * 0.985,   # Low ~1.5% below close
                    "close": price,
                    "volume": 1000000.0 + (i * 50000.0) + (sym_idx * 100000.0),
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
            write_paper_day_outputs(result, output_root, config=paper_track_config)

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

        # Verify aggregated outputs exist
        aggregates_dir = output_root / "aggregates"
        assert aggregates_dir.exists(), "Aggregates directory should exist"

        # Check equity_curve.csv
        equity_curve_path = aggregates_dir / "equity_curve.csv"
        assert equity_curve_path.exists(), "equity_curve.csv should exist"
        equity_curve = pd.read_csv(equity_curve_path)
        assert len(equity_curve) == len(results), (
            f"equity_curve should have {len(results)} rows (one per day), got {len(equity_curve)}"
        )
        assert "date" in equity_curve.columns
        assert "equity" in equity_curve.columns
        assert "cash" in equity_curve.columns

        # Check trades_all.csv
        trades_all_path = aggregates_dir / "trades_all.csv"
        assert trades_all_path.exists(), "trades_all.csv should exist"
        trades_all = pd.read_csv(trades_all_path)
        assert "date" in trades_all.columns
        # Should have at least one row if any trades occurred
        total_trades = sum(r.trades_count for r in results)
        if total_trades > 0:
            assert len(trades_all) > 0, "trades_all should have rows if trades occurred"

        # Check positions_history.csv
        positions_history_path = aggregates_dir / "positions_history.csv"
        assert positions_history_path.exists(), "positions_history.csv should exist"
        positions_history = pd.read_csv(positions_history_path)
        assert "date" in positions_history.columns
        # Should have at least one row per day (may be empty positions)
        unique_dates = positions_history["date"].nunique() if not positions_history.empty else 0
        assert unique_dates == len(results), (
            f"positions_history should have {len(results)} unique dates, got {unique_dates}"
        )

        # Verify no duplicate dates in equity_curve
        assert equity_curve["date"].nunique() == len(equity_curve), (
            "equity_curve should not have duplicate dates"
        )

    finally:
        paper_module.load_eod_prices_for_universe = original_load


@pytest.mark.advanced
def test_paper_track_aggregated_artifacts_parquet(
    paper_track_config: PaperTrackConfig,
    synthetic_prices_5days: pd.DataFrame,
    tmp_path: Path,
):
    """Test that aggregated artifacts can be written as Parquet files."""
    import src.assembled_core.paper.paper_track as paper_module

    def mock_load_prices(universe_file, freq):
        return synthetic_prices_5days.copy()

    original_load = paper_module.load_eod_prices_for_universe
    paper_module.load_eod_prices_for_universe = mock_load_prices

    try:
        # Create config with parquet output format
        config_parquet = PaperTrackConfig(
            **{
                **paper_track_config.__dict__,
                "output_format": "parquet",
            }
        )

        output_root = Path(config_parquet.output_root)
        state_path = output_root / "state" / "state.json"

        results = []
        dates = synthetic_prices_5days["timestamp"].unique()[:3]  # 3 days

        # Run for each day
        for date in sorted(dates):
            result = run_paper_day(
                config=config_parquet,
                as_of=pd.Timestamp(date),
                state_path=state_path,
            )
            results.append(result)

            # Write outputs
            write_paper_day_outputs(result, output_root, config=config_parquet)

            # Save state
            save_paper_state(result.state_after, state_path)

        # Verify aggregated outputs exist as Parquet files
        aggregates_dir = output_root / "aggregates"
        assert aggregates_dir.exists(), "Aggregates directory should exist"

        # Check equity_curve.parquet
        equity_curve_path = aggregates_dir / "equity_curve.parquet"
        assert equity_curve_path.exists(), "equity_curve.parquet should exist"
        equity_curve = pd.read_parquet(equity_curve_path)
        assert len(equity_curve) == len(results), (
            f"equity_curve should have {len(results)} rows (one per day), got {len(equity_curve)}"
        )
        assert "date" in equity_curve.columns
        assert "equity" in equity_curve.columns
        assert "cash" in equity_curve.columns

        # Check trades_all.parquet
        trades_all_path = aggregates_dir / "trades_all.parquet"
        assert trades_all_path.exists(), "trades_all.parquet should exist"
        trades_all = pd.read_parquet(trades_all_path)
        assert "date" in trades_all.columns

        # Check positions_history.parquet
        positions_history_path = aggregates_dir / "positions_history.parquet"
        assert positions_history_path.exists(), "positions_history.parquet should exist"
        positions_history = pd.read_parquet(positions_history_path)
        assert "date" in positions_history.columns
        unique_dates = positions_history["date"].nunique() if not positions_history.empty else 0
        assert unique_dates == len(results), (
            f"positions_history should have {len(results)} unique dates, got {unique_dates}"
        )

    finally:
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
        write_paper_day_outputs(result1, output_root1, config=config1)
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
        write_paper_day_outputs(result2, output_root2, config=config2)
        save_paper_state(result2.state_after, state_path2)

        # Compare results (should be identical within floating point tolerance)
        np.testing.assert_allclose(
            result1.state_after.equity,
            result2.state_after.equity,
            rtol=1e-10,
            err_msg="Equity should be identical",
        )
        np.testing.assert_allclose(
            result1.state_after.cash,
            result2.state_after.cash,
            rtol=1e-10,
            err_msg="Cash should be identical",
        )
        np.testing.assert_allclose(
            result1.daily_pnl,
            result2.daily_pnl,
            rtol=1e-10,
            err_msg="Daily PnL should be identical",
        )
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

    # Ensure deterministic prices
    set_global_seed(42)
    def mock_load_prices(universe_file, freq):
        return synthetic_prices_5days.copy()

    original_load = paper_module.load_eod_prices_for_universe
    paper_module.load_eod_prices_for_universe = mock_load_prices

    try:
        output_root = Path(paper_track_config.output_root)
        state_path = output_root / "state" / "state.json"
        date = pd.Timestamp("2025-01-06", tz="UTC")

        result = run_paper_day(paper_track_config, date, state_path)
        write_paper_day_outputs(result, output_root, config=paper_track_config)

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

        # Check manifest.json exists and has required fields
        manifest_path = run_dir / "manifest.json"
        assert manifest_path.exists(), "manifest.json should exist"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert "date" in manifest
        assert "strategy_name" in manifest
        assert "config_hash" in manifest
        assert "git_commit_hash" in manifest  # May be None if git unavailable
        assert "state_before" in manifest
        assert "state_after" in manifest
        assert "run_summary" in manifest
        assert "artifacts" in manifest
        assert "manifest.json" in manifest["artifacts"]

    finally:
        paper_module.load_eod_prices_for_universe = original_load
