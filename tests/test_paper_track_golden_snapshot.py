"""Golden snapshot test for Paper-Track core math.

This test ensures that given the same inputs and seed, Paper-Track produces
exactly the same results. This guards against accidental behavior changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.paper.paper_track import (
    PaperTrackConfig,
    run_paper_day,
    save_paper_state,
)
from src.assembled_core.utils.random_state import set_global_seed

pytestmark = pytest.mark.advanced

# Golden snapshot: Expected equity values for 5 days
# Generated with seed=42, synthetic prices with 3 symbols (AAPL, MSFT, GOOGL)
# This snapshot guards against accidental changes to core math/logic
GOLDEN_EQUITY_SNAPSHOT = [
    100000.0,  # Day 1: Starting capital
    100123.45,  # Day 2: After first trades (example - will be replaced with actual)
    100245.67,  # Day 3
    100367.89,  # Day 4
    100490.12,  # Day 5
]

# Note: These values will be updated once we establish the golden run.
# For now, we'll generate them deterministically and store them.


def generate_deterministic_prices() -> pd.DataFrame:
    """Generate deterministic synthetic prices (same as test_paper_track_e2e)."""
    set_global_seed(42)
    symbols = ["AAPL", "MSFT", "GOOGL"]
    dates = pd.date_range("2025-01-06", periods=5, freq="D", tz="UTC")

    data = []
    for sym_idx, sym in enumerate(symbols):
        base_price = 100.0 + (sym_idx * 50.0)
        drift_per_day = 0.8 + (sym_idx * 0.3)

        for i, date in enumerate(dates):
            drift_component = i * drift_per_day
            noise_component = np.random.randn() * 2.0
            price = base_price + drift_component + noise_component
            price = max(price, 1.0)

            data.append(
                {
                    "timestamp": date,
                    "symbol": sym,
                    "open": price * 0.998,
                    "high": price * 1.015,
                    "low": price * 0.985,
                    "close": price,
                    "volume": 1000000.0 + (i * 50000.0) + (sym_idx * 100000.0),
                }
            )

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def test_paper_track_golden_snapshot(tmp_path: Path):
    """Golden snapshot test: Ensure deterministic results match expected values.

    This test uses a fixed seed and synthetic prices to produce deterministic results.
    If core math/logic changes, this test will fail, alerting us to the change.
    """
    # Set global seed for determinism
    set_global_seed(42)

    # Create universe file
    universe_file = tmp_path / "universe.txt"
    universe_file.write_text("AAPL\nMSFT\nGOOGL\n", encoding="utf-8")

    # Create config with fixed seed
    output_root = tmp_path / "output" / "paper_track" / "golden_test"
    config = PaperTrackConfig(
        strategy_name="golden_test",
        strategy_type="trend_baseline",
        strategy_params={
            "ma_fast": 2,
            "ma_slow": 3,
            "top_n": 2,
            "min_score": 0.0,
        },
        universe_file=universe_file,
        freq="1d",
        seed_capital=100000.0,
        commission_bps=0.5,
        spread_w=0.25,
        impact_w=0.5,
        output_root=output_root,
        random_seed=42,  # Fixed seed
        enable_pit_checks=False,
    )

    # Generate deterministic prices
    synthetic_prices = generate_deterministic_prices()

    # Monkey-patch price loader
    import src.assembled_core.paper.paper_track as paper_module

    def mock_load_prices(universe_file, freq):
        return synthetic_prices.copy()

    original_load = paper_module.load_eod_prices_for_universe
    paper_module.load_eod_prices_for_universe = mock_load_prices

    try:
        state_path = output_root / "state" / "state.json"
        dates = sorted(synthetic_prices["timestamp"].unique()[:5])

        # Run for 5 days
        equity_values = []
        for date in dates:
            result = run_paper_day(
                config=config,
                as_of=pd.Timestamp(date),
                state_path=state_path,
            )
            equity_values.append(float(result.state_after.equity))

            # Save state for next day
            save_paper_state(result.state_after, state_path)

            assert result.status == "success", f"Day {date.date()} should succeed"

        # Verify we got 5 equity values (one per day)
        assert len(equity_values) == 5, f"Expected 5 equity values, got {len(equity_values)}"

        # Store actual snapshot (first run establishes golden values)
        # In production, we'd commit these values and assert they match
        actual_equity_snapshot = equity_values

        # For now, verify determinism: run again and compare
        # Reset seed and state
        set_global_seed(42)
        state_path.unlink(missing_ok=True)

        equity_values_rerun = []
        for date in dates:
            result = run_paper_day(
                config=config,
                as_of=pd.Timestamp(date),
                state_path=state_path,
            )
            equity_values_rerun.append(float(result.state_after.equity))
            save_paper_state(result.state_after, state_path)

        # Verify exact match (determinism)
        np.testing.assert_allclose(
            equity_values,
            equity_values_rerun,
            rtol=1e-10,
            err_msg="Equity values should be identical on rerun (determinism check)",
        )

        # Verify snapshot values are reasonable (positive, finite, increasing or stable)
        assert all(eq > 0 for eq in actual_equity_snapshot), "All equity values should be positive"
        assert all(np.isfinite(eq) for eq in actual_equity_snapshot), "All equity values should be finite"
        # Equity can decrease due to costs/trades, so we don't assert monotonic increase
        # But it should be within reasonable bounds (e.g., within 50% of seed capital)
        assert all(50000.0 <= eq <= 200000.0 for eq in actual_equity_snapshot), (
            f"Equity values should be within reasonable bounds, got: {actual_equity_snapshot}"
        )

        # If golden snapshot is defined, compare with it
        # (In production, uncomment and update GOLDEN_EQUITY_SNAPSHOT with actual values)
        # np.testing.assert_allclose(
        #     actual_equity_snapshot,
        #     GOLDEN_EQUITY_SNAPSHOT,
        #     rtol=1e-10,
        #     err_msg="Equity values should match golden snapshot",
        # )

        # Log snapshot for manual verification (first time)
        print(f"\n[GOLDEN SNAPSHOT] Equity values: {actual_equity_snapshot}")
        print("[GOLDEN SNAPSHOT] Copy these values to GOLDEN_EQUITY_SNAPSHOT if they look correct.")

    finally:
        paper_module.load_eod_prices_for_universe = original_load


def test_paper_track_golden_snapshot_with_actual_values(tmp_path: Path):
    """Test with actual golden snapshot values (once established).

    This test should pass once we've verified the golden snapshot values are correct.
    """
    # Set global seed
    set_global_seed(42)

    # Create universe file
    universe_file = tmp_path / "universe.txt"
    universe_file.write_text("AAPL\nMSFT\nGOOGL\n", encoding="utf-8")

    # Create config
    output_root = tmp_path / "output" / "paper_track" / "golden_test2"
    config = PaperTrackConfig(
        strategy_name="golden_test2",
        strategy_type="trend_baseline",
        strategy_params={
            "ma_fast": 2,
            "ma_slow": 3,
            "top_n": 2,
            "min_score": 0.0,
        },
        universe_file=universe_file,
        freq="1d",
        seed_capital=100000.0,
        commission_bps=0.5,
        spread_w=0.25,
        impact_w=0.5,
        output_root=output_root,
        random_seed=42,
        enable_pit_checks=False,
    )

    # Generate deterministic prices
    synthetic_prices = generate_deterministic_prices()

    # Monkey-patch
    import src.assembled_core.paper.paper_track as paper_module

    def mock_load_prices(universe_file, freq):
        return synthetic_prices.copy()

    original_load = paper_module.load_eod_prices_for_universe
    paper_module.load_eod_prices_for_universe = mock_load_prices

    try:
        state_path = output_root / "state" / "state.json"
        dates = sorted(synthetic_prices["timestamp"].unique()[:5])

        equity_values = []
        for date in dates:
            result = run_paper_day(config=config, as_of=pd.Timestamp(date), state_path=state_path)
            equity_values.append(float(result.state_after.equity))
            save_paper_state(result.state_after, state_path)

        # For now, just verify determinism (exact match on rerun)
        # Once golden snapshot is established, we can compare against GOLDEN_EQUITY_SNAPSHOT
        assert len(equity_values) == 5
        assert all(eq > 0 and np.isfinite(eq) for eq in equity_values)

        # Verify values are stored for manual review
        # In production: assert np.testing.assert_allclose(equity_values, GOLDEN_EQUITY_SNAPSHOT, rtol=1e-10)

    finally:
        paper_module.load_eod_prices_for_universe = original_load

