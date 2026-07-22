# tests/test_integration_run_daily.py
"""Sprint 11.1: Integration test for run_daily EOD pipeline.

This module tests the complete EOD pipeline flow:
- Loading price data
- Computing features
- Generating signals
- Computing target positions
- Generating orders
- Writing SAFE-CSV files

Tests cover:
- Complete pipeline execution with sample data
- Output file generation and validation
- Order file structure and content
- Cleanup of temporary files
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from scripts.run_eod_pipeline import main

pytestmark = pytest.mark.fast
pytestmark = [pytestmark, pytest.mark.integration]


@pytest.fixture
def sample_price_file(tmp_path: Path):
    """Create a sample price Parquet file for testing."""
    # Create realistic price data for multiple symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    data = []

    for symbol in symbols:
        dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
        base_price = 100.0 if symbol == "AAPL" else 200.0 if symbol == "MSFT" else 150.0

        symbol_data = pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": [symbol] * 100,
                "open": [base_price + i * 0.1 for i in range(100)],
                "high": [base_price + 5.0 + i * 0.1 for i in range(100)],
                "low": [base_price - 1.0 + i * 0.1 for i in range(100)],
                "close": [base_price + 2.0 + i * 0.1 for i in range(100)],
                "volume": [1000000.0 + i * 1000 for i in range(100)],
            }
        )
        data.append(symbol_data)

    df = pd.concat(data, ignore_index=True)
    price_file = tmp_path / "sample_prices.parquet"
    df.to_parquet(price_file, index=False)

    return price_file


@pytest.fixture
def sample_universe_file(tmp_path: Path):
    """Create a sample universe file for testing."""
    universe_file = tmp_path / "universe.txt"
    universe_file.write_text("AAPL\nMSFT\nGOOGL\n")
    return universe_file


def test_run_daily_complete_pipeline(
    sample_price_file, sample_universe_file, tmp_path: Path, monkeypatch
):
    """Test complete EOD pipeline execution with sample data.

    This integration test:
    1. Creates sample price data and universe file
    2. Runs the EOD pipeline via main()
    3. Verifies that order files are generated
    4. Validates order file structure and content
    """
    # Set output directory to tmp_path
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mock the output directory in config if needed
    # For now, we'll pass it via CLI arguments

    # Prepare arguments for run_eod_pipeline
    import sys
    from unittest.mock import patch

    # We need to call main() with proper arguments
    # Since main() uses argparse, we'll need to mock sys.argv
    test_args = [
        "run_eod_pipeline.py",
        "--freq",
        "1d",
        "--price-file",
        str(sample_price_file),
        "--universe",
        str(sample_universe_file),
        "--start-capital",
        "10000.0",
        "--out",
        str(output_dir),
    ]

    # NOTE: previously this block wrapped main() in `except Exception: pass`
    # followed by `assert True` — the only integration test of the EOD
    # entrypoint passed even if the pipeline crashed. We now (a) let any crash
    # surface (no bare swallow) and (b) assert real postconditions: a valid
    # non-fatal exit code AND concrete output artifacts on disk.
    #
    # main() returns an int exit code and does NOT call sys.exit() itself (only
    # the __main__ guard does), so SystemExit should not normally be raised. We
    # still tolerate SystemExit defensively but assert on its code.
    rc: int
    with patch.object(sys, "argv", test_args):
        try:
            rc = main()
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 0

    # main() exit-code contract (run_eod_pipeline.main):
    #   0 = success, 2 = QA error, 3 = QA warning, 1 = FATAL/unhandled crash.
    # With synthetic perfectly-linear prices the QA health checks legitimately
    # flag (Sharpe=None etc.), so exit code 2 is expected here. Exit code 1
    # means an unhandled exception inside main() — i.e. the pipeline crashed —
    # and MUST fail this integration test.
    assert rc != 1, (
        f"run_eod_pipeline.main() returned 1 (FATAL/unhandled crash); "
        f"the EOD pipeline did not complete. Output dir contents: "
        f"{[p.name for p in output_dir.iterdir()] if output_dir.exists() else 'MISSING'}"
    )
    assert rc in (0, 2, 3), (
        f"run_eod_pipeline.main() returned unexpected exit code {rc!r}; "
        "expected 0 (ok), 2 (qa error) or 3 (qa warning)."
    )

    # The pipeline must have actually written its run manifest to output_dir.
    assert output_dir.exists(), "Output directory should be created"
    manifest_files = list(output_dir.glob("run_manifest_*.json"))
    assert manifest_files, (
        "EOD pipeline produced no run_manifest_*.json — the pipeline did not "
        f"reach the manifest-writing step. output_dir contents: "
        f"{[p.name for p in output_dir.iterdir()]}"
    )

    # And it must have produced concrete output artifacts beyond the manifest
    # (ledger / accounting / reconcile / qa report directories or files).
    produced = [p for p in output_dir.iterdir() if p.name not in {".gitkeep"}]
    assert len(produced) >= 2, (
        "EOD pipeline produced fewer artifacts than expected (only "
        f"{[p.name for p in produced]}); expected manifest plus ledger/"
        "accounting/qa output."
    )


def test_run_daily_creates_order_file_structure(
    sample_price_file, sample_universe_file, tmp_path: Path, monkeypatch
):
    """Test that order files have the expected structure.

    This test verifies:
    - Order files are created (if orders are generated)
    - Order files have required columns
    - Order files are not empty (if signals were generated)
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run a simplified version: directly call the pipeline functions
    # This avoids argparse complexity
    from src.assembled_core.data.prices_ingest import load_eod_prices
    from src.assembled_core.features.ta_features import add_all_features
    from src.assembled_core.signals.rules_trend import (
        generate_trend_signals_from_prices,
    )
    from src.assembled_core.portfolio.position_sizing import (
        compute_target_positions_from_trend_signals,
    )
    from src.assembled_core.execution.order_generation import (
        generate_orders_from_signals,
    )
    from src.assembled_core.execution.safe_bridge import write_safe_orders_csv

    # Step 1: Load prices
    prices = load_eod_prices(price_file=sample_price_file)
    assert not prices.empty, "Price data should be loaded"

    # Step 2: Add features
    prices_with_features = add_all_features(prices, ma_windows=(20, 50))
    assert "ma_20" in prices_with_features.columns, "Moving averages should be computed"

    # Step 3: Generate signals (use latest date)
    latest_date = prices_with_features["timestamp"].max()
    latest_prices = prices_with_features[
        prices_with_features["timestamp"] == latest_date
    ]

    signals = generate_trend_signals_from_prices(latest_prices, ma_fast=20, ma_slow=50)
    assert "direction" in signals.columns, "Signals should have direction column"

    # Step 4: Compute target positions
    if not signals[signals["direction"] == "LONG"].empty:
        targets = compute_target_positions_from_trend_signals(
            signals, total_capital=10000.0, top_n=5
        )

        # Step 5: Generate orders
        if not targets.empty:
            orders = generate_orders_from_signals(
                signals,
                total_capital=10000.0,
                top_n=5,
                timestamp=latest_date,
                prices=latest_prices,
            )

            # Step 6: Write SAFE-CSV
            if not orders.empty:
                # write_safe_orders_csv expects a file path, not a directory
                # If output_path is a directory, it will create a file inside it
                # Otherwise, we need to provide a full file path
                order_file_path = (
                    output_dir / f"orders_{latest_date.date().strftime('%Y%m%d')}.csv"
                )
                order_file = write_safe_orders_csv(
                    orders,
                    date=latest_date.date(),
                    output_path=order_file_path,
                    price_type="MARKET",
                    comment="Integration test",
                )

                # Verify file exists
                assert order_file.exists(), (
                    f"Order file should be created: {order_file}"
                )

                # Verify file structure
                order_df = pd.read_csv(order_file)
                assert not order_df.empty, "Order file should not be empty"

                # Check for expected columns (SAFE-CSV format uses: Ticker, Side, Quantity, PriceType, Comment)
                required_cols = ["Ticker", "Side", "Quantity"]
                for col in required_cols:
                    assert col in order_df.columns, (
                        f"Order file should have '{col}' column"
                    )

                # Verify data types
                assert pd.api.types.is_string_dtype(order_df["Ticker"]), (
                    "Ticker should be string"
                )
                assert order_df["Side"].isin(["BUY", "SELL"]).all(), (
                    "Side should be BUY or SELL"
                )
                assert order_df["Quantity"].dtype in [
                    float,
                    "float64",
                ], "Quantity should be numeric"


def test_run_daily_handles_empty_signals(
    sample_price_file, sample_universe_file, tmp_path: Path
):
    """Test that pipeline handles the case when no signals are generated.

    This test verifies:
    - Pipeline completes successfully even with no signals
    - Appropriate handling of empty order lists
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create price data that will generate no signals (e.g., all FLAT)
    # Downward trend should generate FLAT signals
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    prices = [200.0 - i * 0.5 for i in range(100)]  # Downward trend

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["AAPL"] * 100,
            "open": [p - 1.0 for p in prices],
            "high": [p + 2.0 for p in prices],
            "low": [p - 2.0 for p in prices],
            "close": prices,
            "volume": [1000000.0] * 100,
        }
    )

    price_file = tmp_path / "downward_prices.parquet"
    df.to_parquet(price_file, index=False)

    # Run pipeline steps
    from src.assembled_core.data.prices_ingest import load_eod_prices
    from src.assembled_core.features.ta_features import add_all_features
    from src.assembled_core.signals.rules_trend import (
        generate_trend_signals_from_prices,
    )

    prices = load_eod_prices(price_file=price_file)
    prices_with_features = add_all_features(prices, ma_windows=(20, 50))

    latest_date = prices_with_features["timestamp"].max()
    latest_prices = prices_with_features[
        prices_with_features["timestamp"] == latest_date
    ]

    signals = generate_trend_signals_from_prices(latest_prices, ma_fast=20, ma_slow=50)

    # With a strictly downward trend the fast MA sits below the slow MA at the
    # latest date, so the trend rule must not emit any LONG signal.
    assert not signals.empty, "Signal frame should contain a row for AAPL"
    assert signals["direction"].isin(["LONG", "FLAT"]).all(), (
        "direction must only contain LONG or FLAT"
    )
    long_signals = signals[signals["direction"] == "LONG"]
    assert long_signals.empty, (
        "Strict downward trend (MA20 < MA50) must not generate LONG signals"
    )
