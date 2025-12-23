"""Tests for Paper-Track price panel robustness (missing data, NaNs, delisted symbols)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.paper.paper_track import (
    PaperTrackConfig,
    PaperTrackState,
    filter_tradeable_universe,
    run_paper_day,
    write_paper_day_outputs,
)

pytestmark = pytest.mark.advanced


@pytest.fixture
def universe_file_with_3_symbols(tmp_path: Path) -> Path:
    """Create a universe file with 3 symbols."""
    universe_file = tmp_path / "universe.txt"
    universe_file.write_text("AAPL\nMSFT\nGOOGL\n", encoding="utf-8")
    return universe_file


@pytest.fixture
def prices_with_nan_symbol() -> pd.DataFrame:
    """Create price data where one symbol (GOOGL) has NaN prices."""
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"], utc=True)
    symbols = ["AAPL", "MSFT", "GOOGL"]
    data = []
    for date in dates:
        for symbol in symbols:
            if symbol == "GOOGL":
                # GOOGL has NaN close prices
                close = float("nan")
            else:
                close = 100.0 + (dates.get_loc(date) * 0.5)
            data.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "open": close * 0.99 if not pd.isna(close) else float("nan"),
                    "high": close * 1.01 if not pd.isna(close) else float("nan"),
                    "low": close * 0.98 if not pd.isna(close) else float("nan"),
                    "close": close,
                    "volume": 1000000.0 if not pd.isna(close) else 0.0,
                }
            )
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def prices_with_missing_symbol() -> pd.DataFrame:
    """Create price data where one symbol (GOOGL) is completely missing."""
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"], utc=True)
    symbols = ["AAPL", "MSFT"]  # GOOGL missing
    data = []
    for date in dates:
        for symbol in symbols:
            close = 100.0 + (dates.get_loc(date) * 0.5)
            data.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1000000.0,
                }
            )
    df = pd.DataFrame(data)
    return df


def test_filter_tradeable_universe_excludes_nan_symbols(
    prices_with_nan_symbol: pd.DataFrame,
) -> None:
    """Test that filter_tradeable_universe excludes symbols with NaN prices."""
    # Filter to last date
    as_of = pd.Timestamp("2025-01-03", tz="UTC")
    prices_filtered = prices_with_nan_symbol[
        prices_with_nan_symbol["timestamp"] <= as_of
    ].copy()
    prices_filtered = (
        prices_filtered.groupby("symbol", group_keys=False).last().reset_index()
    )

    universe_symbols = ["AAPL", "MSFT", "GOOGL"]
    tradeable, n_requested, n_tradeable, n_missing = filter_tradeable_universe(
        prices_filtered=prices_filtered,
        universe_symbols=universe_symbols,
        min_history_days=0,
    )

    assert n_requested == 3
    assert n_tradeable == 2  # AAPL and MSFT only
    assert n_missing == 1  # GOOGL
    assert len(tradeable) == 2
    assert "GOOGL" not in tradeable["symbol"].values
    assert "AAPL" in tradeable["symbol"].values
    assert "MSFT" in tradeable["symbol"].values


def test_filter_tradeable_universe_excludes_missing_symbols(
    prices_with_missing_symbol: pd.DataFrame,
) -> None:
    """Test that filter_tradeable_universe excludes symbols not in price data."""
    # Filter to last date
    as_of = pd.Timestamp("2025-01-03", tz="UTC")
    prices_filtered = prices_with_missing_symbol[
        prices_with_missing_symbol["timestamp"] <= as_of
    ].copy()
    prices_filtered = (
        prices_filtered.groupby("symbol", group_keys=False).last().reset_index()
    )

    universe_symbols = ["AAPL", "MSFT", "GOOGL"]  # GOOGL not in prices
    tradeable, n_requested, n_tradeable, n_missing = filter_tradeable_universe(
        prices_filtered=prices_filtered,
        universe_symbols=universe_symbols,
        min_history_days=0,
    )

    assert n_requested == 3
    assert n_tradeable == 2  # AAPL and MSFT only
    assert n_missing == 1  # GOOGL
    assert len(tradeable) == 2
    assert "GOOGL" not in tradeable["symbol"].values
    assert "AAPL" in tradeable["symbol"].values
    assert "MSFT" in tradeable["symbol"].values


def test_run_paper_day_with_nan_symbol_no_orders(
    tmp_path: Path,
    universe_file_with_3_symbols: Path,
    prices_with_nan_symbol: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that run_paper_day handles NaN symbols gracefully and produces no orders for them."""
    # Mock load_eod_prices_for_universe to return our synthetic data
    def _mock_load_eod_prices_for_universe(universe_file, freq):
        return prices_with_nan_symbol

    monkeypatch.setattr(
        "src.assembled_core.paper.paper_track.load_eod_prices_for_universe",
        _mock_load_eod_prices_for_universe,
    )

    output_root = tmp_path / "output" / "paper_track" / "test_strategy"
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "state" / "state.json"

    config = PaperTrackConfig(
        strategy_name="test_strategy",
        strategy_type="trend_baseline",
        strategy_params={"ma_fast": 2, "ma_slow": 3, "top_n": 2, "min_score": 0.0},
        universe_file=universe_file_with_3_symbols,
        freq="1d",
        seed_capital=100000.0,
        output_root=output_root,
        enable_pit_checks=False,
    )

    as_of = pd.Timestamp("2025-01-03", tz="UTC")
    result = run_paper_day(config=config, as_of=as_of, state_path=state_path)

    assert result.status == "success"
    assert result.n_symbols_requested == 3
    assert result.n_tradeable == 2  # AAPL and MSFT only
    assert result.n_missing == 1  # GOOGL
    # Should not have orders for GOOGL (NaN symbol)
    if not result.orders.empty:
        assert "GOOGL" not in result.orders["symbol"].values


def test_run_paper_day_with_missing_price_at_as_of(
    tmp_path: Path,
    universe_file_with_3_symbols: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that run_paper_day handles missing price at as_of date gracefully."""
    # Create price data where last date is missing for one symbol
    dates = pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True)  # No 2025-01-03
    symbols = ["AAPL", "MSFT", "GOOGL"]
    data = []
    for date in dates:
        for symbol in symbols:
            close = 100.0 + (dates.get_loc(date) * 0.5)
            data.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1000000.0,
                }
            )
    prices = pd.DataFrame(data)

    # Mock load_eod_prices_for_universe
    def _mock_load_eod_prices_for_universe(universe_file, freq):
        return prices

    monkeypatch.setattr(
        "src.assembled_core.paper.paper_track.load_eod_prices_for_universe",
        _mock_load_eod_prices_for_universe,
    )

    output_root = tmp_path / "output" / "paper_track" / "test_strategy"
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "state" / "state.json"

    config = PaperTrackConfig(
        strategy_name="test_strategy",
        strategy_type="trend_baseline",
        strategy_params={"ma_fast": 2, "ma_slow": 3, "top_n": 2, "min_score": 0.0},
        universe_file=universe_file_with_3_symbols,
        freq="1d",
        seed_capital=100000.0,
        output_root=output_root,
        enable_pit_checks=False,
    )

    # Run for 2025-01-03 (date not in prices - will use last available)
    as_of = pd.Timestamp("2025-01-03", tz="UTC")
    result = run_paper_day(config=config, as_of=as_of, state_path=state_path)

    # Should succeed (uses last available prices)
    assert result.status == "success"
    assert result.n_symbols_requested == 3
    # All symbols should be tradeable (last available price from 2025-01-02)
    assert result.n_tradeable == 3
    assert result.n_missing == 0


def test_daily_summary_includes_universe_stats(
    tmp_path: Path,
    universe_file_with_3_symbols: Path,
    prices_with_nan_symbol: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that daily_summary.json includes n_symbols_requested, n_tradeable, n_missing."""
    # Mock load_eod_prices_for_universe
    def _mock_load_eod_prices_for_universe(universe_file, freq):
        return prices_with_nan_symbol

    monkeypatch.setattr(
        "src.assembled_core.paper.paper_track.load_eod_prices_for_universe",
        _mock_load_eod_prices_for_universe,
    )

    output_root = tmp_path / "output" / "paper_track" / "test_strategy"
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "state" / "state.json"

    config = PaperTrackConfig(
        strategy_name="test_strategy",
        strategy_type="trend_baseline",
        strategy_params={"ma_fast": 2, "ma_slow": 3, "top_n": 2, "min_score": 0.0},
        universe_file=universe_file_with_3_symbols,
        freq="1d",
        seed_capital=100000.0,
        output_root=output_root,
        enable_pit_checks=False,
    )

    as_of = pd.Timestamp("2025-01-03", tz="UTC")
    result = run_paper_day(config=config, as_of=as_of, state_path=state_path)

    assert result.status == "success"

    # Write outputs
    write_paper_day_outputs(result, output_root, config)

    # Check daily_summary.json
    run_dir = output_root / "runs" / "20250103"
    assert (run_dir / "daily_summary.json").exists()

    import json

    with open(run_dir / "daily_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert "n_symbols_requested" in summary
    assert "n_tradeable" in summary
    assert "n_missing" in summary
    assert summary["n_symbols_requested"] == 3
    assert summary["n_tradeable"] == 2
    assert summary["n_missing"] == 1

    # Check daily_summary.md includes universe section
    assert (run_dir / "daily_summary.md").exists()
    with open(run_dir / "daily_summary.md", "r", encoding="utf-8") as f:
        md_content = f.read()

    assert "## Universe" in md_content
    assert "Symbols Requested: 3" in md_content
    assert "Tradeable: 2" in md_content
    assert "Missing/Invalid: 1" in md_content

