"""Tests for run_daily.py factor store integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.assembled_core.pipeline.trading_cycle_shared import TradingCycleResult

pytestmark = pytest.mark.advanced


def _make_mock_trading_result(dates):
    """Create a successful TradingCycleResult with minimal data."""
    orders_df = pd.DataFrame(
        {
            "timestamp": dates[:1],
            "symbol": ["AAPL"],
            "side": ["BUY"],
            "qty": [100.0],
            "price": [102.0],
        }
    )
    signals_df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["AAPL"] * len(dates),
            "direction": ["LONG"] * len(dates),
            "score": [0.5] * len(dates),
        }
    )
    targets_df = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "target_weight": [0.5],
            "target_qty": [100.0],
        }
    )
    return TradingCycleResult(
        signals=signals_df,
        target_positions=targets_df,
        orders=orders_df,
        orders_filtered=orders_df,
        status="success",
    )


def test_run_daily_eod_with_factor_store_flag(tmp_path: Path) -> None:
    """Test that --use-factor-store flag passes use_factor_store=True to TradingContext."""
    from scripts.run_daily import run_daily_eod

    dates = pd.date_range("2024-01-01", "2024-01-05", freq="D", tz="UTC")
    prices = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["AAPL"] * len(dates),
            "open": [100.0] * len(dates),
            "high": [105.0] * len(dates),
            "low": [95.0] * len(dates),
            "close": [102.0] * len(dates),
            "volume": [1000000] * len(dates),
        }
    )

    mock_settings = MagicMock()
    mock_settings.watchlist_file = tmp_path / "watchlist.txt"

    with (
        patch(
            "src.assembled_core.config.settings.get_settings",
            return_value=mock_settings,
        ),
        patch(
            "src.assembled_core.data.prices_ingest.load_eod_prices_for_universe",
            return_value=prices,
        ),
        patch("scripts.run_daily.generate_trend_signals_from_prices") as mock_signals,
        patch(
            "scripts.run_daily.compute_target_positions_from_trend_signals"
        ) as mock_targets,
        patch(
            "scripts.run_daily.run_trading_cycle",
            return_value=_make_mock_trading_result(dates),
        ) as mock_cycle,
        patch(
            "scripts.run_daily.write_safe_orders_csv",
            return_value=tmp_path / "orders.csv",
        ),
    ):
        mock_signals.return_value = pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": ["AAPL"] * len(dates),
                "direction": ["LONG"] * len(dates),
                "score": [0.5] * len(dates),
            }
        )
        mock_targets.return_value = pd.DataFrame(
            {"symbol": ["AAPL"], "target_weight": [0.5], "target_qty": [100.0]}
        )

        _ = run_daily_eod(
            date_str="2024-01-05",
            price_file=None,
            output_dir=tmp_path / "output",
            use_factor_store=True,
            factor_group="core_ta",
            factor_store_root=tmp_path / "factors",
        )

        # Verify run_trading_cycle was called and its TradingContext has factor store enabled
        mock_cycle.assert_called_once()
        ctx = mock_cycle.call_args[0][0]  # First positional arg is the TradingContext
        assert ctx.use_factor_store is True
        assert ctx.factor_group == "core_ta"
        assert ctx.factor_store_root == tmp_path / "factors"


def test_run_daily_eod_without_factor_store_flag(tmp_path: Path) -> None:
    """Test that default behavior (no flag) passes use_factor_store=False to TradingContext."""
    from scripts.run_daily import run_daily_eod

    dates = pd.date_range("2024-01-01", "2024-01-05", freq="D", tz="UTC")
    prices = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["AAPL"] * len(dates),
            "open": [100.0] * len(dates),
            "high": [105.0] * len(dates),
            "low": [95.0] * len(dates),
            "close": [102.0] * len(dates),
            "volume": [1000000] * len(dates),
        }
    )

    mock_settings = MagicMock()
    mock_settings.watchlist_file = tmp_path / "watchlist.txt"

    with (
        patch(
            "src.assembled_core.config.settings.get_settings",
            return_value=mock_settings,
        ),
        patch(
            "src.assembled_core.data.prices_ingest.load_eod_prices_for_universe",
            return_value=prices,
        ),
        patch("scripts.run_daily.generate_trend_signals_from_prices") as mock_signals,
        patch(
            "scripts.run_daily.compute_target_positions_from_trend_signals"
        ) as mock_targets,
        patch(
            "scripts.run_daily.run_trading_cycle",
            return_value=_make_mock_trading_result(dates),
        ) as mock_cycle,
        patch(
            "scripts.run_daily.write_safe_orders_csv",
            return_value=tmp_path / "orders.csv",
        ),
    ):
        mock_signals.return_value = pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": ["AAPL"] * len(dates),
                "direction": ["LONG"] * len(dates),
                "score": [0.5] * len(dates),
            }
        )
        mock_targets.return_value = pd.DataFrame(
            {"symbol": ["AAPL"], "target_weight": [0.5], "target_qty": [100.0]}
        )

        _ = run_daily_eod(
            date_str="2024-01-05",
            price_file=None,
            output_dir=tmp_path / "output",
            use_factor_store=False,
        )

        # Verify TradingContext was built with use_factor_store=False
        mock_cycle.assert_called_once()
        ctx = mock_cycle.call_args[0][0]
        assert ctx.use_factor_store is False


def test_run_daily_eod_timings_metadata_with_factor_store(tmp_path: Path) -> None:
    """Test that timings metadata includes factor store info when enabled."""
    from scripts.run_daily import run_daily_eod

    dates = pd.date_range("2024-01-01", "2024-01-05", freq="D", tz="UTC")
    prices = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["AAPL"] * len(dates),
            "open": [100.0] * len(dates),
            "high": [105.0] * len(dates),
            "low": [95.0] * len(dates),
            "close": [102.0] * len(dates),
            "volume": [1000000] * len(dates),
        }
    )

    mock_settings = MagicMock()
    mock_settings.watchlist_file = tmp_path / "watchlist.txt"

    with (
        patch(
            "src.assembled_core.config.settings.get_settings",
            return_value=mock_settings,
        ),
        patch(
            "src.assembled_core.data.prices_ingest.load_eod_prices_for_universe",
            return_value=prices,
        ),
        patch("scripts.run_daily.generate_trend_signals_from_prices") as mock_signals,
        patch(
            "scripts.run_daily.compute_target_positions_from_trend_signals"
        ) as mock_targets,
        patch(
            "scripts.run_daily.run_trading_cycle",
            return_value=_make_mock_trading_result(dates),
        ),
        patch(
            "scripts.run_daily.write_safe_orders_csv",
            return_value=tmp_path / "orders.csv",
        ),
    ):
        mock_signals.return_value = pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": ["AAPL"] * len(dates),
                "direction": ["LONG"] * len(dates),
                "score": [0.5] * len(dates),
            }
        )
        mock_targets.return_value = pd.DataFrame(
            {"symbol": ["AAPL"], "target_weight": [0.5], "target_qty": [100.0]}
        )

        _ = run_daily_eod(
            date_str="2024-01-05",
            price_file=None,
            output_dir=tmp_path / "output",
            enable_timings=True,
            timings_out=tmp_path / "timings.json",
            use_factor_store=True,
            factor_group="core_ta",
            factor_store_root=tmp_path / "factors",
        )

        assert (tmp_path / "timings.json").exists()

        import json

        with open(tmp_path / "timings.json", "r") as f:
            timings_data = json.load(f)

        # Verify timings file was written with basic job metadata
        assert "job_meta" in timings_data
        job_meta = timings_data["job_meta"]
        assert "date" in job_meta
        assert "total_capital" in job_meta
