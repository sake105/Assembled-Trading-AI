"""Tests for run_backtest_strategy.py factor store integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

pytestmark = pytest.mark.advanced


def test_backtest_with_factor_store_flag(tmp_path: Path) -> None:
    """Test that --use-factor-store flag calls build_or_load_factors in backtest engine."""
    from src.assembled_core.qa.backtest_engine import run_portfolio_backtest

    dates = pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * len(dates),
        "open": [100.0] * len(dates),
        "high": [105.0] * len(dates),
        "low": [95.0] * len(dates),
        "close": [102.0] * len(dates),
        "volume": [1000000] * len(dates),
    })

    def signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": dates[:5],
            "symbol": ["AAPL"] * 5,
            "direction": ["LONG"] * 5,
            "score": [0.5] * 5,
        })

    def position_sizing_fn(signals_df: pd.DataFrame, capital: float) -> pd.DataFrame:
        return pd.DataFrame({
            "symbol": ["AAPL"],
            "target_weight": [0.5],
            "target_qty": [100.0],
        })

    with patch("src.assembled_core.qa.backtest_engine.build_or_load_factors") as mock_build_or_load:
        mock_build_or_load.return_value = prices.copy()

        result = run_portfolio_backtest(
            prices=prices,
            signal_fn=signal_fn,
            position_sizing_fn=position_sizing_fn,
            start_capital=10000.0,
            compute_features=True,
            use_factor_store=True,
            factor_group="core_ta",
            factor_store_root=tmp_path / "factors",
            rebalance_freq="1d",
        )

        # Verify build_or_load_factors was called
        mock_build_or_load.assert_called_once()
        
        # Verify it was called with correct parameters
        call_kwargs = mock_build_or_load.call_args[1]
        assert call_kwargs["factor_group"] == "core_ta"
        assert call_kwargs["freq"] == "1d"
        assert call_kwargs["factors_root"] == tmp_path / "factors"
        assert call_kwargs["builder_fn"] is not None

        # Verify backtest completed successfully
        assert result is not None
        assert "equity" in result.__dict__
        assert "metrics" in result.__dict__


def test_backtest_without_factor_store_flag(tmp_path: Path) -> None:
    """Test that default behavior (no flag) uses add_all_features directly."""
    from src.assembled_core.qa.backtest_engine import run_portfolio_backtest

    dates = pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * len(dates),
        "open": [100.0] * len(dates),
        "high": [105.0] * len(dates),
        "low": [95.0] * len(dates),
        "close": [102.0] * len(dates),
        "volume": [1000000] * len(dates),
    })

    def signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": dates[:5],
            "symbol": ["AAPL"] * 5,
            "direction": ["LONG"] * 5,
            "score": [0.5] * 5,
        })

    def position_sizing_fn(signals_df: pd.DataFrame, capital: float) -> pd.DataFrame:
        return pd.DataFrame({
            "symbol": ["AAPL"],
            "target_weight": [0.5],
            "target_qty": [100.0],
        })

    with patch("src.assembled_core.qa.backtest_engine.add_all_features") as mock_add_features:
        mock_add_features.return_value = prices.copy()

        with patch("src.assembled_core.qa.backtest_engine.build_or_load_factors") as mock_build_or_load:
            result = run_portfolio_backtest(
                prices=prices,
                signal_fn=signal_fn,
                position_sizing_fn=position_sizing_fn,
                start_capital=10000.0,
                compute_features=True,
                use_factor_store=False,  # Explicit default
                rebalance_freq="1d",
            )

            # Verify add_all_features was called (default path)
            mock_add_features.assert_called_once()
            
            # Verify build_or_load_factors was NOT called
            mock_build_or_load.assert_not_called()

            # Verify backtest completed successfully
            assert result is not None
            assert "equity" in result.__dict__


def test_backtest_factor_store_without_ohlc(tmp_path: Path) -> None:
    """Test that factor store works even without OHLC columns (fallback to limited features)."""
    from src.assembled_core.qa.backtest_engine import run_portfolio_backtest

    dates = pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * len(dates),
        "close": [102.0] * len(dates),
        # No OHLC columns
    })

    def signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": dates[:5],
            "symbol": ["AAPL"] * 5,
            "direction": ["LONG"] * 5,
            "score": [0.5] * 5,
        })

    def position_sizing_fn(signals_df: pd.DataFrame, capital: float) -> pd.DataFrame:
        return pd.DataFrame({
            "symbol": ["AAPL"],
            "target_weight": [0.5],
            "target_qty": [100.0],
        })

    with patch("src.assembled_core.qa.backtest_engine.build_or_load_factors") as mock_build_or_load:
        # Mock return value with limited features (no OHLC-based features)
        prices_with_features = prices.copy()
        prices_with_features["log_return"] = 0.0
        prices_with_features["ma_20"] = 100.0
        mock_build_or_load.return_value = prices_with_features

        result = run_portfolio_backtest(
            prices=prices,
            signal_fn=signal_fn,
            position_sizing_fn=position_sizing_fn,
            start_capital=10000.0,
            compute_features=True,
            use_factor_store=True,
            factor_group="core_ta",
            factor_store_root=tmp_path / "factors",
            rebalance_freq="1d",
        )

        # Verify build_or_load_factors was called
        mock_build_or_load.assert_called_once()
        
        # Verify backtest completed successfully
        assert result is not None

