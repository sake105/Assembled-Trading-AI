"""Tests for run_daily.py factor store integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

pytestmark = pytest.mark.advanced


def test_run_daily_eod_with_factor_store_flag(tmp_path: Path) -> None:
    """Test that --use-factor-store flag calls build_or_load_factors."""
    from scripts.run_daily import run_daily_eod

    dates = pd.date_range("2024-01-01", "2024-01-05", freq="D", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * len(dates),
        "open": [100.0] * len(dates),
        "high": [105.0] * len(dates),
        "low": [95.0] * len(dates),
        "close": [102.0] * len(dates),
        "volume": [1000000] * len(dates),
    })

    with patch("src.assembled_core.config.settings.get_settings") as mock_get_settings:
        from unittest.mock import MagicMock
        mock_settings = MagicMock()
        mock_settings.watchlist_file = tmp_path / "watchlist.txt"
        mock_get_settings.return_value = mock_settings

        with patch("src.assembled_core.data.prices_ingest.load_eod_prices_for_universe") as mock_load_universe:
            mock_load_universe.return_value = prices

            with patch("src.assembled_core.features.factor_store_integration.build_or_load_factors") as mock_build_or_load:
                mock_build_or_load.return_value = prices.copy()

                with patch("scripts.run_daily.generate_trend_signals_from_prices") as mock_signals:
                    mock_signals.return_value = pd.DataFrame({
                        "timestamp": dates,
                        "symbol": ["AAPL"] * len(dates),
                        "direction": ["LONG"] * len(dates),
                        "score": [0.5] * len(dates),
                    })

                    with patch("scripts.run_daily.compute_target_positions_from_trend_signals") as mock_targets:
                        mock_targets.return_value = pd.DataFrame({
                            "symbol": ["AAPL"],
                            "target_weight": [0.5],
                            "target_qty": [100.0],
                        })

                        with patch("scripts.run_daily.generate_orders_from_signals") as mock_orders:
                            mock_orders.return_value = pd.DataFrame({
                                "timestamp": dates[:1],
                                "symbol": ["AAPL"],
                                "side": ["BUY"],
                                "qty": [100.0],
                                "price": [102.0],
                            })

                            with patch("scripts.run_daily.write_safe_orders_csv") as mock_write:
                                mock_write.return_value = tmp_path / "orders.csv"

                                with patch("src.assembled_core.execution.risk_controls.filter_orders_with_risk_controls") as mock_risk:
                                    mock_risk.return_value = (pd.DataFrame({
                                        "timestamp": dates[:1],
                                        "symbol": ["AAPL"],
                                        "side": ["BUY"],
                                        "qty": [100.0],
                                        "price": [102.0],
                                    }), {})

                                    _ = run_daily_eod(
                                        date_str="2024-01-05",
                                        price_file=None,
                                        output_dir=tmp_path / "output",
                                        use_factor_store=True,
                                        factor_group="core_ta",
                                        factor_store_root=tmp_path / "factors",
                                    )

                                    mock_build_or_load.assert_called_once()
                                    call_kwargs = mock_build_or_load.call_args[1]
                                    assert call_kwargs["factor_group"] == "core_ta"
                                    assert call_kwargs["freq"] == "1d"
                                    assert call_kwargs["factors_root"] == tmp_path / "factors"
                                    assert call_kwargs["builder_fn"] is not None


def test_run_daily_eod_without_factor_store_flag(tmp_path: Path) -> None:
    """Test that default behavior (no flag) uses add_all_features directly."""
    from scripts.run_daily import run_daily_eod

    dates = pd.date_range("2024-01-01", "2024-01-05", freq="D", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * len(dates),
        "open": [100.0] * len(dates),
        "high": [105.0] * len(dates),
        "low": [95.0] * len(dates),
        "close": [102.0] * len(dates),
        "volume": [1000000] * len(dates),
    })

    with patch("src.assembled_core.config.settings.get_settings") as mock_get_settings:
        from unittest.mock import MagicMock
        mock_settings = MagicMock()
        mock_settings.watchlist_file = tmp_path / "watchlist.txt"
        mock_get_settings.return_value = mock_settings

        with patch("src.assembled_core.data.prices_ingest.load_eod_prices_for_universe") as mock_load_universe:
            mock_load_universe.return_value = prices

            with patch("src.assembled_core.features.ta_features.add_all_features") as mock_add_features:
                mock_add_features.return_value = prices.copy()

                with patch("src.assembled_core.features.factor_store_integration.build_or_load_factors") as mock_build_or_load:
                    with patch("scripts.run_daily.generate_trend_signals_from_prices") as mock_signals:
                        mock_signals.return_value = pd.DataFrame({
                        "timestamp": dates,
                        "symbol": ["AAPL"] * len(dates),
                        "direction": ["LONG"] * len(dates),
                        "score": [0.5] * len(dates),
                    })

                    with patch("scripts.run_daily.compute_target_positions_from_trend_signals") as mock_targets:
                        mock_targets.return_value = pd.DataFrame({
                            "symbol": ["AAPL"],
                            "target_weight": [0.5],
                            "target_qty": [100.0],
                        })

                        with patch("scripts.run_daily.generate_orders_from_signals") as mock_orders:
                            mock_orders.return_value = pd.DataFrame({
                                "timestamp": dates[:1],
                                "symbol": ["AAPL"],
                                "side": ["BUY"],
                                "qty": [100.0],
                                "price": [102.0],
                            })

                            with patch("scripts.run_daily.write_safe_orders_csv") as mock_write:
                                mock_write.return_value = tmp_path / "orders.csv"

                                with patch("src.assembled_core.execution.risk_controls.filter_orders_with_risk_controls") as mock_risk:
                                    mock_risk.return_value = (pd.DataFrame({
                                        "timestamp": dates[:1],
                                        "symbol": ["AAPL"],
                                        "side": ["BUY"],
                                        "qty": [100.0],
                                        "price": [102.0],
                                    }), {})

                                    _ = run_daily_eod(
                                        date_str="2024-01-05",
                                        price_file=None,
                                        output_dir=tmp_path / "output",
                                        use_factor_store=False,
                                    )

                                    mock_add_features.assert_called_once()
                                    mock_build_or_load.assert_not_called()


def test_run_daily_eod_timings_metadata_with_factor_store(tmp_path: Path) -> None:
    """Test that timings metadata includes factor store info when enabled."""
    from scripts.run_daily import run_daily_eod

    dates = pd.date_range("2024-01-01", "2024-01-05", freq="D", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * len(dates),
        "open": [100.0] * len(dates),
        "high": [105.0] * len(dates),
        "low": [95.0] * len(dates),
        "close": [102.0] * len(dates),
        "volume": [1000000] * len(dates),
    })

    with patch("src.assembled_core.config.settings.get_settings") as mock_get_settings:
        from unittest.mock import MagicMock
        mock_settings = MagicMock()
        mock_settings.watchlist_file = tmp_path / "watchlist.txt"
        mock_get_settings.return_value = mock_settings

        with patch("src.assembled_core.data.prices_ingest.load_eod_prices_for_universe") as mock_load_universe:
            mock_load_universe.return_value = prices

            with patch("src.assembled_core.features.factor_store_integration.build_or_load_factors") as mock_build_or_load:
                mock_build_or_load.return_value = prices.copy()

                with patch("scripts.run_daily.generate_trend_signals_from_prices") as mock_signals:
                    mock_signals.return_value = pd.DataFrame({
                    "timestamp": dates,
                    "symbol": ["AAPL"] * len(dates),
                    "direction": ["LONG"] * len(dates),
                    "score": [0.5] * len(dates),
                })

                with patch("scripts.run_daily.compute_target_positions_from_trend_signals") as mock_targets:
                    mock_targets.return_value = pd.DataFrame({
                        "symbol": ["AAPL"],
                        "target_weight": [0.5],
                        "target_qty": [100.0],
                    })

                    with patch("scripts.run_daily.generate_orders_from_signals") as mock_orders:
                        mock_orders.return_value = pd.DataFrame({
                            "timestamp": dates[:1],
                            "symbol": ["AAPL"],
                            "side": ["BUY"],
                            "qty": [100.0],
                            "price": [102.0],
                        })

                        with patch("scripts.run_daily.write_safe_orders_csv") as mock_write:
                            mock_write.return_value = tmp_path / "orders.csv"

                            with patch("src.assembled_core.execution.risk_controls.filter_orders_with_risk_controls") as mock_risk:
                                mock_risk.return_value = (pd.DataFrame({
                                    "timestamp": dates[:1],
                                    "symbol": ["AAPL"],
                                    "side": ["BUY"],
                                    "qty": [100.0],
                                    "price": [102.0],
                                }), {})

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

                                assert "job_meta" in timings_data
                                job_meta = timings_data["job_meta"]
                                assert job_meta["use_factor_store"] is True
                                assert job_meta["factor_group"] == "core_ta"

                                if "build_factors" in timings_data.get("steps", {}):
                                    build_factors_step = timings_data["steps"]["build_factors"]
                                    if "meta" in build_factors_step:
                                        meta = build_factors_step["meta"]
                                        assert "universe_key" in meta
                                        assert "factor_group" in meta
                                        assert meta["factor_group"] == "core_ta"
