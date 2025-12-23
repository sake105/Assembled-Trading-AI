# tests/test_portfolio_report_freq_specific.py
"""Tests for frequency-specific portfolio report naming."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pandas import Timedelta

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.api.app import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


def test_portfolio_report_prefers_freq_specific(
    tmp_path: Path, monkeypatch, client: TestClient
):
    """Test that API prefers freq-specific portfolio_report_{freq}.md over legacy portfolio_report.md."""
    import src.assembled_core.config as config_module

    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)

    try:
        # Create both files with different values
        # Legacy file
        legacy_file = tmp_path / "portfolio_report.md"
        legacy_file.write_text(
            "# Portfolio Report (legacy)\n\n"
            "- Final PF: 0.9000\n"
            "- Sharpe: -0.1000\n"
            "- Trades: 1\n",
            encoding="utf-8",
        )

        # Freq-specific file (should be preferred)
        freq_file = tmp_path / "portfolio_report_1d.md"
        freq_file.write_text(
            "# Portfolio Report (1d)\n\n"
            "- Final PF: 1.0100\n"
            "- Sharpe: 0.5000\n"
            "- Trades: 5\n",
            encoding="utf-8",
        )

        # Create portfolio equity file
        equity_file = tmp_path / "portfolio_equity_1d.csv"
        equity_data = {
            "timestamp": pd.to_datetime(
                [
                    datetime(2025, 11, 28, 14, 0, 0) + Timedelta(days=i)
                    for i in range(5)
                ],
                utc=True,
            ),
            "equity": [10000.0, 10050.0, 10020.0, 10080.0, 10100.0],
        }
        pd.DataFrame(equity_data).to_csv(equity_file, index=False)

        # Create orders file
        orders_file = tmp_path / "orders_1d.csv"
        orders_data = {
            "timestamp": pd.to_datetime(
                [datetime(2025, 11, 28, 14, 0, 0)] * 2, utc=True
            ),
            "symbol": ["AAPL", "MSFT"],
            "side": ["BUY", "SELL"],
            "qty": [1.0, 1.0],
            "price": [100.0, 200.0],
        }
        orders_df = pd.DataFrame(orders_data)
        orders_df["timestamp"] = orders_df["timestamp"].dt.strftime(
            "%Y-%m-%d %H:%M:%S%z"
        )
        orders_df.to_csv(orders_file, index=False)

        # Call API
        response = client.get("/api/v1/portfolio/1d/current")

        assert response.status_code == 200
        data = response.json()

        # Should use freq-specific file values (1.0100, not 0.9000)
        assert data["pf"] == 1.0100
        assert data["sharpe"] == 0.5000
        assert data["total_trades"] == 5

    finally:
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)


def test_portfolio_report_fallback_to_legacy(
    tmp_path: Path, monkeypatch, client: TestClient
):
    """Test that API falls back to legacy portfolio_report.md if freq-specific file doesn't exist."""
    import src.assembled_core.config as config_module

    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)

    try:
        # Create only legacy file
        legacy_file = tmp_path / "portfolio_report.md"
        legacy_file.write_text(
            "# Portfolio Report (legacy)\n\n"
            "- Final PF: 0.9500\n"
            "- Sharpe: 0.2000\n"
            "- Trades: 3\n",
            encoding="utf-8",
        )

        # Create portfolio equity file
        equity_file = tmp_path / "portfolio_equity_1d.csv"
        equity_data = {
            "timestamp": pd.to_datetime(
                [
                    datetime(2025, 11, 28, 14, 0, 0) + Timedelta(days=i)
                    for i in range(5)
                ],
                utc=True,
            ),
            "equity": [10000.0, 10050.0, 10020.0, 10080.0, 10100.0],
        }
        pd.DataFrame(equity_data).to_csv(equity_file, index=False)

        # Create orders file
        orders_file = tmp_path / "orders_1d.csv"
        orders_data = {
            "timestamp": pd.to_datetime(
                [datetime(2025, 11, 28, 14, 0, 0)] * 2, utc=True
            ),
            "symbol": ["AAPL", "MSFT"],
            "side": ["BUY", "SELL"],
            "qty": [1.0, 1.0],
            "price": [100.0, 200.0],
        }
        orders_df = pd.DataFrame(orders_data)
        orders_df["timestamp"] = orders_df["timestamp"].dt.strftime(
            "%Y-%m-%d %H:%M:%S%z"
        )
        orders_df.to_csv(orders_file, index=False)

        # Call API
        response = client.get("/api/v1/portfolio/1d/current")

        assert response.status_code == 200
        data = response.json()

        # Should use legacy file values
        assert data["pf"] == 0.9500
        assert data["sharpe"] == 0.2000
        assert data["total_trades"] == 3

    finally:
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)
