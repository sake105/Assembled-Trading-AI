# tests/test_run_eod_pipeline.py
"""Tests for EOD pipeline orchestration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from pandas import Timedelta

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.pipeline.orchestrator import run_eod_pipeline

pytestmark = pytest.mark.phase4


def test_run_eod_pipeline_smoke(tmp_path: Path, monkeypatch):
    """Test EOD pipeline with minimal synthetic data."""
    # Temporarily override OUTPUT_DIR
    import src.assembled_core.config as config_module

    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)

    try:
        # Create minimal synthetic price file
        price_file = tmp_path / "aggregates" / "daily.parquet"
        price_file.parent.mkdir(parents=True, exist_ok=True)

        # Create enough data for EMA calculation (need at least slow period = 60)
        from datetime import datetime

        data = {
            "timestamp": pd.to_datetime(
                [
                    datetime(2025, 11, 28, 14, 0, 0) + Timedelta(days=i)
                    for i in range(100)
                ],
                utc=True,
            ),
            "symbol": ["AAPL"] * 100,
            "close": [100.0 + i * 0.1 for i in range(100)],  # Upward trend
        }
        df = pd.DataFrame(data)
        df.to_parquet(price_file)

        # Run pipeline
        manifest = run_eod_pipeline(
            freq="1d", start_capital=10000.0, output_dir=tmp_path
        )

        # Assert manifest structure
        assert manifest["freq"] == "1d"
        assert manifest["start_capital"] == 10000.0
        assert "completed_steps" in manifest
        assert "qa_overall_status" in manifest
        assert "timestamps" in manifest
        assert "started" in manifest["timestamps"]
        assert "finished" in manifest["timestamps"]

        # Assert manifest file exists
        manifest_file = tmp_path / "run_manifest_1d.json"
        assert manifest_file.exists(), "Manifest file should be created"

        # Read and validate manifest JSON
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        assert manifest_data["freq"] == "1d"
        assert manifest_data["start_capital"] == 10000.0
        assert "completed_steps" in manifest_data
        assert "qa_overall_status" in manifest_data

        # Assert that at least execute step completed
        assert "execute" in manifest_data["completed_steps"]

        # Assert output files exist
        assert (tmp_path / "orders_1d.csv").exists(), "Orders file should be created"

        # If backtest ran, check for backtest files
        if "backtest" in manifest_data["completed_steps"]:
            assert (tmp_path / "equity_curve_1d.csv").exists()
            assert (tmp_path / "performance_report_1d.md").exists()

        # If portfolio ran, check for portfolio files
        if "portfolio" in manifest_data["completed_steps"]:
            assert (tmp_path / "portfolio_equity_1d.csv").exists()
            assert (tmp_path / "portfolio_report_1d.md").exists()

        # If QA ran, check for QA metrics, gates, and report
        if "qa" in manifest_data["completed_steps"]:
            # Check that qa_metrics and qa_gate_result are present (if available)
            # They might be None if equity data is not available
            assert "qa_metrics" in manifest_data
            assert "qa_gate_result" in manifest_data

            # Check that qa_report_path is present (might be None if report generation failed)
            assert "qa_report_path" in manifest_data

            # If report path is present, check that file exists
            if manifest_data.get("qa_report_path"):
                report_path = tmp_path / manifest_data["qa_report_path"]
                assert report_path.exists(), (
                    f"QA report file should exist: {report_path}"
                )

                # Check report content
                report_content = report_path.read_text(encoding="utf-8")
                assert (
                    "QA Report" in report_content
                    or "Performance Metrics" in report_content
                )

    finally:
        # Restore original OUTPUT_DIR
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)


def test_run_eod_pipeline_skip_steps(tmp_path: Path, monkeypatch):
    """Test EOD pipeline with skipped steps."""
    # Temporarily override OUTPUT_DIR
    import src.assembled_core.config as config_module

    original_output_dir = config_module.OUTPUT_DIR
    monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)

    try:
        # Create minimal synthetic price file
        price_file = tmp_path / "aggregates" / "daily.parquet"
        price_file.parent.mkdir(parents=True, exist_ok=True)

        from datetime import datetime

        data = {
            "timestamp": pd.to_datetime(
                [
                    datetime(2025, 11, 28, 14, 0, 0) + Timedelta(days=i)
                    for i in range(100)
                ],
                utc=True,
            ),
            "symbol": ["AAPL"] * 100,
            "close": [100.0 + i * 0.1 for i in range(100)],
        }
        df = pd.DataFrame(data)
        df.to_parquet(price_file)

        # Run pipeline with skipped steps
        manifest = run_eod_pipeline(
            freq="1d",
            start_capital=10000.0,
            skip_backtest=True,
            skip_portfolio=True,
            skip_qa=True,
            output_dir=tmp_path,
        )

        # Assert only execute step completed
        assert "execute" in manifest["completed_steps"]
        assert "backtest" not in manifest["completed_steps"]
        assert "portfolio" not in manifest["completed_steps"]
        assert "qa" not in manifest["completed_steps"]

        # QA status should be None
        assert manifest["qa_overall_status"] is None

    finally:
        # Restore original OUTPUT_DIR
        monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)
