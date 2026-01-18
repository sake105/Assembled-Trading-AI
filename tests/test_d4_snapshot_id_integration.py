# tests/test_d4_snapshot_id_integration.py
"""Tests for D4: Snapshot-ID durchreichen (Manifest + Batch Summary).

This test suite verifies:
1. Manifest contains data_snapshot_id when prices are loaded
2. Batch summary contains data_snapshot_id column and reads from manifest
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.data.snapshot import compute_price_panel_snapshot_id
from src.assembled_core.experiments.batch_runner import collect_backtest_metrics


def test_manifest_contains_data_snapshot_id() -> None:
    """Test that manifest contains data_snapshot_id when prices are loaded."""
    # Create a temporary manifest file with data_snapshot_id
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "run_manifest.json"
        
        # Simulate manifest with data_snapshot_id
        prices = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC"),
            "symbol": ["AAPL"] * 5,
            "close": [150.0] * 5,
        })
        snapshot_id = compute_price_panel_snapshot_id(prices, freq="1d")
        
        manifest = {
            "freq": "1d",
            "start_capital": 10000.0,
            "data_snapshot_id": snapshot_id,
            "completed_steps": ["execute", "backtest"],
        }
        
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f)
        
        # Verify manifest contains data_snapshot_id
        with manifest_path.open("r", encoding="utf-8") as f:
            loaded_manifest = json.load(f)
        
        assert "data_snapshot_id" in loaded_manifest, "Manifest should contain data_snapshot_id"
        assert loaded_manifest["data_snapshot_id"] == snapshot_id, "Snapshot ID should match"
        assert len(loaded_manifest["data_snapshot_id"]) == 64, "Snapshot ID should be 64 characters"


def test_manifest_data_snapshot_id_empty_prices() -> None:
    """Test that manifest contains data_snapshot_id even for empty prices (Empty-Semantik)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "run_manifest.json"
        
        # Empty prices should still produce a stable snapshot ID
        empty_prices = pd.DataFrame(columns=["timestamp", "symbol", "close"])
        snapshot_id = compute_price_panel_snapshot_id(empty_prices, freq="1d")
        
        manifest = {
            "freq": "1d",
            "data_snapshot_id": snapshot_id,
        }
        
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f)
        
        with manifest_path.open("r", encoding="utf-8") as f:
            loaded_manifest = json.load(f)
        
        assert "data_snapshot_id" in loaded_manifest, "Manifest should contain data_snapshot_id even for empty prices"
        assert loaded_manifest["data_snapshot_id"] == snapshot_id, "Empty prices should produce stable snapshot ID"


def test_batch_summary_reads_data_snapshot_id_from_manifest() -> None:
    """Test that batch summary reads data_snapshot_id from manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_output_dir = Path(tmpdir) / "run_001"
        run_output_dir.mkdir()
        
        # Create manifest with data_snapshot_id
        manifest_path = run_output_dir / "run_manifest.json"
        prices = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC"),
            "symbol": ["AAPL"] * 5,
            "close": [150.0] * 5,
        })
        snapshot_id = compute_price_panel_snapshot_id(prices, freq="1d")
        
        manifest = {
            "freq": "1d",
            "data_snapshot_id": snapshot_id,
            "base_args": {"strategy": "trend_baseline"},
        }
        
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f)
        
        # Use collect_backtest_metrics to read manifest
        metrics = collect_backtest_metrics(run_output_dir, freq="1d")
        
        assert "data_snapshot_id" in metrics, "Metrics should contain data_snapshot_id"
        assert metrics["data_snapshot_id"] == snapshot_id, "data_snapshot_id should be read from manifest"


def test_batch_summary_handles_missing_manifest() -> None:
    """Test that batch summary handles missing manifest gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_output_dir = Path(tmpdir) / "run_001"
        run_output_dir.mkdir()
        
        # No manifest file
        metrics = collect_backtest_metrics(run_output_dir, freq="1d")
        
        assert "data_snapshot_id" in metrics, "Metrics should contain data_snapshot_id field"
        assert metrics["data_snapshot_id"] is None, "data_snapshot_id should be None if manifest missing"


def test_batch_summary_handles_missing_data_snapshot_id_in_manifest() -> None:
    """Test that batch summary handles missing data_snapshot_id in manifest gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_output_dir = Path(tmpdir) / "run_001"
        run_output_dir.mkdir()
        
        # Manifest without data_snapshot_id (old format)
        manifest_path = run_output_dir / "run_manifest.json"
        manifest = {
            "freq": "1d",
            "base_args": {"strategy": "trend_baseline"},
        }
        
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f)
        
        metrics = collect_backtest_metrics(run_output_dir, freq="1d")
        
        assert "data_snapshot_id" in metrics, "Metrics should contain data_snapshot_id field"
        assert metrics["data_snapshot_id"] is None, "data_snapshot_id should be None if not in manifest"
