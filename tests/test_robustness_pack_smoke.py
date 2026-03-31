"""Smoke tests for robustness pack orchestrator (Sprint 12 Final).

These tests verify that build_robustness_pack produces stable outputs
and writes all expected files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.assembled_core.qa.robustness import build_robustness_pack


def test_build_robustness_pack_smoke(tmp_path: Path):
    """Test that build_robustness_pack produces stable outputs."""

    # Create toy backtest function
    def backtest_fn(config: dict[str, Any]) -> dict[str, float | int]:
        return {
            "sharpe": 1.0,
            "cagr": 0.10,
            "max_drawdown": -0.05,
            "turnover": 0.5,
        }

    base_config: dict[str, Any] = {"strategy": "ema"}

    # Create toy prices DataFrame for RB1
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D", tz="UTC")
    prices_df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "AAPL",
            "close": 100.0,
        }
    )

    # Create toy parameter grid for RB2
    param_grid = {
        "ma_fast": [10, 20],
        "ma_slow": [50, 100],
    }

    # Run robustness pack
    manifest_fields = build_robustness_pack(
        backtest_fn=backtest_fn,
        base_config=base_config,
        run_id="test_run",
        output_dir=tmp_path,
        prices_df=prices_df,
        param_grid=param_grid,
        deterministic=True,
    )

    # Verify manifest fields structure
    assert "robustness_pack_path" in manifest_fields
    assert "wf_oos_metrics" in manifest_fields
    assert "plateau_score" in manifest_fields
    assert "sensitivity_summary" in manifest_fields
    assert "crisis_summary" in manifest_fields
    assert "deflated_sharpe" in manifest_fields
    assert "multiple_testing_warning" in manifest_fields
    assert "robustness_ok" in manifest_fields

    # Verify robustness_summary.json exists
    pack_dir = Path(manifest_fields["robustness_pack_path"])
    summary_json = pack_dir / "robustness_summary.json"
    assert summary_json.exists()

    # Verify summary JSON is readable
    import json

    with summary_json.open("r", encoding="utf-8") as f:
        summary_data = json.load(f)

    assert "robustness_ok" in summary_data
    assert isinstance(summary_data["robustness_ok"], bool)

    # Verify other expected files exist (at least some)
    expected_files = [
        "robustness_summary.json",
    ]
    for expected_file in expected_files:
        assert (pack_dir / expected_file).exists()


def test_build_robustness_pack_no_inputs(tmp_path: Path):
    """Test that build_robustness_pack handles missing inputs gracefully."""

    def backtest_fn(config: dict[str, Any]) -> dict[str, float | int]:
        return {"sharpe": 1.0}

    base_config: dict[str, Any] = {}

    # Run without prices_df or param_grid (RB1 and RB2 will be skipped)
    manifest_fields = build_robustness_pack(
        backtest_fn=backtest_fn,
        base_config=base_config,
        run_id="test_run_minimal",
        output_dir=tmp_path,
        prices_df=None,
        param_grid=None,
        deterministic=True,
    )

    # Should still produce manifest fields
    assert "robustness_ok" in manifest_fields
    # RB1 and RB2 are skipped, but RB3 (sensitivity) and RB4 (crisis) still run
    assert isinstance(manifest_fields["robustness_ok"], (bool, type(True)))

    # Verify summary JSON exists
    pack_dir = Path(manifest_fields["robustness_pack_path"])
    summary_json = pack_dir / "robustness_summary.json"
    assert summary_json.exists()
