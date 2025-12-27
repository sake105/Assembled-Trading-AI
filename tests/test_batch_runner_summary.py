"""Tests for batch runner summary generation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from scripts.batch_runner import (
    BatchConfig,
    RunConfig,
    collect_backtest_metrics,
    write_batch_summary,
    write_run_manifest,
)


def test_collect_backtest_metrics_from_json_preferred(tmp_path: Path):
    """Test that metrics.json is preferred over Markdown parsing."""
    run_output_dir = tmp_path / "test_run"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write metrics.json (preferred format)
    metrics_json = {
        "final_pf": 1.234,
        "total_return": 0.234,
        "cagr": 0.15,
        "sharpe_ratio": 1.5,
        "sharpe": 1.5,  # Alias
        "max_drawdown_pct": -10.5,
        "total_trades": 100,
        "trades": 100,  # Alias
        "volatility": 0.20,
        "start_date": "2020-01-01T00:00:00+00:00",
        "end_date": "2023-12-31T00:00:00+00:00",
    }
    
    metrics_json_path = run_output_dir / "metrics.json"
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)
    
    # Also create a Markdown report (should be ignored if JSON exists)
    report_md_path = run_output_dir / "performance_report_1d.md"
    with report_md_path.open("w", encoding="utf-8") as f:
        f.write("# Performance Report\n\n")
        f.write("- Final PF: 2.000\n")  # Different value - should be ignored
        f.write("- Sharpe: 2.0\n")
        f.write("- Trades: 200\n")
    
    # Collect metrics (should use JSON, not Markdown)
    metrics = collect_backtest_metrics(run_output_dir)
    
    # Verify JSON values are used (not Markdown values)
    assert metrics["final_pf"] == 1.234, "Should use JSON value, not Markdown value"
    assert metrics["sharpe"] == 1.5, "Should use JSON value, not Markdown value"
    assert metrics["trades"] == 100, "Should use JSON value, not Markdown value"
    assert metrics["max_drawdown_pct"] == -10.5
    assert metrics["total_return"] == 0.234
    assert metrics["cagr"] == 0.15
    assert metrics["volatility"] == 0.20


def test_collect_backtest_metrics_from_json_reports_subdirectory(tmp_path: Path):
    """Test that metrics.json in reports/ subdirectory is also found."""
    run_output_dir = tmp_path / "test_run"
    reports_dir = run_output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Write metrics.json in reports subdirectory
    metrics_json = {
        "final_pf": 1.234,
        "sharpe_ratio": 1.5,
        "sharpe": 1.5,
        "total_trades": 100,
        "trades": 100,
        "max_drawdown_pct": -10.5,
    }
    
    metrics_json_path = reports_dir / "metrics.json"
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)
    
    # Collect metrics
    metrics = collect_backtest_metrics(run_output_dir)
    
    # Verify metrics from reports/metrics.json are used
    assert metrics["final_pf"] == 1.234
    assert metrics["sharpe"] == 1.5
    assert metrics["trades"] == 100
    assert metrics["max_drawdown_pct"] == -10.5


def test_collect_backtest_metrics_fallback_to_md(tmp_path: Path):
    """Test that Markdown parsing is used as fallback when JSON is not available."""
    run_output_dir = tmp_path / "test_run"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Markdown report (no JSON file)
    report_md_path = run_output_dir / "performance_report_1d.md"
    with report_md_path.open("w", encoding="utf-8") as f:
        f.write("# Performance Report (1d)\n\n")
        f.write("## Performance Metrics\n\n")
        f.write("- Final PF: 1.234\n")
        f.write("- Sharpe Ratio: 1.5\n")
        f.write("- Trades: 100\n")
        f.write("- Max Drawdown: -10.5%\n")
        f.write("- Total Return: 23.4%\n")
        f.write("- CAGR: 15.0%\n")
    
    # Collect metrics (should use Markdown parsing)
    metrics = collect_backtest_metrics(run_output_dir)
    
    # Verify Markdown values are parsed correctly
    assert metrics["final_pf"] == 1.234
    assert metrics["sharpe"] == 1.5
    assert metrics["trades"] == 100
    assert metrics["max_drawdown_pct"] == -10.5
    assert metrics["total_return"] == 23.4
    assert metrics["cagr"] == 15.0


def test_collect_backtest_metrics_no_files(tmp_path: Path):
    """Test that empty dict is returned when no metric files are found."""
    run_output_dir = tmp_path / "test_run"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect metrics (no files exist)
    metrics = collect_backtest_metrics(run_output_dir)
    
    # Should return empty dict (no errors)
    assert isinstance(metrics, dict)
    assert len(metrics) == 0 or all(v is None for v in metrics.values())


def test_collect_backtest_metrics_handles_invalid_json(tmp_path: Path):
    """Test that invalid JSON is handled gracefully (falls back to Markdown)."""
    run_output_dir = tmp_path / "test_run"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write invalid JSON
    metrics_json_path = run_output_dir / "metrics.json"
    with metrics_json_path.open("w", encoding="utf-8") as f:
        f.write("{ invalid json }")
    
    # Create valid Markdown report (should be used as fallback)
    report_md_path = run_output_dir / "performance_report_1d.md"
    with report_md_path.open("w", encoding="utf-8") as f:
        f.write("# Performance Report\n\n")
        f.write("- Final PF: 1.234\n")
        f.write("- Sharpe: 1.5\n")
    
    # Collect metrics (should fallback to Markdown)
    metrics = collect_backtest_metrics(run_output_dir)
    
    # Verify Markdown values are used (fallback worked)
    assert metrics["final_pf"] == 1.234
    assert metrics["sharpe"] == 1.5


def test_write_batch_summary(tmp_path: Path):
    """Test that batch summary is written correctly."""
    # Create mock batch config
    batch_cfg = BatchConfig(
        batch_name="test_batch",
        output_root=tmp_path,
        seed=42,
        runs=[
            RunConfig(
                id="run1",
                strategy="trend_baseline",
                freq="1d",
                start_date="2020-01-01",
                end_date="2023-12-31",
            ),
            RunConfig(
                id="run2",
                strategy="trend_baseline",
                freq="1d",
                start_date="2020-01-01",
                end_date="2023-12-31",
            ),
        ],
    )
    
    batch_output_root = tmp_path / batch_cfg.batch_name
    batch_output_root.mkdir(parents=True, exist_ok=True)
    
    # Create mock run directories with metrics.json
    run1_dir = batch_output_root / "run1"
    run1_dir.mkdir(parents=True, exist_ok=True)
    
    # Write metrics.json for run1
    metrics_json1 = {
        "final_pf": 1.234,
        "sharpe_ratio": 1.5,
        "sharpe": 1.5,
        "total_trades": 100,
        "trades": 100,
        "max_drawdown_pct": -10.5,
        "total_return": 0.234,
        "cagr": 0.15,
    }
    with (run1_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_json1, f)
    
    # Write manifest for run1
    now = datetime.utcnow()
    write_run_manifest(
        run_id="run1",
        run_cfg=batch_cfg.runs[0],
        run_output_dir=run1_dir,
        status="success",
        started_at=now,
        finished_at=now,
        runtime_sec=10.5,
        exit_code=0,
        seed=42,
    )
    
    # Create empty run2 (no metrics)
    run2_dir = batch_output_root / "run2"
    run2_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        run_id="run2",
        run_cfg=batch_cfg.runs[1],
        run_output_dir=run2_dir,
        status="success",
        started_at=now,
        finished_at=now,
        runtime_sec=5.0,
        exit_code=0,
        seed=42,
    )
    
    # Write batch summary
    write_batch_summary(batch_cfg, batch_output_root)
    
    # Verify summary files exist
    summary_csv = batch_output_root / "summary.csv"
    summary_json = batch_output_root / "summary.json"
    
    assert summary_csv.exists(), "summary.csv should be created"
    assert summary_json.exists(), "summary.json should be created"
    
    # Verify CSV content
    df = pd.read_csv(summary_csv)
    assert len(df) == 2, "Should have 2 runs"
    assert "run1" in df["run_id"].values
    assert "run2" in df["run_id"].values
    
    # Verify run1 has metrics from JSON
    run1_row = df[df["run_id"] == "run1"].iloc[0]
    assert run1_row["final_pf"] == 1.234
    assert run1_row["sharpe"] == 1.5
    assert run1_row["trades"] == 100
    assert run1_row["max_drawdown_pct"] == -10.5
    
    # Verify JSON content
    with summary_json.open("r", encoding="utf-8") as f:
        summary_data = json.load(f)
    
    assert summary_data["batch_name"] == "test_batch"
    assert summary_data["seed"] == 42
    assert len(summary_data["runs"]) == 2


def test_write_batch_summary_missing_manifests(tmp_path: Path):
    """Test that batch summary handles missing manifests gracefully."""
    batch_cfg = BatchConfig(
        batch_name="test_batch",
        output_root=tmp_path,
        seed=42,
        runs=[
            RunConfig(
                id="run1",
                strategy="trend_baseline",
                freq="1d",
                start_date="2020-01-01",
                end_date="2023-12-31",
            ),
        ],
    )
    
    batch_output_root = tmp_path / batch_cfg.batch_name
    batch_output_root.mkdir(parents=True, exist_ok=True)
    
    # Create run directory but no manifest
    run1_dir = batch_output_root / "run1"
    run1_dir.mkdir(parents=True, exist_ok=True)
    
    # Write batch summary (should handle missing manifest gracefully)
    write_batch_summary(batch_cfg, batch_output_root)
    
    # Verify summary files exist
    summary_csv = batch_output_root / "summary.csv"
    assert summary_csv.exists()
    
    # Verify CSV has run with unknown status
    df = pd.read_csv(summary_csv)
    assert len(df) == 1
    assert df.iloc[0]["status"] == "unknown"
    assert pd.isna(df.iloc[0]["final_pf"]) or df.iloc[0]["final_pf"] is None


def test_write_batch_summary_paths(tmp_path: Path):
    """Test that paths in summary are relative to batch_output_root."""
    batch_cfg = BatchConfig(
        batch_name="test_batch",
        output_root=tmp_path,
        seed=42,
        runs=[
            RunConfig(
                id="run1",
                strategy="trend_baseline",
                freq="1d",
                start_date="2020-01-01",
                end_date="2023-12-31",
            ),
        ],
    )
    
    batch_output_root = tmp_path / batch_cfg.batch_name
    batch_output_root.mkdir(parents=True, exist_ok=True)
    
    run1_dir = batch_output_root / "run1"
    run1_dir.mkdir(parents=True, exist_ok=True)
    
    # Write manifest
    now = datetime.utcnow()
    write_run_manifest(
        run_id="run1",
        run_cfg=batch_cfg.runs[0],
        run_output_dir=run1_dir,
        status="success",
        started_at=now,
        finished_at=now,
        runtime_sec=10.0,
        exit_code=0,
        seed=42,
    )
    
    # Write batch summary
    write_batch_summary(batch_cfg, batch_output_root)
    
    # Verify paths are relative
    summary_json = batch_output_root / "summary.json"
    with summary_json.open("r", encoding="utf-8") as f:
        summary_data = json.load(f)
    
    run_data = summary_data["runs"][0]
    if run_data.get("manifest_path"):
        assert not Path(run_data["manifest_path"]).is_absolute(), (
            "manifest_path should be relative"
        )
        assert run_data["manifest_path"] == "run1/run_manifest.json"
