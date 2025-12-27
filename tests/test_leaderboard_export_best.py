"""Tests for leaderboard best run config export functionality."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

try:
    import yaml
except ImportError:
    yaml = None
    pytestmark = pytest.mark.skip(reason="PyYAML not installed")


from scripts.leaderboard import (
    export_best_run_config_yaml,
    get_best_run_config,
    load_batch_summary,
)


def create_mock_summary_csv_with_config(tmp_path: Path) -> Path:
    """Create a mock summary.csv with run configuration fields."""
    summary_data = {
        "run_id": ["run1", "run2", "run3"],
        "status": ["success", "success", "failed"],
        "strategy": ["trend_baseline", "trend_baseline", "trend_baseline"],
        "freq": ["1d", "1d", "1d"],
        "start_date": ["2020-01-01", "2020-01-01", "2020-01-01"],
        "end_date": ["2023-12-31", "2023-12-31", "2023-12-31"],
        "universe": [None, "watchlist.txt", None],
        "start_capital": [100000.0, 100000.0, 100000.0],
        "sharpe": [1.5, 2.0, None],  # run2 has best sharpe
        "final_pf": [1.234, 1.456, None],
        "total_return": [0.234, 0.456, None],
    }
    
    df = pd.DataFrame(summary_data)
    summary_csv = tmp_path / "summary.csv"
    df.to_csv(summary_csv, index=False)
    
    return summary_csv.parent


def create_mock_manifest(tmp_path: Path, run_id: str, config: dict) -> None:
    """Create a mock run manifest JSON file."""
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        "run_id": run_id,
        "status": "success",
        "params": config,
        "seed": 42,
    }
    
    import json
    manifest_path = run_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def test_get_best_run_config_basic(tmp_path: Path):
    """Test extracting best run config from summary."""
    batch_output_dir = create_mock_summary_csv_with_config(tmp_path)
    df = load_batch_summary(batch_output_dir)
    
    # Get best run by sharpe (should be run2 with sharpe=2.0)
    config = get_best_run_config(df, sort_by="sharpe", batch_output_dir=batch_output_dir)
    
    # Verify required fields
    assert "id" in config
    assert config["id"] == "run2"  # Best by sharpe
    assert config["strategy"] == "trend_baseline"
    assert config["freq"] == "1d"
    assert config["start_date"] == "2020-01-01"
    assert config["end_date"] == "2023-12-31"
    
    # Verify optional fields (with defaults)
    assert "start_capital" in config
    assert config["universe"] is None or isinstance(config["universe"], str)


def test_get_best_run_config_with_manifest(tmp_path: Path):
    """Test that manifest is used to complete config when available."""
    batch_output_dir = create_mock_summary_csv_with_config(tmp_path)
    
    # Create manifest for run2 with additional fields
    create_mock_manifest(
        tmp_path=batch_output_dir,
        run_id="run2",
        config={
            "strategy": "trend_baseline",
            "freq": "1d",
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
            "universe": "watchlist.txt",
            "start_capital": 100000.0,
            "use_factor_store": True,
            "factor_store_root": "/path/to/factors",
            "factor_group": "core_ta",
        }
    )
    
    df = load_batch_summary(batch_output_dir)
    config = get_best_run_config(df, sort_by="sharpe", batch_output_dir=batch_output_dir)
    
    # Verify manifest fields are loaded
    assert config["universe"] == "watchlist.txt"
    assert config["start_capital"] == 100000.0
    assert config["use_factor_store"] is True
    assert config["factor_store_root"] == "/path/to/factors"
    assert config["factor_group"] == "core_ta"
    assert config.get("seed") == 42


def test_get_best_run_config_no_successful_runs(tmp_path: Path):
    """Test error handling when no successful runs exist."""
    # Create summary with only failed runs
    summary_data = {
        "run_id": ["run1"],
        "status": ["failed"],
        "strategy": ["trend_baseline"],
        "freq": ["1d"],
        "sharpe": [None],
    }
    df = pd.DataFrame(summary_data)
    summary_csv = tmp_path / "summary.csv"
    df.to_csv(summary_csv, index=False)
    
    batch_output_dir = summary_csv.parent
    
    df = load_batch_summary(batch_output_dir)
    
    with pytest.raises(ValueError, match="No successful runs found"):
        get_best_run_config(df, sort_by="sharpe", batch_output_dir=batch_output_dir)


def test_export_best_run_config_yaml(tmp_path: Path):
    """Test exporting best run config as YAML."""
    batch_output_dir = create_mock_summary_csv_with_config(tmp_path)
    
    # Create manifest for best run
    create_mock_manifest(
        tmp_path=batch_output_dir,
        run_id="run2",
        config={
            "strategy": "trend_baseline",
            "freq": "1d",
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
            "universe": "watchlist.txt",
            "start_capital": 100000.0,
        }
    )
    
    df = load_batch_summary(batch_output_dir)
    
    output_path = tmp_path / "best_config.yaml"
    export_best_run_config_yaml(
        df,
        sort_by="sharpe",
        output_path=output_path,
        batch_output_dir=batch_output_dir,
    )
    
    # Verify YAML file exists
    assert output_path.exists()
    
    # Verify YAML is parseable
    with output_path.open("r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    
    assert isinstance(yaml_data, dict)
    assert "runs" in yaml_data
    assert isinstance(yaml_data["runs"], list)
    assert len(yaml_data["runs"]) == 1
    
    config = yaml_data["runs"][0]
    
    # Verify required fields
    assert "id" in config
    assert config["id"] == "run2"  # Best by sharpe
    assert "strategy" in config
    assert config["strategy"] == "trend_baseline"
    assert "freq" in config
    assert config["freq"] == "1d"
    assert "start_date" in config
    assert config["start_date"] == "2020-01-01"
    assert "end_date" in config
    assert config["end_date"] == "2023-12-31"
    
    # Verify optional fields
    assert "universe" in config
    assert "start_capital" in config
    assert config["start_capital"] == 100000.0


def test_export_best_run_config_yaml_no_yaml_library(tmp_path: Path, monkeypatch):
    """Test error handling when PyYAML is not available."""
    batch_output_dir = create_mock_summary_csv_with_config(tmp_path)
    df = load_batch_summary(batch_output_dir)
    
    # Mock yaml module to None
    import scripts.leaderboard as leaderboard_module
    original_yaml = leaderboard_module.yaml
    leaderboard_module.yaml = None
    
    try:
        output_path = tmp_path / "best_config.yaml"
        with pytest.raises(RuntimeError, match="YAML export requires PyYAML"):
            export_best_run_config_yaml(
                df,
                sort_by="sharpe",
                output_path=output_path,
                batch_output_dir=batch_output_dir,
            )
    finally:
        leaderboard_module.yaml = original_yaml


def test_export_best_run_config_yaml_expected_fields(tmp_path: Path):
    """Test that exported YAML contains all expected fields for RunConfig."""
    batch_output_dir = create_mock_summary_csv_with_config(tmp_path)
    
    # Create manifest with all fields
    create_mock_manifest(
        tmp_path=batch_output_dir,
        run_id="run2",
        config={
            "strategy": "trend_baseline",
            "freq": "1d",
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
            "universe": "watchlist.txt",
            "start_capital": 100000.0,
            "use_factor_store": True,
            "factor_store_root": "/path/to/factors",
            "factor_group": "core_ta",
        }
    )
    
    df = load_batch_summary(batch_output_dir)
    
    output_path = tmp_path / "best_config.yaml"
    export_best_run_config_yaml(
        df,
        sort_by="sharpe",
        output_path=output_path,
        batch_output_dir=batch_output_dir,
    )
    
    # Parse YAML
    with output_path.open("r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    
    config = yaml_data["runs"][0]
    
    # Expected fields for RunConfig compatibility
    expected_fields = [
        "id",
        "strategy",
        "freq",
        "start_date",
        "end_date",
    ]
    
    for field in expected_fields:
        assert field in config, f"Expected field '{field}' not found in exported config"
        assert config[field] is not None, f"Field '{field}' should not be None"
    
    # Optional fields (should be present but may be None or default)
    optional_fields = [
        "universe",
        "start_capital",
        "use_factor_store",
        "factor_store_root",
        "factor_group",
    ]
    
    for field in optional_fields:
        assert field in config, f"Optional field '{field}' should be present in config"
    
    # Verify types
    assert isinstance(config["id"], str)
    assert isinstance(config["strategy"], str)
    assert isinstance(config["freq"], str)
    assert isinstance(config["start_date"], str)
    assert isinstance(config["end_date"], str)
    assert isinstance(config["start_capital"], (int, float))
    assert isinstance(config["use_factor_store"], bool)

