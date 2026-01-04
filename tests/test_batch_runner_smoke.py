"""Smoke tests for batch runner MVP."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_batch_runner_load_config(tmp_path: Path) -> None:
    """Test loading a valid batch config."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import load_batch_config

    # Create a minimal valid config
    config_file = tmp_path / "test_batch.yaml"
    config_content = """
batch_name: "test_batch"
output_root: "output/test_batch"
seed: 42

runs:
  - id: "run1"
    strategy: "trend_baseline"
    freq: "1d"
    start_date: "2020-01-01"
    end_date: "2023-12-31"
    universe: "watchlist.txt"
    start_capital: 100000.0
"""
    config_file.write_text(config_content, encoding="utf-8")

    batch_cfg = load_batch_config(config_file)

    assert batch_cfg.batch_name == "test_batch"
    assert batch_cfg.seed == 42
    assert len(batch_cfg.runs) == 1
    assert batch_cfg.runs[0].id == "run1"
    assert batch_cfg.runs[0].strategy == "trend_baseline"
    assert batch_cfg.runs[0].freq == "1d"


def test_batch_runner_invalid_config(tmp_path: Path) -> None:
    """Test loading an invalid batch config."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import load_batch_config

    # Missing batch_name
    config_file = tmp_path / "invalid.yaml"
    config_content = """
output_root: "output/test"
runs: []
"""
    config_file.write_text(config_content, encoding="utf-8")

    with pytest.raises(ValueError, match="batch_name"):
        load_batch_config(config_file)

    # Missing runs
    config_content2 = """
batch_name: "test"
output_root: "output/test"
"""
    config_file.write_text(config_content2, encoding="utf-8")

    with pytest.raises(ValueError, match="runs"):
        load_batch_config(config_file)


def test_batch_runner_write_manifest(tmp_path: Path) -> None:
    """Test writing a run manifest."""
    import sys
    sys.path.insert(0, str(ROOT))

    from datetime import datetime
    from scripts.batch_runner import RunConfig, write_run_manifest

    run_output_dir = tmp_path / "run1"
    run_output_dir.mkdir(parents=True)

    run_cfg = RunConfig(
        id="run1",
        strategy="trend_baseline",
        freq="1d",
        start_date="2020-01-01",
        end_date="2023-12-31",
        start_capital=100000.0,
    )

    started_at = datetime.utcnow()
    finished_at = datetime.utcnow()

    write_run_manifest(
        run_id="run1",
        run_cfg=run_cfg,
        run_output_dir=run_output_dir,
        status="success",
        started_at=started_at,
        finished_at=finished_at,
        runtime_sec=150.5,
        exit_code=0,
        seed=42,
    )

    manifest_path = run_output_dir / "run_manifest.json"
    assert manifest_path.exists()

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["run_id"] == "run1"
    assert manifest["status"] == "success"
    assert manifest["runtime_sec"] == 150.5
    assert manifest["exit_code"] == 0
    assert "params" in manifest
    assert manifest["params"]["strategy"] == "trend_baseline"
    assert manifest["params"]["freq"] == "1d"
    # git_commit_hash may be None or a string
    assert manifest["git_commit_hash"] is None or isinstance(manifest["git_commit_hash"], str)


def test_batch_runner_build_args(tmp_path: Path) -> None:
    """Test building argparse.Namespace from RunConfig."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import RunConfig, build_args_from_run_config

    run_cfg = RunConfig(
        id="run1",
        strategy="trend_baseline",
        freq="1d",
        start_date="2020-01-01",
        end_date="2023-12-31",
        universe="watchlist.txt",
        start_capital=100000.0,
        use_factor_store=True,
        factor_store_root="data/factors",
        factor_group="core_ta",
    )

    output_dir = tmp_path / "output"
    args = build_args_from_run_config(run_cfg, output_dir)

    assert args.freq == "1d"
    assert args.strategy == "trend_baseline"
    assert args.start_date == "2020-01-01"
    assert args.end_date == "2023-12-31"
    assert args.start_capital == 100000.0
    assert args.use_factor_store is True
    assert args.factor_store_root is not None
    assert args.factor_group == "core_ta"
    assert args.out == output_dir


def test_batch_runner_dry_run(tmp_path: Path) -> None:
    """Test dry-run mode (no actual execution)."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import BatchConfig, RunConfig, run_batch

    run_cfg = RunConfig(
        id="run1",
        strategy="trend_baseline",
        freq="1d",
        start_date="2020-01-01",
        end_date="2023-12-31",
        start_capital=100000.0,
    )

    batch_cfg = BatchConfig(
        batch_name="test_batch",
        output_root=tmp_path / "output",
        seed=42,
        runs=[run_cfg],
    )

    # Dry-run should not fail (even if backtest would fail)
    exit_code = run_batch(batch_cfg, dry_run=True)
    assert exit_code == 0

    # Run directory should exist
    run_dir = batch_cfg.output_root / batch_cfg.batch_name / "runs" / run_cfg.id
    assert run_dir.exists()


def test_batch_runner_cli_help(capsys) -> None:
    """Test CLI help output."""
    import sys
    sys.path.insert(0, str(ROOT))

    from scripts.batch_runner import parse_args

    # Should not raise
    with pytest.raises(SystemExit):
        parse_args(["--help"])

