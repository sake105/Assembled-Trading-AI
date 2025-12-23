"""Tests for Paper-Track Run-ID and Run Summary JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.advanced


def test_run_summary_json_exists_and_contains_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that run_summary.json exists and contains run_id after a range run."""
    import scripts.run_paper_track as rpt_module

    # Mock generate_run_id to return a deterministic value
    def mock_generate_run_id(prefix: str = "run") -> str:
        return f"{prefix}_20250115_120000_12345678"

    monkeypatch.setattr(rpt_module, "generate_run_id", mock_generate_run_id)
    monkeypatch.setattr("scripts.run_paper_track.generate_run_id", mock_generate_run_id)

    # Mock run_paper_day to simulate successful runs
    def mock_run_paper_day(config, as_of, state_path=None):
        from src.assembled_core.paper.paper_track import (
            PaperTrackDayResult,
            PaperTrackState,
        )

        state = PaperTrackState(
            strategy_name=config.strategy_name,
            last_run_date=as_of,
            cash=100000.0,
            equity=100000.0,
            seed_capital=100000.0,
        )
        return PaperTrackDayResult(
            date=as_of,
            config=config,
            state_before=PaperTrackState(
                strategy_name=config.strategy_name, last_run_date=None
            ),
            state_after=state,
            orders=pd.DataFrame(),
            daily_return_pct=0.0,
            daily_pnl=0.0,
            trades_count=0,
            buy_count=0,
            sell_count=0,
            status="success",
        )

    monkeypatch.setattr(rpt_module, "run_paper_day", mock_run_paper_day)

    # Mock write_paper_day_outputs to create minimal files
    def mock_write_outputs(result, output_dir, config=None, run_id=None):
        run_date_str = result.date.strftime("%Y%m%d")
        run_dir = output_dir / "runs" / run_date_str
        run_dir.mkdir(parents=True, exist_ok=True)
        # Create minimal manifest.json with run_id
        manifest = {
            "date": result.date.strftime("%Y-%m-%d"),
            "run_id": run_id,
            "strategy_name": config.strategy_name if config else "test",
        }
        with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

    monkeypatch.setattr(rpt_module, "write_paper_day_outputs", mock_write_outputs)
    monkeypatch.setattr(
        "src.assembled_core.paper.paper_track.write_paper_day_outputs", mock_write_outputs
    )

    # Mock save_paper_state
    def mock_save_state(state, state_path):
        state_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(rpt_module, "save_paper_state", mock_save_state)
    monkeypatch.setattr(
        "src.assembled_core.paper.paper_track.save_paper_state", mock_save_state
    )

    # Mock load_paper_track_config
    def mock_load_config(config_file):
        from src.assembled_core.paper.paper_track import PaperTrackConfig

        return PaperTrackConfig(
            strategy_name="test_strategy",
            strategy_type="trend_baseline",
            universe_file=tmp_path / "universe.txt",
            freq="1d",
        )

    monkeypatch.setattr(rpt_module, "load_paper_track_config", mock_load_config)

    # Create universe file
    (tmp_path / "universe.txt").write_text("AAPL\n", encoding="utf-8")

    # Mock ROOT
    original_root = rpt_module.ROOT
    rpt_module.ROOT = tmp_path

    # Create config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "strategy_name: test_strategy\nstrategy_type: trend_baseline\nuniverse:\n  file: universe.txt\n",
        encoding="utf-8",
    )

    try:
        # Run for a date range (3 days)
        output_root = tmp_path / "output" / "paper_track" / "test_strategy"
        output_root.mkdir(parents=True, exist_ok=True)

        exit_code = rpt_module.run_paper_track_from_cli(
            config_file=config_file,
            start_date="2025-01-15",
            end_date="2025-01-17",
            dry_run=False,
            fail_fast=False,
        )

        assert exit_code == 0

        # Check that run_summary.json exists
        summary_path = output_root / "paper_track_run_summary.json"
        assert summary_path.exists(), "run_summary.json should exist"

        # Load and check contents
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert "run_id" in summary
        assert summary["run_id"] == "paper_track_20250115_120000_12345678"
        assert summary["strategy_name"] == "test_strategy"
        assert summary["days_attempted"] == 3
        assert summary["days_successful"] == 3
        assert summary["days_failed"] == 0
        assert summary["days_skipped"] == 0
        assert summary["dry_run"] is False
        assert summary["rerun"] is False
        assert "start_time" in summary
        assert "end_time" in summary
        assert "duration_seconds" in summary
        assert "per_day_statuses" in summary
        assert len(summary["per_day_statuses"]) == 3

        # Check per-day statuses
        for day_status in summary["per_day_statuses"]:
            assert "date" in day_status
            assert "status" in day_status
            assert day_status["status"] == "success"

    finally:
        rpt_module.ROOT = original_root


def test_run_summary_json_includes_skipped_days(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that run_summary.json includes skipped days in counts and per_day_statuses."""
    import scripts.run_paper_track as rpt_module

    # Mock generate_run_id
    def mock_generate_run_id(prefix: str = "run") -> str:
        return f"{prefix}_20250115_120000_12345678"

    monkeypatch.setattr(rpt_module, "generate_run_id", mock_generate_run_id)
    monkeypatch.setattr("scripts.run_paper_track.generate_run_id", mock_generate_run_id)

    # Mock run_paper_day (but it won't be called for existing days)
    def mock_run_paper_day(config, as_of, state_path=None):
        from src.assembled_core.paper.paper_track import (
            PaperTrackDayResult,
            PaperTrackState,
        )

        state = PaperTrackState(
            strategy_name=config.strategy_name,
            last_run_date=as_of,
            cash=100000.0,
            equity=100000.0,
            seed_capital=100000.0,
        )
        return PaperTrackDayResult(
            date=as_of,
            config=config,
            state_before=PaperTrackState(
                strategy_name=config.strategy_name, last_run_date=None
            ),
            state_after=state,
            orders=pd.DataFrame(),
            daily_return_pct=0.0,
            daily_pnl=0.0,
            trades_count=0,
            buy_count=0,
            sell_count=0,
            status="success",
        )

    monkeypatch.setattr(rpt_module, "run_paper_day", mock_run_paper_day)

    # Mock write_paper_day_outputs
    def mock_write_outputs(result, output_dir, config=None, run_id=None):
        run_date_str = result.date.strftime("%Y%m%d")
        run_dir = output_dir / "runs" / run_date_str
        run_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "src.assembled_core.paper.paper_track.write_paper_day_outputs", mock_write_outputs
    )

    # Mock save_paper_state
    def mock_save_state(state, state_path):
        state_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "src.assembled_core.paper.paper_track.save_paper_state", mock_save_state
    )

    # Mock load_paper_track_config
    def mock_load_config(config_file):
        from src.assembled_core.paper.paper_track import PaperTrackConfig

        return PaperTrackConfig(
            strategy_name="test_strategy",
            strategy_type="trend_baseline",
            universe_file=tmp_path / "universe.txt",
            freq="1d",
        )

    monkeypatch.setattr(rpt_module, "load_paper_track_config", mock_load_config)

    # Create universe file
    (tmp_path / "universe.txt").write_text("AAPL\n", encoding="utf-8")

    # Mock ROOT
    original_root = rpt_module.ROOT
    rpt_module.ROOT = tmp_path

    # Create config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "strategy_name: test_strategy\nstrategy_type: trend_baseline\nuniverse:\n  file: universe.txt\n",
        encoding="utf-8",
    )

    try:
        # Pre-create a run directory for one day (to simulate skipped day)
        output_root = tmp_path / "output" / "paper_track" / "test_strategy"
        (output_root / "runs" / "20250115").mkdir(parents=True, exist_ok=True)

        exit_code = rpt_module.run_paper_track_from_cli(
            config_file=config_file,
            start_date="2025-01-15",
            end_date="2025-01-17",
            dry_run=False,
            fail_fast=False,
            rerun=False,  # Don't rerun existing days
        )

        assert exit_code == 0

        # Check that run_summary.json exists
        summary_path = output_root / "paper_track_run_summary.json"
        assert summary_path.exists(), "run_summary.json should exist"

        # Load and check contents
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["days_attempted"] == 3
        assert summary["days_skipped"] == 1  # One day was skipped
        assert summary["days_successful"] == 2  # Two days were processed

        # Check per_day_statuses includes skipped day
        skipped_statuses = [s for s in summary["per_day_statuses"] if s["status"] == "skipped"]
        assert len(skipped_statuses) == 1
        assert skipped_statuses[0]["date"] == "2025-01-15"

    finally:
        rpt_module.ROOT = original_root


def test_run_id_in_manifest_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that run_id is included in daily manifest.json files."""
    from src.assembled_core.paper.paper_track import (
        PaperTrackConfig,
        PaperTrackDayResult,
        PaperTrackState,
        _write_run_manifest,
    )

    # Create a mock result
    state = PaperTrackState(
        strategy_name="test_strategy",
        last_run_date=pd.Timestamp("2025-01-15", tz="UTC"),
        cash=100000.0,
        equity=100000.0,
        seed_capital=100000.0,
    )
    result = PaperTrackDayResult(
        date=pd.Timestamp("2025-01-15", tz="UTC"),
        config=PaperTrackConfig(
            strategy_name="test_strategy",
            strategy_type="trend_baseline",
            universe_file=tmp_path / "universe.txt",
            freq="1d",
        ),
        state_before=PaperTrackState(strategy_name="test_strategy", last_run_date=None),
        state_after=state,
        orders=pd.DataFrame(),
        daily_return_pct=0.0,
        daily_pnl=0.0,
        trades_count=0,
        buy_count=0,
        sell_count=0,
        status="success",
    )

    run_dir = tmp_path / "runs" / "20250115"
    run_dir.mkdir(parents=True, exist_ok=True)

    test_run_id = "test_run_id_12345678"
    _write_run_manifest(result, run_dir, result.config, run_id=test_run_id)

    # Check manifest.json
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "run_id" in manifest
    assert manifest["run_id"] == test_run_id


def test_run_id_not_in_manifest_if_not_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that manifest.json works even if run_id is not provided (backwards compatibility)."""
    from src.assembled_core.paper.paper_track import (
        PaperTrackConfig,
        PaperTrackDayResult,
        PaperTrackState,
        _write_run_manifest,
    )

    # Create a mock result
    state = PaperTrackState(
        strategy_name="test_strategy",
        last_run_date=pd.Timestamp("2025-01-15", tz="UTC"),
        cash=100000.0,
        equity=100000.0,
        seed_capital=100000.0,
    )
    result = PaperTrackDayResult(
        date=pd.Timestamp("2025-01-15", tz="UTC"),
        config=PaperTrackConfig(
            strategy_name="test_strategy",
            strategy_type="trend_baseline",
            universe_file=tmp_path / "universe.txt",
            freq="1d",
        ),
        state_before=PaperTrackState(strategy_name="test_strategy", last_run_date=None),
        state_after=state,
        orders=pd.DataFrame(),
        daily_return_pct=0.0,
        daily_pnl=0.0,
        trades_count=0,
        buy_count=0,
        sell_count=0,
        status="success",
    )

    run_dir = tmp_path / "runs" / "20250115"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Call without run_id (backwards compatibility)
    _write_run_manifest(result, run_dir, result.config, run_id=None)

    # Check manifest.json
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # run_id should be None if not provided
    assert "run_id" in manifest
    assert manifest["run_id"] is None

