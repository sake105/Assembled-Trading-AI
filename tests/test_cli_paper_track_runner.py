"""Tests for Paper Track Runner CLI.

Tests the CLI interface for paper track operations, including help, dry-run,
single day execution, and date range execution.

Uses in-process execution with monkeypatching for robustness.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import scripts.run_paper_track as rpt

pytestmark = pytest.mark.advanced


def _write_minimal_outputs(
    output_root: Path, strategy_name: str, as_of: pd.Timestamp
) -> None:
    """Write minimal output files for testing."""
    # Note: output_root already includes strategy_name from config
    run_dir = output_root / "runs" / as_of.strftime("%Y%m%d")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "equity_snapshot.json").write_text(
        json.dumps(
            {"timestamp": as_of.isoformat(), "equity": 100000.0, "cash": 100000.0}
        ),
        encoding="utf-8",
    )
    (run_dir / "positions.csv").write_text("symbol,qty\n", encoding="utf-8")
    (run_dir / "daily_summary.json").write_text(
        json.dumps(
            {
                "date": as_of.strftime("%Y-%m-%d"),
                "status": "success",
                "equity": 100000.0,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "daily_summary.md").write_text(
        f"# Paper Track {strategy_name} {as_of.date()}\n\nstatus: success\n",
        encoding="utf-8",
    )


def _save_minimal_state(
    state_path: Path, strategy_name: str, as_of: pd.Timestamp
) -> None:
    """Save minimal state file for testing."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_data = {
        "strategy_name": strategy_name,
        "last_run_date": as_of.isoformat(),
        "version": "1.0",
        "positions": [],
        "cash": 100000.0,
        "equity": 100000.0,
        "seed_capital": 100000.0,
        "created_at": as_of.isoformat(),
        "updated_at": as_of.isoformat(),
        "total_trades": 0,
        "total_pnl": 0.0,
    }
    state_path.write_text(json.dumps(state_data, indent=2), encoding="utf-8")


def _patch_paper_track_runtime(monkeypatch, output_root: Path, strategy_name: str):
    """Patch paper track runtime functions for testing."""
    from src.assembled_core.paper.paper_track import (
        PaperTrackDayResult,
        PaperTrackState,
        PaperTrackConfig,
    )

    def fake_run_paper_day(
        config: PaperTrackConfig, as_of: pd.Timestamp, state_path: Path | None = None
    ):
        """Fake run_paper_day that returns success without real data."""
        # Create minimal state
        state_before = PaperTrackState(
            strategy_name=config.strategy_name,
            last_run_date=None,
            cash=config.seed_capital,
            equity=config.seed_capital,
            seed_capital=config.seed_capital,
        )
        state_after = PaperTrackState(
            strategy_name=config.strategy_name,
            last_run_date=as_of,
            cash=config.seed_capital,
            equity=config.seed_capital,
            seed_capital=config.seed_capital,
        )

        return PaperTrackDayResult(
            date=as_of,
            config=config,
            state_before=state_before,
            state_after=state_after,
            orders=pd.DataFrame(
                columns=[
                    "timestamp",
                    "symbol",
                    "side",
                    "qty",
                    "price",
                    "fill_price",
                    "costs",
                ]
            ),
            daily_return_pct=0.0,
            daily_pnl=0.0,
            trades_count=0,
            buy_count=0,
            sell_count=0,
            status="success",
            error_message=None,
        )

    def fake_write_paper_day_outputs(result, output_dir: Path):
        """Fake write_paper_day_outputs that writes minimal files."""
        # output_dir already includes strategy_name from config (output.root/strategy_dir)
        # So we write directly to output_dir/runs/YYYYMMDD
        _write_minimal_outputs(output_dir, result.config.strategy_name, result.date)

    def fake_save_paper_state(state, state_path: Path):
        """Fake save_paper_state that writes minimal state."""
        _save_minimal_state(
            state_path,
            state.strategy_name,
            state.last_run_date or pd.Timestamp.utcnow().normalize(),
        )

    # Patch the imported functions in the run_paper_track module
    # These functions are imported at module level, so we patch them directly
    monkeypatch.setattr(rpt, "run_paper_day", fake_run_paper_day, raising=False)
    monkeypatch.setattr(
        rpt, "write_paper_day_outputs", fake_write_paper_day_outputs, raising=False
    )
    monkeypatch.setattr(rpt, "save_paper_state", fake_save_paper_state, raising=False)


def _write_json_config(
    path: Path, *, strategy_name: str, output_root: Path, universe_file: Path
):
    """Write minimal JSON config file for testing."""
    cfg = {
        "strategy_name": strategy_name,
        "strategy_type": "trend_baseline",
        "strategy": {
            "params": {
                "ma_fast": 2,
                "ma_slow": 3,
                "top_n": 2,
                "min_score": 0.0,
            }
        },
        "universe": {
            "file": str(universe_file),
        },
        "trading": {
            "freq": "1d",
        },
        "costs": {
            "commission_bps": 0.5,
            "spread_w": 0.25,
            "impact_w": 0.5,
        },
        "portfolio": {
            "seed_capital": 100000.0,
        },
        "output": {
            "root": str(output_root.parent)
            if output_root.name == strategy_name
            else str(output_root),
            "strategy_dir": strategy_name,
        },
        "random_seed": 42,
        "integration": {
            "enable_pit_checks": False,
        },
    }
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


@pytest.mark.advanced
def test_paper_track_cli_help():
    """Test that CLI --help works."""
    cli_path = Path(__file__).parent.parent / "scripts" / "cli.py"

    cmd = [
        sys.executable,
        str(cli_path),
        "paper_track",
        "--help",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
    )

    assert result.returncode == 0, (
        f"Help should exit with code 0. stderr: {result.stderr}"
    )
    assert "config-file" in result.stdout, "Help should mention --config-file"
    assert "as-of" in result.stdout, "Help should mention --as-of"
    assert "start-date" in result.stdout, "Help should mention --start-date"
    assert "end-date" in result.stdout, "Help should mention --end-date"
    assert "dry-run" in result.stdout, "Help should mention --dry-run"


@pytest.mark.advanced
def test_paper_track_cli_single_day_dry_run(tmp_path: Path, monkeypatch):
    """Test that dry-run mode returns 0 and writes no files."""
    output_root = tmp_path / "out"
    cfg_path = tmp_path / "paper.json"
    universe = tmp_path / "universe.txt"
    universe.write_text("AAPL\nMSFT\n", encoding="utf-8")

    _write_json_config(
        cfg_path,
        strategy_name="paper_test",
        output_root=output_root,
        universe_file=universe,
    )

    _patch_paper_track_runtime(monkeypatch, output_root, "paper_test")

    args = rpt.parse_args(
        [
            "--config-file",
            str(cfg_path),
            "--as-of",
            "2024-01-03",
            "--dry-run",
        ]
    )

    exit_code = rpt.run_paper_track_from_cli(
        config_file=args.config_file,
        as_of=args.as_of,
        start_date=args.start_date,
        end_date=args.end_date,
        dry_run=args.dry_run,
        fail_fast=args.fail_fast,
    )

    assert exit_code == 0, "Dry-run should exit with code 0"

    # dry-run: keine Outputs
    strategy_dir = output_root / "paper_test"
    assert not strategy_dir.exists() or not (strategy_dir / "runs").exists(), (
        "Dry-run should not create run directories"
    )


@pytest.mark.advanced
def test_paper_track_cli_single_day_writes_outputs(tmp_path: Path, monkeypatch):
    """Test that single day execution writes expected output files."""
    output_root = tmp_path / "out"
    cfg_path = tmp_path / "paper.json"
    universe = tmp_path / "universe.txt"
    universe.write_text("AAPL\nMSFT\n", encoding="utf-8")

    _write_json_config(
        cfg_path,
        strategy_name="paper_test",
        output_root=output_root,
        universe_file=universe,
    )

    _patch_paper_track_runtime(monkeypatch, output_root, "paper_test")

    args = rpt.parse_args(
        [
            "--config-file",
            str(cfg_path),
            "--as-of",
            "2024-01-03",
        ]
    )

    exit_code = rpt.run_paper_track_from_cli(
        config_file=args.config_file,
        as_of=args.as_of,
        start_date=args.start_date,
        end_date=args.end_date,
        dry_run=args.dry_run,
        fail_fast=args.fail_fast,
    )

    assert exit_code == 0, "Execution should exit with code 0"

    run_dir = output_root / "paper_test" / "runs" / "20240103"
    assert run_dir.exists(), "Run directory should exist"
    assert (run_dir / "daily_summary.json").exists(), "daily_summary.json should exist"
    assert (run_dir / "daily_summary.md").exists(), "daily_summary.md should exist"
    assert (run_dir / "equity_snapshot.json").exists(), (
        "equity_snapshot.json should exist"
    )

    state_file = output_root / "paper_test" / "state" / "state.json"
    assert state_file.exists(), "State file should exist"


@pytest.mark.advanced
def test_paper_track_cli_range_creates_multiple_run_dirs(tmp_path: Path, monkeypatch):
    """Test that date range execution creates multiple run directories."""
    # output_root in config will be set to output_base/strategy_dir
    output_base = tmp_path / "out"
    expected_output_root = output_base / "paper_test"  # This is what config will set
    cfg_path = tmp_path / "paper.json"
    universe = tmp_path / "universe.txt"
    universe.write_text("AAPL\nMSFT\n", encoding="utf-8")

    _write_json_config(
        cfg_path,
        strategy_name="paper_test",
        output_root=output_base,
        universe_file=universe,
    )

    _patch_paper_track_runtime(monkeypatch, expected_output_root, "paper_test")

    args = rpt.parse_args(
        [
            "--config-file",
            str(cfg_path),
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-03",
        ]
    )

    exit_code = rpt.run_paper_track_from_cli(
        config_file=args.config_file,
        as_of=args.as_of,
        start_date=args.start_date,
        end_date=args.end_date,
        dry_run=args.dry_run,
        fail_fast=args.fail_fast,
    )

    assert exit_code == 0, "Execution should exit with code 0"

    # output_root from config is output_base/strategy_dir (output_base/paper_test)
    # So we check expected_output_root/runs
    base = expected_output_root / "runs"

    # Debug: list what exists
    if not base.exists():
        # Check if expected_output_root exists at all
        if expected_output_root.exists():
            existing = list(expected_output_root.iterdir())
            pytest.fail(
                f"Runs directory does not exist: {base}. Output root exists: {expected_output_root}. Contents: {existing}"
            )
        else:
            pytest.fail(f"Output root does not exist: {expected_output_root}")

    # List all run directories
    run_dirs = sorted([d for d in base.iterdir() if d.is_dir()])
    assert len(run_dirs) >= 3, (
        f"Should have at least 3 run directories, got {len(run_dirs)}: {[d.name for d in run_dirs]}"
    )

    # Check specific dates
    assert (base / "20240101").exists(), "Run directory for 2024-01-01 should exist"
    assert (base / "20240102").exists(), "Run directory for 2024-01-02 should exist"
    assert (base / "20240103").exists(), "Run directory for 2024-01-03 should exist"

    # Check run summary
    summary_file = expected_output_root / "paper_track_run_summary.csv"
    assert summary_file.exists(), "Run summary CSV should exist"

    summary_df = pd.read_csv(summary_file)
    assert len(summary_df) == 3, f"Summary should have 3 rows, got {len(summary_df)}"
    assert all(summary_df["status"] == "success"), "All runs should be successful"

    # State file should exist and be updated
    state_file = expected_output_root / "state" / "state.json"
    assert state_file.exists(), "State file should exist"

    with open(state_file, "r", encoding="utf-8") as f:
        state_data = json.load(f)

    assert state_data["last_run_date"] == "2024-01-03T00:00:00+00:00", (
        "State should be updated to last run date"
    )
