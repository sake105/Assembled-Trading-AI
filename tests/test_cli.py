# tests/test_cli.py
"""Tests for central CLI (scripts/cli.py)."""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.phase4


def test_cli_importable():
    """Test that cli module can be imported."""
    import scripts.cli

    assert scripts.cli is not None


def test_cli_parser_creation():
    """Test that argument parser can be created without errors."""
    from scripts.cli import create_parser

    parser = create_parser()
    assert parser is not None

    # Test that subcommands exist
    assert "run_daily" in parser.format_help()
    assert "run_backtest" in parser.format_help()
    assert "run_phase4_tests" in parser.format_help()


def test_cli_help():
    """Test that CLI help works."""
    script_path = ROOT / "scripts" / "cli.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "run_daily" in result.stdout
    assert "run_backtest" in result.stdout
    assert "run_phase4_tests" in result.stdout


def test_cli_run_daily_help():
    """Test that run_daily subcommand help works."""
    script_path = ROOT / "scripts" / "cli.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "run_daily", "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "--freq" in result.stdout
    assert "--start-capital" in result.stdout


@pytest.mark.phase10
def test_cli_run_daily_pre_trade_flags():
    """Test that run_daily subcommand has pre-trade check flags."""
    # Test run_daily.py directly (not via cli.py which uses run_eod_pipeline)
    script_path = ROOT / "scripts" / "run_daily.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "--disable-pre-trade-checks" in result.stdout
    assert "--ignore-kill-switch" in result.stdout


def test_cli_run_backtest_help():
    """Test that run_backtest subcommand help works."""
    script_path = ROOT / "scripts" / "cli.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "run_backtest", "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "--freq" in result.stdout
    assert "--strategy" in result.stdout


def test_cli_run_phase4_tests_help():
    """Test that run_phase4_tests subcommand help works."""
    script_path = ROOT / "scripts" / "cli.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "run_phase4_tests", "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "--durations" in result.stdout or "--verbose" in result.stdout


def test_cli_run_phase4_tests_smoke(tmp_path: Path):
    """Test that run_phase4_tests subcommand can be invoked (smoke test)."""
    script_path = ROOT / "scripts" / "cli.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "run_phase4_tests", "-q"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Should succeed (tests pass) or fail (tests fail), but not crash
    assert result.returncode in [0, 1, 2, 3, 4, 5]  # pytest exit codes
    # Should have some output
    assert len(result.stdout) > 0 or len(result.stderr) > 0


@pytest.mark.phase10
def test_cli_runtime_profile_backtest():
    """Test that run_backtest automatically sets profile=BACKTEST."""
    script_path = ROOT / "scripts" / "cli.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "run_backtest", "--freq", "1d", "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    # Check that profile is logged (we'll check this via actual run with sample data)

    # Test with a minimal run that should log the profile
    # Use --price-file with a non-existent file to trigger early exit but still log profile
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "run_backtest",
            "--freq",
            "1d",
            "--price-file",
            "nonexistent.parquet",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )

    # Should log "Runtime Profile: BACKTEST"
    assert (
        "Runtime Profile: BACKTEST" in result.stdout
        or "Runtime Profile: BACKTEST" in result.stderr
    )


@pytest.mark.phase10
def test_cli_runtime_profile_build_ml_dataset():
    """Test that build_ml_dataset automatically sets profile=BACKTEST."""
    script_path = ROOT / "scripts" / "cli.py"

    # Test with a minimal run that should log the profile
    # Use --price-file with a non-existent file to trigger early exit but still log profile
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "build_ml_dataset",
            "--strategy",
            "trend_baseline",
            "--freq",
            "1d",
            "--price-file",
            "nonexistent.parquet",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )

    # Should log "Runtime Profile: BACKTEST"
    assert (
        "Runtime Profile: BACKTEST" in result.stdout
        or "Runtime Profile: BACKTEST" in result.stderr
    )


@pytest.mark.phase10
def test_cli_runtime_profile_run_daily_default():
    """Test that run_daily uses DEV profile by default."""
    script_path = ROOT / "scripts" / "cli.py"

    # Test with a minimal run that should log the profile
    # Use --price-file with a non-existent file to trigger early exit but still log profile
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "run_daily",
            "--freq",
            "1d",
            "--price-file",
            "nonexistent.parquet",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )

    # Should log "Runtime Profile: DEV" (default)
    assert (
        "Runtime Profile: DEV" in result.stdout
        or "Runtime Profile: DEV" in result.stderr
    )


@pytest.mark.phase10
def test_cli_runtime_profile_run_daily_explicit():
    """Test that run_daily accepts --profile argument."""
    script_path = ROOT / "scripts" / "cli.py"

    # Test with explicit BACKTEST profile
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "run_daily",
            "--freq",
            "1d",
            "--profile",
            "BACKTEST",
            "--price-file",
            "nonexistent.parquet",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )

    # Should log "Runtime Profile: BACKTEST"
    assert (
        "Runtime Profile: BACKTEST" in result.stdout
        or "Runtime Profile: BACKTEST" in result.stderr
    )
