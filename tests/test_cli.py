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
        timeout=10
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
        timeout=10
    )
    
    assert result.returncode == 0
    assert "--freq" in result.stdout
    assert "--start-capital" in result.stdout


def test_cli_run_backtest_help():
    """Test that run_backtest subcommand help works."""
    script_path = ROOT / "scripts" / "cli.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "run_backtest", "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=10
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
        timeout=10
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
        timeout=60
    )
    
    # Should succeed (tests pass) or fail (tests fail), but not crash
    assert result.returncode in [0, 1, 2, 3, 4, 5]  # pytest exit codes
    # Should have some output
    assert len(result.stdout) > 0 or len(result.stderr) > 0

