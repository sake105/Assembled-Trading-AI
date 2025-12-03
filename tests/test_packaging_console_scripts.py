# tests/test_packaging_console_scripts.py
"""Tests for console scripts defined in pyproject.toml."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.phase4


class TestConsoleScriptsImport:
    """Test that console script entry points can be imported."""

    def test_cli_main_importable(self):
        """Test that scripts.cli.main can be imported."""
        from scripts.cli import main
        
        assert callable(main)
        assert main.__name__ == "main"

    def test_run_backtest_main_importable(self):
        """Test that scripts.run_backtest_strategy.main can be imported."""
        from scripts.run_backtest_strategy import main
        
        assert callable(main)
        assert main.__name__ == "main"

    def test_run_eod_pipeline_main_importable(self):
        """Test that scripts.run_eod_pipeline.main can be imported."""
        from scripts.run_eod_pipeline import main
        
        assert callable(main)
        assert main.__name__ == "main"


class TestConsoleScriptsExecution:
    """Test that console scripts can be executed via subprocess."""

    def test_assembled_cli_help(self):
        """Test that assembled-cli --help works."""
        # Try to find the script in PATH (if installed) or use python -m
        # Since scripts may not be installed, we'll test via python -m scripts.cli
        result = subprocess.run(
            [sys.executable, "-m", "scripts.cli", "--help"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "usage:" in result.stdout.lower() or "subcommand" in result.stdout.lower()
        assert "run_daily" in result.stdout or "run_backtest" in result.stdout

    def test_assembled_cli_info(self):
        """Test that assembled-cli info works."""
        result = subprocess.run(
            [sys.executable, "-m", "scripts.cli", "info"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert len(result.stdout) > 0, "Should produce output"

    def test_assembled_run_backtest_help(self):
        """Test that assembled-run-backtest --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "scripts.run_backtest_strategy", "--help"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "usage:" in result.stdout.lower()
        assert "--freq" in result.stdout

    def test_assembled_run_daily_help(self):
        """Test that assembled-run-daily --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "scripts.run_eod_pipeline", "--help"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "usage:" in result.stdout.lower()
        assert "--freq" in result.stdout

    def test_assembled_cli_version(self):
        """Test that assembled-cli --version works."""
        result = subprocess.run(
            [sys.executable, "-m", "scripts.cli", "--version"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "0.0.1" in result.stdout or "version" in result.stdout.lower()

    def test_assembled_cli_invalid_subcommand(self):
        """Test that assembled-cli with invalid subcommand fails gracefully."""
        result = subprocess.run(
            [sys.executable, "-m", "scripts.cli", "invalid_subcommand"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Should fail with non-zero exit code
        assert result.returncode != 0
        # Should produce error message
        assert len(result.stderr) > 0 or len(result.stdout) > 0


class TestConsoleScriptsEntryPoints:
    """Test that entry points are correctly defined (if package is installed)."""
    
    def test_entry_points_defined(self):
        """Test that entry points are defined in pyproject.toml."""
        pyproject_path = ROOT / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml should exist"
        
        # Try tomllib (Python 3.11+) first, fall back to tomli
        try:
            import tomllib
            with open(pyproject_path, "rb") as f:
                pyproject = tomllib.load(f)
        except ImportError:
            import tomli
            with open(pyproject_path, "rb") as f:
                pyproject = tomli.load(f)
        
        # Check that project.scripts is defined
        assert "project" in pyproject, "project section should exist"
        assert "scripts" in pyproject["project"], "project.scripts should be defined"
        
        scripts = pyproject["project"]["scripts"]
        
        # Check that all expected scripts are defined
        assert "assembled-cli" in scripts, "assembled-cli should be defined"
        assert "assembled-run-backtest" in scripts, "assembled-run-backtest should be defined"
        assert "assembled-run-daily" in scripts, "assembled-run-daily should be defined"
        
        # Check that entry points point to correct modules
        assert scripts["assembled-cli"] == "scripts.cli:main"
        assert scripts["assembled-run-backtest"] == "scripts.run_backtest_strategy:main"
        assert scripts["assembled-run-daily"] == "scripts.run_eod_pipeline:main"

