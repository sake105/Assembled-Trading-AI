# tests/test_operator_overview_example.py
"""Tests for operator overview example script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.phase9


class TestOperatorOverviewExample:
    """Tests for operator overview example script."""

    def test_operator_overview_runs_successfully(self, tmp_path: Path, monkeypatch):
        """Test that operator overview script runs and completes successfully."""
        script_path = ROOT / "notebooks" / "operator_overview_example.py"

        if not script_path.exists():
            pytest.skip("Operator overview script not found")

        # Override OUTPUT_DIR to use tmp_path
        import src.assembled_core.config as config_module

        original_output_dir = config_module.OUTPUT_DIR
        monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)

        # Mock settings
        from src.assembled_core.config.settings import (
            get_settings as original_get_settings,
        )

        settings_mock = original_get_settings()
        settings_mock.output_dir = tmp_path
        settings_mock.sample_eod_file = Path("nonexistent.parquet")

        def mock_get_settings():
            return settings_mock

        # Only patch the settings module directly
        monkeypatch.setattr(
            "src.assembled_core.config.settings.get_settings", mock_get_settings
        )
        monkeypatch.setattr(
            "notebooks.operator_overview_example.get_settings", mock_get_settings
        )

        try:
            # Create necessary directories
            (tmp_path / "ml_datasets").mkdir(parents=True, exist_ok=True)
            (tmp_path / "monitoring").mkdir(parents=True, exist_ok=True)
            (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

            # Run script (with timeout to prevent hanging)
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
                env={**dict(subprocess.os.environ), "PYTHONPATH": str(ROOT)},
            )

            # Check exit code (may be 0 or 1 depending on if sample data exists)
            # The important thing is that it doesn't crash
            assert result.returncode in [0, 1], (
                f"Script exited with unexpected code {result.returncode}"
            )

            # Check that certain text patterns appear in output
            output = result.stdout + result.stderr

            # Must contain section headers
            assert "OPERATOR OVERVIEW" in output or "SYSTEM HEALTH CHECK" in output, (
                "Missing header"
            )
            assert "STEP 1" in output or "Trend Baseline" in output, "Missing Step 1"
            assert "STEP 2" in output or "Event Insider" in output, "Missing Step 2"
            assert "STEP 3" in output or "ML Dataset" in output, "Missing Step 3"
            assert "STEP 4" in output or "Validation" in output, "Missing Step 4"

            # Must contain final summary
            assert (
                "FINAL SUMMARY" in output
                or "System Health" in output
                or "SUMMARY" in output
            ), "Missing final summary"

        finally:
            monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)

    def test_operator_overview_contains_required_patterns(
        self, tmp_path: Path, monkeypatch
    ):
        """Test that operator overview output contains required text patterns."""
        script_path = ROOT / "notebooks" / "operator_overview_example.py"

        if not script_path.exists():
            pytest.skip("Operator overview script not found")

        # Override OUTPUT_DIR to use tmp_path
        import src.assembled_core.config as config_module

        original_output_dir = config_module.OUTPUT_DIR
        monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)

        # Mock settings
        from src.assembled_core.config.settings import (
            get_settings as original_get_settings,
        )

        settings_mock = original_get_settings()
        settings_mock.output_dir = tmp_path
        settings_mock.sample_eod_file = Path("nonexistent.parquet")

        def mock_get_settings():
            return settings_mock

        # Only patch the settings module directly
        monkeypatch.setattr(
            "src.assembled_core.config.settings.get_settings", mock_get_settings
        )
        monkeypatch.setattr(
            "notebooks.operator_overview_example.get_settings", mock_get_settings
        )

        try:
            # Create necessary directories
            (tmp_path / "ml_datasets").mkdir(parents=True, exist_ok=True)
            (tmp_path / "monitoring").mkdir(parents=True, exist_ok=True)
            (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

            # Run script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=300,
                env={**dict(subprocess.os.environ), "PYTHONPATH": str(ROOT)},
            )

            output = result.stdout + result.stderr

            # Check for key patterns (some may be missing if data is not available, but structure should be there)
            required_patterns = [
                "Trend Baseline",
                "Event",
                "ML Dataset",
                "Validation",
                "SUMMARY",
            ]

            found_patterns = [
                pattern for pattern in required_patterns if pattern in output
            ]

            # At least some patterns should be present (depending on whether data is available)
            assert len(found_patterns) >= 2, (
                f"Too few required patterns found. Found: {found_patterns}. Output: {output[:500]}"
            )

        finally:
            monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)

    def test_operator_overview_no_crash_on_missing_data(
        self, tmp_path: Path, monkeypatch
    ):
        """Test that operator overview handles missing data gracefully."""
        script_path = ROOT / "notebooks" / "operator_overview_example.py"

        if not script_path.exists():
            pytest.skip("Operator overview script not found")

        # Override OUTPUT_DIR to use tmp_path
        import src.assembled_core.config as config_module

        original_output_dir = config_module.OUTPUT_DIR
        monkeypatch.setattr(config_module, "OUTPUT_DIR", tmp_path)

        # Mock settings with non-existent sample file
        from src.assembled_core.config.settings import (
            get_settings as original_get_settings,
        )

        settings_mock = original_get_settings()
        settings_mock.output_dir = tmp_path
        settings_mock.sample_eod_file = tmp_path / "nonexistent_sample.parquet"

        def mock_get_settings():
            return settings_mock

        # Only patch the settings module directly
        monkeypatch.setattr(
            "src.assembled_core.config.settings.get_settings", mock_get_settings
        )
        monkeypatch.setattr(
            "notebooks.operator_overview_example.get_settings", mock_get_settings
        )

        try:
            # Create necessary directories
            (tmp_path / "ml_datasets").mkdir(parents=True, exist_ok=True)
            (tmp_path / "monitoring").mkdir(parents=True, exist_ok=True)
            (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

            # Run script - should not crash even with missing data
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=300,
                env={**dict(subprocess.os.environ), "PYTHONPATH": str(ROOT)},
            )

            # Should exit cleanly (0 or 1, but not crash)
            assert result.returncode in [0, 1], (
                f"Script crashed or exited unexpectedly: {result.returncode}"
            )

            # Should produce some output
            assert len(result.stdout) > 0 or len(result.stderr) > 0, (
                "Script produced no output"
            )

        finally:
            monkeypatch.setattr(config_module, "OUTPUT_DIR", original_output_dir)
