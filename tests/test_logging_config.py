# tests/test_logging_config.py
"""Sprint 11.3: Tests for centralized logging configuration.

This module tests the logging configuration functions in src/assembled_core/logging_config.py:
- setup_logging: Configure logging with console and file handlers
- generate_run_id: Generate unique Run-IDs

Tests cover:
- Happy path scenarios
- Run-ID generation
- Log file creation
- Console and file handlers
- Edge cases (missing directories, custom log directories)
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.assembled_core.logging_config import generate_run_id, setup_logging

pytestmark = pytest.mark.phase11


class TestGenerateRunId:
    """Tests for generate_run_id function."""

    def test_generate_run_id_default_prefix(self):
        """Test Run-ID generation with default prefix."""
        run_id = generate_run_id()
        
        assert run_id.startswith("run_")
        assert len(run_id) > 20  # Should have timestamp + UUID

    def test_generate_run_id_custom_prefix(self):
        """Test Run-ID generation with custom prefix."""
        run_id = generate_run_id(prefix="backtest")
        
        assert run_id.startswith("backtest_")
        assert "backtest_" in run_id

    def test_generate_run_id_unique(self):
        """Test that generated Run-IDs are unique."""
        run_id1 = generate_run_id()
        run_id2 = generate_run_id()
        
        assert run_id1 != run_id2


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_with_run_id(self, tmp_path: Path):
        """Test logging setup with explicit Run-ID."""
        run_id = "test_run_12345"
        log_dir = tmp_path / "logs"
        
        setup_logging(run_id=run_id, level="INFO", log_dir=log_dir)
        
        # Check that log file was created
        log_file = log_dir / f"{run_id}.log"
        assert log_file.exists(), f"Log file should be created: {log_file}"
        
        # Check that log file has content
        log_content = log_file.read_text(encoding="utf-8")
        assert "Logging initialized" in log_content
        assert run_id in log_content

    def test_setup_logging_auto_run_id(self, tmp_path: Path):
        """Test logging setup with auto-generated Run-ID."""
        log_dir = tmp_path / "logs"
        
        setup_logging(level="INFO", log_dir=log_dir)
        
        # Should create a log file with auto-generated Run-ID
        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) > 0, "At least one log file should be created"
        
        # Check log file content
        log_file = log_files[0]
        log_content = log_file.read_text(encoding="utf-8")
        assert "Logging initialized" in log_content

    def test_setup_logging_creates_directory(self, tmp_path: Path):
        """Test that setup_logging creates log directory if it doesn't exist."""
        log_dir = tmp_path / "new_logs" / "subdir"
        
        # Directory shouldn't exist yet
        assert not log_dir.exists()
        
        setup_logging(run_id="test", level="INFO", log_dir=log_dir)
        
        # Directory should be created
        assert log_dir.exists()
        assert log_dir.is_dir()

    def test_setup_logging_console_and_file_handlers(self, tmp_path: Path):
        """Test that both console and file handlers are configured."""
        import logging
        
        log_dir = tmp_path / "logs"
        setup_logging(run_id="test", level="INFO", log_dir=log_dir)
        
        root_logger = logging.getLogger()
        handlers = root_logger.handlers
        
        # Should have at least 2 handlers (console + file)
        assert len(handlers) >= 2
        
        # Check handler types
        handler_types = [type(h).__name__ for h in handlers]
        assert "StreamHandler" in handler_types or "ConsoleHandler" in handler_types
        assert "FileHandler" in handler_types

    def test_setup_logging_logs_to_file(self, tmp_path: Path, caplog):
        """Test that logs are written to file."""
        import logging
        
        run_id = "test_file_logging"
        log_dir = tmp_path / "logs"
        
        setup_logging(run_id=run_id, level="INFO", log_dir=log_dir)
        logger = logging.getLogger(__name__)
        
        test_message = "Test log message for file"
        logger.info(test_message)
        
        # Check log file
        log_file = log_dir / f"{run_id}.log"
        assert log_file.exists()
        
        log_content = log_file.read_text(encoding="utf-8")
        assert test_message in log_content

    def test_setup_logging_different_levels(self, tmp_path: Path):
        """Test that different log levels work correctly."""
        import logging
        
        log_dir = tmp_path / "logs"
        
        # Test DEBUG level
        setup_logging(run_id="test_debug", level="DEBUG", log_dir=log_dir)
        logger = logging.getLogger(__name__)
        
        # Check effective level (logger.level may be NOTSET if inherited from root)
        assert logger.getEffectiveLevel() == logging.DEBUG
        
        # Test ERROR level
        setup_logging(run_id="test_error", level="ERROR", log_dir=log_dir)
        logger = logging.getLogger(__name__)
        
        # Check effective level
        assert logger.getEffectiveLevel() == logging.ERROR

    def test_setup_logging_run_id_in_logs(self, tmp_path: Path):
        """Test that Run-ID appears in log file entries."""
        import logging
        
        run_id = "test_run_id_tracking"
        log_dir = tmp_path / "logs"
        
        setup_logging(run_id=run_id, level="INFO", log_dir=log_dir)
        logger = logging.getLogger(__name__)
        
        logger.info("Test message")
        
        # Check log file
        log_file = log_dir / f"{run_id}.log"
        log_content = log_file.read_text(encoding="utf-8")
        
        # Run-ID should appear in log entries (via filter)
        assert run_id in log_content

    def test_setup_logging_multiple_calls(self, tmp_path: Path):
        """Test that multiple calls to setup_logging work correctly."""
        import logging
        
        log_dir = tmp_path / "logs"
        
        # First call
        setup_logging(run_id="first", level="INFO", log_dir=log_dir)
        logger1 = logging.getLogger(__name__)
        logger1.info("First message")
        
        # Second call (should reconfigure)
        setup_logging(run_id="second", level="INFO", log_dir=log_dir)
        logger2 = logging.getLogger(__name__)
        logger2.info("Second message")
        
        # Both log files should exist
        assert (log_dir / "first.log").exists()
        assert (log_dir / "second.log").exists()

