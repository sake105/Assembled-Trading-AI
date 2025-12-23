# src/assembled_core/logging_config.py
"""Central logging configuration for Assembled Trading AI Backend.

This module provides a centralized logging setup that:
- Configures console and file handlers
- Supports Run-IDs for tracking execution runs
- Creates log files in logs/ directory with date/run-id in filename
- Uses consistent log format across all modules

Usage:
    >>> from src.assembled_core.logging_config import setup_logging
    >>> setup_logging(run_id="run_20250115_143022", level="INFO")
    >>> import logging
    >>> logger = logging.getLogger(__name__)
    >>> logger.info("Pipeline started")
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4


def setup_logging(
    run_id: str | None = None,
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    log_dir: Path | str | None = None,
) -> None:
    """Setup centralized logging configuration with console and file handlers.

    This function configures Python's logging module with:
    - Console handler (stdout) with simplified format
    - File handler (logs/ directory) with detailed format including Run-ID
    - Automatic log directory creation
    - Log file naming: logs/run_{run_id}.log or logs/run_YYYYMMDD_HHMMSS.log

    Args:
        run_id: Optional Run-ID for tracking execution runs. If None, generates
            a timestamp-based ID (YYYYMMDD_HHMMSS).
        level: Logging level (DEBUG, INFO, WARNING, ERROR), default: INFO
        log_dir: Optional log directory path. If None, uses logs/ in project root.

    Side effects:
        - Creates logs/ directory if it doesn't exist
        - Configures root logger with console and file handlers
        - Overrides any existing logging configuration (force=True)

    Example:
        >>> setup_logging(run_id="backtest_20250115", level="INFO")
        >>> import logging
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Starting backtest")
        [INFO] Starting backtest  # Console output
        # Also written to logs/run_backtest_20250115.log
    """
    # Generate Run-ID if not provided
    if run_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}"

    # Determine log directory
    if log_dir is None:
        # Try to get from settings, fallback to project root
        try:
            from src.assembled_core.config.settings import get_settings

            settings = get_settings()
            log_dir = settings.logs_dir
        except (ImportError, AttributeError, ModuleNotFoundError):
            # Fallback: assume we're in project root
            log_dir = Path(__file__).resolve().parents[3] / "logs"
    else:
        log_dir = Path(log_dir)

    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Log file path
    log_file = log_dir / f"{run_id}.log"

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Console format: [LEVEL] message (simple, for CLI output)
    console_format = "[%(levelname)s] %(message)s"

    # File format: timestamp | level | logger | run_id | message (detailed)
    file_format = (
        "%(asctime)s | %(levelname)-8s | %(name)-30s | [%(run_id)s] %(message)s"
    )

    # Create formatters
    console_formatter = logging.Formatter(console_format)
    file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(file_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Close existing handlers before clearing to avoid ResourceWarning
    for handler in root_logger.handlers[:]:
        try:
            handler.close()
        except (AttributeError, RuntimeError):
            pass  # Ignore errors when closing handlers

    root_logger.handlers.clear()  # Remove any existing handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Add Run-ID to all log records via a filter
    class RunIDFilter(logging.Filter):
        """Filter to add Run-ID to all log records."""

        def __init__(self, run_id: str):
            super().__init__()
            self.run_id = run_id

        def filter(self, record: logging.LogRecord) -> bool:
            record.run_id = self.run_id  # type: ignore[attr-defined]
            return True

    # Remove any existing RunIDFilter to avoid duplicates
    root_logger.filters = [
        f for f in root_logger.filters if not isinstance(f, RunIDFilter)
    ]
    for handler in root_logger.handlers:
        handler.filters = [f for f in handler.filters if not isinstance(f, RunIDFilter)]

    # Add filter to file handler (where run_id is used in format)
    run_id_filter = RunIDFilter(run_id)
    file_handler.addFilter(run_id_filter)

    # Also add to root logger for console handler (optional, but consistent)
    root_logger.addFilter(run_id_filter)

    # Log initial message
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging initialized: Run-ID={run_id}, Level={level}, Log file={log_file}"
    )


def generate_run_id(prefix: str = "run") -> str:
    """Generate a unique Run-ID.

    Args:
        prefix: Optional prefix for the Run-ID (default: "run")

    Returns:
        Run-ID string in format: {prefix}_{YYYYMMDD}_{HHMMSS}_{uuid4_short}

    Example:
        >>> generate_run_id("backtest")
        'backtest_20250115_143022_a1b2c3d4'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uuid_short = str(uuid4()).replace("-", "")[:8]
    return f"{prefix}_{timestamp}_{uuid_short}"
