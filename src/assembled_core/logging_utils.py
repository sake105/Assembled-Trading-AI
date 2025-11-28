# src/assembled_core/logging_utils.py
"""Logging utilities for CLI scripts and core modules."""
from __future__ import annotations

import logging
import sys
from typing import Literal


def setup_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    format_string: str | None = None
) -> logging.Logger:
    """Setup logging configuration for CLI scripts.
    
    This function configures Python's logging module with a simple format
    suitable for CLI scripts. It sets up a single StreamHandler that writes
    to stdout.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR), default: INFO
        format_string: Optional custom format string. If None, uses default:
            "[%(levelname)s] %(message)s"
    
    Returns:
        Logger instance named "assembled_core"
    
    Example:
        >>> logger = setup_logging(level="INFO")
        >>> logger.info("Starting pipeline")
        [INFO] Starting pipeline
        >>> logger.warning("Missing data for symbol AAPL")
        [WARNING] Missing data for symbol AAPL
        >>> logger.error("Failed to load prices")
        [ERROR] Failed to load prices
    """
    # Default format: [LEVEL] message
    if format_string is None:
        format_string = "[%(levelname)s] %(message)s"
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True  # Override any existing configuration
    )
    
    # Get logger
    logger = logging.getLogger("assembled_core")
    logger.setLevel(numeric_level)
    
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.
    
    If logging has not been configured yet, this will use Python's default
    logging configuration. For CLI scripts, call setup_logging() first.
    
    Args:
        name: Logger name (default: "assembled_core")
    
    Returns:
        Logger instance
    """
    if name is None:
        name = "assembled_core"
    return logging.getLogger(name)

