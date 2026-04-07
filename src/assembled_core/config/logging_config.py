"""Structured JSON Logging (Plan 11.7).

Configurable JSON log formatter for pipeline and ops.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Format log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "event": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        return json.dumps(log_entry)


def configure_json_logging(
    level: str = "INFO",
    logger_name: str | None = None,
) -> logging.Logger:
    """Configure a logger with JSON formatting.

    Args:
        level: Log level string.
        logger_name: Logger name (None = root).

    Returns:
        Configured logger.
    """
    log = logging.getLogger(logger_name)
    log.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    # Avoid duplicate handlers
    if not any(isinstance(h, logging.StreamHandler) and isinstance(h.formatter, JSONFormatter) for h in log.handlers):
        log.addHandler(handler)

    return log


__all__ = ["JSONFormatter", "configure_json_logging"]
