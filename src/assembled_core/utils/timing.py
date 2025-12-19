"""Timing utilities for performance profiling."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def timed_block(name: str):
    """
    Context manager to log the runtime of a code block.

    This utility can be used throughout the codebase to measure execution time
    of specific operations without modifying business logic. The timing
    information is logged at INFO level with a standardized format.

    Parameters
    ----------
    name : str
        Name/label for the timed block (will appear in logs)

    Examples
    --------
    >>> from src.assembled_core.utils.timing import timed_block
    >>> 
    >>> with timed_block("build_factors"):
    ...     factors_df = compute_factors(...)
    ... 
    >>> # Log output: TIMING | build_factors | 1.234 sec
    >>> 
    >>> with timed_block("run_backtest"):
    ...     result = run_backtest(...)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        logger.info("TIMING | %s | %.3f sec", name, duration)

