"""Timing utilities for profiling pipeline steps.

This module provides a context manager for timing code blocks and collecting
timing data that can be written to JSON files for analysis.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@contextmanager
def timed_step(
    name: str,
    timings: dict[str, Any],
    logger_instance: logging.Logger | None = None,
    meta: dict[str, Any] | None = None,
):
    """Context manager to time a code block and record timing data.

    Records start timestamp, end timestamp, and duration (in milliseconds)
    in the provided timings dictionary under the given step name.

    Args:
        name: Step name (e.g., "load_data", "build_factors")
        timings: Dictionary to store timing data (will be mutated)
        logger_instance: Optional logger instance to log step start/end
        meta: Optional dictionary of metadata to include in timing record

    Example:
        >>> timings = {}
        >>> with timed_step("load_data", timings, logger):
        ...     data = load_prices()
        >>> print(timings["load_data"]["duration_ms"])
        123.45
    """
    log = logger_instance or logger

    # Record start time
    start_ts = time.perf_counter()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

    if log:
        log.debug(f"[TIMING] Step '{name}' started at {start_iso}")

    try:
        yield
    finally:
        # Record end time and compute duration
        end_ts = time.perf_counter()
        end_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        duration_ms = (end_ts - start_ts) * 1000.0

        # Store timing data
        timing_record: dict[str, Any] = {
            "start_ts": start_iso,
            "end_ts": end_iso,
            "start_ts_epoch": start_ts,
            "end_ts_epoch": end_ts,
            "duration_ms": duration_ms,
        }

        if meta:
            timing_record["meta"] = meta

        timings[name] = timing_record

        if log:
            log.debug(
                f"[TIMING] Step '{name}' completed in {duration_ms:.2f}ms (ended at {end_iso})"
            )


def write_timings_json(
    timings: dict[str, Any],
    output_path: Path,
    job_name: str | None = None,
    job_meta: dict[str, Any] | None = None,
) -> None:
    """Write timing data to a JSON file.

    Creates a structured JSON output with:
    - job_name: Optional job identifier
    - job_meta: Optional job metadata
    - steps: Dictionary of step timings

    Args:
        timings: Dictionary of timing data (from timed_step calls)
        output_path: Path to write JSON file
        job_name: Optional job name for the output
        job_meta: Optional metadata about the job (e.g., config, parameters)

    Example:
        >>> timings = {"load_data": {"duration_ms": 123.45, ...}}
        >>> write_timings_json(timings, Path("output/timings.json"), job_name="EOD_SMALL")
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data: dict[str, Any] = {
        "steps": timings,
    }

    if job_name:
        output_data["job_name"] = job_name

    if job_meta:
        output_data["job_meta"] = job_meta

    # Add summary statistics (always include, even if empty)
    durations = [step.get("duration_ms", 0) for step in timings.values()]
    output_data["summary"] = {
        "total_steps": len(timings),
        "total_duration_ms": sum(durations),
        "avg_duration_ms": sum(durations) / len(durations) if durations else 0.0,
        "min_duration_ms": min(durations) if durations else 0.0,
        "max_duration_ms": max(durations) if durations else 0.0,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Timings written to {output_path}")


def load_timings_json(input_path: Path) -> dict[str, Any]:
    """Load timing data from a JSON file.

    Args:
        input_path: Path to JSON file

    Returns:
        Dictionary containing timing data (same structure as written by write_timings_json)
    """
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)
