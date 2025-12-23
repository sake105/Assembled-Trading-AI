# src/assembled_core/config.py
"""Central configuration for the trading pipeline."""

from __future__ import annotations

from pathlib import Path

# Base directory: repo root (assumes this file is in src/assembled_core/)
# Go up: src/assembled_core -> src -> repo root
_BASE_DIR = Path(__file__).resolve().parents[2]

# Output directory: output/ relative to repo root
OUTPUT_DIR = _BASE_DIR / "output"

# Supported frequencies
SUPPORTED_FREQS = ("1d", "5min")


def get_output_path(*parts: str) -> Path:
    """Get a path within the output directory.

    Args:
        *parts: Path components relative to output directory

    Returns:
        Path object: OUTPUT_DIR / part1 / part2 / ...

    Examples:
        >>> get_output_path("aggregates", "5min.parquet")
        Path("output/aggregates/5min.parquet")
        >>> get_output_path("orders_5min.csv")
        Path("output/orders_5min.csv")
    """
    return OUTPUT_DIR.joinpath(*parts)


def get_base_dir() -> Path:
    """Get the repository root directory.

    Returns:
        Path to repository root
    """
    return _BASE_DIR
