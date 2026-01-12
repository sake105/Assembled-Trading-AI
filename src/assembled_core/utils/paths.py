"""Path utility functions (shared across layers)."""

from __future__ import annotations

from pathlib import Path

from src.assembled_core.config import OUTPUT_DIR


def get_default_price_path(freq: str, output_dir: Path | str | None = None) -> Path:
    """Get default price file path for a frequency.

    Args:
        freq: Frequency string ("1d" or "5min")
        output_dir: Base output directory (default: None, uses config.OUTPUT_DIR)

    Returns:
        Path to price parquet file

    Raises:
        ValueError: If freq is not supported
    """
    base = Path(output_dir) if output_dir else OUTPUT_DIR
    if freq == "1d":
        return base / "aggregates" / "daily.parquet"
    if freq == "5min":
        return base / "aggregates" / "5min.parquet"
    raise ValueError(f"Unbekannte freq: {freq}")
