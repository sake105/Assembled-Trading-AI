"""Data Versioning (Plan 10.9).

SHA-256 hash over price data per run for lineage tracking.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)


def compute_data_hash(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Compute SHA-256 hash of a DataFrame for versioning.

    Args:
        df: Data to hash.
        columns: Specific columns to include (default: all).

    Returns:
        Hex digest string.
    """
    if df.empty:
        return hashlib.sha256(b"empty").hexdigest()

    subset = df[columns] if columns else df
    # Canonicalise: sort columns alphabetically and drop the row index so
    # the hash depends on content only, not on column order or on the
    # reload-dependent pandas index. Prior behaviour produced different
    # hashes for the same data reloaded from Parquet.
    subset = subset.reindex(sorted(subset.columns), axis=1)
    content = subset.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def create_lineage_record(
    data_hash: str,
    source: str,
    n_rows: int,
    n_symbols: int = 0,
    date_range: str = "",
) -> dict:
    """Create a lineage metadata record.

    Args:
        data_hash: SHA-256 hash of the data.
        source: Data source identifier.
        n_rows: Number of rows.
        n_symbols: Number of unique symbols.
        date_range: Date range string.

    Returns:
        Lineage record dict.
    """
    return {
        "data_hash": data_hash,
        "source": source,
        "n_rows": n_rows,
        "n_symbols": n_symbols,
        "date_range": date_range,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


__all__ = ["compute_data_hash", "create_lineage_record"]
