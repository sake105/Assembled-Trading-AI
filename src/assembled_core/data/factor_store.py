"""Factor store for caching computed factors (Parquet-based).

Provides compute_universe_key, load_factors, store_factors for the factor
cache layer used by features.factor_store_integration and pipeline.trading_cycle.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_FACTORS_ROOT = Path("output/factors")


def get_factor_store_root(factors_root: Path | None = None) -> Path:
    """Return the root directory for the factor store."""
    root = factors_root or _DEFAULT_FACTORS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def compute_universe_key(symbols: list[str] | None = None) -> str:
    """Compute a deterministic hash key for a universe of symbols.

    Args:
        symbols: Sorted list of ticker symbols.

    Returns:
        Short hex digest identifying the universe.
    """
    if symbols is None:
        symbols = []
    joined = ",".join(sorted(symbols))
    return hashlib.sha256(joined.encode()).hexdigest()[:12]


def _factor_path(
    factor_group: str,
    freq: str,
    universe_key: str,
    factors_root: Path | None = None,
) -> Path:
    root = get_factor_store_root(factors_root)
    return root / factor_group / freq / f"{universe_key}.parquet"


def load_factors(
    factor_group: str,
    freq: str,
    universe_key: str,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    as_of: pd.Timestamp | None = None,
    factors_root: Path | None = None,
) -> pd.DataFrame | None:
    """Load cached factors from the store.

    Returns None if no cache file exists.
    """
    path = _factor_path(factor_group, freq, universe_key, factors_root)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            if start_date is not None:
                df = df[df["timestamp"] >= start_date]
            if end_date is not None:
                df = df[df["timestamp"] <= end_date]
            if as_of is not None:
                df = df[df["timestamp"] <= as_of]
        return df
    except Exception:
        logger.warning("Failed to load factors from %s", path, exc_info=True)
        return None


def load_factors_parquet(path: Path | str) -> pd.DataFrame:
    """Load a single parquet factor file."""
    return pd.read_parquet(path)


def store_factors(
    df: pd.DataFrame,
    factor_group: str,
    freq: str,
    universe_key: str,
    mode: str = "overwrite",
    factors_root: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Store computed factors to the cache.

    Args:
        df: DataFrame with factor data.
        factor_group: Factor group name.
        freq: Trading frequency.
        universe_key: Universe hash key.
        mode: 'overwrite' or 'append'.
        factors_root: Optional root directory.
        metadata: Optional metadata dict (stored as parquet metadata).

    Returns:
        Path to the written parquet file.
    """
    path = _factor_path(factor_group, freq, universe_key, factors_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "append" and path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df]).drop_duplicates(
            subset=["timestamp", "symbol"], keep="last"
        )

    df.to_parquet(path, index=False)
    logger.info("[factor_store] Stored %d rows to %s", len(df), path)
    return path
