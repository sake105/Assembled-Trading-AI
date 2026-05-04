"""DuckDB + Parquet Feature-Store with ASOF-JOIN PIT safety.

From 12_FREE_INFRASTRUKTUR.md §12.6 and 14_FREE_UNIVERSUM.md §14.7.

Storage layout (Hive partitioning):
  features/
    view=rsi/
      year=2025/month=01/ticker=AAPL.parquet
    view=residual_mom/
      ...

ASOF-JOIN pattern guarantees PIT safety:
  available_at <= inference_ts - embargo
  (prevents look-ahead bias structurally)

Install: pip install duckdb==1.1.3 pyarrow==17.0.0
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SAFE_VIEW_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")
_SAFE_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,16}$")


def _validate_view(view: str) -> str:
    if not _SAFE_VIEW_RE.match(view):
        raise ValueError(
            f"Invalid feature view name: {view!r} — must match [a-zA-Z][a-zA-Z0-9_]{{0,63}}"
        )
    return view


def _sanitize_tickers(tickers: list[str]) -> list[str]:
    bad = [t for t in tickers if not _SAFE_TICKER_RE.match(t)]
    if bad:
        raise ValueError(f"Invalid ticker(s): {bad!r} — must match [A-Z0-9.-]{{1,16}}")
    return tickers


import pandas as pd

logger = logging.getLogger(__name__)

# Default feature store root — override via env var FEATURE_STORE_PATH
_DEFAULT_ROOT = Path("data/features")
FEATURE_STORE_PATH = Path(os.environ.get("FEATURE_STORE_PATH", str(_DEFAULT_ROOT)))


def _try_duckdb():
    try:
        import duckdb

        return duckdb
    except ImportError:
        logger.warning("duckdb not installed — pip install duckdb==1.1.3")
        return None


def _try_pyarrow():
    try:
        import pyarrow

        return pyarrow
    except ImportError:
        logger.warning("pyarrow not installed — pip install pyarrow==17.0.0")
        return None


# ---------------------------------------------------------------------------
# Write: store feature DataFrame
# ---------------------------------------------------------------------------


def write_features(
    df: pd.DataFrame,
    view: str,
    ticker: str,
    available_at_col: str = "available_at",
    root: Path | str | None = None,
) -> Path | None:
    """Write feature DataFrame to Parquet with Hive partition layout.

    Args:
        df: Feature DataFrame. Must have `available_at` column (UTC datetime).
        view: Feature view name (e.g. 'rsi', 'residual_mom', 'shap_vals').
        ticker: Ticker symbol (e.g. 'AAPL').
        available_at_col: Column containing the availability timestamp.
        root: Override feature store root directory.

    Returns:
        Path to written Parquet file, or None on failure.
    """
    if df.empty:
        logger.warning(
            "write_features: empty DataFrame for view=%s ticker=%s — skipping",
            view,
            ticker,
        )
        return None

    pa = _try_pyarrow()
    if pa is None:
        return None

    store_root = Path(root) if root else FEATURE_STORE_PATH

    if available_at_col not in df.columns:
        logger.warning(
            "write_features: '%s' column missing — adding as now()", available_at_col
        )
        df = df.copy()
        df[available_at_col] = datetime.now(tz=timezone.utc)

    if df.empty:
        logger.warning(
            "write_features: empty DataFrame passed for view=%s ticker=%s — skipping",
            view,
            ticker,
        )
        return None

    # Determine partition key from first available_at timestamp
    first_ts = pd.to_datetime(df[available_at_col].iloc[0])
    year = first_ts.year
    month = f"{first_ts.month:02d}"

    partition_dir = store_root / f"view={view}" / f"year={year}" / f"month={month}"
    partition_dir.mkdir(parents=True, exist_ok=True)

    file_path = partition_dir / f"ticker={ticker}.parquet"

    try:
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df, preserve_index=True)
        pq.write_table(table, str(file_path), compression="snappy")
        logger.debug("Feature store write: %s → %s", view, file_path)
        return file_path
    except Exception as exc:
        logger.warning("Feature store write failed (%s, %s): %s", view, ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Read: ASOF-JOIN feature query
# ---------------------------------------------------------------------------


def read_features_asof(
    view: str,
    entities: pd.DataFrame,
    inference_ts_col: str = "inference_ts",
    embargo_minutes: int = 1,
    root: Path | str | None = None,
) -> pd.DataFrame | None:
    """ASOF-JOIN feature lookup: returns features available at inference_ts - embargo.

    This is the PIT-safe read pattern. Features with available_at > inference_ts
    are excluded, preventing look-ahead bias.

    Args:
        view: Feature view name.
        entities: DataFrame with at minimum (ticker, inference_ts_col).
        inference_ts_col: Column containing the inference timestamp.
        embargo_minutes: Minutes to subtract from inference_ts for the ASOF boundary.
        root: Override feature store root.

    Returns:
        Merged DataFrame with feature columns, or None on failure.
    """
    _validate_view(view)
    duckdb = _try_duckdb()
    if duckdb is None:
        return None

    store_root = Path(root) if root else FEATURE_STORE_PATH
    pattern = str(store_root / f"view={view}" / "**" / "*.parquet")

    try:
        con = duckdb.connect()
        con.execute(f"""
            CREATE OR REPLACE VIEW fv_{view} AS
            SELECT * FROM read_parquet('{pattern}', hive_partitioning=1)
        """)

        # Register entities as a table
        con.register("entities_df", entities)

        embargo_interval = f"INTERVAL '{embargo_minutes} minutes'"
        query = f"""
            SELECT e.*, fv_{view}.*
            EXCLUDE (fv_{view}.ticker, fv_{view}.available_at)
            FROM entities_df e
            ASOF LEFT JOIN fv_{view}
                ON e.ticker = fv_{view}.ticker
                AND e.{inference_ts_col} - {embargo_interval} >= fv_{view}.available_at
            ORDER BY e.{inference_ts_col}
        """
        result = con.execute(query).df()
        con.close()
        return result

    except Exception as exc:
        logger.warning("Feature store ASOF read failed (view=%s): %s", view, exc)
        return None


def read_features_latest(
    view: str,
    tickers: list[str],
    root: Path | str | None = None,
) -> pd.DataFrame | None:
    """Read the most recent feature row per ticker from a view.

    Args:
        view: Feature view name.
        tickers: List of ticker symbols to fetch.
        root: Override feature store root.

    Returns:
        DataFrame with latest features per ticker, or None on failure.
    """
    _validate_view(view)
    _sanitize_tickers(tickers)
    duckdb = _try_duckdb()
    if duckdb is None:
        return None

    store_root = Path(root) if root else FEATURE_STORE_PATH
    pattern = str(store_root / f"view={view}" / "**" / "*.parquet")
    tickers_str = ", ".join(f"'{t}'" for t in tickers)

    try:
        con = duckdb.connect()
        query = f"""
            SELECT *
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY ticker ORDER BY available_at DESC
                ) AS rn
                FROM read_parquet('{pattern}', hive_partitioning=1)
                WHERE ticker IN ({tickers_str})
            )
            WHERE rn = 1
        """
        result = con.execute(query).df()
        con.close()
        return result

    except Exception as exc:
        logger.debug("Feature store latest read failed (view=%s): %s", view, exc)
        return None


# ---------------------------------------------------------------------------
# Meta: list views and diagnose
# ---------------------------------------------------------------------------


def list_views(root: Path | str | None = None) -> list[str]:
    """List available feature views in the store."""
    store_root = Path(root) if root else FEATURE_STORE_PATH
    if not store_root.exists():
        return []

    views = []
    for p in store_root.iterdir():
        if p.is_dir() and p.name.startswith("view="):
            views.append(p.name[5:])
    return sorted(views)


def feature_store_stats(root: Path | str | None = None) -> dict[str, Any]:
    """Return summary stats about the feature store."""
    store_root = Path(root) if root else FEATURE_STORE_PATH
    views = list_views(store_root)
    total_files = (
        sum(1 for _ in store_root.rglob("*.parquet")) if store_root.exists() else 0
    )
    total_bytes = (
        sum(f.stat().st_size for f in store_root.rglob("*.parquet"))
        if store_root.exists()
        else 0
    )

    return {
        "root": str(store_root),
        "views": views,
        "n_views": len(views),
        "n_parquet_files": total_files,
        "total_size_mb": round(total_bytes / 1024 / 1024, 2),
    }


__all__ = [
    "FEATURE_STORE_PATH",
    "write_features",
    "read_features_asof",
    "read_features_latest",
    "list_views",
    "feature_store_stats",
]
