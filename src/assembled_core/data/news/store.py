"""News persistence store — month-partitioned Parquet storage."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd


def news_partition_path(
    root: Path,
    source: str,
    year: int,
    month: int,
) -> Path:
    """Return the path to a monthly news partition file.

    Layout: <root>/<source>/<YYYY>/<MM>/news_<source>_<YYYY>_<MM>.parquet
    """
    return (
        root
        / source
        / f"{year}"
        / f"{month:02d}"
        / f"news_{source}_{year}_{month:02d}.parquet"
    )


def store_news_parquet(
    df: pd.DataFrame,
    root: Path,
    source: str,
    year: int,
    month: int,
    mode: str = "replace",
    dedupe_keep: str = "last",
) -> Path:
    """Store news events to a monthly partition atomically.

    Args:
        df: News DataFrame (must not be empty).
        root: Root directory.
        source: News source name (e.g. 'reuters').
        year: Partition year.
        month: Partition month.
        mode: 'replace' (overwrite) or 'append' (merge + dedup).
        dedupe_keep: 'first' or 'last' for deduplication in append mode.

    Returns:
        Path to the written partition file.

    Raises:
        ValueError: If df is empty.
    """
    if df.empty:
        raise ValueError("Cannot store empty DataFrame")

    target = news_partition_path(root, source, year, month)

    out = df.copy()

    if mode == "append" and target.exists():
        existing = pd.read_parquet(target)
        out = pd.concat([existing, out], ignore_index=True)
        # Dedup by provider_id if available, else by (publish_ts, source, headline)
        if "provider_id" in out.columns and not out["provider_id"].isna().all():
            out = out.drop_duplicates(
                subset=["publish_ts", "source", "provider_id"], keep=dedupe_keep
            )
        else:
            out = out.drop_duplicates(
                subset=["publish_ts", "source", "headline"], keep=dedupe_keep
            )

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp.parquet")
    try:
        os.close(fd)
        out.to_parquet(tmp, index=False)
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    return target


def load_news_parquet(
    root: Path,
    source: str,
    year: int,
    month: int,
) -> pd.DataFrame:
    """Load a monthly news partition.

    Returns empty DataFrame if partition does not exist.
    """
    path = news_partition_path(root, source, year, month)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def list_news_partitions(
    root: Path,
    source: str | None = None,
) -> list[Path]:
    """List all news partition files.

    Args:
        root: Root directory.
        source: If given, only list partitions for this source.

    Returns:
        Sorted list of partition file paths.
    """
    if not root.exists():
        return []

    pattern = "**/*.parquet"
    if source is not None:
        base = root / source
        if not base.exists():
            return []
        paths = sorted(base.glob(pattern))
    else:
        paths = sorted(root.glob(pattern))

    return [p for p in paths if p.is_file()]


# Keep backward-compatible alias
def load_news(path: str | Path | None = None) -> pd.DataFrame:
    """Load stored news from parquet/csv (legacy interface)."""
    if path is None:
        return pd.DataFrame(columns=["timestamp", "symbol", "headline", "sentiment"])
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["timestamp", "symbol", "headline", "sentiment"])
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
