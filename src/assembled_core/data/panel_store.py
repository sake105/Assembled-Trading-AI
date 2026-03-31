"""Panel store for price panel persistence (Parquet-based).

Layout: <root>/panels/<freq>/<universe>/panel.parquet
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytz

_UTC = pytz.UTC

_REQUIRED_COLS = {"timestamp", "symbol", "close"}


def panel_path(
    freq: str,
    universe: str | None = None,
    root: Path | None = None,
) -> Path:
    """Return the path to a price panel file.

    Args:
        freq: Trading frequency (e.g. '1d').
        universe: Universe name (defaults to 'default').
        root: Optional root directory.

    Returns:
        Path: <root>/panels/<freq>/<universe>/panel.parquet
    """
    base = (root or Path("output")) / "panels"
    return base / freq / (universe or "default") / "panel.parquet"


def panel_exists(
    freq: str,
    universe: str | None = None,
    root: Path | None = None,
) -> bool:
    """Return True if the panel file exists."""
    return panel_path(freq=freq, universe=universe, root=root).exists()


def store_price_panel_parquet(
    df: pd.DataFrame,
    freq: str,
    universe: str | None = None,
    root: Path | None = None,
    mode: str = "replace",
) -> Path:
    """Store a price panel to Parquet atomically.

    Args:
        df: DataFrame with at least 'timestamp', 'symbol', 'close'.
        freq: Trading frequency.
        universe: Universe name.
        root: Optional root directory.
        mode: 'replace' (overwrite) or 'append' (merge, dedup).

    Returns:
        Path to the written panel file.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    df = df.copy()
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(_UTC)
        else:
            ts = ts.dt.tz_convert(_UTC)
        df["timestamp"] = ts

    target = panel_path(freq=freq, universe=universe, root=root)

    if mode == "append" and target.exists():
        existing = pd.read_parquet(target)
        if "timestamp" in existing.columns:
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        df = (
            pd.concat([existing, df], ignore_index=True)
            .drop_duplicates(subset=["timestamp", "symbol"], keep="last")
        )

    if "timestamp" in df.columns and "symbol" in df.columns:
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp.parquet")
    try:
        os.close(fd)
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return target


def load_price_panel_parquet(
    freq: str,
    universe: str | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """Load a price panel from Parquet.

    Args:
        freq: Trading frequency.
        universe: Universe name.
        root: Optional root directory.

    Returns:
        DataFrame sorted by (symbol, timestamp) with UTC timestamps.

    Raises:
        FileNotFoundError: If the panel file does not exist.
    """
    path = panel_path(freq=freq, universe=universe, root=root)
    if not path.exists():
        raise FileNotFoundError(f"Panel file not found: {path}")
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(_UTC)
        else:
            ts = ts.dt.tz_convert(_UTC)
        df["timestamp"] = ts
    if "timestamp" in df.columns and "symbol" in df.columns:
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df
