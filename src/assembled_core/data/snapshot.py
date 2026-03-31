"""Data snapshot ID computation for reproducibility."""

from __future__ import annotations

import hashlib

import pandas as pd

_REQUIRED_COLS = {"timestamp", "symbol", "close"}
_HASH_COLS = ["symbol", "timestamp", "close"]


def compute_price_panel_snapshot_id(
    prices: pd.DataFrame,
    source_meta: dict | None = None,
    freq: str | None = None,
) -> str:
    """Compute a deterministic snapshot ID for a price panel.

    Args:
        prices: Price DataFrame. Must contain columns: timestamp, symbol, close.
        source_meta: Optional metadata dict included in the hash.
        freq: Optional frequency string (e.g. "1d") included in the hash.

    Returns:
        64-character hex digest (SHA-256) identifying the data snapshot.

    Raises:
        ValueError: If required columns (timestamp, symbol, close) are missing.
    """
    missing = _REQUIRED_COLS - set(prices.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if prices.empty:
        h = hashlib.sha256(b"empty")
        if freq is not None:
            h.update(freq.encode())
        return h.hexdigest()

    # Work on required columns only — ignore optional columns (order-invariant)
    df = prices[_HASH_COLS].copy()

    # Normalize timestamps: convert to UTC, then strip timezone for consistent str repr
    ts = df["timestamp"]
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        df["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)

    # Normalize close to float64 (dtype-invariant: int 150 == float 150.0)
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype(float)

    # Deduplicate: keep last per (symbol, timestamp)
    df = df.drop_duplicates(subset=["symbol", "timestamp"], keep="last")

    # Sort for row-order-invariance
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    h = hashlib.sha256()
    h.update(str(len(df)).encode())
    h.update(",".join(sorted(df["symbol"].unique())).encode())
    h.update(str(df["timestamp"].min()).encode())
    h.update(str(df["timestamp"].max()).encode())

    # Hash close values as sorted string — NaN/inf convert to "nan"/"inf" consistently
    close_str = ",".join(df["close"].astype(str).tolist())
    h.update(close_str.encode())

    if source_meta:
        h.update(str(sorted(source_meta.items())).encode())
    if freq is not None:
        h.update(freq.encode())

    return h.hexdigest()
