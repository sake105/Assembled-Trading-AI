"""Data snapshot ID computation for reproducibility."""

from __future__ import annotations

import hashlib

import pandas as pd


def compute_price_panel_snapshot_id(
    prices: pd.DataFrame,
    source_meta: dict | None = None,
) -> str:
    """Compute a deterministic snapshot ID for a price panel.

    Args:
        prices: Price DataFrame.
        source_meta: Optional metadata dict.

    Returns:
        Hex digest identifying the data snapshot.
    """
    if prices.empty:
        return hashlib.sha256(b"empty").hexdigest()[:16]

    h = hashlib.sha256()
    h.update(str(len(prices)).encode())
    h.update(str(sorted(prices.columns.tolist())).encode())
    if "symbol" in prices.columns:
        h.update(",".join(sorted(prices["symbol"].unique())).encode())
    if "timestamp" in prices.columns:
        h.update(str(prices["timestamp"].min()).encode())
        h.update(str(prices["timestamp"].max()).encode())
    if source_meta:
        h.update(str(sorted(source_meta.items())).encode())
    return h.hexdigest()[:16]
