"""Panel store for price panel persistence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_price_panel_parquet(path: str | Path) -> pd.DataFrame:
    """Load a price panel from parquet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Price panel not found: {path}")
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df
