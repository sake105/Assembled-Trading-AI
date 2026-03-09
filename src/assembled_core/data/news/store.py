"""News persistence store stub."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_news(path: str | Path | None = None) -> pd.DataFrame:
    """Load stored news from parquet/csv."""
    if path is None:
        return pd.DataFrame(columns=["timestamp", "symbol", "headline", "sentiment"])
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["timestamp", "symbol", "headline", "sentiment"])
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
