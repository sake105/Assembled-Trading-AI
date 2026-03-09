"""Lightweight security master for symbol metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def get_default_security_master_path() -> Path:
    """Return the default security master CSV path."""
    return Path("data") / "security_master.csv"


def load_security_master(path: Path | str | None = None) -> pd.DataFrame:
    """Load security master from CSV.

    Returns DataFrame with at least columns: symbol.
    Returns empty DataFrame if file does not exist.
    """
    if path is None:
        path = get_default_security_master_path()
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "name", "sector", "exchange"])
    return pd.read_csv(path)
