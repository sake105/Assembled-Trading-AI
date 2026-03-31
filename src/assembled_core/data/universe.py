"""Universe management — symbol lists with start/end date ranges.

Storage layout: <root>/<universe_name>.parquet (or .csv)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _universe_path(universe_name: str, root: Path, fmt: str) -> Path:
    ext = ".csv" if fmt == "csv" else ".parquet"
    return root / f"{universe_name}{ext}"


def store_universe_history(
    df: pd.DataFrame | list[str],
    universe_name: str = "default",
    root: Path | None = None,
    format: str = "parquet",
    valid_from: pd.Timestamp | None = None,
) -> None:
    """Persist universe history to disk.

    Args:
        df: DataFrame with columns symbol, start_date, end_date — or list of symbols.
        universe_name: Name of the universe.
        root: Directory to store the file.
        format: 'parquet' or 'csv'.
        valid_from: Ignored (backward compatibility).
    """
    base = root or Path("data") / "universe"
    base.mkdir(parents=True, exist_ok=True)
    path = _universe_path(universe_name, base, format)

    if isinstance(df, list):
        out = pd.DataFrame(
            {"symbol": df, "start_date": [None] * len(df), "end_date": [None] * len(df)}
        )
    else:
        out = df.copy()

    if format == "csv":
        out.to_csv(path, index=False)
    else:
        out.to_parquet(path, index=False)


def load_universe_history(
    universe_name: str = "default",
    root: Path | None = None,
) -> pd.DataFrame:
    """Load universe history from disk.

    Returns empty DataFrame with columns {symbol, start_date, end_date} if not found.
    """
    base = root or Path("data") / "universe"
    empty = pd.DataFrame(columns=["symbol", "start_date", "end_date"])

    for fmt in ("parquet", "csv"):
        path = _universe_path(universe_name, base, fmt)
        if path.exists():
            return pd.read_csv(path) if fmt == "csv" else pd.read_parquet(path)

    return empty


def get_universe_members(
    as_of: pd.Timestamp | str | None = None,
    universe_name: str = "default",
    root: Path | None = None,
) -> list[str]:
    """Return sorted list of symbols active in the universe at *as_of*.

    Rules:
      - Symbol is active if start_date <= as_of < end_date (end_date exclusive)
      - end_date=None means still active indefinitely
      - as_of=None falls back to watchlist.txt

    Args:
        as_of: Point-in-time timestamp. Naive timestamps treated as UTC.
        universe_name: Universe to query.
        root: Directory containing universe files.
    """
    if as_of is None:
        wl = Path("watchlist.txt")
        if wl.exists():
            return sorted(
                line.strip()
                for line in wl.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            )
        return []

    if isinstance(as_of, str):
        as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")

    history = load_universe_history(universe_name=universe_name, root=root)
    if history.empty:
        return []

    history = history.copy()
    history["start_date"] = pd.to_datetime(history["start_date"], utc=True)
    history["end_date"] = pd.to_datetime(history["end_date"], utc=True, errors="coerce")

    started = history["start_date"] <= as_of
    not_ended = history["end_date"].isna() | (history["end_date"] > as_of)
    active = history.loc[started & not_ended, "symbol"]

    return sorted(str(s).strip().upper() for s in active)
