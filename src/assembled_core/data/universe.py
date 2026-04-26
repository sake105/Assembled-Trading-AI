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
            return pd.read_csv(path, dtype={"symbol": "string"}) if fmt == "csv" else pd.read_parquet(path)

    return empty


def get_universe_members(
    as_of: pd.Timestamp | str | None = None,
    universe_name: str = "default",
    root: Path | None = None,
    require_active_status: bool = False,
) -> list[str]:
    """Return sorted list of symbols active in the universe at *as_of*.

    Rules:
      - Symbol is active if start_date <= as_of < end_date (end_date exclusive)
      - end_date=NaT with status='active' (or no status column) → still active
      - end_date=NaT with require_active_status=True AND status not 'active'
        → conservatively excluded (avoids survivorship bias from delistings
        without explicit end_date)
      - as_of=None falls back to watchlist.txt

    Args:
        as_of: Point-in-time timestamp. Naive timestamps treated as UTC.
        universe_name: Universe to query.
        root: Directory containing universe files.
        require_active_status: If True, symbols with end_date=NaT must have
            status='active' to be included. Prevents survivorship bias when
            delisted symbols have no recorded end_date.
    """
    if as_of is None:
        # A caller that forgets to pass `as_of` in a historical backtest path
        # silently gets the *live* watchlist — classic survivorship bias
        # (delisted symbols are invisible to the caller). The PIT-safe API is
        # `get_universe_members_pit`; surface this fallback so the risk is
        # observable in logs.
        import logging
        logging.getLogger(__name__).warning(
            "[Universe] get_universe_members called with as_of=None — "
            "falling back to watchlist.txt; this is NOT PIT-safe and hides "
            "delistings in historical contexts. Use get_universe_members_pit "
            "for point-in-time membership."
        )
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

    if require_active_status and "status" in history.columns:
        # When end_date is NaT, only include if status is explicitly 'active'
        # This prevents survivorship bias from delistings without end_date
        null_end = history["end_date"].isna()
        explicitly_active = history["status"].str.lower().eq("active")
        not_ended = (null_end & explicitly_active) | (~null_end & (history["end_date"] > as_of))

    active = history.loc[started & not_ended, "symbol"]

    return sorted(str(s).strip().upper() for s in active)


def get_universe_members_pit(
    as_of: pd.Timestamp | str,
    universe_name: str = "default",
    root: Path | None = None,
    require_active_status: bool = True,
) -> list[str]:
    """Strict point-in-time universe lookup — raises when no members are found.

    Wraps ``get_universe_members`` with two hardening guarantees that make it
    safer for production backtest and live decision paths:

    1. ``as_of`` is mandatory. No silent fallback to ``watchlist.txt`` — a
       missing timestamp is an error, not a default.
    2. An empty result set raises :class:`UniverseLookupError` instead of
       returning ``[]``. This prevents a survivorship-bias-free backtest from
       silently proceeding on an empty universe (which would otherwise look
       like a perfectly-legal zero-position day).
    3. ``require_active_status=True`` by default, so symbols with
       ``end_date=NaT`` must have ``status='active'`` to be included. This
       closes the second survivorship gap: delisted symbols that never got an
       explicit ``end_date`` recorded.

    Raises:
        UniverseLookupError: if the universe history is missing, empty, or
            contains no members that satisfy the PIT filter at ``as_of``.
    """
    from src.assembled_core.errors import UniverseLookupError

    if as_of is None:
        raise UniverseLookupError(
            universe_name=universe_name,
            as_of="None",
            details="as_of is required — strict PIT lookup has no fallback",
        )

    members = get_universe_members(
        as_of=as_of,
        universe_name=universe_name,
        root=root,
        require_active_status=require_active_status,
    )
    if not members:
        raise UniverseLookupError(
            universe_name=universe_name,
            as_of=str(as_of),
            details="PIT filter yielded zero members — check universe history coverage",
        )
    return members


# ---------------------------------------------------------------------------
# Universe Reconstitution (Plan 10.1)
# ---------------------------------------------------------------------------


def build_monthly_snapshots(
    history: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> dict[str, list[str]]:
    """Build monthly universe snapshots for survivorship-bias-free backtesting.

    Args:
        history: Universe history with columns: symbol, start_date, end_date, status.
        start_date: Start of backtest period (YYYY-MM-DD).
        end_date: End of backtest period (YYYY-MM-DD).

    Returns:
        Dict mapping month string (YYYY-MM) -> list of active symbols.
    """
    months = pd.date_range(start_date, end_date, freq="MS", tz="UTC")
    snapshots: dict[str, list[str]] = {}

    hist = history.copy()
    hist["start_date"] = pd.to_datetime(hist["start_date"], utc=True)
    hist["end_date"] = pd.to_datetime(hist["end_date"], utc=True, errors="coerce")

    for month_start in months:
        key = month_start.strftime("%Y-%m")
        started = hist["start_date"] <= month_start
        not_ended = hist["end_date"].isna() | (hist["end_date"] > month_start)
        if "status" in hist.columns:
            null_end = hist["end_date"].isna()
            active_status = hist["status"].str.lower().eq("active")
            not_ended = (null_end & active_status) | (~null_end & (hist["end_date"] > month_start))
        active = hist.loc[started & not_ended, "symbol"]
        snapshots[key] = sorted(str(s).strip().upper() for s in active)

    return snapshots


def get_pit_members_for_date(
    snapshots: dict[str, list[str]],
    date: pd.Timestamp,
) -> list[str]:
    """Look up PIT universe members for a specific date from monthly snapshots.

    Args:
        snapshots: Output from build_monthly_snapshots().
        date: Date to look up.

    Returns:
        List of symbols active in the month containing ``date``.
    """
    key = pd.Timestamp(date).strftime("%Y-%m")
    return snapshots.get(key, [])


# ---------------------------------------------------------------------------
# 10.7  Survivorship-Bias-Free Delisting Detection
# ---------------------------------------------------------------------------

def detect_delisted_symbols(
    prices: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    max_stale_days: int = 90,
    terminal_return: float = -0.30,
) -> dict:
    """Detect delisted symbols from price staleness.

    Args:
        prices: Price DataFrame with timestamp, symbol, close columns.
        as_of_date: Current date for staleness check.
        max_stale_days: Days without price update to flag as delisted.
        terminal_return: Return to apply in backtest for delisted stocks.

    Returns:
        Dict with delisted list and terminal values.
    """
    as_of = pd.Timestamp(as_of_date)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")

    prices_copy = prices.copy()
    prices_copy["timestamp"] = pd.to_datetime(prices_copy["timestamp"], utc=True)

    # Point-in-time filter: a historical as_of must not peek at price rows
    # after that date. Without this, a symbol still trading in the full
    # panel but delisted at as_of would be missed because group.max() sees
    # later rows.
    prices_copy = prices_copy[prices_copy["timestamp"] <= as_of]

    delisted = []
    terminal_values = {}

    for sym, group in prices_copy.groupby("symbol"):
        if group.empty:
            continue
        last_date = group["timestamp"].max()
        days_stale = (as_of - last_date).days

        if days_stale > max_stale_days:
            last_price = float(group.sort_values("timestamp").iloc[-1]["close"])
            terminal_price = last_price * (1.0 + terminal_return)
            delisted.append(str(sym))
            terminal_values[str(sym)] = {
                "last_price": round(last_price, 2),
                "last_date": str(last_date.date()),
                "days_stale": days_stale,
                "terminal_price": round(terminal_price, 2),
            }

    return {
        "delisted": delisted,
        "n_delisted": len(delisted),
        "terminal_values": terminal_values,
    }
