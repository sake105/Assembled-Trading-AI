"""Universe management — symbol lists with start/end date ranges.

Storage layout: <root>/<universe_name>.parquet (or .csv)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


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
            return (
                pd.read_csv(path, dtype={"symbol": "string"})
                if fmt == "csv"
                else pd.read_parquet(path)
            )

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
                for line in wl.read_text(encoding="utf-8").splitlines()
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
        not_ended = (null_end & explicitly_active) | (
            ~null_end & (history["end_date"] > as_of)
        )

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
# PIT Universe — build from panel + signal_fn wrapper
# ---------------------------------------------------------------------------


def build_universe_history_from_prices(
    prices_df: pd.DataFrame,
    *,
    status: str = "active",
) -> pd.DataFrame:
    """Derive symbol membership windows from a price panel.

    Uses each symbol's first and last timestamp as start_date / end_date.
    Symbols whose last row equals the panel maximum are treated as still listed
    (end_date = NaT). All others get end_date = last_ts + 1 business day so
    the membership interval is half-open: [start_date, end_date).

    Args:
        prices_df: Long-format price panel with at least 'timestamp' and 'symbol'.
        status: Value written to the 'status' column (default 'active').

    Returns:
        DataFrame with columns: symbol, start_date, end_date, status.
    """
    ts = pd.to_datetime(prices_df["timestamp"], utc=True)
    panel_end = ts.max()

    agg = prices_df.assign(_ts=ts).groupby("symbol")["_ts"].agg(["min", "max"])
    agg.columns = ["start_date", "last_ts"]

    # Symbols still active at panel end → end_date = NaT (still listed)
    still_active = agg["last_ts"] >= panel_end
    agg["end_date"] = agg["last_ts"].where(~still_active, other=pd.NaT)
    # For delisted symbols, advance end_date by 1 business day (exclusive boundary)
    delisted_mask = ~still_active
    if delisted_mask.any():
        agg.loc[delisted_mask, "end_date"] = agg.loc[
            delisted_mask, "last_ts"
        ] + pd.offsets.BDay(1)

    agg["status"] = status
    agg = agg.drop(columns=["last_ts"]).reset_index()
    return agg[["symbol", "start_date", "end_date", "status"]]


def _pit_members_from_history(
    universe_history: pd.DataFrame,
    as_of: pd.Timestamp,
) -> set[str]:
    """Return set of PIT-valid symbols at as_of (in-memory, no disk I/O)."""
    if universe_history.empty:
        return set()
    hist = universe_history.copy()
    hist["start_date"] = pd.to_datetime(hist["start_date"], utc=True)
    hist["end_date"] = pd.to_datetime(hist["end_date"], utc=True, errors="coerce")
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    started = hist["start_date"] <= as_of
    not_ended = hist["end_date"].isna() | (hist["end_date"] > as_of)
    if "status" in hist.columns:
        null_end = hist["end_date"].isna()
        active_status = hist["status"].str.lower().eq("active")
        not_ended = (null_end & active_status) | (
            ~null_end & (hist["end_date"] > as_of)
        )
    return set(hist.loc[started & not_ended, "symbol"].str.strip().str.upper())


def wrap_signal_fn_with_pit_filter(
    signal_fn: Callable[[pd.DataFrame], pd.DataFrame],
    universe_history: pd.DataFrame,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Wrap a signal function so its output is filtered to PIT universe members.

    The wrapper derives the rebalance date from the signals DataFrame's latest
    timestamp, then keeps only rows whose symbol was active in the universe at
    that date.

    Degrades gracefully: if universe_history is empty a WARNING is logged and
    the original signals are returned unfiltered (backwards-compatible).

    Args:
        signal_fn: Original signal function (prices_df -> signals_df).
        universe_history: Output of build_universe_history_from_prices or
            load_universe_history — columns: symbol, start_date, end_date[, status].

    Returns:
        Wrapped callable with the same signature as signal_fn.
    """
    if universe_history.empty:
        logger.warning(
            "[PIT] universe_history is empty — PIT filter disabled, signals unfiltered. "
            "Run build_universe_history_from_prices() to enable."
        )
        return signal_fn

    def _wrapped(prices_df: pd.DataFrame) -> pd.DataFrame:
        signals = signal_fn(prices_df)
        if signals.empty or "symbol" not in signals.columns:
            return signals
        if "timestamp" in signals.columns and not signals["timestamp"].isna().all():
            as_of = pd.to_datetime(signals["timestamp"].max(), utc=True)
        elif "timestamp" in prices_df.columns:
            as_of = pd.to_datetime(prices_df["timestamp"].max(), utc=True)
        else:
            return signals

        valid = _pit_members_from_history(universe_history, as_of)
        before = len(signals)
        filtered = signals[signals["symbol"].str.strip().str.upper().isin(valid)]
        dropped = before - len(filtered)
        if dropped:
            logger.debug(
                "[PIT] Filtered %d signal(s) not in universe at %s",
                dropped,
                as_of.date(),
            )
        return filtered

    return _wrapped


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
            not_ended = (null_end & active_status) | (
                ~null_end & (hist["end_date"] > month_start)
            )
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

    last_dates = prices_copy.groupby("symbol")["timestamp"].max()
    days_stale_s = (as_of - last_dates).dt.days
    stale_syms = days_stale_s[days_stale_s > max_stale_days].index

    last_prices_s = (
        prices_copy[prices_copy["symbol"].isin(stale_syms)]
        .sort_values("timestamp")
        .groupby("symbol")["close"]
        .last()
        if len(stale_syms) > 0
        else pd.Series(dtype=float)
    )

    delisted = []
    terminal_values = {}
    for sym in stale_syms:
        lp = float(last_prices_s[sym])
        tp = lp * (1.0 + terminal_return)
        delisted.append(str(sym))
        terminal_values[str(sym)] = {
            "last_price": round(lp, 2),
            "last_date": str(last_dates[sym].date()),
            "days_stale": int(days_stale_s[sym]),
            "terminal_price": round(tp, 2),
        }

    return {
        "delisted": delisted,
        "n_delisted": len(delisted),
        "terminal_values": terminal_values,
    }


def select_top_adv_symbols(
    prices: pd.DataFrame,
    top_n: int,
    *,
    lookback_days: int = 20,
) -> list[str]:
    """§9.6 (a) ADV universe filter — return top-N symbols by trailing dollar-volume.

    Dollar-volume per row = close × volume; per-symbol trailing mean over the
    last ``lookback_days`` bars. Symbols are sorted descending by ADV; the
    top ``top_n`` are returned in that order.

    Use case: live paper pilot and backtest universes typically include
    illiquid names whose realized slippage costs erode strategy edge. The
    backtest sweep that motivated §9.6 (a) showed mfv2 Top-50 ADV +
    weekly = +20.33% CAGR vs all-195-daily = -7.74% CAGR — most of the
    delta was illiquidity drag on the long tail. Even for trend_baseline
    (current primary post-Phase-2), restricting to liquid names reduces
    transaction-cost noise.

    Args:
        prices: panel with timestamp, symbol, close, volume columns.
        top_n: keep this many symbols (positive int). Returns fewer when
            the input has fewer.
        lookback_days: trailing window for ADV computation. Default 20
            (~1 trading month).

    Returns:
        list of symbols sorted DESC by trailing ADV. Empty list when
        prices is empty, missing required columns, or top_n <= 0.
    """
    if prices is None or prices.empty:
        return []
    required = {"timestamp", "symbol", "close", "volume"}
    if not required.issubset(set(prices.columns)):
        return []
    if top_n <= 0:
        return []

    df = prices[["timestamp", "symbol", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["close", "volume"])
    if df.empty:
        return []

    df["dollar_volume"] = df["close"].astype(float) * df["volume"].astype(float)
    df = df.sort_values(["symbol", "timestamp"])
    # Trailing lookback per symbol — take the last `lookback_days` rows each.
    recent = df.groupby("symbol", group_keys=False).tail(lookback_days)
    adv = recent.groupby("symbol")["dollar_volume"].mean().sort_values(ascending=False)
    return adv.head(top_n).index.tolist()
