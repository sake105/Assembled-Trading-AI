"""Shared loader for cached alt-data (earnings, insider, news, macro).

Reads from output/ parquet files if present.
Returns schema-correct empty DataFrames if files are missing — callers
already handle empty frames gracefully, so degradation is silent.

All functions are PIT-safe: data is filtered to as_of <= cutoff.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_OUTPUT_ROOT = Path("output")

# Column aliases: what the output parquets contain → what wrappers expect
_EARNINGS_COL_MAP = {
    "disclosure_date": "filing_date",
    "eps_surprise_pct": "surprise_pct",
}
_INSIDER_COL_MAP = {
    "shares": "shares_delta",
}


def _resolve(root: Path | str | None, filename: str) -> Path:
    base = Path(root) if root else _OUTPUT_ROOT
    return base / filename


def load_earnings_history(
    symbols: list[str],
    as_of: pd.Timestamp,
    lookback_days: int = 90,
    *,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """Load earnings events for symbols, PIT-safe.

    Returns DataFrame with columns: [symbol, filing_date, surprise_pct].
    """
    empty = pd.DataFrame(columns=["symbol", "filing_date", "surprise_pct"])
    fpath = _resolve(root, "events_earnings.parquet")
    if not fpath.exists():
        logger.debug("[altdata] earnings file not found: %s", fpath)
        return empty

    try:
        df = pd.read_parquet(fpath)
    except Exception as exc:
        logger.warning("[altdata] cannot read earnings: %s", exc)
        return empty

    # Normalise timezone-aware timestamps
    date_col = "disclosure_date" if "disclosure_date" in df.columns else "event_date"
    if date_col not in df.columns:
        logger.warning("[altdata] earnings: no usable date column")
        return empty

    df[date_col] = pd.to_datetime(
        df[date_col], utc=True, errors="coerce"
    ).dt.tz_localize(None)
    as_of_naive = as_of.tz_localize(None) if as_of.tzinfo else as_of
    cutoff = as_of_naive - pd.Timedelta(days=lookback_days)

    mask = (df[date_col] <= as_of_naive) & (df[date_col] >= cutoff)
    if symbols:
        mask &= df["symbol"].isin(symbols)
    df = df.loc[mask].copy()

    # Map columns to expected schema
    if "disclosure_date" in df.columns:
        df = df.rename(columns={"disclosure_date": "filing_date"})
    elif "event_date" in df.columns:
        df = df.rename(columns={"event_date": "filing_date"})

    if "eps_surprise_pct" in df.columns:
        df = df.rename(columns={"eps_surprise_pct": "surprise_pct"})

    keep = [c for c in ["symbol", "filing_date", "surprise_pct"] if c in df.columns]
    df = df[keep].dropna(subset=["symbol", "filing_date"])
    logger.debug(
        "[altdata] earnings loaded: %d rows for %d symbols", len(df), len(symbols)
    )
    return df.reset_index(drop=True)


def load_insider_filings(
    symbols: list[str],
    as_of: pd.Timestamp,
    lookback_days: int = 90,
    *,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """Load insider filings for symbols, PIT-safe.

    Returns DataFrame with columns: [symbol, filing_date, shares_delta].
    Note: transaction_type is currently 'unknown' for all rows (data quality issue).
    """
    empty = pd.DataFrame(columns=["symbol", "filing_date", "shares_delta"])
    fpath = _resolve(root, "insider_trading.parquet")
    if not fpath.exists():
        logger.debug("[altdata] insider file not found: %s", fpath)
        return empty

    try:
        df = pd.read_parquet(fpath)
    except Exception as exc:
        logger.warning("[altdata] cannot read insider: %s", exc)
        return empty

    if "filing_date" not in df.columns:
        logger.warning("[altdata] insider: no filing_date column")
        return empty

    df["filing_date"] = pd.to_datetime(
        df["filing_date"], utc=True, errors="coerce"
    ).dt.tz_localize(None)
    as_of_naive = as_of.tz_localize(None) if as_of.tzinfo else as_of
    cutoff = as_of_naive - pd.Timedelta(days=lookback_days)

    mask = (df["filing_date"] <= as_of_naive) & (df["filing_date"] >= cutoff)
    if symbols:
        mask &= df["symbol"].isin(symbols)
    df = df.loc[mask].copy()

    if "shares" in df.columns and "shares_delta" not in df.columns:
        df = df.rename(columns={"shares": "shares_delta"})

    keep = [c for c in ["symbol", "filing_date", "shares_delta"] if c in df.columns]
    df = df[keep].dropna(subset=["symbol", "filing_date"])
    logger.debug(
        "[altdata] insider loaded: %d rows for %d symbols", len(df), len(symbols)
    )
    return df.reset_index(drop=True)


def load_news_sentiment(
    symbols: list[str],
    as_of: pd.Timestamp,
    lookback_days: int = 30,
    *,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """Load news sentiment events for symbols, PIT-safe.

    Returns DataFrame with columns: [symbol, timestamp, sentiment_score].
    """
    empty = pd.DataFrame(columns=["symbol", "timestamp", "sentiment_score"])
    fpath = _resolve(root, "news_sentiment_daily.parquet")
    if not fpath.exists():
        logger.debug("[altdata] news_sentiment file not found: %s", fpath)
        return empty

    try:
        df = pd.read_parquet(fpath)
    except Exception as exc:
        logger.warning("[altdata] cannot read news_sentiment: %s", exc)
        return empty

    if "timestamp" not in df.columns:
        logger.warning("[altdata] news_sentiment: no timestamp column")
        return empty

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    ).dt.tz_localize(None)
    as_of_naive = as_of.tz_localize(None) if as_of.tzinfo else as_of
    cutoff = as_of_naive - pd.Timedelta(days=lookback_days)

    mask = (df["timestamp"] <= as_of_naive) & (df["timestamp"] >= cutoff)
    if symbols:
        mask &= df["symbol"].isin(symbols)
    df = df.loc[mask].copy()

    keep = [c for c in ["symbol", "timestamp", "sentiment_score"] if c in df.columns]
    df = df[keep].dropna(subset=["symbol", "timestamp"])
    logger.debug(
        "[altdata] news_sentiment loaded: %d rows for %d symbols", len(df), len(symbols)
    )
    return df.reset_index(drop=True)


def load_macro_indicators(
    as_of: pd.Timestamp,
    lookback_days: int = 365,
    *,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """Load macro indicators in long format, PIT-safe.

    The output/macro.parquet is wide-format; this function melts it to long:
    columns: [timestamp, macro_code, value, country].
    """
    empty = pd.DataFrame(columns=["timestamp", "macro_code", "value", "country"])
    fpath = _resolve(root, "macro.parquet")
    if not fpath.exists():
        logger.debug("[altdata] macro file not found: %s", fpath)
        return empty

    try:
        df = pd.read_parquet(fpath)
    except Exception as exc:
        logger.warning("[altdata] cannot read macro: %s", exc)
        return empty

    if "timestamp" not in df.columns:
        logger.warning("[altdata] macro: no timestamp column")
        return empty

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    ).dt.tz_localize(None)
    as_of_naive = as_of.tz_localize(None) if as_of.tzinfo else as_of
    cutoff = as_of_naive - pd.Timedelta(days=lookback_days)

    df = df[(df["timestamp"] <= as_of_naive) & (df["timestamp"] >= cutoff)].copy()

    # Melt wide → long
    value_cols = [c for c in df.columns if c != "timestamp"]
    if not value_cols:
        return empty

    long = df.melt(
        id_vars=["timestamp"],
        value_vars=value_cols,
        var_name="macro_code",
        value_name="value",
    )
    long = long.dropna(subset=["value"])
    long["country"] = "US"  # all indicators are US-based

    logger.debug(
        "[altdata] macro loaded: %d rows, %d indicators",
        len(long),
        long["macro_code"].nunique(),
    )
    return long[["timestamp", "macro_code", "value", "country"]].reset_index(drop=True)


__all__ = [
    "load_earnings_history",
    "load_insider_filings",
    "load_news_sentiment",
    "load_macro_indicators",
]
