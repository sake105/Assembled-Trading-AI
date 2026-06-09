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

# Observability for silently-missing alt-data caches (theme: "silent degradation
# logged at DEBUG"). A permanently-missing / never-written cache file yields an
# empty frame and therefore zeroed downstream factors. At DEBUG that is invisible
# — a broken/never-built cache is indistinguishable from legitimately-absent
# data. The one-shot guard below surfaces the FIRST missing-cache per cache type
# at WARNING; subsequent misses stay at DEBUG to avoid per-call spam. Behaviour
# is unchanged: the schema-correct empty frame is still returned by the caller.
_MISSING_CACHE_WARNED: set[str] = set()


def _warn_missing_cache(cache_type: str, fpath: Path) -> None:
    """Surface the first missing alt-data cache of ``cache_type`` at WARNING.

    Observability-only: callers must still return their schema-correct empty
    frame. This does not change loader behaviour, only makes a permanently
    absent cache visible once in default-level logs instead of silent at DEBUG.
    """
    if cache_type not in _MISSING_CACHE_WARNED:
        _MISSING_CACHE_WARNED.add(cache_type)
        logger.warning(
            "[altdata] %s cache missing: %s — factor(s) degrade to empty/0.0. "
            "Further misses of this cache suppressed to DEBUG.",
            cache_type,
            fpath,
        )
    else:
        logger.debug("[altdata] %s cache still missing: %s", cache_type, fpath)


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
        _warn_missing_cache("earnings", fpath)
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

    # Batch-12 PIT fix (Diagnostik §data MINOR-(b), altdata_loader.py:60):
    # an event_date-only feed (no real disclosure_date) bypasses disclosure-PIT —
    # the raw event_date would be used directly as the availability date in the
    # as_of cutoff below. Production parquet carries disclosure_date, so this is
    # contingent; when it fires, apply a conservative 1-calendar-day disclosure
    # lag (earnings land after-close/pre-market → next-bar availability) so the
    # event only becomes visible at event_date + 1. When a real disclosure_date
    # column is present, behaviour is unchanged (no shift).
    if date_col == "event_date":
        df[date_col] = df[date_col] + pd.Timedelta(days=1)

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

    Returns a DataFrame carrying the columns the live insider factor requires:
    ``[symbol, filing_date, transaction_type, value_usd, shares_delta]``. These
    are present even on the empty/degraded path so that
    ``earnings_insider_wrapper.compute_earnings_insider_factors`` — which
    validates them UNCONDITIONALLY — never raises.

    Source preference: the EDGAR Form 4 ingester output
    ``output/insider_form4.parquet`` (real ``transaction_type`` ∈ {P,S,unknown}
    and gross ``value_usd``), falling back to the legacy
    ``output/insider_trading.parquet`` for back-compat (legacy lacks
    ``value_usd`` → synthesized as NaN; its ``transaction_type`` is 'unknown' for
    all rows, so the factor stays degraded until the Form 4 feed is generated).
    """
    required_cols = [
        "symbol",
        "filing_date",
        "transaction_type",
        "value_usd",
        "shares_delta",
    ]
    empty = pd.DataFrame(columns=required_cols)

    fpath = _resolve(root, "insider_form4.parquet")
    if not fpath.exists():
        legacy = _resolve(root, "insider_trading.parquet")
        if legacy.exists():
            logger.debug(
                "[altdata] insider_form4.parquet absent — using legacy %s", legacy
            )
            fpath = legacy
        else:
            _warn_missing_cache("insider", fpath)
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

    # shares_delta = signed net change. Prefer the signed ``net_shares`` from the
    # Form 4 feed; fall back to a raw ``shares`` column for the legacy file.
    if "shares_delta" not in df.columns:
        if "net_shares" in df.columns:
            df["shares_delta"] = df["net_shares"]
        elif "shares" in df.columns:
            df["shares_delta"] = df["shares"]

    # Guarantee the wrapper-required columns exist even on the legacy path.
    if "transaction_type" not in df.columns:
        df["transaction_type"] = "unknown"
    if "value_usd" not in df.columns:
        df["value_usd"] = pd.NA

    keep = [c for c in required_cols if c in df.columns]
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
        _warn_missing_cache("news_sentiment", fpath)
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
    release_lag_days: int = 32,
) -> pd.DataFrame:
    """Load macro indicators in long format, PIT-safe.

    The output/macro.parquet is wide-format; this function melts it to long:
    columns: [timestamp, macro_code, value, country].

    Batch-12 PIT fix (Diagnostik §data "Macro-Loader ohne Release-Lag", §294):
    macro indicators are stamped at their observation date, but the public
    release happens later (month-T values during month T+1). Previously the raw
    observation date was used directly as the availability date in the ``as_of``
    cutoff, leaking future macro data into a backtest bar. ``release_lag_days``
    (default 32, mirroring ``merge_gpr_index_into_panel``) is applied as a
    SPLIT-BOUND availability delay: it shifts only the *upper* (``as_of``) bound
    so a value is visible only once it would realistically have been published,
    while the *lower* (lookback) bound keeps comparing the RAW observation date.
    This is a PIT correction to a LIVE non-zero-weighted macro factor
    (multifactor_v2 macro_growth_momentum / macro_inflation_surprise): it removes
    look-ahead and therefore DOES change the macro z-score input — it is NOT
    production-invariant. The returned ``timestamp`` stays the RAW observation
    date so downstream alignment (which applies its own availability lag) is
    unchanged. Pass ``release_lag_days=0`` only for parity tests / research that
    knowingly use raw observation dates; with ``release_lag_days=0`` the filter
    reproduces the legacy raw-observation behaviour byte-for-byte.
    """
    empty = pd.DataFrame(columns=["timestamp", "macro_code", "value", "country"])
    fpath = _resolve(root, "macro.parquet")
    if not fpath.exists():
        _warn_missing_cache("macro", fpath)
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

    # Batch-12 PIT fix (SPLIT-BOUND): the publication lag is an availability
    # delay that applies ONLY to the upper (as_of) bound. The lower (lookback)
    # bound compares the RAW observation date, so the lookback window selects the
    # same raw observations as before the lag — the lag can only hide recent
    # unreleased obs, never pull older history into the window. We compute a
    # local availability series and never mutate df["timestamp"], so the returned
    # schema keeps the RAW observation date (downstream applies its own
    # availability lag during merge). With release_lag_days=0 the availability
    # series equals the raw timestamp and the filter is byte-identical to legacy.
    raw_ts = df["timestamp"]
    available = (
        raw_ts + pd.Timedelta(days=release_lag_days) if release_lag_days else raw_ts
    )
    df = df[(available <= as_of_naive) & (raw_ts >= cutoff)].copy()

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
