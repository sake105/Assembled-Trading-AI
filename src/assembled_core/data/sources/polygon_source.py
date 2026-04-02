"""Polygon.io OHLCV price data source.

Fetches daily aggregate bars for a list of symbols via the Polygon.io REST API.
The free tier allows 5 requests/minute; this module enforces a 12-second delay
between per-symbol requests to stay within that limit.

Requires environment variable::

    POLYGON_API_KEY=<your key>

Get a free key at https://polygon.io/

Usage::

    from assembled_core.data.sources.polygon_source import fetch_prices_polygon

    df = fetch_prices_polygon(["AAPL", "SPY"], "2024-01-01", "2024-12-31")
"""

from __future__ import annotations

import logging
import os
import time

import pandas as pd

logger = logging.getLogger(__name__)

# Free tier: 5 requests / minute → 12 s between requests is safe
_FREE_TIER_DELAY_S = 12.0
_RESULTS_LIMIT = 50000  # max bars per request for Polygon aggs endpoint


def _get_api_key() -> str | None:
    """Return POLYGON_API_KEY from environment, or None if not set."""
    key = os.environ.get("POLYGON_API_KEY", "").strip()
    return key if key else None


def _fetch_single_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    timespan: str,
    api_key: str,
) -> pd.DataFrame | None:
    """Fetch aggregate bars for one symbol from Polygon.io.

    Returns None on any error or if no data is available.
    """
    try:
        from polygon import RESTClient  # noqa: PLC0415
    except ImportError:
        logger.error(
            "[ERROR] polygon-api-client not installed. Run: pip install polygon-api-client>=1.12.0"
        )
        return None

    try:
        client = RESTClient(api_key=api_key)
        aggs = list(
            client.list_aggs(
                ticker=symbol,
                multiplier=1,
                timespan=timespan,
                from_=start_date,
                to=end_date,
                limit=_RESULTS_LIMIT,
            )
        )

        if not aggs:
            logger.warning(
                "[WARN] polygon: no data for %s (%s – %s)", symbol, start_date, end_date
            )
            return None

        rows = []
        for bar in aggs:
            # bar.timestamp is milliseconds since epoch (UTC)
            ts = pd.Timestamp(bar.timestamp, unit="ms", tz="UTC").normalize()
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
            )

        df = pd.DataFrame(rows)
        return df

    except Exception as exc:  # noqa: BLE001
        logger.error("[ERROR] polygon: failed to fetch %s — %s", symbol, exc)
        return None


def fetch_prices_polygon(
    symbols: list[str],
    start_date: str,
    end_date: str,
    *,
    timespan: str = "day",
) -> pd.DataFrame:
    """Fetch OHLCV data via Polygon.io free API. Returns empty DataFrame on failure.

    Rate-limiting: 12-second sleep between per-symbol requests to respect the
    free-tier limit of 5 requests/minute.

    Args:
        symbols:    List of ticker symbols (e.g. ["AAPL", "SPY"]).
        start_date: Inclusive start date, "YYYY-MM-DD".
        end_date:   Inclusive end date, "YYYY-MM-DD".
        timespan:   Polygon timespan ("day", "hour", "minute"). Default "day".

    Returns:
        DataFrame with columns: timestamp (UTC date), symbol, open, high,
        low, close, volume.  Empty DataFrame if key missing or nothing fetched.
    """
    _empty = pd.DataFrame(
        columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    )

    if not symbols:
        return _empty

    api_key = _get_api_key()
    if api_key is None:
        logger.warning("[WARN] polygon: POLYGON_API_KEY not set — skipping fetch.")
        return _empty

    frames: list[pd.DataFrame] = []
    for idx, sym in enumerate(symbols):
        if idx > 0:
            # Respect free-tier rate limit between requests
            time.sleep(_FREE_TIER_DELAY_S)

        df = _fetch_single_symbol(sym, start_date, end_date, timespan, api_key)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        logger.warning(
            "[WARN] polygon: no data returned for any of %d requested symbols.",
            len(symbols),
        )
        return _empty

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    logger.info(
        "[OK] polygon: fetched %d rows for %d symbols.",
        len(result),
        result["symbol"].nunique(),
    )
    return result
