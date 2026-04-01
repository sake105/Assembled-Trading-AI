"""Alpha Vantage market data source.

Fetches daily adjusted price data and fundamentals via Alpha Vantage.

Requires environment variable::

    ALPHAVANTAGE_KEY=<your key>

Free tier: 25 requests/day, 5 requests/minute.
Get a free key at https://www.alphavantage.co/support/#api-key

Note: yfinance covers the same price data without rate limits.
Alpha Vantage is most useful as a fallback or for its fundamentals endpoint.

Usage::

    from assembled_core.data.sources.alphavantage_source import fetch_prices_alphavantage

    df = fetch_prices_alphavantage(["AAPL", "MSFT"], "2024-01-01", "2024-01-31")
"""

from __future__ import annotations

import logging
import os
import time

import pandas as pd

logger = logging.getLogger(__name__)

_EMPTY = pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
_BASE_URL = "https://www.alphavantage.co/query"
_RATE_LIMIT_SLEEP = 12.5  # seconds between requests (5/min free tier)


def _get_api_key() -> str | None:
    key = os.environ.get("ALPHAVANTAGE_KEY", "").strip()
    return key if key else None


def fetch_prices_alphavantage(
    symbols: list[str],
    start_date: str,
    end_date: str,
    *,
    outputsize: str = "compact",
) -> pd.DataFrame:
    """Fetch daily adjusted OHLCV prices from Alpha Vantage.

    Args:
        symbols:    List of ticker symbols, e.g. ["SPY", "QQQ"].
        start_date: Inclusive start date, "YYYY-MM-DD".
        end_date:   Inclusive end date, "YYYY-MM-DD".
        outputsize: "compact" (last 100 days) or "full" (20+ years). Default: "full".

    Returns:
        DataFrame with columns: timestamp (UTC), symbol, open, high, low, close, volume.
        Empty DataFrame if key missing or all fetches fail.

    Note:
        Free tier allows 25 requests/day and 5/min. A 12.5s sleep is inserted between
        requests to respect the rate limit.
    """
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        logger.error("[ERROR] requests not installed.")
        return _EMPTY.copy()

    if not symbols:
        return _EMPTY.copy()

    api_key = _get_api_key()
    if api_key is None:
        logger.warning("[WARN] alphavantage: ALPHAVANTAGE_KEY not set — returning empty DataFrame.")
        return _EMPTY.copy()

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    frames: list[pd.DataFrame] = []

    for i, symbol in enumerate(symbols):
        if i > 0:
            time.sleep(_RATE_LIMIT_SLEEP)

        try:
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": outputsize,
                "apikey": api_key,
                "datatype": "json",
            }
            resp = requests.get(_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if "Note" in data:
                logger.warning("[WARN] alphavantage: rate limit hit for %s — %s", symbol, data["Note"])
                continue
            if "Information" in data:
                logger.warning("[WARN] alphavantage: API message for %s — %s", symbol, data["Information"])
                continue

            ts_data = data.get("Time Series (Daily)") or {}
            if not ts_data:
                logger.warning("[WARN] alphavantage: no price data for %s", symbol)
                continue

            rows = []
            for date_str, ohlcv in ts_data.items():
                ts = pd.Timestamp(date_str, tz="UTC")
                if ts < start_ts.tz_localize("UTC") or ts > end_ts.tz_localize("UTC"):
                    continue
                rows.append({
                    "timestamp": ts,
                    "symbol": symbol,
                    "open": float(ohlcv.get("1. open", 0)),
                    "high": float(ohlcv.get("2. high", 0)),
                    "low": float(ohlcv.get("3. low", 0)),
                    "close": float(ohlcv.get("4. close", 0)),
                    "volume": float(ohlcv.get("5. volume", 0)),
                })

            if rows:
                frames.append(pd.DataFrame(rows))
                logger.debug("[OK] alphavantage: %d rows for %s", len(rows), symbol)
            else:
                logger.warning("[WARN] alphavantage: no data in date range for %s", symbol)

        except Exception as exc:
            logger.error("[ERROR] alphavantage: failed to fetch %s — %s", symbol, exc)

    if not frames:
        logger.warning("[WARN] alphavantage: no data returned for any of %d symbols.", len(symbols))
        return _EMPTY.copy()

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    logger.info("[OK] alphavantage: fetched %d rows for %d symbols.", len(result), result["symbol"].nunique())
    return result
