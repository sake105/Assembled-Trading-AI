"""yfinance-based OHLCV price data source.

Fetches daily (or intraday) OHLCV bars for a list of symbols via yfinance.
Used as a free fallback data source when local Parquet panels are not available.

Usage::

    from assembled_core.data.sources.yfinance_source import fetch_prices_yfinance

    df = fetch_prices_yfinance(["AAPL", "SPY"], "2024-01-01", "2024-12-31")
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_RETRY_MAX = 3
_RETRY_BACKOFF_BASE = 2.0  # seconds; doubles each retry


def _fetch_single_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
) -> pd.DataFrame | None:
    """Download OHLCV data for a single symbol with retry/backoff.

    Returns None if the symbol fails after all retries.
    """
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError:
        logger.error(
            "[ERROR] yfinance not installed. Run: pip install yfinance>=0.2.40"
        )
        return None

    last_exc: Exception | None = None
    for attempt in range(1, _RETRY_MAX + 1):
        try:
            ticker = yf.Ticker(symbol)
            raw = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                actions=False,
            )
            if raw is None or raw.empty:
                logger.warning(
                    "[WARN] yfinance: no data for %s (%s – %s)",
                    symbol,
                    start_date,
                    end_date,
                )
                return None

            raw = raw.reset_index()
            # Column may be "Date" or "Datetime" depending on interval
            date_col = "Date" if "Date" in raw.columns else "Datetime"
            raw = raw.rename(columns={date_col: "timestamp"})
            raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.normalize()
            raw["symbol"] = symbol

            cols = ["timestamp", "symbol", "Open", "High", "Low", "Close", "Volume"]
            available = [c for c in cols if c in raw.columns]
            df = raw[available].copy()
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            return df

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = _RETRY_BACKOFF_BASE**attempt
            logger.warning(
                "[WARN] yfinance: attempt %d/%d failed for %s — %s. Retrying in %.1fs.",
                attempt,
                _RETRY_MAX,
                symbol,
                exc,
                wait,
            )
            if attempt < _RETRY_MAX:
                time.sleep(wait)

    logger.error(
        "[ERROR] yfinance: all %d retries exhausted for %s — %s",
        _RETRY_MAX,
        symbol,
        last_exc,
    )
    return None


def fetch_prices_yfinance(
    symbols: list[str],
    start_date: str,
    end_date: str,
    *,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV data via yfinance. Returns empty DataFrame on failure.

    Args:
        symbols:    List of ticker symbols (e.g. ["AAPL", "SPY"]).
        start_date: Inclusive start date, "YYYY-MM-DD".
        end_date:   Inclusive end date, "YYYY-MM-DD".
        interval:   yfinance interval string. Default "1d" for daily EOD.

    Returns:
        DataFrame with columns: timestamp (UTC date), symbol, open, high,
        low, close, volume.  Empty DataFrame if nothing could be fetched.
    """
    if not symbols:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        )

    frames: list[pd.DataFrame] = []
    for sym in symbols:
        df = _fetch_single_symbol(sym, start_date, end_date, interval)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        logger.warning(
            "[WARN] yfinance: no data returned for any of %d requested symbols.",
            len(symbols),
        )
        return pd.DataFrame(
            columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        )

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    logger.info(
        "[OK] yfinance: fetched %d rows for %d symbols.",
        len(result),
        result["symbol"].nunique(),
    )
    return result
