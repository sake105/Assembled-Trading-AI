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

from src.assembled_core.data.feed_status import (
    FEED_EMPTY,
    FEED_ERROR,
    FEED_OK,
    stamp_feed_status,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_PRICE_COLS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]

_RETRY_MAX = 3
_RETRY_BACKOFF_BASE = 2.0  # seconds; doubles each retry


class YFinanceRateLimitError(Exception):
    """Raised when yfinance returns HTTP 429. Caller should try an alternative source."""


def _fetch_single_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
) -> pd.DataFrame | None:
    """Download OHLCV data for a single symbol with retry/backoff.

    DAT-005 outage-vs-empty distinction: returns an **empty DataFrame** when the
    symbol legitimately has no bars in the window, and ``None`` only on a genuine
    **error** (yfinance missing, or all retries exhausted). The caller relies on
    this to tag a total outage differently from an empty window. A 429 still
    raises :class:`YFinanceRateLimitError`.
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
                # DAT-005: legitimate empty window -> empty frame (NOT None/error).
                return pd.DataFrame(columns=_PRICE_COLS)

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
            exc_str = str(exc).lower()
            if "429" in exc_str or "too many requests" in exc_str:
                raise YFinanceRateLimitError(
                    f"yfinance rate-limited (HTTP 429) for {symbol}: {exc}"
                ) from exc
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
        # Caller asked for nothing — an empty window, not an outage. No
        # protocol either: zero requests leave nothing to account for.
        return stamp_feed_status(
            pd.DataFrame(columns=_PRICE_COLS),
            "yfinance",
            FEED_EMPTY,
            reason="no_symbols_requested",
        )

    # E-112: yfinance is the only LIVE price path — it must leave a request
    # protocol like every other ingest (§0.06c closed 2026-08-16). PullLog is
    # non-raising by contract; the write sits in `finally` so the protocol
    # survives every exit, including the rate-limit abort (E-147).
    from src.assembled_core.data.pull_log import PullLog

    plog = PullLog(source="yfinance")
    window = (start_date, end_date)

    frames: list[pd.DataFrame] = []
    any_error = False  # DAT-005: a None from _fetch_single_symbol is an outage
    try:
        for _i, sym in enumerate(symbols):
            # YFinanceRateLimitError propagates: caller should try an
            # alternative source. Record before re-raising.
            try:
                df = _fetch_single_symbol(sym, start_date, end_date, interval)
            except YFinanceRateLimitError:
                plog.record(sym, window=window, http_status=429, error="rate_limited")
                # F-senior-1 (Stage 2, 2026-08-16): die NICHT mehr angefragten
                # Symbole als skipped protokollieren — sonst kuerzt der Abbruch
                # den Nenner jeder Fehlerquote, und je spaeter (schlimmer) der
                # Ausfall, desto stiller wird er (E-158-Klasse). Laufindex statt
                # symbols.index(sym): bei Duplikaten wuerde .index() bereits
                # geholte Symbole als skipped nachbuchen (F-auditor-4).
                from src.assembled_core.data.pull_log import STATUS_SKIPPED

                for rest in symbols[_i + 1 :]:
                    plog.record(
                        rest,
                        window=window,
                        status=STATUS_SKIPPED,
                        error="not_requested_after_rate_limit_abort",
                    )
                raise
            # None == this symbol errored (outage); empty df == no bars in
            # window (legitimate). Track so a total outage is distinguishable
            # from "no data".
            if df is None:
                any_error = True
                plog.record(sym, window=window, error="all_retries_exhausted")
            elif not df.empty:
                frames.append(df)
                plog.record(sym, window=window, n_rows=len(df))
            else:
                plog.record(sym, window=window, n_rows=0)
    finally:
        plog.write()

    if not frames:
        logger.warning(
            "[WARN] yfinance: no data returned for any of %d requested symbols.",
            len(symbols),
        )
        status = FEED_ERROR if any_error else FEED_EMPTY
        reason = "all_symbols_errored" if any_error else "no_rows_in_window"
        return stamp_feed_status(
            pd.DataFrame(columns=_PRICE_COLS), "yfinance", status, reason=reason
        )

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    logger.info(
        "[OK] yfinance: fetched %d rows for %d symbols.",
        len(result),
        result["symbol"].nunique(),
    )
    # Some symbols may have errored even though others returned bars (partial
    # outage) — record it on the OK stamp's reason so it stays observable.
    return stamp_feed_status(
        result,
        "yfinance",
        FEED_OK,
        reason="partial_outage" if any_error else None,
    )
