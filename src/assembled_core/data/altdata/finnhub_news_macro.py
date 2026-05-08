"""Finnhub news and macro data client."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
from src.assembled_core.data.altdata.finnhub_common import (
    FINNHUB_BASE_URL,
    get_finnhub_session,
)

logger = logging.getLogger(__name__)


def _get_finnhub_session(settings) -> tuple:
    """Return (session, api_key) for Finnhub. Raises RuntimeError if key missing."""
    return get_finnhub_session(settings)


def fetch_news(
    symbols: list[str] | None,
    start_date: date,
    end_date: date,
    settings,
) -> pd.DataFrame:
    """Fetch company or market news from Finnhub.

    If symbols is None or empty, fetches general market news.

    Returns:
        DataFrame with columns: timestamp, symbol, headline, news_id, event_type,
        source, url, summary.
        Empty DataFrame on HTTP error.

    Raises:
        RuntimeError: If FINNHUB_API_KEY is not configured.
    """
    session, api_key = _get_finnhub_session(settings)

    from_str = (
        start_date.strftime("%Y-%m-%d")
        if not isinstance(start_date, str)
        else start_date
    )
    to_str = (
        end_date.strftime("%Y-%m-%d") if not isinstance(end_date, str) else end_date
    )

    rows = []

    if symbols:
        for sym in symbols:
            url = f"{FINNHUB_BASE_URL}/company-news"
            params = {"symbol": sym, "from": from_str, "to": to_str, "token": api_key}
            try:
                response = session.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                items = response.json()
            except Exception as exc:
                logger.warning(
                    "[finnhub_news_macro] fetch_news(%s) failed: %s", sym, exc
                )
                continue
            rows.extend(_parse_news_items(items, default_symbol=sym))
    else:
        url = f"{FINNHUB_BASE_URL}/news"
        params = {"category": "general", "token": api_key}
        try:
            response = session.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            items = response.json()
        except Exception as exc:
            logger.warning("[finnhub_news_macro] fetch_news(market) failed: %s", exc)
            return pd.DataFrame(
                columns=["timestamp", "symbol", "headline", "news_id", "event_type"]
            )
        rows.extend(_parse_news_items(items, default_symbol=None))

    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "headline", "news_id", "event_type"]
        )

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _parse_news_items(items: list, default_symbol: str | None) -> list[dict]:
    rows = []
    if not isinstance(items, list):
        return rows
    for item in items:
        dt_ms = item.get("datetime")
        if dt_ms:
            ts = pd.Timestamp(dt_ms, unit="ms", tz="UTC")
        else:
            ts = pd.NaT
        symbol = item.get("related") or default_symbol or ""
        rows.append(
            {
                "timestamp": ts,
                "symbol": symbol,
                "headline": item.get("headline", ""),
                "news_id": str(item.get("id", "")),
                "event_type": "news",
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "summary": item.get("summary", ""),
            }
        )
    return rows


def fetch_news_sentiment(
    symbols: list[str],
    start_date: date,
    end_date: date,
    settings,
) -> pd.DataFrame:
    """Compute daily news sentiment by aggregating news counts.

    Returns:
        DataFrame with columns: timestamp, symbol, sentiment_score, sentiment_volume.
        Empty DataFrame if no news found.

    Raises:
        RuntimeError: If FINNHUB_API_KEY is not configured.
    """
    news_df = fetch_news(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        settings=settings,
    )

    if news_df.empty:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "sentiment_score", "sentiment_volume"]
        )

    news_df["_date"] = news_df["timestamp"].dt.normalize()
    group_cols = ["_date", "symbol"] if "symbol" in news_df.columns else ["_date"]

    agg = news_df.groupby(group_cols).size().reset_index(name="sentiment_volume")
    agg["sentiment_score"] = 0.0
    agg = agg.rename(columns={"_date": "timestamp"})

    return agg.reset_index(drop=True)


def fetch_macro_series(
    codes: list[str],
    start_date: date,
    end_date: date,
    settings,
) -> pd.DataFrame:
    """Fetch macro / economic calendar data from Finnhub.

    Args:
        codes: List of macro event codes (e.g. ['CPI', 'GDP']).

    Returns:
        DataFrame with columns: timestamp, macro_code, value, country.
        Empty DataFrame on HTTP error.

    Raises:
        RuntimeError: If FINNHUB_API_KEY is not configured.
    """
    session, api_key = _get_finnhub_session(settings)

    from_str = (
        start_date.strftime("%Y-%m-%d")
        if not isinstance(start_date, str)
        else start_date
    )
    to_str = (
        end_date.strftime("%Y-%m-%d") if not isinstance(end_date, str) else end_date
    )

    url = f"{FINNHUB_BASE_URL}/calendar/economic"
    params = {"from": from_str, "to": to_str, "token": api_key}

    try:
        response = session.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logger.warning("[finnhub_news_macro] fetch_macro_series failed: %s", exc)
        return pd.DataFrame(columns=["timestamp", "macro_code", "value", "country"])

    # Handle both list (economic calendar) and dict (economic indicator) responses
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "data" in data:
        items = data["data"]
    else:
        items = []

    if not items:
        return pd.DataFrame(columns=["timestamp", "macro_code", "value", "country"])

    rows = []
    code_set = set(codes) if codes else None

    for item in items:
        # Economic calendar format
        if "time" in item or "event" in item:
            event_code = item.get("event", "")
            if code_set and event_code not in code_set:
                continue
            ts_raw = item.get("time", "")
            ts = pd.Timestamp(ts_raw, tz="UTC") if ts_raw else pd.NaT
            rows.append(
                {
                    "timestamp": ts,
                    "macro_code": event_code,
                    "value": item.get("actual"),
                    "estimate": item.get("estimate"),
                    "impact": item.get("impact"),
                    "country": item.get("country", ""),
                }
            )
        # Economic indicator format
        elif "date" in item:
            for code in codes or [""]:
                ts_raw = item.get("date", "")
                ts = pd.Timestamp(ts_raw, tz="UTC") if ts_raw else pd.NaT
                rows.append(
                    {
                        "timestamp": ts,
                        "macro_code": code,
                        "value": item.get("value"),
                        "country": "",
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["timestamp", "macro_code", "value", "country"])

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


# Keep backward-compatible stub functions
def fetch_finnhub_news(symbols: list[str] | None = None) -> pd.DataFrame:
    """Fetch news from Finnhub API (legacy stub)."""
    return pd.DataFrame(columns=["timestamp", "symbol", "headline", "sentiment"])


def fetch_finnhub_macro() -> pd.DataFrame:
    """Fetch macro data from Finnhub (legacy stub)."""
    return pd.DataFrame(columns=["timestamp", "indicator", "value"])
