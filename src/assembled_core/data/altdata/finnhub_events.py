"""Finnhub corporate events client — earnings and insider transactions."""

from __future__ import annotations

import hashlib
import logging
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from src.assembled_core.data.altdata.finnhub_common import (
    FINNHUB_BASE_URL,
    get_finnhub_session,
)

if TYPE_CHECKING:
    from src.assembled_core.config.settings import Settings

logger = logging.getLogger(__name__)

_EMPTY_EVENTS = pd.DataFrame(
    columns=["timestamp", "symbol", "event_type", "event_id"]
)


def _get_finnhub_session(settings) -> tuple:
    """Return (session, api_key) for Finnhub. Raises RuntimeError if key missing."""
    return get_finnhub_session(settings)


def fetch_earnings_events(
    symbols: list[str],
    start_date: date,
    end_date: date,
    settings,
) -> pd.DataFrame:
    """Fetch earnings calendar events for given symbols.

    Returns:
        DataFrame with columns: timestamp, symbol, event_type, event_id,
        eps_actual, eps_estimate, eps_surprise, eps_surprise_percent,
        revenue_actual, revenue_estimate, fiscal_period.
        Empty DataFrame on HTTP error or empty response.

    Raises:
        RuntimeError: If FINNHUB_API_KEY is not configured.
    """
    session, api_key = _get_finnhub_session(settings)

    from_str = start_date.strftime("%Y-%m-%d") if not isinstance(start_date, str) else start_date
    to_str = end_date.strftime("%Y-%m-%d") if not isinstance(end_date, str) else end_date

    url = f"{FINNHUB_BASE_URL}/calendar/earnings"
    params = {"from": from_str, "to": to_str, "token": api_key}

    try:
        response = session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logger.warning("[finnhub_events] fetch_earnings_events failed: %s", exc)
        return _EMPTY_EVENTS.copy()

    calendar = data.get("earningsCalendar", []) if isinstance(data, dict) else []
    if not calendar:
        return _EMPTY_EVENTS.copy()

    symbol_set = set(symbols) if symbols else None

    rows = []
    for item in calendar:
        sym = item.get("symbol") or ""
        if symbol_set is not None and sym not in symbol_set:
            continue

        event_date = item.get("date", "")
        ts = pd.Timestamp(event_date, tz="UTC") if event_date else pd.NaT
        event_id = hashlib.md5(
            f"earnings_{sym}_{event_date}_{item.get('fiscalPeriod','')}".encode()
        ).hexdigest()

        rows.append(
            {
                "timestamp": ts,
                "symbol": sym,
                "event_type": "earnings",
                "event_id": event_id,
                "eps_actual": item.get("epsActual"),
                "eps_estimate": item.get("epsEstimate"),
                "eps_surprise": item.get("epsSurprise"),
                "eps_surprise_percent": item.get("epsSurprisePercent"),
                "revenue_actual": item.get("revenueActual"),
                "revenue_estimate": item.get("revenueEstimate"),
                "fiscal_period": item.get("fiscalPeriod"),
            }
        )

    if not rows:
        return _EMPTY_EVENTS.copy()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_insider_events(
    symbols: list[str],
    start_date: date,
    end_date: date,
    settings,
) -> pd.DataFrame:
    """Fetch insider transaction events per symbol.

    Returns:
        DataFrame with columns: timestamp, symbol, event_type, event_id,
        insider_name, shares, transaction_code, transaction_price.
        Empty DataFrame on HTTP error or empty response.

    Raises:
        RuntimeError: If FINNHUB_API_KEY is not configured.
    """
    session, api_key = _get_finnhub_session(settings)

    from_str = start_date.strftime("%Y-%m-%d") if not isinstance(start_date, str) else start_date
    to_str = end_date.strftime("%Y-%m-%d") if not isinstance(end_date, str) else end_date

    rows = []
    for sym in symbols:
        url = f"{FINNHUB_BASE_URL}/stock/insider-transactions"
        params = {"symbol": sym, "from": from_str, "to": to_str, "token": api_key}

        try:
            response = session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.warning(
                "[finnhub_events] fetch_insider_events(%s) failed: %s", sym, exc
            )
            continue

        txns = data.get("data", []) if isinstance(data, dict) else []
        for item in txns:
            tx_date = item.get("transactionDate") or item.get("filingDate", "")
            ts = pd.Timestamp(tx_date, tz="UTC") if tx_date else pd.NaT
            tx_code = item.get("transactionCode", "")
            if tx_code == "P":
                event_type = "insider_purchase"
            elif tx_code == "S":
                event_type = "insider_sale"
            else:
                event_type = f"insider_{tx_code.lower()}" if tx_code else "insider_other"

            event_id = hashlib.md5(
                f"insider_{sym}_{tx_date}_{item.get('name','')}_{item.get('share','')}".encode()
            ).hexdigest()

            rows.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "event_type": event_type,
                    "event_id": event_id,
                    "insider_name": item.get("name"),
                    "shares": item.get("share"),
                    "transaction_code": tx_code,
                    "transaction_price": item.get("transactionPrice"),
                }
            )

    if not rows:
        return _EMPTY_EVENTS.copy()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)
