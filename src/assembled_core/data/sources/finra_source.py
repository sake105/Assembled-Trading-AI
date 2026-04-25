"""FINRA Short-Interest data source — free, no API key required.

From 10_FREE_DATEN.md §10.3.

Endpoints:
  - regShoDaily (daily, most reactive)
  - EquityShortInterest (bi-monthly, more stable)

Update cadence: bi-monthly with T+8 business-day lag (EquityShortInterest)
                daily for regShoDaily.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.finra.org/data/group/otcMarket/name"
_TIMEOUT = 30
_HEADERS = {"Content-Type": "application/json"}


def _post(endpoint: str, body: dict) -> list[dict]:
    url = f"{_BASE_URL}/{endpoint}"
    try:
        r = requests.post(url, json=body, headers=_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json() if r.text else []
    except Exception as exc:
        logger.warning("FINRA %s failed: %s", endpoint, exc)
        return []


def get_short_interest(
    ticker: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Fetch bi-monthly short interest for a ticker.

    Returns list of records with keys:
      symbolCode, shortInterestQty, shortInterestSharesPct, avgDailyVol, daysToClose, settlementDate
    """
    body = {
        "compareFilters": [{"fieldName": "symbolCode", "compareType": "equal", "fieldValue": ticker}],
        "limit": limit,
        "offset": 0,
        "fields": ["symbolCode", "shortInterestQty", "shortInterestSharesPct", "avgDailyVol", "daysToClose", "settlementDate"],
        "domainFilters": [],
        "sortFields": [{"fieldName": "settlementDate", "sortOrder": "DESC"}],
    }
    return _post("EquityShortInterest", body)


def get_reg_sho_daily(
    ticker: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Fetch daily Reg-SHO threshold list data.

    More granular and reactive than bi-monthly EquityShortInterest.
    """
    body = {
        "compareFilters": [{"fieldName": "issueSymbolIdentifier", "compareType": "equal", "fieldValue": ticker}],
        "limit": limit,
        "offset": 0,
        "sortFields": [{"fieldName": "tradeReportDate", "sortOrder": "DESC"}],
    }
    return _post("regShoDaily", body)


def short_interest_features(ticker: str) -> dict[str, float]:
    """Compute short-interest features for signal construction.

    Returns dict with:
      days_to_cover: Short interest / avg daily volume
      si_pct_float: Short interest as % of float (proxy)
      si_change_pct: Period-over-period change (if 2+ records available)

    From 13_FREE_MODULE.md §13.11.
    """
    records = get_short_interest(ticker, limit=2)
    if not records:
        return {}

    latest = records[0]
    si_qty = float(latest.get("shortInterestQty", 0) or 0)
    si_pct = float(latest.get("shortInterestSharesPct", 0) or 0)
    avg_vol = float(latest.get("avgDailyVol", 1) or 1)
    days_to_cover = si_qty / max(avg_vol, 1)

    result: dict[str, float] = {
        "si_qty": si_qty,
        "si_pct_float": si_pct,
        "days_to_cover": days_to_cover,
    }

    if len(records) >= 2:
        prior_qty = float(records[1].get("shortInterestQty", 0) or 0)
        if prior_qty > 0:
            result["si_change_pct"] = (si_qty - prior_qty) / prior_qty
        else:
            result["si_change_pct"] = 0.0

    return result


def batch_short_interest_features(tickers: list[str]) -> pd.DataFrame:
    """Fetch short-interest features for multiple tickers.

    Returns DataFrame indexed by ticker.
    """
    rows = {}
    for ticker in tickers:
        feats = short_interest_features(ticker)
        if feats:
            rows[ticker] = feats
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).T


__all__ = [
    "get_short_interest",
    "get_reg_sho_daily",
    "short_interest_features",
    "batch_short_interest_features",
]
