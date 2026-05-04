"""Stooq EU/US EOD data source — free, unofficial endpoint.

From 10_FREE_DATEN.md §10.8.
Coverage: Germany, UK, Poland, teils USA (EOD only, no intraday, no delisted).
Legal: no explicit scraping clause — use as EU-EOD fallback only.

Usage: Fallback when EODHD paid is not available.
       EU ticker format: e.g. "SAP.DE", "VODAFONE.UK"

Install: pip install pandas-datareader
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)


def _suffix_map(exchange: str) -> str:
    """Map exchange name to Stooq suffix."""
    return {
        "XETRA": ".DE",
        "LSE": ".UK",
        "WSE": ".PL",
        "NASDAQ": ".US",
        "NYSE": ".US",
    }.get(exchange.upper(), "")


def fetch_stooq_eod(
    ticker: str,
    start: str | date = "2020-01-01",
    end: str | date | None = None,
) -> pd.DataFrame:
    """Fetch EOD OHLCV data from Stooq.

    Args:
        ticker: Stooq symbol (e.g. "SAP.DE", "^SPX" for S&P 500)
        start: Start date string (YYYY-MM-DD) or date object
        end: End date (default today)

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume. Empty on failure.
    """
    try:
        from pandas_datareader import data as pdr
    except ImportError:
        logger.warning(
            "pandas-datareader not installed — pip install pandas-datareader"
        )
        return pd.DataFrame()

    try:
        df = pdr.DataReader(
            ticker, "stooq", start=str(start), end=str(end or date.today())
        )
        if df.empty:
            return pd.DataFrame()
        df = df.sort_index()
        # Standardize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        df.index.name = "date"
        return df
    except Exception as exc:
        logger.debug("Stooq fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def fetch_eu_eod(
    tickers: list[str],
    start: str = "2020-01-01",
) -> dict[str, pd.DataFrame]:
    """Fetch EOD for a list of EU tickers.

    Args:
        tickers: List of Stooq-formatted symbols (e.g. ["SAP.DE", "ADS.DE"])
        start: Start date

    Returns:
        Dict mapping ticker → OHLCV DataFrame.
    """
    result = {}
    for ticker in tickers:
        df = fetch_stooq_eod(ticker, start=start)
        if not df.empty:
            result[ticker] = df
        else:
            logger.debug("Stooq: no data for %s", ticker)
    return result


def build_euro_stoxx50_tickers_stooq() -> list[str]:
    """Return EURO STOXX 50 tickers in Stooq format (.DE/.PA/.MI etc.).

    Hardcoded list — update when index composition changes.
    """
    return [
        "AIR.PA",
        "ALV.DE",
        "ADS.DE",
        "AD.AS",
        "ASML.AS",
        "ATO.PA",
        "CS.PA",
        "AXA.PA",
        "BNP.PA",
        "BAS.DE",
        "BAYN.DE",
        "BMW.DE",
        "CAP.PA",
        "CRH.DE",
        "DAI.DE",
        "DAN.PA",
        "DB1.DE",
        "DTE.DE",
        "ENEL.MI",
        "ENI.MI",
        "EL.PA",
        "FRE.DE",
        "IBE.MC",
        "IFX.DE",
        "INGA.AS",
        "ISP.MI",
        "KER.PA",
        "LIN.DE",
        "LOR.PA",
        "MC.PA",
        "MUV2.DE",
        "OR.PA",
        "ORA.PA",
        "PHIA.AS",
        "PRX.AS",
        "RWE.DE",
        "SGO.PA",
        "SAN.MC",
        "SU.PA",
        "SAF.PA",
        "SAP.DE",
        "SIE.DE",
        "STLA.MI",
        "TEF.MC",
        "TOTF.PA",
        "UCG.MI",
        "UNA.AS",
        "URW.PA",
        "VIV.PA",
        "VOW3.DE",
    ]


__all__ = [
    "fetch_stooq_eod",
    "fetch_eu_eod",
    "build_euro_stoxx50_tickers_stooq",
]
