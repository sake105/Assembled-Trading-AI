"""Insider Form-4 Cluster-Buy Signal.

From 13_FREE_MODULE.md §13.4.
Cohen/Malloy/Pomorski 2012 (JF 67(3)): 82 bps/month for opportunistic insiders.

Key signals:
- cluster_buy_score: distinct insiders buying in 30d (>=3 = strong bullish)
- net_officer_usd: CEO/CFO/COO net purchase USD in 90d

Rules:
- Only 'P' (Purchase) and 'S' (Sale) codes — not 'A' (Award), 'M' (Exercise).
- Filter 10b5-1 plan dispositions (preplanned, non-informative).

Data source: SEC EDGAR via edgartools (free, public domain).
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# Officer title patterns
_OFFICER_TITLES = re.compile(
    r"\b(CEO|CFO|COO|President|Chief\s+Executive|Chief\s+Financial|Chief\s+Operating)\b",
    re.IGNORECASE,
)


def _try_edgartools():
    try:
        import edgar
        return edgar
    except ImportError:
        logger.warning("edgartools not installed — pip install edgartools")
        return None


def _set_edgar_identity():
    edgar = _try_edgartools()
    if edgar is None:
        return False
    try:
        edgar.set_identity("AssembledTradingAI research@example.com")
        return True
    except Exception:
        return False


def _get_form4_filings(ticker: str, days: int = 30) -> list:
    """Fetch recent Form-4 filings for a ticker."""
    edgar = _try_edgartools()
    if edgar is None:
        return []

    try:
        _set_edgar_identity()
        entity = edgar.Company(ticker)
        filings = entity.get_filings(
            form="4",
            date_range=(
                str(date.today() - timedelta(days=days)),
                str(date.today()),
            ),
        )
        if filings is None:
            return []
        return list(filings)
    except Exception as exc:
        logger.debug("Form-4 fetch failed for %s: %s", ticker, exc)
        return []


def _extract_transactions(filing) -> list[dict]:
    """Extract individual transactions from a Form-4 filing."""
    try:
        transactions = []
        # edgartools Form-4 may expose transactions directly
        txns = getattr(filing, "transactions", None) or getattr(filing, "rows", None)
        if txns is None:
            return []

        for txn in txns:
            code = str(getattr(txn, "transaction_code", "") or "").upper()
            if code not in ("P", "S"):
                continue

            # Skip 10b5-1 plan transactions
            is_10b51 = getattr(txn, "is_10b5_1", False) or getattr(txn, "footnote", "")
            if is_10b51 and "10b5-1" in str(is_10b51).lower():
                continue

            shares = float(getattr(txn, "shares", 0) or 0)
            price = float(getattr(txn, "price", 0) or 0)
            reporter = str(getattr(filing, "reporting_name", "") or
                           getattr(filing, "reporter_name", "") or "")
            title = str(getattr(filing, "reporting_title", "") or
                        getattr(filing, "reporter_title", "") or "")
            is_officer = bool(_OFFICER_TITLES.search(title))

            transactions.append({
                "code": code,
                "shares": shares,
                "price": price,
                "value": shares * price,
                "reporter": reporter,
                "title": title,
                "is_officer": is_officer,
                "filing_date": str(getattr(filing, "filing_date", "")),
            })

        return transactions
    except Exception as exc:
        logger.debug("Transaction extraction failed: %s", exc)
        return []


def cluster_buy_score(
    ticker: str,
    lookback_days: int = 30,
) -> int:
    """Count distinct insiders making open-market purchases in last N days.

    Args:
        ticker: Stock ticker symbol
        lookback_days: Lookback window (default 30 days)

    Returns:
        Count of distinct buyers. 0 if no filings or edgartools unavailable.
        Interpretation: >=3 = strong bullish, >=2 = weak bullish, 0 = neutral/bearish.
    """
    filings = _get_form4_filings(ticker, days=lookback_days)
    if not filings:
        return 0

    buyers: set[str] = set()
    for filing in filings:
        transactions = _extract_transactions(filing)
        for txn in transactions:
            if txn["code"] == "P" and txn["value"] > 0:
                buyers.add(txn["reporter"])

    return len(buyers)


def net_officer_usd(
    ticker: str,
    lookback_days: int = 90,
) -> float:
    """Net USD purchased by C-suite officers in last N days.

    CEO/CFO/COO only (most informative insider tier).
    Purchases are positive, sales are negative.

    Args:
        ticker: Stock ticker
        lookback_days: Lookback window (default 90 days)

    Returns:
        Net USD value. Positive = net buying, negative = net selling.
        Returns 0.0 if no filings or edgartools unavailable.
    """
    filings = _get_form4_filings(ticker, days=lookback_days)
    if not filings:
        return 0.0

    net = 0.0
    for filing in filings:
        transactions = _extract_transactions(filing)
        for txn in transactions:
            if not txn["is_officer"]:
                continue
            if txn["code"] == "P":
                net += txn["value"]
            elif txn["code"] == "S":
                net -= txn["value"]

    return net


def insider_cluster_signal(
    ticker: str,
    days: int = 30,
    buyback_score: float = 0.0,
) -> float:
    """Composite insider cluster signal score [0, 1].

    Args:
        ticker: Stock ticker
        days: Lookback window
        buyback_score: Optional buyback signal for combo boost

    Returns:
        0.0 = no signal
        0.4 = 2 distinct buyers (weak)
        0.7 = ≥3 distinct buyers (strong cluster)
        0.9 = ≥3 buyers + positive net officer USD
        1.0 = cluster + officer + buyback combo
    """
    buyers = cluster_buy_score(ticker, lookback_days=days)
    net_usd = net_officer_usd(ticker, lookback_days=days * 3)

    if buyers == 0:
        return 0.0

    score = 0.4 if buyers >= 2 else 0.0
    if buyers >= 3:
        score = 0.7
    if buyers >= 3 and net_usd > 250_000:
        score = 0.9
    if score >= 0.7 and buyback_score > 0.5:
        score = min(1.0, score + 0.1)

    return score


def batch_insider_signals(
    tickers: list[str],
    days: int = 30,
) -> pd.DataFrame:
    """Compute insider cluster signals for a list of tickers.

    Returns:
        DataFrame with columns: ticker, cluster_buyers, net_officer_usd, signal_score.
    """
    rows = []
    for ticker in tickers:
        try:
            buyers = cluster_buy_score(ticker, lookback_days=days)
            net_usd = net_officer_usd(ticker, lookback_days=days * 3)
            score = insider_cluster_signal(ticker, days=days)
            rows.append({
                "ticker": ticker,
                "cluster_buyers": buyers,
                "net_officer_usd": net_usd,
                "signal_score": score,
            })
        except Exception as exc:
            logger.debug("Insider signal failed for %s: %s", ticker, exc)
            rows.append({
                "ticker": ticker,
                "cluster_buyers": 0,
                "net_officer_usd": 0.0,
                "signal_score": 0.0,
            })

    return pd.DataFrame(rows)


__all__ = [
    "cluster_buy_score",
    "net_officer_usd",
    "insider_cluster_signal",
    "batch_insider_signals",
]
