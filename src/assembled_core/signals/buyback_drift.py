"""Buyback-Drift signal from SEC 8-K filings.

From 13_FREE_MODULE.md §13.12.
Peyer/Vermaelen 2009: 3-6% abnormal returns 12-24 months post-announcement.

Datenquelle: SEC EDGAR 8-K via edgartools (free, public domain).
Item 8.01 = "Other Events" — buyback announcements typically filed here.

Filter: pct_market_cap > 5% = strong bullish signal.
Combo with Insider-Buying = very strong signal.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Regex for extracting USD buyback amounts from 8-K text
_USD_RE = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)\s*(million|billion|mn|bn|M|B)\b",
    re.IGNORECASE,
)


def _parse_usd_amount(text: str) -> float:
    """Extract USD amount from text like '$500 million' → 500_000_000."""
    for match in _USD_RE.finditer(text):
        try:
            amount_str = match.group(1).replace(",", "")
            amount = float(amount_str)
            unit = match.group(2).lower()
            if unit in ("billion", "bn", "b"):
                amount *= 1e9
            elif unit in ("million", "mn", "m"):
                amount *= 1e6
            return amount
        except ValueError:
            continue
    return 0.0


def _try_edgartools():
    try:
        import edgar

        return edgar
    except ImportError:
        logger.warning("edgartools not installed — pip install edgartools")
        return None


def detect_buyback_announcement(
    ticker: str,
    days: int = 30,
    market_cap: float | None = None,
) -> dict | None:
    """Detect buyback announcements in recent 8-K filings.

    Args:
        ticker: Stock ticker symbol
        days: Lookback window in days
        market_cap: Current market cap in USD (for pct calculation)

    Returns:
        Dict with buyback info or None if no announcement found.
        Keys: filing_date, amount_usd, pct_market_cap, raw_text_excerpt
    """
    edgar = _try_edgartools()
    if edgar is None:
        return None

    try:
        from datetime import date, timedelta

        edgar.set_identity("AssembledTradingAI research@example.com")

        entity = edgar.Company(ticker)
        filings = entity.get_filings(
            form="8-K",
            date_range=(
                str(date.today() - timedelta(days=days)),
                str(date.today()),
            ),
        )

        if filings is None or len(filings) == 0:
            return None

        for filing in filings:
            try:
                # Check item codes — 8.01 = Other Events where buybacks often appear
                items = getattr(filing, "items", []) or []
                if not any(
                    "8.01" in str(item) or "7.01" in str(item) for item in items
                ):
                    # Try reading full text
                    text = ""
                else:
                    text = ""

                # Try to get document text
                doc = filing.document
                if doc is not None:
                    text = str(doc)

                text_lower = text.lower()
                if not any(
                    kw in text_lower
                    for kw in [
                        "repurchase",
                        "buyback",
                        "share repurchase",
                        "stock repurchase",
                    ]
                ):
                    continue

                amount = _parse_usd_amount(text)
                if amount <= 0:
                    continue

                pct = amount / market_cap if market_cap and market_cap > 0 else None

                return {
                    "filing_date": str(getattr(filing, "filing_date", "")),
                    "amount_usd": amount,
                    "pct_market_cap": pct,
                    "signal_strength": "strong" if (pct and pct > 0.05) else "moderate",
                }
            except Exception as exc:
                logger.debug("Buyback filing parse failed for %s: %s", ticker, exc)
                continue

        return None
    except Exception as exc:
        logger.debug("Buyback detection failed for %s: %s", ticker, exc)
        return None


def buyback_signal_score(
    ticker: str,
    days: int = 30,
    market_cap: float | None = None,
    insider_cluster_score: float = 0.0,
) -> float:
    """Compute buyback signal score [0, 1].

    Args:
        ticker: Stock ticker
        days: Lookback window
        market_cap: For pct calculation
        insider_cluster_score: Optional insider buy score for combo boost

    Returns:
        Signal score:
          0.0 = no buyback detected
          0.5 = buyback < 5% market cap
          0.8 = buyback >= 5% market cap
          1.0 = buyback >= 5% + insider buying combo
    """
    result = detect_buyback_announcement(ticker, days=days, market_cap=market_cap)
    if result is None:
        return 0.0

    pct = result.get("pct_market_cap")
    base_score = 0.8 if (pct is not None and pct > 0.05) else 0.5

    # Combo boost with insider buying
    if insider_cluster_score > 0 and base_score >= 0.8:
        return min(1.0, base_score + 0.2)

    return base_score


__all__ = [
    "detect_buyback_announcement",
    "buyback_signal_score",
]
