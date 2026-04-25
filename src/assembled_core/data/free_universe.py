"""Free-tier universe management — ticker lists without paid data.

From 14_FREE_UNIVERSUM.md.

Tier-1 Core (~588 tickers):
  - S&P 500 from Wikipedia (tagesaktuell)
  - EURO STOXX 50 hardcoded (update quarterly)
  - 35 ETF-Core hardcoded

Tier-2: iShares ETF holdings CSVs (weekly download)
Tier-3: On-demand via yfinance batch-pull at news/volume trigger

Liquidity filter from 14_FREE_UNIVERSUM.md §14.4.
Priority scoring from §14.8 — reduce compute by 65%.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ETF Core (35 tickers) — hardcoded, update as needed
# ---------------------------------------------------------------------------

ETF_CORE: list[str] = [
    # US-Broad
    "SPY", "VOO", "IVV", "QQQ", "IWM", "VTI", "DIA",
    # Int-Developed
    "VGK", "VEA", "EFA", "IEFA", "VXUS",
    # Emerging
    "EEM", "VWO", "IEMG",
    # Sectors
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
    "XLI", "XLB", "XLRE", "XLU", "XLC",
    # Commodities
    "GLD", "SLV", "USO", "UNG", "DBC",
    # Bonds
    "TLT", "HYG", "LQD",
    # Vol
    "VXX",
    # Crypto spot ETF (post 2024)
    "IBIT",
]

# ---------------------------------------------------------------------------
# EURO STOXX 50 — hardcoded (update quarterly when composition changes)
# ---------------------------------------------------------------------------

EURO_STOXX_50: list[str] = [
    "AIR.PA", "ALV.DE", "ADS.DE", "AD.AS", "ASML.AS",
    "ATO.PA", "CS.PA", "AXA.PA", "BNP.PA", "BAS.DE",
    "BAYN.DE", "BMW.DE", "CAP.PA", "CRH.DE", "DAI.DE",
    "DAN.PA", "DB1.DE", "DTE.DE", "ENEL.MI", "ENI.MI",
    "EL.PA", "FRE.DE", "IBE.MC", "IFX.DE", "INGA.AS",
    "ISP.MI", "KER.PA", "LIN.DE", "LOR.PA", "MC.PA",
    "MUV2.DE", "OR.PA", "ORA.PA", "PHIA.AS", "PRX.AS",
    "RWE.DE", "SGO.PA", "SAN.MC", "SU.PA", "SAF.PA",
    "SAP.DE", "SIE.DE", "STLA.MI", "TEF.MC", "TOTF.PA",
    "UCG.MI", "UNA.AS", "URW.PA", "VIV.PA", "VOW3.DE",
]


def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 tickers from Wikipedia.

    Falls back to cached list if network unavailable.
    """
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"},
        )
        sp500 = tables[0]["Symbol"].tolist()
        # Normalize: BRK.B → BRK-B for Alpaca/yfinance compatibility
        return [t.replace(".", "-") for t in sp500]
    except Exception as exc:
        logger.warning("Wikipedia S&P 500 fetch failed (%s) — using fallback list", exc)
        return _SP500_FALLBACK


def get_tier1_universe() -> list[str]:
    """Return full Tier-1 universe: S&P 500 + EURO STOXX 50 + ETF Core."""
    sp500 = get_sp500_tickers()
    all_tickers = list(dict.fromkeys(sp500 + EURO_STOXX_50 + ETF_CORE))
    logger.info("Tier-1 universe: %d tickers", len(all_tickers))
    return all_tickers


def get_russell2000_tickers() -> list[str]:
    """Fetch Russell 2000 tickers from iShares IWM holdings CSV.

    NOTE: survivorship-biased — delisted tickers not included.
    Only use for current-state analysis, not historical backtesting.
    """
    url = (
        "https://www.ishares.com/us/products/239710/"
        "ishares-russell-2000-etf/1467271812596.ajax"
        "?fileType=csv&fileName=IWM_holdings&dataType=fund"
    )
    try:
        df = pd.read_csv(url, skiprows=9)
        tickers = df["Ticker"].dropna().tolist()
        # Filter out non-equity rows
        tickers = [t for t in tickers if isinstance(t, str) and t.isalpha() and 1 <= len(t) <= 5]
        logger.info("Russell 2000 from iShares: %d tickers", len(tickers))
        return tickers
    except Exception as exc:
        logger.warning("Russell 2000 iShares fetch failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Liquidity Filter — §14.4
# ---------------------------------------------------------------------------

def liquidity_filter(
    ticker_data: dict[str, Any],
    permissive: bool = False,
) -> bool:
    """Return True if ticker passes liquidity requirements.

    Args:
        ticker_data: Dict with keys:
          avg_dollar_volume_30d (float, USD/day)
          market_cap (float, USD)
          avg_bid_ask_spread_bps (float)
          price (float)
          trading_days_ytd_pct (float, 0-1)
        permissive: Use Small-Cap thresholds if True.

    Returns:
        True = passes liquidity gate, False = skip.
    """
    min_vol = 500_000 if permissive else 1_000_000
    min_cap = 100_000_000 if permissive else 300_000_000

    return (
        float(ticker_data.get("avg_dollar_volume_30d", 0)) > min_vol
        and float(ticker_data.get("market_cap", 0)) > min_cap
        and float(ticker_data.get("avg_bid_ask_spread_bps", 999)) < 20
        and float(ticker_data.get("price", 0)) > 5
        and float(ticker_data.get("trading_days_ytd_pct", 0)) > 0.9
    )


# ---------------------------------------------------------------------------
# Priority Score — §14.8
# ---------------------------------------------------------------------------

def priority_score(
    ticker: str,
    news_velocity: float = 0.0,
    last_ta_score: float = 0.0,
    avg_dollar_volume: float = 1_000_000,
    has_earnings_today: bool = False,
    has_fomc_impact: bool = False,
) -> float:
    """Compute priority score for batch processing.

    High scores → analyzed first. Reduces compute by 65% by focusing on
    the 200 most active tickers.

    Returns:
        Priority score (higher = process first).
    """
    import math
    base = (
        0.4 * news_velocity
        + 0.3 * abs(last_ta_score)
        + 0.3 * math.log1p(avg_dollar_volume)
    )
    if has_earnings_today or has_fomc_impact:
        base += 10.0
    return base


def get_top_n_tickers(
    tickers: list[str],
    scores: dict[str, float],
    n: int = 200,
) -> list[str]:
    """Return top-N tickers by priority score."""
    return sorted(tickers, key=lambda t: scores.get(t, 0.0), reverse=True)[:n]


# ---------------------------------------------------------------------------
# Minimal S&P 500 fallback (first 20 for import safety only)
# ---------------------------------------------------------------------------

_SP500_FALLBACK: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG",
    "BRK-B", "LLY", "JPM", "XOM", "V", "UNH", "TSLA", "AVGO",
    "PG", "MA", "COST", "JNJ", "HD",
]


__all__ = [
    "ETF_CORE",
    "EURO_STOXX_50",
    "get_sp500_tickers",
    "get_tier1_universe",
    "get_russell2000_tickers",
    "liquidity_filter",
    "priority_score",
    "get_top_n_tickers",
]
