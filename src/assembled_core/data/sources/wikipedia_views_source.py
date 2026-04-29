"""Wikipedia Page Views — retail attention signal.

From 10_FREE_DATEN.md §10.14 and 13_FREE_MODULE.md §13.14.
Moat et al. 2013 (Nature Sci Rep): 1-day lag views predict drawdowns.
Long-short Sharpe ~0.3.

Rate limit: 100 req/s, no key required.
Install: pip install mwviews==0.3

Feature: views_z = zscore(views_7d_mean / views_90d_mean)
Especially useful for Small/Mid-Caps without news coverage.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# Hardcoded company→Wikipedia article mapping for top tickers
# Extend this as needed — or use OpenFIGI/Wikidata SPARQL for automated mapping
_TICKER_TO_WIKI: dict[str, str] = {
    "AAPL": "Apple_Inc.",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet_Inc.",
    "AMZN": "Amazon_(company)",
    "NVDA": "Nvidia",
    "META": "Meta_Platforms",
    "TSLA": "Tesla,_Inc.",
    "BRK-B": "Berkshire_Hathaway",
    "JPM": "JPMorgan_Chase",
    "V": "Visa_Inc.",
    "JNJ": "Johnson_%26_Johnson",
    "WMT": "Walmart",
    "PG": "Procter_%26_Gamble",
    "MA": "Mastercard",
    "HD": "Home_Depot",
    "UNH": "UnitedHealth_Group",
    "BAC": "Bank_of_America",
    "XOM": "ExxonMobil",
    "CVX": "Chevron_Corporation",
    "KO": "The_Coca-Cola_Company",
}


def _try_mwviews():
    try:
        from mwviews.api import PageviewsClient
        return PageviewsClient
    except ImportError:
        logger.warning("mwviews not installed — pip install mwviews==0.3")
        return None


def fetch_article_views(
    articles: list[str],
    days: int = 90,
    language: str = "en",
) -> pd.DataFrame:
    """Fetch Wikipedia daily page views for a list of article names.

    Args:
        articles: List of Wikipedia article names (URL-encoded if needed)
        days: Number of trailing days to fetch
        language: Wikipedia language edition (default 'en')

    Returns:
        DataFrame with dates as index, article names as columns, views as values.
    """
    PageviewsClient = _try_mwviews()
    if PageviewsClient is None:
        return pd.DataFrame()

    end = date.today()
    start = end - timedelta(days=days)

    try:
        client = PageviewsClient(user_agent="AssembledTradingAI/1.0 (research)")
        views = client.article_views(
            language,
            articles,
            granularity="daily",
            start=start.strftime("%Y%m%d"),
            end=end.strftime("%Y%m%d"),
        )
        # views is dict: article → {date_str: count}
        dfs = []
        for article, data in views.items():
            s = pd.Series(data, name=article)
            s.index = pd.to_datetime(list(s.index))
            dfs.append(s)
        if not dfs:
            return pd.DataFrame()
        return pd.DataFrame(dfs).T.sort_index()
    except Exception as exc:
        logger.debug("mwviews fetch failed: %s", exc)
        return pd.DataFrame()


def wikipedia_attention_feature(
    ticker: str,
    days: int = 90,
    short_window: int = 7,
) -> float:
    """Compute Wikipedia attention Z-score for a single ticker.

    Formula: zscore(views_7d_mean / views_90d_mean)
    Returns 0.0 if ticker not in mapping or data unavailable.
    """
    article = _TICKER_TO_WIKI.get(ticker)
    if article is None:
        return 0.0

    df = fetch_article_views([article], days=days)
    if df.empty or article not in df.columns:
        return 0.0

    views = df[article].dropna()
    if len(views) < short_window + 1:
        return 0.0

    recent_mean = float(views.iloc[-short_window:].mean())
    long_mean = float(views.mean())
    if long_mean < 1:
        return 0.0

    ratio = recent_mean / long_mean
    # Simple Z-score relative to trailing std of ratio
    # For single-ticker: return normalized ratio directly (no panel Z)
    return float(ratio - 1.0)  # positive = above-average attention


def batch_wikipedia_attention(
    tickers: list[str],
    days: int = 90,
    short_window: int = 7,
) -> pd.Series:
    """Compute Wikipedia attention features for multiple tickers.

    Returns Series indexed by ticker. Tickers without a Wiki mapping return 0.0.
    """
    # Fetch all available articles in one request
    ticker_to_article = {t: _TICKER_TO_WIKI[t] for t in tickers if t in _TICKER_TO_WIKI}
    if not ticker_to_article:
        return pd.Series({t: 0.0 for t in tickers}, name="wiki_attention")

    articles = list(ticker_to_article.values())
    df = fetch_article_views(articles, days=days)

    scores = {}
    for ticker in tickers:
        article = ticker_to_article.get(ticker)
        if article is None or df.empty or article not in df.columns:
            scores[ticker] = 0.0
            continue
        views = df[article].dropna()
        if len(views) < short_window + 1:
            scores[ticker] = 0.0
            continue
        recent_mean = float(views.iloc[-short_window:].mean())
        long_mean = float(views.mean())
        scores[ticker] = float(recent_mean / max(long_mean, 1) - 1.0)

    # Cross-sectional Z-score
    s = pd.Series(scores, name="wiki_attention")
    if s.std() > 0:
        s = (s - s.mean()) / s.std()
    return s


def add_ticker_wiki_mapping(ticker: str, article: str) -> None:
    """Register a new ticker→Wikipedia article mapping at runtime."""
    _TICKER_TO_WIKI[ticker] = article


__all__ = [
    "fetch_article_views",
    "wikipedia_attention_feature",
    "batch_wikipedia_attention",
    "add_ticker_wiki_mapping",
    "_TICKER_TO_WIKI",
]
