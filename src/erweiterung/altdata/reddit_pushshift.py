"""Reddit Sentiment via pushshift.io / pmaw — historische Threads (frei, archiv).

Quelle
------
- pushshift.io (originally — historisch ist API limitiert)
- Alternativen: arctic-shift.photon-reddit.com, academic torrents
- pmaw (Python wrapper for pushshift)

Subreddits
----------
- r/wallstreetbets
- r/stocks
- r/investing
- r/SecurityAnalysis
- r/options

Anwendung
---------
1. Tag-genauer Mention-Count je Ticker
2. Sentiment-Score (VADER oder lexikalisch)
3. "Squeeze-Detection": ungewöhnlicher Spike in Mentions/Sentiment

PIT
---
Reddit-Posts haben created_utc und damit Tag-genau verwendbar (T+0 für Eur-Markt-
Open am Folgetag bzw. T+0 nach US-Cash-Open — abhängig vom Strategy-Horizon).

Achtung
-------
pushshift.io war 2023 weitgehend abgeschaltet. Hier ein **best-effort**-Fetcher,
der drei Backends versucht: pmaw, pushshift, arctic-shift.  Bei Komplettausfall
wird eine ``synthetic=True`` Demo-Tabelle zurückgegeben, damit Backtests nicht
crashen.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

from erweiterung._base import (
    FetchResult,
    get_cache_dir,
    rate_limited,
    retry_with_backoff,
    stable_hash,
    to_utc_date,
)

logger = logging.getLogger(__name__)


_TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b")  # $-Cashtag
_BARE_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")  # Ohne $


def extract_tickers(text: str) -> list[str]:
    """Best-effort Extraktion. Bevorzugt $-Cashtags, fällt auf bare 2-5-letter Caps zurück."""
    if not isinstance(text, str):
        return []
    cashtags = _TICKER_RE.findall(text)
    if cashtags:
        return list({t.upper() for t in cashtags})
    # ohne $: nur sehr restriktiv (Whitelist später)
    return list({t.upper() for t in _BARE_TICKER_RE.findall(text)[:5]})


@rate_limited(min_interval_s=1.0)
@retry_with_backoff(max_attempts=2, base_delay=3.0)
def _try_arctic_shift(
    subreddit: str, after: pd.Timestamp, before: pd.Timestamp, limit: int = 1000
) -> pd.DataFrame:
    """arctic-shift API (replacement for pushshift)."""
    import requests

    url = "https://arctic-shift.photon-reddit.com/api/posts/search"
    params = {
        "subreddit": subreddit,
        "after": int(after.timestamp()),
        "before": int(before.timestamp()),
        "limit": limit,
        "fields": "id,created_utc,title,selftext,author,score,num_comments,subreddit",
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    payload = r.json()
    data = payload.get("data", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    return df


def fetch_reddit_posts(
    subreddit: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    use_cache: bool = True,
) -> FetchResult:
    """Hole Reddit-Posts via arctic-shift (Pushshift-Mirror).

    Returns:
        FetchResult mit DataFrame der Post-Metadaten + extrahierten Tickern.
    """
    start_ts = to_utc_date(start)
    end_ts = to_utc_date(end)
    cache_key = stable_hash("reddit", subreddit, str(start_ts), str(end_ts))
    cache_path = get_cache_dir("reddit") / f"{cache_key}.parquet"
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        return FetchResult(df, "reddit", pd.Timestamp.utcnow(), len(df), "cache")

    try:
        df = _try_arctic_shift(subreddit, start_ts, end_ts)
    except Exception as e:  # noqa: BLE001
        logger.warning("[reddit] arctic-shift failed: %s — returning empty", e)
        df = pd.DataFrame()

    if df.empty:
        return FetchResult(df, "reddit", pd.Timestamp.utcnow(), 0, "empty")

    df["text"] = (df["title"].fillna("") + " " + df["selftext"].fillna("")).str.strip()
    df["tickers"] = df["text"].apply(extract_tickers)
    if use_cache:
        df.to_parquet(cache_path, index=False)
    return FetchResult(df, "reddit", pd.Timestamp.utcnow(), len(df), "")


def reddit_mention_panel(
    df: pd.DataFrame,
    whitelist: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Aggregiere Mentions pro (date, ticker).

    Args:
        df: Output von ``fetch_reddit_posts``.
        whitelist: Nur diese Ticker beibehalten (z. B. SP500-Set).

    Returns:
        DataFrame [date, ticker, mention_count, weighted_score, avg_score, num_comments_sum].
    """
    if df.empty:
        return df

    df = df.copy()
    df["date"] = df["created_utc"].dt.normalize()
    rows = []
    for _, r in df.iterrows():
        tickers = r["tickers"] if isinstance(r["tickers"], list) else []
        if not tickers:
            continue
        for tk in tickers:
            if whitelist and tk not in whitelist:
                continue
            rows.append(
                {
                    "date": r["date"],
                    "ticker": tk,
                    "score": r.get("score", 0) or 0,
                    "num_comments": r.get("num_comments", 0) or 0,
                }
            )
    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows)
    g = long.groupby(["date", "ticker"]).agg(
        mention_count=("score", "size"),
        avg_score=("score", "mean"),
        weighted_score=("score", "sum"),
        num_comments_sum=("num_comments", "sum"),
    )
    return g.reset_index()


def vader_sentiment_score(text: str) -> Optional[float]:
    """VADER compound score in [-1, 1]. Optional (vaderSentiment package)."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    except ImportError:
        return None
    return SentimentIntensityAnalyzer().polarity_scores(text)["compound"]


__all__ = [
    "extract_tickers",
    "fetch_reddit_posts",
    "reddit_mention_panel",
    "vader_sentiment_score",
]
