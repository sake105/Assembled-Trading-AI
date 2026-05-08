"""Polygon.io Ticker News fetcher.

Free tier: unlimited reads (with free API key), 5 requests/minute.
Returns article-level news with publisher sentiment signals.

Usage:
    python scripts/fetch_news_polygon.py
    python scripts/fetch_news_polygon.py --tickers AAPL,MSFT,NVDA
    python scripts/fetch_news_polygon.py --days-back 14
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

POLYGON_BASE = "https://api.polygon.io/v2/reference/news"
OUTPUT_DIR = ROOT / "output" / "news" / "polygon"
SENTIMENT_OUT = ROOT / "output" / "news_sentiment_polygon.parquet"

_DELAY_BETWEEN_CALLS = 12.5  # 5 req/min free tier → 12s between calls


def _get_api_key() -> str | None:
    return os.environ.get("POLYGON_API_KEY")


# Simple keyword-based sentiment scorer for articles without explicit score
_POSITIVE_WORDS = {
    "beat",
    "beats",
    "surpassed",
    "record",
    "growth",
    "profit",
    "gain",
    "upgrade",
    "bullish",
    "strong",
    "rally",
    "soared",
    "climbed",
    "raised",
    "raised guidance",
    "partnership",
    "win",
    "won",
    "approved",
    "breakthrough",
}
_NEGATIVE_WORDS = {
    "miss",
    "missed",
    "fell",
    "slumped",
    "loss",
    "losses",
    "cut",
    "layoff",
    "lawsuit",
    "downgrade",
    "bearish",
    "dropped",
    "declined",
    "warning",
    "recall",
    "probe",
    "investigation",
    "default",
    "bankruptcy",
    "weak",
}


def _score_title(title: str) -> float:
    """Simple rule-based sentiment: +1 positive, -1 negative, 0 neutral."""
    lower = title.lower()
    pos = sum(1 for w in _POSITIVE_WORDS if w in lower)
    neg = sum(1 for w in _NEGATIVE_WORDS if w in lower)
    if pos == neg:
        return 0.0
    return min(max((pos - neg) / max(pos + neg, 1), -1.0), 1.0)


def fetch_ticker_news(
    ticker: str,
    api_key: str,
    limit: int = 50,
    published_utc_gte: str | None = None,
) -> list[dict]:
    """Fetch news from Polygon.io for a ticker.

    Returns list of article dicts with sentiment_score.
    """
    params: dict[str, str] = {
        "ticker": ticker,
        "limit": str(min(limit, 1000)),
        "sort": "published_utc",
        "order": "desc",
        "apiKey": api_key,
    }
    if published_utc_gte:
        params["published_utc.gte"] = published_utc_gte  # ISO format

    url = POLYGON_BASE + "?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "AssembledTradingAI/1.0"}
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            log.warning("[WARN] Polygon 403 for %s — check API key permissions", ticker)
        else:
            log.warning("[WARN] Polygon %d for %s: %s", exc.code, ticker, exc)
        return []
    except Exception as exc:
        log.warning("[WARN] Polygon fetch %s: %s", ticker, exc)
        return []

    status = data.get("status", "")
    if status not in ("OK", "ok", ""):
        log.warning("[WARN] Polygon status=%s for %s", status, ticker)

    articles = data.get("results", [])
    results = []
    for art in articles:
        # Extract sentiment from insights list if available
        sentiment_score = 0.0
        insights = art.get("insights", [])
        for insight in insights:
            if insight.get("ticker") == ticker:
                stype = insight.get("sentiment", "neutral").lower()
                sentiment_score = {
                    "positive": 0.6,
                    "negative": -0.6,
                    "neutral": 0.0,
                }.get(stype, 0.0)
                break
        else:
            # Fallback: keyword scoring on title
            sentiment_score = _score_title(art.get("title", ""))

        published_str = art.get("published_utc", "")
        try:
            published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        except ValueError:
            published_at = datetime.now(timezone.utc)

        results.append(
            {
                "symbol": ticker,
                "published_at": published_at.isoformat(),
                "title": art.get("title", ""),
                "url": art.get("article_url", ""),
                "publisher": (art.get("publisher") or {}).get("name", ""),
                "sentiment_score": round(sentiment_score, 4),
                "relevance_score": 1.0,
            }
        )
    return results


def articles_to_daily_sentiment(articles: list[dict]) -> "pd.DataFrame":  # noqa: F821
    """Aggregate article-level data to daily (timestamp, symbol) rows."""
    import pandas as pd

    if not articles:
        return pd.DataFrame()

    df = pd.DataFrame(articles)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["date"] = df["published_at"].dt.normalize()

    agg = (
        df.groupby(["date", "symbol"])
        .agg(
            sentiment_score=("sentiment_score", "mean"),
            sentiment_volume=("sentiment_score", "count"),
            count=("sentiment_score", "count"),
        )
        .reset_index()
    )
    return agg.rename(columns={"date": "timestamp"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Polygon.io news sentiment fetcher")
    parser.add_argument("--tickers", help="Comma-separated tickers")
    parser.add_argument("--days-back", type=int, default=14)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--delay", type=float, default=_DELAY_BETWEEN_CALLS)
    args = parser.parse_args(argv)

    api_key = _get_api_key()
    if not api_key or api_key.startswith("your_"):
        log.error("[ERROR] POLYGON_API_KEY not set. Get a free key at polygon.io")
        return 1

    # Build ticker list
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        try:
            from src.assembled_core.data.master_universe_loader import (
                load_master_universe,
            )

            tickers, _ = load_master_universe()
        except Exception:
            tickers = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN"]

    cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days_back)).isoformat()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_articles: list[dict] = []
    total = len(tickers)
    log.info("[START] Polygon news: %d tickers, %d days back", total, args.days_back)

    for i, ticker in enumerate(tickers, 1):
        articles = fetch_ticker_news(
            ticker, api_key, limit=args.limit, published_utc_gte=cutoff
        )
        all_articles.extend(articles)
        if articles:
            log.info("  [%d/%d] %s: %d articles", i, total, ticker, len(articles))
        if i < total:
            time.sleep(args.delay)

    if not all_articles:
        log.warning("[WARN] No articles fetched")
        return 0

    raw_path = (
        OUTPUT_DIR / f"articles_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    )
    raw_path.write_text(
        json.dumps(all_articles, indent=2, default=str), encoding="utf-8"
    )
    log.info("[OK] Raw articles: %d → %s", len(all_articles), raw_path)

    daily = articles_to_daily_sentiment(all_articles)
    if not daily.empty:
        SENTIMENT_OUT.parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(SENTIMENT_OUT, index=False)
        log.info(
            "[OK] Daily sentiment: %d rows, %d symbols → %s",
            len(daily),
            daily["symbol"].nunique(),
            SENTIMENT_OUT,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
