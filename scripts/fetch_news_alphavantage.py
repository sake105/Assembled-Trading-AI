"""Alpha Vantage News & Sentiment fetcher.

Free tier: 25 API calls/day (standard free key).
Fetches ticker-specific news with per-article sentiment scores.

Usage:
    python scripts/fetch_news_alphavantage.py
    python scripts/fetch_news_alphavantage.py --tickers AAPL,MSFT,NVDA
    python scripts/fetch_news_alphavantage.py --tickers AAPL --limit 50
    python scripts/fetch_news_alphavantage.py --universe configs/universes/full_us_universe.txt
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

AV_BASE = "https://www.alphavantage.co/query"
OUTPUT_DIR = ROOT / "output" / "news" / "alphavantage"
SENTIMENT_OUT = ROOT / "output" / "news_sentiment_alphavantage.parquet"

# Limit per ticker per run (free tier: 25 calls/day total)
_DEFAULT_LIMIT = 50
_DELAY_BETWEEN_CALLS = 13.0  # safe for 25 calls/day (~5 calls/min free tier)


def _get_api_key() -> str | None:
    # alphavantage_source.py uses ALPHAVANTAGE_KEY (no _API_ suffix)
    return os.environ.get("ALPHAVANTAGE_KEY") or os.environ.get("ALPHAVANTAGE_API_KEY")


def fetch_ticker_news(
    ticker: str,
    api_key: str,
    limit: int = _DEFAULT_LIMIT,
    time_from: str | None = None,
) -> list[dict]:
    """Fetch news articles for a ticker from Alpha Vantage NEWS_SENTIMENT.

    Returns list of article dicts with: title, published_at, sentiment_score,
    sentiment_label, relevance_score, url.
    """
    params: dict[str, str] = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "limit": str(limit),
        "apikey": api_key,
        "sort": "LATEST",
    }
    if time_from:
        params["time_from"] = time_from  # format: YYYYMMDDTHHMM

    url = AV_BASE + "?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AssembledTradingAI/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log.warning("[WARN] AV fetch %s: %s", ticker, exc)
        return []

    if "Note" in data or "Information" in data:
        log.warning("[WARN] AV rate limit hit: %s", data.get("Note") or data.get("Information"))
        return []

    articles = data.get("feed", [])
    results = []
    for art in articles:
        # Extract per-ticker sentiment from ticker_sentiment list
        ticker_sents = art.get("ticker_sentiment", [])
        ts_entry = next((t for t in ticker_sents if t.get("ticker") == ticker), None)
        if ts_entry is None and ticker_sents:
            ts_entry = ticker_sents[0]

        sentiment_score = 0.0
        relevance = 0.0
        if ts_entry:
            try:
                sentiment_score = float(ts_entry.get("ticker_sentiment_score", 0.0))
                relevance = float(ts_entry.get("relevance_score", 0.0))
            except (TypeError, ValueError):
                pass
        else:
            try:
                sentiment_score = float(art.get("overall_sentiment_score", 0.0))
            except (TypeError, ValueError):
                pass

        # Parse time_published: "20240105T120000"
        published_str = art.get("time_published", "")
        try:
            published_at = datetime.strptime(published_str, "%Y%m%dT%H%M%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            published_at = datetime.now(timezone.utc)

        results.append(
            {
                "symbol": ticker,
                "published_at": published_at.isoformat(),
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "source": art.get("source", ""),
                "sentiment_score": round(sentiment_score, 4),
                "relevance_score": round(relevance, 4),
                "sentiment_label": art.get("overall_sentiment_label", "Neutral"),
            }
        )
    return results


def articles_to_daily_sentiment(articles: list[dict]) -> "pd.DataFrame":
    """Aggregate article-level sentiment to daily (date, symbol) rows."""
    import pandas as pd

    if not articles:
        return pd.DataFrame()

    df = pd.DataFrame(articles)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["date"] = df["published_at"].dt.normalize()

    # Weight by relevance score if available
    df["weighted_score"] = df["sentiment_score"] * df["relevance_score"].clip(lower=0.1)

    agg = (
        df.groupby(["date", "symbol"])
        .agg(
            sentiment_score=("weighted_score", "mean"),
            sentiment_volume=("sentiment_score", "count"),
            count=("sentiment_score", "count"),
        )
        .reset_index()
    )
    agg = agg.rename(columns={"date": "timestamp"})
    return agg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alpha Vantage news sentiment fetcher")
    parser.add_argument("--tickers", help="Comma-separated tickers")
    parser.add_argument("--universe", help="Path to flat ticker file (one per line)")
    parser.add_argument("--limit", type=int, default=_DEFAULT_LIMIT)
    parser.add_argument("--days-back", type=int, default=7, help="How many days to look back")
    parser.add_argument("--delay", type=float, default=_DELAY_BETWEEN_CALLS)
    args = parser.parse_args(argv)

    api_key = _get_api_key()
    if not api_key or api_key.startswith("your_"):
        log.error("[ERROR] ALPHAVANTAGE_KEY not set. Get a free key at alphavantage.co")
        return 1

    # Build ticker list
    tickers: list[str] = []
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    elif args.universe:
        uni_path = ROOT / args.universe
        if uni_path.exists():
            tickers = [
                l.strip().upper()
                for l in uni_path.read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.startswith("#")
            ]
    else:
        # Default: load master universe
        try:
            from src.assembled_core.data.master_universe_loader import load_master_universe
            tickers, _ = load_master_universe()
        except Exception:
            tickers = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN"]

    # Time window
    time_from = (datetime.now(timezone.utc) - timedelta(days=args.days_back)).strftime(
        "%Y%m%dT%H%M"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_articles: list[dict] = []
    total = len(tickers)

    log.info("[START] Alpha Vantage news: %d tickers, %d days back", total, args.days_back)

    for i, ticker in enumerate(tickers, 1):
        articles = fetch_ticker_news(ticker, api_key, limit=args.limit, time_from=time_from)
        all_articles.extend(articles)
        log.info("  [%d/%d] %s: %d articles", i, total, ticker, len(articles))
        if i < total:
            time.sleep(args.delay)

    if not all_articles:
        log.warning("[WARN] No articles fetched")
        return 0

    # Save raw articles
    raw_path = OUTPUT_DIR / f"articles_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    raw_path.write_text(
        json.dumps(all_articles, indent=2, default=str), encoding="utf-8"
    )
    log.info("[OK] Raw articles: %d → %s", len(all_articles), raw_path)

    # Save daily sentiment parquet
    import pandas as pd
    daily = articles_to_daily_sentiment(all_articles)
    if not daily.empty:
        SENTIMENT_OUT.parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(SENTIMENT_OUT, index=False)
        log.info(
            "[OK] Daily sentiment: %d rows, %d symbols → %s",
            len(daily), daily["symbol"].nunique(), SENTIMENT_OUT,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
