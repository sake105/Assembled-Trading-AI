"""NewsAPI.org fetcher — company/ticker news search.

Free tier: 100 requests/day, 1-month history.
Fetches keyword-based news for each ticker using company name + ticker symbol.

Usage:
    python scripts/fetch_news_newsapi.py
    python scripts/fetch_news_newsapi.py --tickers AAPL,MSFT,NVDA
    python scripts/fetch_news_newsapi.py --days-back 7
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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

NEWSAPI_BASE = "https://newsapi.org/v2/everything"
OUTPUT_DIR = ROOT / "output" / "news" / "newsapi"
SENTIMENT_OUT = ROOT / "output" / "news_sentiment_newsapi.parquet"

_DELAY = 2.0  # 100 calls/day → ~1 per 14 min; but burst is fine

# Ticker → company name for better search queries
_TICKER_TO_NAME: dict[str, str] = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "GOOGL": "Google Alphabet",
    "META": "Meta Facebook", "AMZN": "Amazon", "TSLA": "Tesla", "AVGO": "Broadcom",
    "AMD": "AMD Advanced Micro Devices", "QCOM": "Qualcomm", "INTC": "Intel",
    "ORCL": "Oracle", "CRM": "Salesforce", "NOW": "ServiceNow", "ADBE": "Adobe",
    "PLTR": "Palantir", "CRWD": "CrowdStrike", "PANW": "Palo Alto Networks",
    "JPM": "JPMorgan Chase", "BAC": "Bank of America", "GS": "Goldman Sachs",
    "V": "Visa", "MA": "Mastercard", "LLY": "Eli Lilly", "JNJ": "Johnson Johnson",
    "XOM": "ExxonMobil", "CVX": "Chevron", "NEE": "NextEra Energy",
    "LMT": "Lockheed Martin", "RTX": "Raytheon", "NOC": "Northrop Grumman",
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq ETF", "GLD": "Gold ETF",
}

_POSITIVE_WORDS = {
    "beat", "surpass", "record", "growth", "profit", "gain", "upgrade",
    "rally", "soar", "climb", "raise", "approve", "breakthrough", "win",
}
_NEGATIVE_WORDS = {
    "miss", "fall", "slump", "loss", "cut", "layoff", "lawsuit", "downgrade",
    "drop", "decline", "warn", "recall", "probe", "investigation", "default",
}


def _score_title(title: str) -> float:
    lower = title.lower()
    pos = sum(1 for w in _POSITIVE_WORDS if w in lower)
    neg = sum(1 for w in _NEGATIVE_WORDS if w in lower)
    if pos == neg:
        return 0.0
    return min(max((pos - neg) / max(pos + neg, 1), -1.0), 1.0)


def _get_api_key() -> str | None:
    return os.environ.get("NEWSAPI_KEY")


def fetch_ticker_news(
    ticker: str,
    api_key: str,
    days_back: int = 7,
    page_size: int = 20,
) -> list[dict]:
    """Search NewsAPI for articles about a ticker/company."""
    company = _TICKER_TO_NAME.get(ticker, ticker)
    query = f'"{ticker}" OR "{company}"'
    from_dt = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "q": query,
        "from": from_dt,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": str(min(page_size, 100)),
        "apiKey": api_key,
    }
    url = NEWSAPI_BASE + "?" + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AssembledTradingAI/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log.warning("[WARN] NewsAPI %s: %s", ticker, exc)
        return []

    if data.get("status") != "ok":
        log.warning("[WARN] NewsAPI %s: %s", ticker, data.get("message", "unknown"))
        return []

    results = []
    for art in data.get("articles", []):
        title = art.get("title") or ""
        desc = art.get("description") or ""
        text = title + " " + desc
        sentiment_score = _score_title(text)

        published_str = art.get("publishedAt", "")
        try:
            published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        except ValueError:
            published_at = datetime.now(timezone.utc)

        results.append(
            {
                "symbol": ticker,
                "published_at": published_at.isoformat(),
                "title": title,
                "url": art.get("url", ""),
                "source": (art.get("source") or {}).get("name", ""),
                "sentiment_score": round(sentiment_score, 4),
                "relevance_score": 0.7,  # fixed relevance for keyword search
            }
        )
    return results


def articles_to_daily_sentiment(articles: list[dict]) -> "pd.DataFrame":
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
    parser = argparse.ArgumentParser(description="NewsAPI.org news sentiment fetcher")
    parser.add_argument("--tickers", help="Comma-separated tickers")
    parser.add_argument("--days-back", type=int, default=7)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--delay", type=float, default=_DELAY)
    args = parser.parse_args(argv)

    api_key = _get_api_key()
    if not api_key or api_key.startswith("your_"):
        log.error("[ERROR] NEWSAPI_KEY not set. Get a free key at newsapi.org")
        return 1

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        try:
            from src.assembled_core.data.master_universe_loader import load_master_universe
            tickers, _ = load_master_universe()
            # NewsAPI: 100 calls/day → limit to top 80 by importance
            tickers = tickers[:80]
        except Exception:
            tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_articles: list[dict] = []
    total = len(tickers)
    log.info("[START] NewsAPI: %d tickers, %d days back", total, args.days_back)

    for i, ticker in enumerate(tickers, 1):
        articles = fetch_ticker_news(ticker, api_key, days_back=args.days_back,
                                      page_size=args.page_size)
        all_articles.extend(articles)
        if articles:
            log.info("  [%d/%d] %s: %d articles", i, total, ticker, len(articles))
        if i < total:
            time.sleep(args.delay)

    if not all_articles:
        log.warning("[WARN] No articles fetched")
        return 0

    raw_path = OUTPUT_DIR / f"articles_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    raw_path.write_text(json.dumps(all_articles, indent=2, default=str), encoding="utf-8")
    log.info("[OK] %d articles → %s", len(all_articles), raw_path)

    import pandas as pd
    daily = articles_to_daily_sentiment(all_articles)
    if not daily.empty:
        SENTIMENT_OUT.parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(SENTIMENT_OUT, index=False)
        log.info("[OK] Daily sentiment: %d rows, %d symbols → %s",
                 len(daily), daily["symbol"].nunique(), SENTIMENT_OUT)

    return 0


if __name__ == "__main__":
    sys.exit(main())
