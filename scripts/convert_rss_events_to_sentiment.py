"""Bridge: RSS events JSON → news_sentiment_daily.parquet.

Reads output/intel/news/events_latest.json (produced by run_news_worker.py),
runs the news_classifier inline to extract tickers + sentiment, and appends
to output/news_sentiment_rss.parquet.

This bridges the gap between the intel pipeline (events_latest.json) and
the ML pipeline (news_sentiment_daily.parquet).

Usage:
    python scripts/convert_rss_events_to_sentiment.py
    python scripts/convert_rss_events_to_sentiment.py --events-file output/intel/news/events_latest.json
    python scripts/convert_rss_events_to_sentiment.py --also-run-fetch
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

EVENTS_FILE = ROOT / "output" / "intel" / "news" / "events_latest.json"
SENTIMENT_OUT = ROOT / "output" / "news_sentiment_rss.parquet"

# Positive/negative keyword scorer (fallback when classifier has no sentiment)
_POS = {"beat", "growth", "profit", "upgrade", "rally", "soar", "rise", "gain", "approved"}
_NEG = {"miss", "loss", "fall", "drop", "recall", "probe", "layoff", "cut", "decline", "warn"}


def _keyword_sentiment(text: str) -> float:
    t = text.lower()
    pos = sum(1 for w in _POS if w in t)
    neg = sum(1 for w in _NEG if w in t)
    if pos == neg:
        return 0.0
    return round(min(max((pos - neg) / max(pos + neg, 1), -1.0), 1.0), 4)


def _load_events(events_file: Path) -> list[dict]:
    if not events_file.exists():
        log.warning("[WARN] events file not found: %s", events_file)
        return []
    try:
        data = json.loads(events_file.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return data.get("items", data.get("events", []))
    except Exception as exc:
        log.error("[ERROR] loading events: %s", exc)
        return []


def _classify_and_extract(events: list[dict]) -> list[dict]:
    """Run classifier on events; extract (symbol, published_at, sentiment_score)."""
    try:
        from src.assembled_core.intel.news_classifier import classify_news_event
        classifier_available = True
    except ImportError:
        classifier_available = False
        log.warning("[WARN] news_classifier not available — using keyword scoring only")

    results: list[dict] = []

    for evt in events:
        title = evt.get("title", "") or evt.get("summary", "") or ""
        if not title:
            continue

        published_str = (
            evt.get("published_utc") or evt.get("published_at") or evt.get("fetched_utc", "")
        )
        try:
            published_at = datetime.fromisoformat(
                published_str.replace("Z", "+00:00") if published_str else ""
            )
        except ValueError:
            published_at = datetime.now(timezone.utc)

        # Extract tickers from event or run classifier
        tickers: list[str] = []
        sentiment_score = 0.0

        pre_tickers = evt.get("tickers") or evt.get("entities", []) or []
        if isinstance(pre_tickers, list):
            tickers = [t for t in pre_tickers if isinstance(t, str) and len(t) <= 6]

        if classifier_available:
            try:
                clf = classify_news_event(
                    title,
                    geo_tags=evt.get("countries", []) or [],
                    source_tier="T2",
                    tickers=tickers,
                )
                tickers = tickers or clf.affected_assets
                # Derive sentiment from market_direction
                direction = clf.market_direction.lower()
                if direction == "bullish":
                    sentiment_score = 0.5 + clf.confidence * 0.5
                elif direction == "bearish":
                    sentiment_score = -(0.5 + clf.confidence * 0.5)
                else:
                    sentiment_score = _keyword_sentiment(title)
            except Exception:
                sentiment_score = _keyword_sentiment(title)
        else:
            sentiment_score = _keyword_sentiment(title)

        if not tickers:
            # No tickers resolved — skip (can't map to symbol)
            continue

        for sym in tickers:
            results.append(
                {
                    "symbol": sym.upper(),
                    "published_at": published_at.isoformat(),
                    "sentiment_score": round(float(sentiment_score), 4),
                    "title": title[:200],
                }
            )

    return results


def results_to_daily(results: list[dict]) -> "pd.DataFrame":
    import pandas as pd

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
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
    parser = argparse.ArgumentParser(description="RSS events → sentiment parquet bridge")
    parser.add_argument("--events-file", default=str(EVENTS_FILE))
    parser.add_argument(
        "--also-run-fetch",
        action="store_true",
        help="Run run_news_worker.py --once before converting",
    )
    parser.add_argument("--out", default=str(SENTIMENT_OUT))
    args = parser.parse_args(argv)

    if args.also_run_fetch:
        log.info("[START] Running news worker to refresh events...")
        result = subprocess.run(
            [sys.executable, "scripts/run_news_worker.py", "--once"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=ROOT,
        )
        if result.returncode != 0:
            log.warning("[WARN] news worker exited %d: %s", result.returncode, result.stderr[:300])
        else:
            log.info("[OK] news worker done")

    events_file = Path(args.events_file)
    events = _load_events(events_file)
    log.info("[START] Loaded %d events from %s", len(events), events_file)

    if not events:
        log.warning("[WARN] No events to process")
        return 0

    results = _classify_and_extract(events)
    log.info("[OK] Extracted %d symbol-article rows from %d events", len(results), len(events))

    import pandas as pd
    daily = results_to_daily(results)

    if daily.empty:
        log.warning("[WARN] No daily sentiment rows produced (no tickers resolved)")
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out_path, index=False)
    log.info(
        "[OK] RSS sentiment: %d rows, %d symbols → %s",
        len(daily), daily["symbol"].nunique(), out_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
