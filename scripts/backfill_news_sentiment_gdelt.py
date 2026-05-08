"""Backfill news_sentiment_daily.parquet using free GDELT 2.0 API.

Downloads weekly average tone and article counts for each symbol from GDELT,
normalizes to match the existing Finnhub sentiment format, and merges into
output/news_sentiment_daily.parquet.

No API key required. Rate-limited internally.

Usage:
    python scripts/backfill_news_sentiment_gdelt.py \
        --start-date 2025-01-01 \
        --end-date 2025-12-21 \
        --output output/news_sentiment_daily.parquet

    # Dry run (print stats, don't save):
    python scripts/backfill_news_sentiment_gdelt.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ticker → company name mapping (used as GDELT search query)
# ---------------------------------------------------------------------------
TICKER_NAMES: dict[str, str] = {
    "ACN": "Accenture",
    "ADBE": "Adobe",
    "AMD": "AMD Advanced Micro Devices",
    "AMZN": "Amazon",
    "ANET": "Arista Networks",
    "ASML": "ASML",
    "AVGO": "Broadcom",
    "BAC": "Bank of America",
    "CAT": "Caterpillar",
    "COST": "Costco",
    "CVX": "Chevron",
    "DBX": "Dropbox",
    "DELL": "Dell Technologies",
    "DIS": "Disney",
    "GOOGL": "Google Alphabet",
    "GS": "Goldman Sachs",
    "HD": "Home Depot",
    "HON": "Honeywell",
    "INTC": "Intel",
    "JNJ": "Johnson Johnson",
    "JPM": "JPMorgan Chase",
    "KO": "Coca-Cola",
    "LLY": "Eli Lilly",
    "MCD": "McDonald's",
    "META": "Meta Facebook",
    "MSTR": "MicroStrategy",
    "MU": "Micron Technology",
    "NEE": "NextEra Energy",
    "NFLX": "Netflix",
    "NVDA": "Nvidia",
    "ORCL": "Oracle",
    "PG": "Procter Gamble",
    "PLTR": "Palantir",
    "QBTS": "D-Wave Quantum",
    "QCOM": "Qualcomm",
    "QMCO": "Quantum Corporation",
    "RTX": "Raytheon Technologies",
    "SMCI": "Super Micro Computer",
    "TSM": "TSMC Taiwan Semiconductor",
    "U": "Unity Software",
    "UNH": "UnitedHealth",
    "WFC": "Wells Fargo",
    "WMT": "Walmart",
    "XOM": "ExxonMobil",
}

GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
_REQUEST_DELAY = 3.0  # seconds between requests

# Positive/negative keywords for artlist mode sentiment scoring
_POS_WORDS = {
    "beat",
    "surge",
    "growth",
    "profit",
    "gain",
    "upgrade",
    "rally",
    "soar",
    "rise",
    "approved",
    "record",
    "strong",
    "bullish",
    "deal",
    "win",
}
_NEG_WORDS = {
    "miss",
    "fall",
    "slump",
    "loss",
    "cut",
    "layoff",
    "lawsuit",
    "downgrade",
    "drop",
    "decline",
    "warn",
    "recall",
    "probe",
    "investigation",
    "default",
    "bearish",
    "weak",
    "risk",
    "concern",
}


def _keyword_sentiment(text: str) -> float:
    t = text.lower()
    pos = sum(1 for w in _POS_WORDS if w in t)
    neg = sum(1 for w in _NEG_WORDS if w in t)
    if pos == neg:
        return 0.0
    return round(min(max((pos - neg) / max(pos + neg, 1), -1.0), 1.0), 4)


def _gdelt_url(query: str, mode: str, start: str, end: str) -> str:
    import urllib.parse

    params = {
        "query": query,
        "mode": mode,
        "startdatetime": start,
        "enddatetime": end,
        "timelinesmooth": "0",
        "format": "json",
    }
    return GDELT_BASE + "?" + urllib.parse.urlencode(params)


def _gdelt_artlist_url(query: str, start: str, end: str, max_records: int = 250) -> str:
    """Article-list API URL — returns article titles + tones, less rate-limited than timeline."""
    import urllib.parse

    params = {
        "query": query,
        "mode": "artlist",
        "startdatetime": start,
        "enddatetime": end,
        "maxrecords": str(min(max_records, 250)),
        "sort": "DateDesc",
        "format": "json",
    }
    return GDELT_BASE + "?" + urllib.parse.urlencode(params)


def fetch_gdelt_artlist(
    ticker: str,
    start_date: str,
    end_date: str,
    max_records: int = 250,
) -> pd.DataFrame:
    """Fetch article list from GDELT (artlist mode) — more reliable than timeline.

    Returns DataFrame with columns: timestamp, symbol, sentiment_score,
    sentiment_volume, count.
    """
    name = TICKER_NAMES.get(ticker, ticker)
    query = f'"{name}" sourcelang:eng'

    start_dt = start_date.replace("-", "") + "000000"
    end_dt = end_date.replace("-", "") + "235959"
    url = _gdelt_artlist_url(query, start_dt, end_dt, max_records)

    data = _fetch_json(url)
    time.sleep(_REQUEST_DELAY)

    if data is None:
        return pd.DataFrame()

    articles = data.get("articles", [])
    if not articles:
        return pd.DataFrame()

    rows = []
    for art in articles:
        date_str = art.get("seendate", "")
        title = art.get("title", "") or ""
        # GDELT tone: -100..+100, positive = positive sentiment
        tone_raw = art.get("tone", None)
        if tone_raw is not None:
            try:
                sentiment_score = float(np.tanh(float(tone_raw) / 5.0))
            except (TypeError, ValueError):
                sentiment_score = _keyword_sentiment(title)
        else:
            sentiment_score = _keyword_sentiment(title)

        try:
            # seendate format: YYYYMMDDTHHMMSSZ
            ts = pd.Timestamp(date_str[:8], tz="UTC")
        except Exception:
            continue

        rows.append(
            {
                "timestamp": ts,
                "symbol": ticker,
                "sentiment_score": round(sentiment_score, 4),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Aggregate to daily
    daily = (
        df.groupby(["timestamp", "symbol"])
        .agg(
            sentiment_score=("sentiment_score", "mean"),
            count=("sentiment_score", "count"),
        )
        .reset_index()
    )
    daily["sentiment_volume"] = np.log1p(daily["count"]).clip(1, 10).round(2)
    return daily[
        ["timestamp", "symbol", "sentiment_score", "sentiment_volume", "count"]
    ]


def _fetch_json(url: str, timeout: int = 30, max_retries: int = 3) -> dict | None:
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "research-sentiment-backfill/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                wait = (attempt + 1) * 10.0
                log.warning("Rate-limited (429). Waiting %.0fs...", wait)
                time.sleep(wait)
            else:
                log.debug("GDELT HTTP error: %s — %s", url, exc)
                return None
        except Exception as exc:
            log.debug("GDELT fetch failed: %s — %s", url, exc)
            return None
    log.warning("Max retries exceeded for URL: %s", url)
    return None


def fetch_gdelt_sentiment(
    ticker: str,
    start_date: str,  # YYYY-MM-DD
    end_date: str,
) -> pd.DataFrame:
    """Fetch weekly tone + count from GDELT for one ticker.

    Returns DataFrame with columns: timestamp, symbol, tone_raw, count_raw.
    May be empty if GDELT has no coverage.
    """
    name = TICKER_NAMES.get(ticker, ticker)
    query = f'"{name}" stock'

    # GDELT datetime format: YYYYMMDDHHMMSS
    start_dt = start_date.replace("-", "") + "000000"
    end_dt = end_date.replace("-", "") + "235959"

    rows = []
    for mode, col in [("timelinetoneavg", "tone"), ("timelinecountraw", "count")]:
        url = _gdelt_url(query, mode, start_dt, end_dt)
        data = _fetch_json(url)
        time.sleep(_REQUEST_DELAY)

        if data is None:
            continue

        timeline = data.get("timeline", [])
        if not timeline:
            continue

        # timeline is list of dicts: [{data: "...YYYYMMDDHHMMSS...", value: ...}]
        # GDELT returns each entry as {"data": "...", ...} or {"date": "..."}
        for entry in timeline:
            # Handle both key names GDELT uses
            date_str = entry.get("date") or entry.get("data", "")
            val = entry.get("value", 0.0)
            if len(date_str) >= 8:
                try:
                    ts = pd.Timestamp(date_str[:8], tz="UTC")
                    rows.append({"timestamp": ts, "symbol": ticker, col: float(val)})
                except Exception:
                    pass

    if not rows:
        return pd.DataFrame(columns=["timestamp", "symbol", "tone", "count"])

    df = pd.DataFrame(rows)
    # Pivot so tone and count are columns
    tone_df = df[df.columns.intersection(["timestamp", "symbol", "tone"])].copy()
    count_df = df[df.columns.intersection(["timestamp", "symbol", "count"])].copy()

    if "tone" in tone_df.columns and "count" in count_df.columns:
        merged = tone_df.merge(
            count_df.drop(columns="symbol"), on="timestamp", how="outer"
        )
    elif "tone" in tone_df.columns:
        merged = tone_df.rename(columns={"tone": "tone"})
        merged["count"] = 1.0
    else:
        return pd.DataFrame(columns=["timestamp", "symbol", "tone", "count"])

    merged["symbol"] = ticker
    return merged.sort_values("timestamp").reset_index(drop=True)


def normalize_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Convert GDELT tone (-10..+10 typical) to Finnhub format (-1..+1).

    Finnhub uses -1..+1 with 0 as neutral.
    GDELT tone is typically -5..+5 for corporate news.
    We use tanh(tone / 5) to compress to [-1, +1].
    """
    out = df.copy()
    tone_raw = pd.to_numeric(out.get("tone", 0), errors="coerce").fillna(0.0)
    # tanh(x/5): at x=5 → 0.96, at x=2.5 → 0.54, at x=-5 → -0.96
    out["sentiment_score"] = np.tanh(tone_raw / 5.0).round(4)

    # sentiment_volume: log-scaled count, clipped to 1..10
    count_raw = (
        pd.to_numeric(out.get("count", 1), errors="coerce").fillna(1.0).clip(lower=1)
    )
    out["sentiment_volume"] = np.log1p(count_raw).clip(1, 10).round(2)
    out["count"] = count_raw.round(0).astype(int)

    return out[["timestamp", "symbol", "sentiment_score", "sentiment_volume", "count"]]


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill weekly GDELT data to daily business-day resolution."""
    if df.empty:
        return df

    results = []
    for sym, grp in df.groupby("symbol"):
        grp = grp.sort_values("timestamp").drop_duplicates("timestamp")
        grp = grp.set_index("timestamp")

        # Resample to business days, forward-fill from weekly
        start = grp.index.min()
        end = grp.index.max()
        daily_idx = pd.date_range(start, end, freq="B", tz="UTC")
        daily = grp.reindex(daily_idx).ffill()
        daily.index.name = "timestamp"
        daily = daily.reset_index()
        daily["symbol"] = sym
        results.append(daily)

    if not results:
        return df
    return pd.concat(results, ignore_index=True)


def merge_with_existing(
    new_df: pd.DataFrame,
    existing_path: Path,
) -> pd.DataFrame:
    """Merge new GDELT rows with existing Finnhub rows.

    Finnhub rows take precedence (higher quality). Only add GDELT rows
    for dates not covered by Finnhub.
    """
    if not existing_path.exists():
        return new_df

    existing = pd.read_parquet(existing_path)
    existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)

    # Find date range already covered in existing
    existing_min = existing["timestamp"].min()

    # Only keep new rows BEFORE the existing data starts
    new_before = new_df[new_df["timestamp"] < existing_min].copy()
    log.info(
        "Existing: %d rows from %s | New GDELT: %d rows before %s",
        len(existing),
        existing_min.date(),
        len(new_before),
        existing_min.date(),
    )

    if new_before.empty:
        log.info(
            "No new rows to prepend — existing data already covers the full range."
        )
        return existing

    combined = pd.concat([new_before, existing], ignore_index=True)
    combined = combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return combined


def _load_panel_from_dir(panel_path: str, symbols: list[str]) -> pd.DataFrame:
    """Load per-symbol parquet files from a directory (yfinance layout)."""
    p = Path(panel_path)
    frames = []
    sym_set = set(symbols)
    for fpath in sorted(p.glob("*.parquet")):
        ticker = fpath.stem.upper()
        if sym_set and ticker not in sym_set:
            continue
        try:
            df = pd.read_parquet(fpath)
            # Normalize date column name
            if "date" in df.columns and "timestamp" not in df.columns:
                df = df.rename(columns={"date": "timestamp"})
            if "symbol" not in df.columns:
                df["symbol"] = ticker
            frames.append(df[["timestamp", "symbol", "close"]])
        except Exception:
            pass
    if not frames:
        return pd.DataFrame(columns=["timestamp", "symbol", "close"])
    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    return panel


def build_price_proxy_sentiment(
    panel_path: str,
    symbols: list[str],
    start_date: str,
    end_date: str,
    lookback: int = 5,
) -> pd.DataFrame:
    """Generate synthetic news sentiment from lagged price returns.

    This is a fallback when GDELT is unavailable. Uses tanh(5d_return / 5%)
    as a sentiment proxy — lagged by 1 day to avoid look-ahead.
    Clearly labelled as a synthetic proxy, not real news.

    Args:
        panel_path: Path to price panel parquet OR directory of per-symbol parquets.
        symbols: List of symbols to include.
        start_date: Start date YYYY-MM-DD.
        end_date: End date YYYY-MM-DD.
        lookback: Return lookback window in days.

    Returns:
        DataFrame in news_sentiment_daily format.
    """
    p = Path(panel_path)
    log.info("Building synthetic price-proxy sentiment from %s", panel_path)
    try:
        if p.is_dir():
            panel = _load_panel_from_dir(panel_path, symbols)
        else:
            panel = pd.read_parquet(panel_path)
            panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
            if "symbol" not in panel.columns and "date" in panel.columns:
                panel = panel.rename(columns={"date": "timestamp"})
    except Exception as exc:
        log.error("Cannot read panel: %s", exc)
        return pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "sentiment_score",
                "sentiment_volume",
                "count",
            ]
        )

    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")

    sym_set = set(symbols)
    if "symbol" in panel.columns:
        panel = panel[panel["symbol"].isin(sym_set)].copy()
    else:
        log.error("Panel has no 'symbol' column")
        return pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "sentiment_score",
                "sentiment_volume",
                "count",
            ]
        )

    rows = []
    for sym, grp in panel.groupby("symbol"):
        grp = grp.sort_values("timestamp").drop_duplicates("timestamp")
        grp = grp[grp["timestamp"] <= end_ts].copy()
        if "close" not in grp.columns or len(grp) < lookback + 2:
            continue
        closes = grp["close"].ffill()
        # Compute lookback-day return, shift by 1 to avoid look-ahead
        pct_ret = closes.pct_change(lookback).shift(1)
        grp = grp.copy()
        grp["_ret"] = pct_ret.values
        # Filter to start_date and beyond
        grp = grp[grp["timestamp"] >= start_ts]
        grp = grp[grp["_ret"].notna()]
        if grp.empty:
            continue
        sentiment = np.tanh(grp["_ret"].values / 0.05)  # tanh(ret/5%)
        volume = (1.0 + np.abs(grp["_ret"].values) * 20).clip(1, 10)
        for i, (ts, s, v) in enumerate(zip(grp["timestamp"], sentiment, volume)):
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "sentiment_score": round(float(s), 4),
                    "sentiment_volume": round(float(v), 2),
                    "count": 1,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "sentiment_score",
                "sentiment_volume",
                "count",
            ]
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    log.info(
        "Price proxy: %d rows, %d symbols, score range [%.3f, %.3f]",
        len(df),
        df["symbol"].nunique(),
        df["sentiment_score"].min(),
        df["sentiment_score"].max(),
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill news sentiment from GDELT")
    parser.add_argument(
        "--start-date", default="2025-01-01", help="Start date YYYY-MM-DD"
    )
    parser.add_argument("--end-date", default="2025-12-21", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--output",
        default="output/news_sentiment_daily.parquet",
        help="Output parquet path (merged with existing)",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Symbols to fetch (default: all 44 from TICKER_NAMES)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data but don't save, print stats only",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Seconds between GDELT requests (default: 3)",
    )
    parser.add_argument(
        "--mode",
        choices=["timeline", "artlist"],
        default="artlist",
        help=(
            "GDELT fetch mode: 'artlist' (default) returns article titles+tone, "
            "less rate-limited than 'timeline'. Use 'timeline' for weekly aggregates."
        ),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=250,
        help="Max articles per symbol in artlist mode (max 250)",
    )
    parser.add_argument(
        "--gdelt-out",
        default="output/news_sentiment_gdelt.parquet",
        help="Output path for GDELT-only parquet (before merging)",
    )
    parser.add_argument(
        "--use-price-proxy",
        action="store_true",
        help=(
            "Fallback: build synthetic sentiment from lagged price returns instead of GDELT. "
            "Requires --panel-path. Result is a momentum proxy, not real news sentiment."
        ),
    )
    parser.add_argument(
        "--panel-path",
        default="output/watchlist_ai_tech.parquet",
        help="Price panel parquet path for --use-price-proxy (default: output/watchlist_ai_tech.parquet)",
    )
    args = parser.parse_args()

    global _REQUEST_DELAY
    _REQUEST_DELAY = args.delay

    symbols = args.symbols or sorted(TICKER_NAMES.keys())
    output_path = Path(args.output)

    if args.use_price_proxy:
        # --- Price proxy mode ---
        new_df = build_price_proxy_sentiment(
            panel_path=args.panel_path,
            symbols=symbols,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        if new_df.empty:
            log.error("Price proxy returned no data. Check --panel-path.")
            sys.exit(1)
    else:
        # --- GDELT mode ---
        mode = getattr(args, "mode", "artlist")
        log.info(
            "Fetching GDELT (%s mode) for %d symbols, %s to %s",
            mode,
            len(symbols),
            args.start_date,
            args.end_date,
        )

        all_frames = []
        for i, ticker in enumerate(symbols, 1):
            log.info("[%d/%d] %s ...", i, len(symbols), ticker)
            try:
                if mode == "artlist":
                    daily = fetch_gdelt_artlist(
                        ticker,
                        args.start_date,
                        args.end_date,
                        max_records=getattr(args, "max_records", 250),
                    )
                    if daily.empty:
                        log.info("  no data for %s", ticker)
                        continue
                    log.info("  %d daily rows", len(daily))
                    all_frames.append(daily)
                else:
                    raw = fetch_gdelt_sentiment(ticker, args.start_date, args.end_date)
                    if raw.empty:
                        log.info("  no data for %s", ticker)
                        continue
                    normalized = normalize_sentiment(raw)
                    daily = resample_to_daily(normalized)
                    log.info("  %d weekly -> %d daily rows", len(raw), len(daily))
                    all_frames.append(daily)
            except Exception as exc:
                log.warning("  failed for %s: %s", ticker, exc)

        if not all_frames:
            log.error("No data fetched from GDELT. Try --use-price-proxy as fallback.")
            sys.exit(1)

        new_df = pd.concat(all_frames, ignore_index=True)

        # Save GDELT-only output for fusion
        gdelt_out = Path(
            getattr(args, "gdelt_out", "output/news_sentiment_gdelt.parquet")
        )
        gdelt_out.parent.mkdir(parents=True, exist_ok=True)
        new_df.to_parquet(gdelt_out, index=False)
        log.info("[OK] GDELT standalone: %d rows -> %s", len(new_df), gdelt_out)
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], utc=True)
        new_df = new_df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    log.info(
        "GDELT total: %d rows, %d symbols, %s to %s",
        len(new_df),
        new_df["symbol"].nunique(),
        new_df["timestamp"].min().date(),
        new_df["timestamp"].max().date(),
    )

    if args.dry_run:
        log.info("DRY RUN — not saving. Sample:")
        print(new_df.head(20).to_string())
        return

    merged = merge_with_existing(new_df, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    log.info(
        "Saved merged parquet: %d rows (%s to %s) → %s",
        len(merged),
        merged["timestamp"].min().date(),
        merged["timestamp"].max().date(),
        output_path,
    )


if __name__ == "__main__":
    main()
