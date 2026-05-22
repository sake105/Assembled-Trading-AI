"""Fuse news sentiment from all sources into unified news_sentiment_daily.parquet.

Sources (priority order, highest wins in conflict):
  1. Finnhub      — output/news_sentiment_finnhub.parquet (or existing daily)
  2. Alpha Vantage — output/news_sentiment_alphavantage.parquet
  3. Polygon      — output/news_sentiment_polygon.parquet
  4. NewsAPI      — output/news_sentiment_newsapi.parquet
  5. RSS/Intel    — output/news_sentiment_rss.parquet
  6. GDELT        — output/news_sentiment_gdelt.parquet

For each (date, symbol) with multiple sources:
  - sentiment_score: weighted average (weights by source priority + count)
  - sentiment_volume: sum
  - count: sum
  - source: pipe-delimited list of contributing sources

Usage:
    python scripts/fuse_news_sentiment.py
    python scripts/fuse_news_sentiment.py --dry-run
    python scripts/fuse_news_sentiment.py --out output/news_sentiment_fused.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Source registry: (path, weight, label)
_SOURCES: list[tuple[Path, float, str]] = [
    (ROOT / "output" / "news_sentiment_daily.parquet", 1.0, "finnhub"),
    (ROOT / "output" / "news_sentiment_alphavantage.parquet", 0.9, "alphavantage"),
    (ROOT / "output" / "news_sentiment_polygon.parquet", 0.8, "polygon"),
    (ROOT / "output" / "news_sentiment_newsapi.parquet", 0.7, "newsapi"),
    (ROOT / "output" / "news_sentiment_rss.parquet", 0.6, "rss"),
    (ROOT / "output" / "news_sentiment_gdelt.parquet", 0.5, "gdelt"),
]

FUSED_OUT = ROOT / "output" / "news_sentiment_fused.parquet"


def _load_source(path: Path, label: str) -> "pd.DataFrame | None":  # noqa: F821
    import pandas as pd

    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        log.warning("[WARN] %s: load failed: %s", label, exc)
        return None

    # Normalize columns
    col_map = {
        "date": "timestamp",
        "published_at": "timestamp",
        "ticker": "symbol",
    }
    df = df.rename(columns=col_map)

    required = {"timestamp", "symbol", "sentiment_score"}
    if not required.issubset(df.columns):
        log.warning("[WARN] %s: missing cols %s", label, required - set(df.columns))
        return None

    if "sentiment_volume" not in df.columns:
        df["sentiment_volume"] = df.get("count", 1.0)
    if "count" not in df.columns:
        df["count"] = 1

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "symbol", "sentiment_score"])
    df["timestamp"] = df["timestamp"].dt.normalize()
    df["symbol"] = df["symbol"].str.upper()
    df["_source"] = label
    return df[
        [
            "timestamp",
            "symbol",
            "sentiment_score",
            "sentiment_volume",
            "count",
            "_source",
        ]
    ]


def fuse(dry_run: bool = False, out_path: Path = FUSED_OUT) -> "pd.DataFrame":  # noqa: F821
    import pandas as pd

    frames: list[pd.DataFrame] = []
    for path, weight, label in _SOURCES:
        df = _load_source(path, label)
        if df is not None:
            df["_weight"] = weight
            frames.append(df)
            log.info(
                "  [loaded] %s: %d rows, %d symbols",
                label,
                len(df),
                df["symbol"].nunique(),
            )

    if not frames:
        log.warning("[WARN] No source files found")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Weighted average sentiment per (timestamp, symbol)
    combined["_weighted_score"] = (
        combined["sentiment_score"] * combined["_weight"] * combined["count"]
    )
    combined["_total_weight"] = combined["_weight"] * combined["count"]

    agg = (
        combined.groupby(["timestamp", "symbol"])
        .agg(
            _weighted_score_sum=("_weighted_score", "sum"),
            _total_weight_sum=("_total_weight", "sum"),
            sentiment_volume=("sentiment_volume", "sum"),
            count=("count", "sum"),
            sources=("_source", lambda x: "|".join(sorted(set(x)))),
        )
        .reset_index()
    )

    agg["sentiment_score"] = (
        agg["_weighted_score_sum"] / agg["_total_weight_sum"].clip(lower=1e-9)
    ).round(4)
    agg = agg.drop(columns=["_weighted_score_sum", "_total_weight_sum"])
    agg = agg.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    log.info(
        "[OK] Fused: %d rows, %d symbols, %d dates",
        len(agg),
        agg["symbol"].nunique(),
        agg["timestamp"].nunique(),
    )

    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_parquet(out_path, index=False)
        log.info("[OK] Written → %s", out_path)

    return agg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fuse news sentiment from all sources")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default=str(FUSED_OUT))
    parser.add_argument(
        "--update-primary",
        action="store_true",
        help="Overwrite output/news_sentiment_daily.parquet with fused result",
    )
    args = parser.parse_args(argv)

    log.info("[START] Fusing news sentiment sources")
    for path, _, label in _SOURCES:
        status = "[found]" if path.exists() else "[missing]"
        log.info("  %s %s: %s", status, label, path.name)

    fused = fuse(dry_run=args.dry_run, out_path=Path(args.out))

    if not fused.empty and args.update_primary and not args.dry_run:
        primary = ROOT / "output" / "news_sentiment_daily.parquet"
        fused.drop(columns=["sources"], errors="ignore").to_parquet(
            primary, index=False
        )
        log.info("[OK] Primary updated → %s (%d rows)", primary, len(fused))

    return 0 if not fused.empty else 1


if __name__ == "__main__":
    sys.exit(main())
