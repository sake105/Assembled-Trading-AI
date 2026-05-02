"""Event study script for news sentiment validation (Level B).

Usage:
  python scripts/news_validation/level_b_event_study.py \
      --events tests/news_gold/events_labeled.jsonl \
      --prices data/prices/spy_and_universe.parquet \
      --market SPY

Output:
  docs/validation/news_event_study_<date>.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from assembled_core.signals.news_validation import (
    car_significance_report,
    event_study,
    gate_summary,
    news_feature_production_ready,
)


def load_events(path: Path) -> pd.DataFrame:
    """Load events JSONL: {ticker, event_date, sentiment_label}."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows)


def load_prices(path: Path, market_ticker: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load parquet price file. Returns (returns_df, market_returns)."""
    prices = pd.read_parquet(path)
    if "date" in prices.columns:
        prices = prices.set_index("date")
    prices.index = pd.to_datetime(prices.index)
    returns_df = prices.pct_change(fill_method=None).dropna(how="all")
    if market_ticker not in returns_df.columns:
        raise ValueError(f"Market ticker {market_ticker!r} not found in prices file")
    market_returns = returns_df[market_ticker]
    return returns_df, market_returns


def build_report(significance: dict, events_df: pd.DataFrame) -> str:
    lines = [
        f"# News Event Study — {date.today()}",
        "",
        f"Events analysed: {len(events_df)}",
        "",
        "## CAR by Sentiment Label",
        "",
        "| Label | N | Mean CAR | Median CAR | T-stat | p-value | Significant |",
        "|-------|---|----------|------------|--------|---------|-------------|",
    ]
    for label, stats in sorted(significance.items()):
        sig = "YES" if stats["significant_5pct"] else "no"
        lines.append(
            f"| {label} | {stats['n']} | {stats['mean_car']*100:+.3f}% "
            f"| {stats['median_car']*100:+.3f}% "
            f"| {stats.get('t_stat', 'n/a')} "
            f"| {stats.get('p_value', 'n/a')} "
            f"| {sig} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--prices", required=True)
    parser.add_argument("--market", default="SPY")
    parser.add_argument("--out-dir", default="docs/validation")
    args = parser.parse_args()

    events_df = load_events(Path(args.events))
    returns_df, market_returns = load_prices(Path(args.prices), args.market)

    es_df = event_study(events_df, returns_df, market_returns)
    significance = car_significance_report(es_df)

    # Build validation_results dict for production gate
    pos_stats = significance.get("positive", {})
    neg_stats = significance.get("negative", {})
    worst_p = max(
        pos_stats.get("p_value") or 1.0,
        neg_stats.get("p_value") or 1.0,
    )
    car_magnitude_bps = min(
        abs(pos_stats.get("mean_car", 0)) * 10_000,
        abs(neg_stats.get("mean_car", 0)) * 10_000,
    )

    validation_results = {
        "level_b_car_significance_p": worst_p,
        "level_b_car_magnitude_bps": car_magnitude_bps,
    }

    all_passed, per_criterion = news_feature_production_ready(
        "finbert_sentiment", validation_results
    )

    report = build_report(significance, es_df)
    report += "\n\n## Production Gate (Level B only)\n\n"
    report += "```\n" + gate_summary("finbert_sentiment", all_passed, per_criterion) + "\n```\n"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"news_event_study_{date.today()}.md"
    out_file.write_text(report, encoding="utf-8")
    print(f"Report written to {out_file}")


if __name__ == "__main__":
    main()
