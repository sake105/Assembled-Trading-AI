"""Daily Decision Log — Item 103.

Reads from the audit trail (``output/audit/trading_decisions.jsonl``) and
produces a daily ``decisions_YYYY-MM-DD.json`` in ``output/decisions/``.

Each daily file contains:
- Date of the decision session.
- Top-5 most-active symbols by number of decisions.
- Per-symbol summary: max/mean signal score, EDCL trigger count, sizing-cap-hit count.
- Top factors mentioned across all reasoning dicts (if provided by callers).

Usage::

    # Produce today's decision log:
    python scripts/daily_decision_log.py

    # Produce log for a specific date:
    python scripts/daily_decision_log.py --date 2026-05-07

    # Dry run — print to stdout, don't write file:
    python scripts/daily_decision_log.py --dry-run

Output format example::

    {
      "date": "2026-05-07",
      "total_decisions": 12,
      "top_symbols": [
        {
          "symbol": "NVDA",
          "n_decisions": 3,
          "max_signal_score": 0.91,
          "mean_signal_score": 0.87,
          "edcl_trigger_count": 2,
          "sizing_cap_hit_count": 0,
          "top_factors": ["momentum", "news_sentiment"]
        },
        ...
      ],
      "global_top_factors": ["momentum", "news_sentiment", "macro"],
      "generated_at": "2026-05-07T08:15:00+00:00"
    }

If the audit trail is empty or does not exist, the file is still written with
``total_decisions: 0`` so downstream consumers have a stable contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.ops.audit_trail import read_decisions  # noqa: E402

OUTPUT_DIR = ROOT / "output" / "decisions"


def build_daily_summary(date_str: str) -> dict:
    """Build the daily decision summary for *date_str* (YYYY-MM-DD)."""
    records = read_decisions(date_str=date_str)

    # Group by symbol
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_symbol[r.get("symbol", "UNKNOWN")].append(r)

    # Collect global factor counts
    factor_counter: Counter = Counter()
    for r in records:
        reasoning = r.get("reasoning") or {}
        for factor in reasoning.get("top_factors", []):
            factor_counter[factor] += 1

    # Build per-symbol summaries, sorted by decision count desc
    symbol_summaries = []
    for sym, recs in sorted(by_symbol.items(), key=lambda kv: -len(kv[1])):
        scores = [r.get("signal_score", 0.0) for r in recs]
        sym_factors: Counter = Counter()
        for r in recs:
            for f in (r.get("reasoning") or {}).get("top_factors", []):
                sym_factors[f] += 1

        symbol_summaries.append(
            {
                "symbol": sym,
                "n_decisions": len(recs),
                "max_signal_score": round(max(scores), 4) if scores else 0.0,
                "mean_signal_score": (
                    round(sum(scores) / len(scores), 4) if scores else 0.0
                ),
                "edcl_trigger_count": sum(1 for r in recs if r.get("edcl_trigger")),
                "sizing_cap_hit_count": sum(1 for r in recs if r.get("sizing_cap_hit")),
                "top_factors": [f for f, _ in sym_factors.most_common(5)],
            }
        )

    return {
        "date": date_str,
        "total_decisions": len(records),
        "top_symbols": symbol_summaries[:5],  # top-5 by activity
        "global_top_factors": [f for f, _ in factor_counter.most_common(10)],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Daily trading decision log generator")
    parser.add_argument(
        "--date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date to generate log for (YYYY-MM-DD, default: today UTC)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print JSON to stdout instead of writing file",
    )
    args = parser.parse_args(argv)

    summary = build_daily_summary(args.date)

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"decisions_{args.date}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[daily_decision_log] Written: {out_path} ({summary['total_decisions']} decisions)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
