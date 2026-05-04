#!/usr/bin/env python3
"""Standalone RSS feed fetcher — fetch all enabled feeds and save to JSON.

Usage:
    python scripts/run_rss_fetch.py                     # fetch all enabled feeds
    python scripts/run_rss_fetch.py --tier T1           # T1 only
    python scripts/run_rss_fetch.py --focus energy      # by focus keyword
    python scripts/run_rss_fetch.py --feed reuters_world # single feed
    python scripts/run_rss_fetch.py --dry-run           # print summary, no file write

Output: data/intel/rss_events_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.assembled_core.intel.models import SourceTier  # noqa: E402
from src.assembled_core.intel.news_entity_mapper import SimpleEntityLinker  # noqa: E402
from src.assembled_core.intel.rss_fetcher import RSSFetcher  # noqa: E402

logger = logging.getLogger(__name__)

_OUTPUT_DIR = _REPO_ROOT / "data" / "intel"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assembled-Trading-AI standalone RSS fetcher"
    )
    parser.add_argument(
        "--tier", choices=["T0", "T1", "T2", "T3"], help="Fetch only this tier"
    )
    parser.add_argument("--focus", help="Fetch feeds whose focus contains this keyword")
    parser.add_argument("--feed", help="Fetch a single feed by id")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print summary, do not write files"
    )
    parser.add_argument(
        "--no-skip-seen", action="store_true", help="Include already-seen entries"
    )
    parser.add_argument("--output-dir", default=str(_OUTPUT_DIR))
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    fetcher = RSSFetcher(
        timeout=15,
        retries=2,
        entity_linker=SimpleEntityLinker(),
    )
    skip = not args.no_skip_seen

    if args.feed:
        events = fetcher.fetch_feed(args.feed, skip_seen=skip)
    elif args.tier:
        tier = SourceTier(args.tier)
        events = fetcher.fetch_by_tier(tier, skip_seen=skip)
    elif args.focus:
        events = fetcher.fetch_by_focus(args.focus, skip_seen=skip)
    else:
        events = fetcher.fetch_all(skip_seen=skip)

    logger.info("[OK] Fetched %d events", len(events))

    # Summary by tier
    tier_counts: dict[str, int] = {}
    for e in events:
        tier_counts[e.source_tier.value] = tier_counts.get(e.source_tier.value, 0) + 1
    for tier, cnt in sorted(tier_counts.items()):
        logger.info("  %s: %d events", tier, cnt)

    if args.dry_run:
        logger.info("[SKIP] dry-run: no files written")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"rss_events_{ts}.json"

    payload = {
        "fetched_utc": ts,
        "total_events": len(events),
        "tier_breakdown": tier_counts,
        "events": [e.model_dump(mode="json") for e in events],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("[OK] Written to %s", out_path)


if __name__ == "__main__":
    main()
