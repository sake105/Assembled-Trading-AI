"""News coverage report — shows feed count by tier and focus category.

Usage:
    python scripts/news_coverage_report.py
    python scripts/news_coverage_report.py --feeds configs/intel/rss_feeds.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

_ECONOMIC_FOCUSES = {
    "economic",
    "financial_economic",
    "central_bank_policy",
    "economic_policy",
    "financial_alternative",
    "earnings_corporate",
    "earnings_corporate_filings",
    "technology_industry",
    "technology_business",
    "technology_ai",
    "technology_semiconductors",
    "technology_science",
    "energy_commodities",
    "energy_renewables",
    "energy_natural_gas",
    "energy_statistics",
    "mining_commodities",
}

_GEO_FOCUSES = {
    "geopolitical",
    "geopolitical_alternative",
    "geopolitical_analysis",
    "middle_east_geopolitical",
    "ukraine_russia_conflict",
    "conflict_military",
    "asia_geopolitical",
    "china_geopolitical",
    "south_asia_geopolitical",
    "africa_geopolitical",
    "political",
    "political_economic",
    "national_security_legal",
    "defense_military",
    "defense_technology",
    "defense_strategy",
    "shipping_maritime",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="News feed coverage report")
    parser.add_argument("--feeds", default="configs/intel/rss_feeds.yaml")
    args = parser.parse_args(argv)

    feeds_path = ROOT / args.feeds
    if not feeds_path.exists():
        print(f"[ERROR] Not found: {feeds_path}")
        return 1

    with feeds_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    feeds: list[dict] = data.get("feeds", [])
    total = len(feeds)
    enabled = [f for f in feeds if f.get("enabled", False)]

    by_tier: dict[str, int] = {}
    econ_count = 0
    geo_count = 0
    other_count = 0

    for f in enabled:
        tier = f.get("tier", "?")
        by_tier[tier] = by_tier.get(tier, 0) + 1
        focus = f.get("focus", "")
        if focus in _ECONOMIC_FOCUSES:
            econ_count += 1
        elif focus in _GEO_FOCUSES:
            geo_count += 1
        else:
            other_count += 1

    enabled_total = len(enabled)
    print(f"\n{'='*50}")
    print("NEWS COVERAGE REPORT")
    print(f"{'='*50}")
    print(f"Total feeds registered : {total}")
    print(f"Enabled                : {enabled_total}")
    print(f"Disabled               : {total - enabled_total}")
    print()
    print("By tier (enabled only):")
    for tier in sorted(by_tier):
        print(f"  {tier}: {by_tier[tier]}")
    print()
    print("By focus category (enabled only):")
    print(f"  Economic/Financial : {econ_count}")
    print(f"  Geopolitical       : {geo_count}")
    print(f"  Other              : {other_count}")
    print()

    if geo_count > 0 and econ_count > 0:
        ratio = econ_count / geo_count
        print(f"Econ:Geo ratio: {ratio:.2f}:1  (target: >=0.625:1, i.e. 1:1.6)")
        if ratio >= 0.625:
            print("[OK] Economic coverage meets Wave-3 target (>= 1:1.6 ratio)")
        else:
            print("[WARN] Geopolitical feeds dominate -- add more economic feeds")
    else:
        print(f"[INFO] econ={econ_count}, geo={geo_count}")

    print(f"{'='*50}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
