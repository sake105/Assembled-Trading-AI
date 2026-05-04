# scripts/commands/news.py
"""News and disclosures pipeline subcommands."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run_news_pipeline_subcommand(args: argparse.Namespace) -> int:
    """Run NEWS v1 pipeline and emit artifacts under output/intel/news/."""
    from src.assembled_core.events.news import run_news_pipeline

    logger = logging.getLogger(__name__)

    result = run_news_pipeline(
        sources_path=args.sources,
        news_path=args.news,
        cadence=args.cadence,
    )
    health = result.get("health")
    status = getattr(health, "status", None)
    logger.info("NEWS v1 pipeline finished with status=%s", status)
    if status == "ERROR":
        return 1
    if status == "DEGRADED":
        logger.warning(
            "NEWS pipeline health is DEGRADED; check failures/notes in artifacts."
        )
    return 0


def run_disclosures_pipeline_subcommand(args: argparse.Namespace) -> int:
    """Run disclosures pipeline; exit 1 on ERROR health."""
    from src.assembled_core.events.disclosures import run_disclosures_pipeline

    logger = logging.getLogger(__name__)

    result = run_disclosures_pipeline(
        sources_path=args.sources,
        disclosures_path=args.disclosures,
        cadence=args.cadence,
    )
    health = result.get("health")
    status = getattr(health, "status", None)
    logger.info("Disclosures pipeline finished with status=%s", status)
    if status == "ERROR":
        return 1
    if status == "DEGRADED":
        logger.warning(
            "Disclosures pipeline health is DEGRADED; check failures/notes in artifacts."
        )
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register news and disclosures subcommands."""
    # run_news_pipeline
    news_parser = subparsers.add_parser(
        "run_news_pipeline",
        help="Run NEWS v1 pipeline (fetch -> normalize -> dedupe -> health -> emit)",
        description=(
            "Fetch news from configured sources (RSS + optional GDELT), normalize and "
            "dedupe events, compute health status, and write JSON artifacts under "
            "output/intel/news/ (events_latest.json, health_latest.json, clusters_latest.json, "
            "triggers_latest.json). No trading impact in this sprint."
        ),
    )
    news_parser.add_argument(
        "--sources",
        type=Path,
        default=Path("configs/news/sources.yaml"),
        help="Path to news sources YAML config (default: configs/news/sources.yaml)",
    )
    news_parser.add_argument(
        "--news",
        type=Path,
        default=Path("configs/news/news.yaml"),
        help="Path to news parameter YAML config (default: configs/news/news.yaml)",
    )
    news_parser.add_argument(
        "--cadence",
        type=str,
        default="hourly",
        choices=["hourly", "daily"],
        help="Cadence label for this run (default: hourly)",
    )
    news_parser.set_defaults(func=run_news_pipeline_subcommand)

    # run_disclosures_pipeline
    disclosures_parser = subparsers.add_parser(
        "run_disclosures_pipeline",
        help="Run disclosures pipeline (House PTR / SEC EDGAR stubs -> normalize -> dedupe -> emit)",
        description=(
            "Fetch disclosures from configured sources (House PTR, SEC EDGAR — stubs in v0), "
            "normalize, dedupe, compute health, and write JSON under output/intel/disclosures/. "
            "Exit code 1 on ERROR health, else 0."
        ),
    )
    disclosures_parser.add_argument(
        "--sources",
        type=Path,
        default=Path("configs/disclosures/sources.yaml"),
        help="Path to disclosures sources YAML (default: configs/disclosures/sources.yaml)",
    )
    disclosures_parser.add_argument(
        "--disclosures",
        type=Path,
        default=Path("configs/disclosures/disclosures.yaml"),
        help="Path to disclosures params YAML (default: configs/disclosures/disclosures.yaml)",
    )
    disclosures_parser.add_argument(
        "--cadence",
        type=str,
        default="hourly",
        choices=["hourly", "daily"],
        help="Cadence label (default: hourly)",
    )
    disclosures_parser.set_defaults(func=run_disclosures_pipeline_subcommand)

    # run_news (alias for run_news_pipeline)
    run_news_parser = subparsers.add_parser(
        "run_news",
        help="Alias for run_news_pipeline (NEWS v1 pipeline)",
        description="Alias for run_news_pipeline; accepts the same arguments.",
    )
    run_news_parser.add_argument(
        "--sources",
        type=Path,
        default=Path("configs/news/sources.yaml"),
        help="Path to news sources YAML config (default: configs/news/sources.yaml)",
    )
    run_news_parser.add_argument(
        "--news",
        type=Path,
        default=Path("configs/news/news.yaml"),
        help="Path to news parameter YAML config (default: configs/news/news.yaml)",
    )
    run_news_parser.add_argument(
        "--cadence",
        type=str,
        default="hourly",
        choices=["hourly", "daily"],
        help="Cadence label for this run (default: hourly)",
    )
    run_news_parser.set_defaults(func=run_news_pipeline_subcommand)
