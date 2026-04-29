# scripts/commands/run_daily.py
"""run_daily subcommand: daily EOD pipeline."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.assembled_core.config.settings import get_runtime_profile
from src.assembled_core.logging_config import generate_run_id, setup_logging


def run_daily_subcommand(args: argparse.Namespace) -> int:
    """Run daily EOD pipeline subcommand."""
    run_id = generate_run_id(prefix="eod")
    setup_logging(run_id=run_id, level="INFO")
    logger = logging.getLogger(__name__)

    profile = get_runtime_profile(
        profile=getattr(args, "profile", None), env_var="ASSEMBLED_RUNTIME_PROFILE"
    )

    logger.info("=" * 60)
    logger.info("EOD Pipeline (run_daily)")
    logger.info(f"Run-ID: {run_id}")
    logger.info(f"Runtime Profile: {profile.value}")
    logger.info("=" * 60)

    experiment_run = None

    if getattr(args, "track_experiment", False):
        if not getattr(args, "experiment_name", None):
            logger.error("--experiment-name is required when --track-experiment is set")
            return 1

        from src.assembled_core.config.settings import get_settings
        from src.assembled_core.qa.experiment_tracking import ExperimentTracker

        settings = get_settings()
        tracker = ExperimentTracker(settings.experiments_dir)
        tags = (
            getattr(args, "experiment_tags", "").split(",")
            if getattr(args, "experiment_tags", None)
            else []
        )
        tags = [t.strip() for t in tags if t.strip()]

        config = {
            "freq": args.freq,
            "data_source": getattr(args, "data_source", None) or settings.data_source,
            "start_capital": getattr(args, "start_capital", 10000.0),
            "symbols": getattr(args, "symbols", None),
            "start_date": getattr(args, "start_date", None),
            "end_date": getattr(args, "end_date", None),
        }

        experiment_run = tracker.start_run(
            name=args.experiment_name, config=config, tags=tags
        )

        logger.info("")
        logger.info("Experiment Tracking: ENABLED")
        logger.info(f"  Run-ID: {experiment_run.run_id}")
        logger.info(f"  Name: {experiment_run.name}")
        logger.info(
            f"  Tags: {', '.join(experiment_run.tags) if experiment_run.tags else 'none'}"
        )
        logger.info(
            f"  Run Directory: {settings.experiments_dir / experiment_run.run_id}"
        )
        logger.info("")

    from scripts.run_eod_pipeline import run_eod_from_args

    try:
        manifest = run_eod_from_args(args)

        if experiment_run and manifest:
            from src.assembled_core.config.settings import get_settings
            from src.assembled_core.qa.experiment_tracking import ExperimentTracker

            settings = get_settings()
            tracker = ExperimentTracker(settings.experiments_dir)

            metrics_dict = {}
            if manifest.get("qa_metrics"):
                qa_metrics = manifest["qa_metrics"]
                if isinstance(qa_metrics, dict):
                    metrics_dict.update(
                        {
                            "total_return": qa_metrics.get("total_return", 0.0),
                            "cagr": qa_metrics.get("cagr", 0.0),
                            "sharpe_ratio": qa_metrics.get("sharpe_ratio", 0.0),
                            "max_drawdown_pct": qa_metrics.get("max_drawdown_pct", 0.0),
                            "total_trades": qa_metrics.get("total_trades", 0),
                        }
                    )

            if metrics_dict:
                tracker.log_metrics(experiment_run, metrics_dict)
                logger.info(f"Logged metrics to experiment run {experiment_run.run_id}")

            if manifest.get("qa_report_path"):
                report_path = settings.output_dir / manifest["qa_report_path"]
                if report_path.exists():
                    tracker.log_artifact(experiment_run, report_path, "qa_report.md")
                    logger.info(
                        f"Logged QA report as artifact to experiment run {experiment_run.run_id}"
                    )

            final_status = "finished" if not manifest.get("failure") else "failed"
            tracker.finish_run(experiment_run, status=final_status)
            logger.info(
                f"Experiment run {experiment_run.run_id} finished with status '{final_status}'."
            )

        return 0
    except RuntimeError:
        if experiment_run:
            from src.assembled_core.config.settings import get_settings
            from src.assembled_core.qa.experiment_tracking import ExperimentTracker

            settings = get_settings()
            tracker = ExperimentTracker(settings.experiments_dir)
            tracker.finish_run(experiment_run, status="failed")
        return 1
    except Exception as e:
        logger.error(f"EOD pipeline failed: {e}", exc_info=True)
        if experiment_run:
            from src.assembled_core.config.settings import get_settings
            from src.assembled_core.qa.experiment_tracking import ExperimentTracker

            settings = get_settings()
            tracker = ExperimentTracker(settings.experiments_dir)
            tracker.finish_run(experiment_run, status="failed")
        return 1


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the run_daily subcommand."""
    daily_parser = subparsers.add_parser(
        "run_daily",
        help="Run daily EOD pipeline (execute, backtest, portfolio, QA)",
        description="Runs the full EOD pipeline: execute, backtest, portfolio simulation, and QA checks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/cli.py run_daily --freq 1d
  python scripts/cli.py run_daily --freq 1d --universe watchlist.txt --start-capital 50000
  python scripts/cli.py run_daily --freq 5min --price-file data/sample/eod_sample.parquet
        """,
    )
    daily_parser.add_argument(
        "--freq",
        type=str,
        required=True,
        choices=["1d", "5min"],
        help="Trading frequency: '1d' for daily or '5min' for 5-minute bars",
    )
    daily_parser.add_argument(
        "--universe",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to universe file (default: watchlist.txt in repo root)",
    )
    daily_parser.add_argument(
        "--price-file",
        type=str,
        default=None,
        metavar="FILE",
        help="Optional explicit path to price file (overrides default path)",
    )
    daily_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Start date for price data filtering (YYYY-MM-DD or 'today', optional)",
    )
    daily_parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="End date for price data filtering (YYYY-MM-DD or 'today', optional). Use 'today' for live data.",
    )
    daily_parser.add_argument(
        "--data-source",
        type=str,
        choices=["local", "yahoo"],
        default=None,
        help="Data source type: 'local' (Parquet files) or 'yahoo' (Yahoo Finance API). Default: from settings.data_source",
    )
    daily_parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        metavar="SYMBOL",
        help="List of symbols to load (e.g., --symbols AAPL MSFT GOOGL). Overrides universe file.",
    )
    daily_parser.add_argument(
        "--start-capital",
        type=float,
        default=10000.0,
        metavar="AMOUNT",
        help="Starting capital in USD (default: 10000.0)",
    )
    daily_parser.add_argument(
        "--skip-backtest", action="store_true", help="Skip backtest step in pipeline"
    )
    daily_parser.add_argument(
        "--skip-portfolio", action="store_true", help="Skip portfolio simulation step"
    )
    daily_parser.add_argument(
        "--skip-qa", action="store_true", help="Skip QA checks step"
    )
    daily_parser.add_argument(
        "--commission-bps",
        type=float,
        default=None,
        metavar="BPS",
        help="Commission in basis points (overrides default cost model)",
    )
    daily_parser.add_argument(
        "--spread-w",
        type=float,
        default=None,
        metavar="WEIGHT",
        help="Spread weight for cost model (overrides default)",
    )
    daily_parser.add_argument(
        "--impact-w",
        type=float,
        default=None,
        metavar="WEIGHT",
        help="Market impact weight for cost model (overrides default)",
    )
    daily_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory (default: from config.OUTPUT_DIR)",
    )
    daily_parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["BACKTEST", "PAPER", "DEV"],
        metavar="PROFILE",
        help="Runtime profile: BACKTEST (offline), PAPER (simulated), or DEV (development, default)",
    )
    daily_parser.add_argument(
        "--track-experiment",
        action="store_true",
        default=False,
        help="Enable experiment tracking (stores run config, metrics, and artifacts)",
    )
    daily_parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        metavar="NAME",
        help="Name for the experiment run (required if --track-experiment is set)",
    )
    daily_parser.add_argument(
        "--experiment-tags",
        type=str,
        default=None,
        metavar="TAGS",
        help="Comma-separated tags for the experiment (e.g., 'daily,live,yahoo')",
    )
    daily_parser.set_defaults(func=run_daily_subcommand)
