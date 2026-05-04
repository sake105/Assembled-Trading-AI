# scripts/commands/backtest.py
"""Backtest subcommands: run_backtest, batch_backtest, batch_run, leaderboard."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.assembled_core.config.settings import RuntimeProfile
from src.assembled_core.logging_config import generate_run_id, setup_logging


def run_backtest_subcommand(args: argparse.Namespace) -> int:
    """Run strategy backtest subcommand."""
    run_id = generate_run_id(prefix="backtest")
    setup_logging(run_id=run_id, level="INFO")
    logger = logging.getLogger(__name__)

    profile = RuntimeProfile.BACKTEST

    logger.info("=" * 60)
    logger.info("Strategy Backtest (run_backtest)")
    logger.info(f"Run-ID: {run_id}")
    logger.info(f"Runtime Profile: {profile.value}")
    logger.info("=" * 60)

    from scripts.run_backtest_strategy import run_backtest_from_args

    experiment_run = None

    try:
        if args.track_experiment:
            if not args.experiment_name:
                logger.error(
                    "--experiment-name is required when --track-experiment is set"
                )
                return 1

            from src.assembled_core.config.settings import get_settings
            from src.assembled_core.qa.experiment_tracking import ExperimentTracker

            settings = get_settings()
            tracker = ExperimentTracker(settings.experiments_dir)
            tags = args.experiment_tags.split(",") if args.experiment_tags else []
            tags = [t.strip() for t in tags if t.strip()]

            config = {
                "freq": args.freq,
                "strategy": args.strategy,
                "start_capital": args.start_capital,
                "with_costs": args.with_costs,
                "use_meta_model": args.use_meta_model,
                "meta_model_path": (
                    str(args.meta_model_path) if args.meta_model_path else None
                ),
                "meta_ensemble_mode": args.meta_ensemble_mode,
                "meta_min_confidence": args.meta_min_confidence,
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

        if args.use_meta_model:
            logger.info("")
            logger.info("Meta-Model Ensemble: ENABLED")
            logger.info(f"  Model Path: {args.meta_model_path}")
            logger.info(f"  Min Confidence: {args.meta_min_confidence}")
            logger.info(f"  Mode: {args.meta_ensemble_mode}")
        else:
            logger.info("")
            logger.info(
                "Meta-Model Ensemble: DISABLED (use --use-meta-model to enable)"
            )

        return run_backtest_from_args(args)
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return 1


def batch_run_subcommand(args: argparse.Namespace) -> int:
    """Run batch backtests with resume support (MVP)."""
    from scripts.batch_runner import _setup_logging, load_batch_config, run_batch

    logger = logging.getLogger(__name__)

    verbosity = args.verbose if hasattr(args, "verbose") else 0
    _setup_logging(verbosity)

    try:
        batch_cfg = load_batch_config(args.config_file)
    except Exception as exc:
        logger.error("Failed to load batch config: %s", exc, exc_info=True)
        return 1

    if args.output_root is not None:
        batch_cfg.output_root = args.output_root.resolve()

    try:
        max_workers = args.max_workers if hasattr(args, "max_workers") else 1
        if max_workers < 1:
            logger.error("max_workers must be >= 1")
            return 1

        return run_batch(
            batch_cfg,
            max_workers=max_workers,
            dry_run=args.dry_run if hasattr(args, "dry_run") else False,
            resume=args.resume if hasattr(args, "resume") else False,
            rerun_failed=args.rerun_failed if hasattr(args, "rerun_failed") else False,
        )
    except Exception as exc:
        logger.error("Batch execution failed: %s", exc, exc_info=True)
        return 1


def batch_backtest_subcommand(args: argparse.Namespace) -> int:
    """Run batch of strategy backtests from config file (blessed entry point)."""
    return batch_run_subcommand(args)


def leaderboard_subcommand(args: argparse.Namespace) -> int:
    """Rank and display best runs from batch backtest results."""
    from scripts.leaderboard import (
        export_leaderboard_json,
        format_leaderboard_table,
        load_batch_summary,
        rank_runs,
    )

    logger = logging.getLogger(__name__)

    if not args.batch_output.exists():
        logger.error("Batch output directory does not exist: %s", args.batch_output)
        return 1

    if not args.batch_output.is_dir():
        logger.error("Batch output path is not a directory: %s", args.batch_output)
        return 1

    try:
        df = load_batch_summary(args.batch_output)
    except FileNotFoundError as exc:
        logger.error("Failed to load batch summary: %s", exc)
        return 1
    except ValueError as exc:
        logger.error("Invalid batch summary: %s", exc)
        return 1

    try:
        ranked_df = rank_runs(df, sort_by=args.sort_by, top_k=args.top_k)
    except ValueError as exc:
        logger.error("Failed to rank runs: %s", exc)
        return 1

    table_str = format_leaderboard_table(ranked_df, args.sort_by)
    print(f"\nTop {len(ranked_df)} runs (sorted by {args.sort_by}):\n")
    print(table_str)

    if args.json:
        try:
            export_leaderboard_json(ranked_df, args.json)
            logger.info("Leaderboard exported to %s", args.json)
        except Exception as exc:
            logger.error("Failed to export JSON: %s", exc, exc_info=True)
            return 1

    if args.export_best:
        try:
            from scripts.leaderboard import export_best_run_config_yaml

            export_best_run_config_yaml(
                df,
                sort_by=args.sort_by,
                output_path=args.export_best,
                batch_output_dir=args.batch_output,
            )
            logger.info("Best run config exported to %s", args.export_best)
        except ValueError as exc:
            logger.error("Failed to export best run config: %s", exc)
            return 1
        except RuntimeError as exc:
            logger.error("Error: %s", exc)
            return 1
        except Exception as exc:
            logger.error("Failed to export best run config: %s", exc, exc_info=True)
            return 1

    return 0


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register all backtest subcommands."""
    # run_backtest
    backtest_parser = subparsers.add_parser(
        "run_backtest",
        help="Run strategy backtest",
        description="Runs a strategy backtest using the portfolio-level backtest engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/cli.py run_backtest --freq 1d --universe watchlist.txt
  python scripts/cli.py run_backtest --freq 1d --price-file data/sample/eod_sample.parquet --generate-report
  python scripts/cli.py run_backtest --freq 5min --start-capital 50000 --no-costs
        """,
    )
    backtest_parser.add_argument(
        "--freq",
        type=str,
        required=True,
        choices=["1d", "5min"],
        help="Trading frequency: '1d' for daily or '5min' for 5-minute bars",
    )
    backtest_parser.add_argument(
        "--price-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Explicit path to price file (overrides default path)",
    )
    backtest_parser.add_argument(
        "--universe",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to universe file (default: watchlist.txt in repo root)",
    )
    backtest_parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        metavar="SYMBOL",
        help="List of symbols to load (e.g., --symbols NVDA AAPL MSFT). Priority: --symbols > --symbols-file > --universe.",
    )
    backtest_parser.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to text file with one symbol per line (e.g., config/universe_ai_tech_tickers.txt). "
        "Priority: --symbols > --symbols-file > --universe.",
    )
    backtest_parser.add_argument(
        "--data-source",
        type=str,
        choices=["local", "yahoo"],
        default=None,
        help="Data source type: 'local' (Parquet files) or 'yahoo' (Yahoo Finance API). Default: from settings.data_source",
    )
    backtest_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Start date for data loading (default: use all available data)",
    )
    backtest_parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="End date for data loading (default: use all available data)",
    )
    backtest_parser.add_argument(
        "--strategy",
        type=str,
        default="trend_baseline",
        choices=["trend_baseline", "event_insider_shipping", "multifactor_long_short"],
        metavar="NAME",
        help="Strategy: trend_baseline | event_insider_shipping | multifactor_long_short",
    )
    backtest_parser.add_argument(
        "--bundle-path",
        type=str,
        default=None,
        dest="bundle_path",
        metavar="FILE",
        help="Factor bundle YAML (required for multifactor_long_short)",
    )
    backtest_parser.add_argument(
        "--top-quantile",
        type=float,
        default=0.2,
        dest="top_quantile",
        help="Top quantile for long positions (default: 0.2)",
    )
    backtest_parser.add_argument(
        "--bottom-quantile",
        type=float,
        default=0.2,
        dest="bottom_quantile",
        help="Bottom quantile for short positions (default: 0.2)",
    )
    backtest_parser.add_argument(
        "--max-gross-exposure",
        type=float,
        default=1.5,
        dest="max_gross_exposure",
        help="Max gross exposure (default: 1.5)",
    )
    backtest_parser.add_argument(
        "--use-regime-overlay",
        action="store_true",
        default=False,
        dest="use_regime_overlay",
        help="Enable regime overlay for multifactor strategy",
    )
    backtest_parser.add_argument(
        "--regime-config-file",
        type=str,
        default=None,
        dest="regime_config_file",
        help="Path to regime config JSON/YAML",
    )
    backtest_parser.add_argument(
        "--start-capital",
        type=float,
        default=10000.0,
        metavar="AMOUNT",
        help="Starting capital in USD (default: 10000.0)",
    )
    backtest_parser.add_argument(
        "--with-costs",
        action="store_true",
        default=True,
        help="Include transaction costs in backtest (default: True)",
    )
    backtest_parser.add_argument(
        "--no-costs",
        action="store_false",
        dest="with_costs",
        help="Disable transaction costs (use cost-free simulation)",
    )
    backtest_parser.add_argument(
        "--commission-bps",
        type=float,
        default=None,
        metavar="BPS",
        help="Commission in basis points (overrides default cost model)",
    )
    backtest_parser.add_argument(
        "--spread-w",
        type=float,
        default=None,
        metavar="WEIGHT",
        help="Spread weight for cost model (overrides default)",
    )
    backtest_parser.add_argument(
        "--impact-w",
        type=float,
        default=None,
        metavar="WEIGHT",
        help="Market impact weight for cost model (overrides default)",
    )
    backtest_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory (default: from config.OUTPUT_DIR)",
    )
    backtest_parser.add_argument(
        "--generate-report",
        action="store_true",
        default=False,
        help="Generate QA report after backtest",
    )
    backtest_parser.add_argument(
        "--use-meta-model",
        action="store_true",
        default=False,
        help="Enable meta-model ensemble (filter signals by confidence score)",
    )
    backtest_parser.add_argument(
        "--meta-model-path",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to trained meta-model file (required if --use-meta-model is set)",
    )
    backtest_parser.add_argument(
        "--meta-min-confidence",
        type=float,
        default=0.5,
        metavar="THRESHOLD",
        help="Minimum confidence threshold for meta-model filter (default: 0.5)",
    )
    backtest_parser.add_argument(
        "--meta-ensemble-mode",
        type=str,
        choices=["filter", "scaling"],
        default="filter",
        help="Meta-model ensemble mode: 'filter' (remove low-confidence signals) or 'scaling' (scale positions by confidence, default: 'filter')",
    )
    backtest_parser.add_argument(
        "--track-experiment",
        action="store_true",
        default=False,
        help="Enable experiment tracking (stores run config, metrics, and artifacts)",
    )
    backtest_parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        metavar="NAME",
        help="Name for the experiment run (required if --track-experiment is set)",
    )
    backtest_parser.add_argument(
        "--experiment-tags",
        type=str,
        default=None,
        metavar="TAGS",
        help="Comma-separated tags for the experiment (e.g., 'trend,baseline,ma20_50')",
    )
    backtest_parser.add_argument(
        "--no-ledger",
        action="store_true",
        default=False,
        help="Skip ledger/accounting output (faster, for quick checks)",
    )
    backtest_parser.add_argument(
        "--no-qa-gate",
        action="store_true",
        default=False,
        dest="no_qa_gate",
        help="Override QA data-quality gate (for research/backtesting with imperfect data)",
    )
    backtest_parser.add_argument(
        "--rebalance-freq",
        type=str,
        default="1d",
        dest="rebalance_freq",
        metavar="FREQ",
        help="Rebalance frequency (default: 1d)",
    )
    backtest_parser.add_argument(
        "--broker-snapshot-policy",
        type=str,
        default="never",
        dest="broker_snapshot_policy",
        help="When to write broker snapshots: never|on_rebal|always (default: never)",
    )
    backtest_parser.add_argument(
        "--write-broker-snapshot",
        action="store_true",
        default=False,
        dest="write_broker_snapshot",
        help="Write broker snapshot after backtest",
    )
    backtest_parser.set_defaults(func=run_backtest_subcommand)

    # batch_backtest (blessed entry point)
    batch_parser = subparsers.add_parser(
        "batch_backtest",
        help="Run batch of strategy backtests from config file (blessed entry point)",
        description="Runs multiple backtests from a YAML config file with resume/rerun support. "
        "This is the recommended entry point (alias for batch_run). "
        "Each run gets a deterministic run_id based on parameters for reproducibility.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run batch from config
  python scripts/cli.py batch_backtest --config-file configs/batch_example.yaml

  # Parallel execution with 4 workers
  python scripts/cli.py batch_backtest --config-file configs/batch_example.yaml --max-workers 4

  # Resume from previous run (skip successful runs)
  python scripts/cli.py batch_backtest --config-file configs/batch_example.yaml --resume

  # Resume and rerun failed runs
  python scripts/cli.py batch_backtest --config-file configs/batch_example.yaml --resume --rerun-failed

  # Dry-run (show plan without execution)
  python scripts/cli.py batch_backtest --config-file configs/batch_example.yaml --dry-run

See docs/BATCH_RUNNER_P4.md for detailed documentation.
        """,
    )
    batch_parser.add_argument(
        "--config-file",
        type=Path,
        required=True,
        help="Path to YAML config file",
    )
    batch_parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output_root from config",
    )
    batch_parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of parallel workers (1 = serial execution, default: 1)",
    )
    batch_parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs that already succeeded (resume from previous run)",
    )
    batch_parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="Rerun failed runs even with --resume (default: skip failed runs)",
    )
    batch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without executing backtests",
    )
    batch_parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times)",
    )
    batch_parser.set_defaults(func=batch_backtest_subcommand)

    # batch_run (MVP runner)
    batch_run_parser = subparsers.add_parser(
        "batch_run",
        help="Run batch backtests with resume support (MVP)",
        description="Runs multiple backtests from a YAML config file with resume/rerun support. "
        "Each run gets a deterministic run_id based on parameters for reproducibility.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run batch from config
  python scripts/cli.py batch_run --config-file configs/batch_example.yaml

  # Parallel execution with 4 workers
  python scripts/cli.py batch_run --config-file configs/batch_example.yaml --max-workers 4

  # Resume from previous run (skip successful runs)
  python scripts/cli.py batch_run --config-file configs/batch_example.yaml --resume

  # Resume and rerun failed runs
  python scripts/cli.py batch_run --config-file configs/batch_example.yaml --resume --rerun-failed

  # Dry-run (show plan without execution)
  python scripts/cli.py batch_run --config-file configs/batch_example.yaml --dry-run

See docs/BATCH_RUNNER_P4.md for detailed documentation.
        """,
    )
    batch_run_parser.add_argument(
        "--config-file",
        type=Path,
        required=True,
        help="Path to YAML config file",
    )
    batch_run_parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output_root from config",
    )
    batch_run_parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of parallel workers (1 = serial execution, default: 1)",
    )
    batch_run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs that already succeeded (resume from previous run)",
    )
    batch_run_parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="Rerun failed runs even with --resume (default: skip failed runs)",
    )
    batch_run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without executing backtests",
    )
    batch_run_parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times)",
    )
    batch_run_parser.set_defaults(func=batch_run_subcommand)

    # leaderboard
    leaderboard_parser = subparsers.add_parser(
        "leaderboard",
        help="Rank and display best runs from batch backtest results",
        description="Reads summary.csv from batch output directory and ranks runs by various metrics (Sharpe, total return, final PF, etc.).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Top 10 runs by Sharpe ratio
  python scripts/cli.py leaderboard --batch-output output/batch_backtests/my_batch --sort-by sharpe --top-k 10

  # Top 20 runs by total return
  python scripts/cli.py leaderboard --batch-output output/batch_backtests/my_batch --sort-by total_return --top-k 20

  # Top runs by final PF with JSON export
  python scripts/cli.py leaderboard --batch-output output/batch_backtests/my_batch --sort-by final_pf --top-k 10 --json leaderboard.json

See scripts/leaderboard.py for more details.
        """,
    )
    leaderboard_parser.add_argument(
        "--batch-output",
        type=Path,
        required=True,
        metavar="DIR",
        help="Path to batch output directory (contains summary.csv)",
    )
    leaderboard_parser.add_argument(
        "--sort-by",
        type=str,
        default="sharpe",
        choices=[
            "sharpe",
            "total_return",
            "final_pf",
            "max_drawdown_pct",
            "cagr",
            "trades",
        ],
        help="Metric to sort by (default: sharpe)",
    )
    leaderboard_parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        metavar="N",
        help="Number of top runs to display (default: 20)",
    )
    leaderboard_parser.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional: Export leaderboard to JSON file",
    )
    leaderboard_parser.add_argument(
        "--export-best",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional: Export best run configuration as YAML file (for reproducible reruns)",
    )
    leaderboard_parser.set_defaults(func=leaderboard_subcommand)
