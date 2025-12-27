# scripts/cli.py
"""Central CLI for Assembled Trading AI Backend.

This script provides a unified command-line interface for the most important backend operations:
- run_daily: Daily EOD pipeline
- run_backtest: Strategy backtest
- run_phase4_tests: Phase-4 test suite
- info: Show project information

Usage:
    python scripts/cli.py run_daily --freq 1d
    python scripts/cli.py run_backtest --freq 1d --universe watchlist.txt
    python scripts/cli.py run_phase4_tests
    python scripts/cli.py info
    python scripts/cli.py --version
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Import core modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.config.settings import RuntimeProfile, get_runtime_profile
from src.assembled_core.logging_config import generate_run_id, setup_logging

# Initialize logging with Run-ID at module level
# Individual subcommands may override with their own Run-ID
_run_id = generate_run_id(prefix="cli")
setup_logging(run_id=_run_id, level="INFO")
import logging

logger = logging.getLogger(__name__)

# Project version (from pyproject.toml / __init__.py)
__version__ = "0.0.1"


def get_git_branch() -> str | None:
    """Try to get current git branch.

    Returns:
        Branch name if git is available, None otherwise
    """
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return None


def print_version() -> None:
    """Print version and project information."""
    branch = get_git_branch()
    print("Assembled Trading AI - Central CLI")
    print(f"Version: {__version__}")
    if branch:
        print(f"Git Branch: {branch}")
    print("Status: Phase-4/Phase-6 ready")
    print(f"Python: {sys.version.split()[0]}")


def info_subcommand(args: argparse.Namespace) -> int:
    """Show project information subcommand.

    Args:
        args: Parsed command-line arguments (unused)

    Returns:
        Exit code (always 0)
    """
    print("=" * 60)
    print("Assembled Trading AI - Project Information")
    print("=" * 60)
    print()
    print("Main Subcommands:")
    print(
        "  run_daily          - Run daily EOD pipeline (execute, backtest, portfolio, QA)"
    )
    print("  run_backtest       - Run strategy backtest with portfolio-level engine")
    print("  batch_backtest     - Run batch of strategy backtests from config file")
    print(
        "  leaderboard        - Rank and display best runs from batch backtest results"
    )
    print(
        "  walk_forward       - Run walk-forward analysis (out-of-sample validation, research tool)"
    )
    print(
        "  risk_report        - Generate comprehensive risk report from backtest results"
    )
    print(
        "  tca_report         - Generate transaction cost analysis (TCA) report from backtest results"
    )
    print("  factor_report      - Generate factor analysis report (IC/IR statistics)")
    print(
        "  analyze_factors    - Comprehensive factor analysis (IC + Portfolio evaluation)"
    )
    print(
        "  ml_validate_factors - ML validation on factor panels (predict forward returns)"
    )
    print(
        "  ml_model_zoo       - Compare multiple ML models on factor panels (model zoo)"
    )
    print(
        "  check_health       - Check backend health status (read-only, operations monitoring)"
    )
    print("  run_phase4_tests   - Run Phase-4 regression test suite (~13s, 110 tests)")
    print("  info               - Show this information")
    print()
    print("Documentation:")
    print("  - Backend Architecture: docs/ARCHITECTURE_BACKEND.md")
    print("  - Legacy Overview: docs/LEGACY_OVERVIEW.md")
    print("  - Legacy Mapping: docs/LEGACY_TO_CORE_MAPPING.md")
    print("  - PowerShell Wrappers: docs/POWERSHELL_WRAPPERS.md")
    print("  - Testing Commands: docs/TESTING_COMMANDS.md")
    print()
    print("Examples:")
    print("  python scripts/cli.py run_daily --freq 1d")
    print(
        "  python scripts/cli.py run_backtest --freq 1d --universe watchlist.txt --generate-report"
    )
    print("  python scripts/cli.py run_phase4_tests --verbose --durations 5")
    print()
    print("For detailed help on a subcommand:")
    print("  python scripts/cli.py <subcommand> --help")
    print()
    return 0


def run_daily_subcommand(args: argparse.Namespace) -> int:
    """Run daily EOD pipeline subcommand.

    Args:
        args: Parsed command-line arguments for run_daily

    Returns:
        Exit code (0 for success, 1 for failure)

    Note:
        Orders generated by this command are currently written to SAFE-CSV files.
        If orders are later routed to the Paper-Trading-API (Phase 10), the source
        field should be set to "CLI_EOD" to identify the origin.
    """
    # Generate Run-ID for this execution
    run_id = generate_run_id(prefix="eod")
    setup_logging(run_id=run_id, level="INFO")
    logger = logging.getLogger(__name__)

    # Determine runtime profile (from CLI arg or default: DEV)
    profile = get_runtime_profile(
        profile=getattr(args, "profile", None), env_var="ASSEMBLED_RUNTIME_PROFILE"
    )

    logger.info("=" * 60)
    logger.info("EOD Pipeline (run_daily)")
    logger.info(f"Run-ID: {run_id}")
    logger.info(f"Runtime Profile: {profile.value}")
    logger.info("=" * 60)

    # Initialize experiment_run to None (will be set if tracking is enabled)
    experiment_run = None

    # Setup experiment tracking if enabled
    if getattr(args, "track_experiment", False):
        if not getattr(args, "experiment_name", None):
            logger.error("--experiment-name is required when --track-experiment is set")
            return 1

        from src.assembled_core.qa.experiment_tracking import ExperimentTracker
        from src.assembled_core.config.settings import get_settings

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

    # Import here to avoid circular imports
    from scripts.run_eod_pipeline import run_eod_from_args

    try:
        manifest = run_eod_from_args(args)

        # Log metrics to experiment tracking if enabled
        if experiment_run and manifest:
            from src.assembled_core.qa.experiment_tracking import ExperimentTracker
            from src.assembled_core.config.settings import get_settings

            settings = get_settings()
            tracker = ExperimentTracker(settings.experiments_dir)

            # Log pipeline metrics from manifest
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

            # Log QA report as artifact if available
            if manifest.get("qa_report_path"):
                report_path = settings.output_dir / manifest["qa_report_path"]
                if report_path.exists():
                    tracker.log_artifact(experiment_run, report_path, "qa_report.md")
                    logger.info(
                        f"Logged QA report as artifact to experiment run {experiment_run.run_id}"
                    )

            # Determine final status
            final_status = "finished" if not manifest.get("failure") else "failed"

            # Finish run
            tracker.finish_run(experiment_run, status=final_status)
            logger.info(
                f"Experiment run {experiment_run.run_id} finished with status '{final_status}'."
            )

        return 0
    except RuntimeError:
        # Expected error from run_eod_from_args when pipeline fails
        if experiment_run:
            from src.assembled_core.qa.experiment_tracking import ExperimentTracker
            from src.assembled_core.config.settings import get_settings

            settings = get_settings()
            tracker = ExperimentTracker(settings.experiments_dir)
            tracker.finish_run(experiment_run, status="failed")
        return 1
    except Exception as e:
        logger.error(f"EOD pipeline failed: {e}", exc_info=True)
        if experiment_run:
            from src.assembled_core.qa.experiment_tracking import ExperimentTracker
            from src.assembled_core.config.settings import get_settings

            settings = get_settings()
            tracker = ExperimentTracker(settings.experiments_dir)
            tracker.finish_run(experiment_run, status="failed")
        return 1


def build_ml_dataset_subcommand(args: argparse.Namespace) -> int:
    """Build ML dataset from backtest results subcommand.

    Args:
        args: Parsed command-line arguments for build_ml_dataset

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Generate Run-ID for this execution
    run_id = generate_run_id(prefix="ml_dataset")
    setup_logging(run_id=run_id, level="INFO")
    logger = logging.getLogger(__name__)

    # ML dataset building always uses BACKTEST profile
    profile = RuntimeProfile.BACKTEST

    logger.info("=" * 60)
    logger.info("ML Dataset Builder (build_ml_dataset)")
    logger.info(f"Run-ID: {run_id}")
    logger.info(f"Runtime Profile: {profile.value}")
    logger.info("=" * 60)

    try:
        from src.assembled_core.config import OUTPUT_DIR
        from src.assembled_core.qa.dataset_builder import build_ml_dataset_from_backtest

        # Set output path
        if args.out:
            output_path = Path(args.out)
        else:
            output_dir = OUTPUT_DIR / "ml_datasets"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{args.strategy}_{args.freq}.parquet"

        logger.info(f"Strategy: {args.strategy}")
        logger.info(f"Frequency: {args.freq}")
        logger.info(f"Label Horizon: {args.label_horizon_days} days")
        logger.info(f"Success Threshold: {args.success_threshold:.2%}")
        logger.info(f"Label Type: {args.label_type}")
        logger.info(f"Output Path: {output_path}")
        logger.info(f"Output Format: {args.format}")

        # Determine if we should use the new build_ml_dataset_for_strategy() function
        # Use it if start-date/end-date are provided OR if symbols are provided
        use_new_builder = (
            args.start_date is not None
            or args.end_date is not None
            or args.symbols is not None
        )

        if use_new_builder:
            # Use new high-level builder function
            logger.info("")
            logger.info(
                "Using strategy-based dataset builder (build_ml_dataset_for_strategy)..."
            )

            from src.assembled_core.qa.dataset_builder import (
                build_ml_dataset_for_strategy,
                export_ml_dataset,
            )

            # Determine date range
            if args.start_date is None:
                # Use earliest available date (will be determined from data)
                args.start_date = (
                    "2020-01-01"  # Fallback, will be filtered by available data
                )
            if args.end_date is None:
                # Use latest available date
                args.end_date = (
                    "2099-12-31"  # Fallback, will be filtered by available data
                )

            # Determine universe
            universe_list = None
            if args.symbols:
                universe_list = args.symbols
            elif args.universe:
                # Read universe file
                universe_list = [
                    line.strip().upper()
                    for line in open(args.universe, "r").readlines()
                    if line.strip()
                ]

            # Build label params
            label_params = {
                "horizon_days": args.label_horizon_days,
                "threshold_pct": args.success_threshold,
                "label_type": args.label_type,
            }

            # Build dataset
            ml_dataset = build_ml_dataset_for_strategy(
                strategy_name=args.strategy,
                start_date=args.start_date,
                end_date=args.end_date,
                universe=universe_list,
                universe_file=args.universe if not args.symbols else None,
                label_params=label_params,
                price_file=args.price_file,
                freq=args.freq,
            )

            if ml_dataset.empty:
                logger.error("ML dataset is empty")
                return 1

            logger.info(
                f"ML dataset built: {len(ml_dataset)} records, {len(ml_dataset.columns)} columns"
            )

            # Count labels
            if "label" in ml_dataset.columns:
                label_counts = ml_dataset["label"].value_counts()
                logger.info(f"Label distribution: {dict(label_counts)}")

            # Export dataset
            logger.info("")
            logger.info("Exporting ML dataset...")
            export_ml_dataset(ml_dataset, output_path, format=args.format)

        else:
            # Use existing backtest-based builder (for backward compatibility)
            logger.info("")
            logger.info(
                "Using backtest-based dataset builder (build_ml_dataset_from_backtest)..."
            )

            # Run backtest and get prices_with_features and trades
            logger.info("Running backtest...")
            prices_with_features, trades = _run_backtest_for_ml_dataset(
                strategy=args.strategy,
                freq=args.freq,
                price_file=args.price_file,
                universe=args.universe,
                start_capital=args.start_capital,
                with_costs=args.with_costs,
                output_dir=OUTPUT_DIR,
            )

            if prices_with_features.empty:
                logger.error("No price data with features available")
                return 1

            if trades.empty:
                logger.error("No trades generated from backtest")
                return 1

            logger.info(
                f"Prices with features: {len(prices_with_features)} rows, {prices_with_features['symbol'].nunique()} symbols"
            )
            logger.info(f"Trades: {len(trades)} trades")

            # Build ML dataset
            logger.info("")
            logger.info("Building ML dataset...")
            ml_dataset = build_ml_dataset_from_backtest(
                prices_with_features=prices_with_features,
                trades=trades,
                label_horizon_days=args.label_horizon_days,
                success_threshold=args.success_threshold,
                feature_prefixes=("ta_", "insider_", "congress_", "shipping_", "news_"),
            )

            if ml_dataset.empty:
                logger.error("ML dataset is empty")
                return 1

            logger.info(
                f"ML dataset built: {len(ml_dataset)} records, {len(ml_dataset.columns)} columns"
            )

            # Count labels
            if "label" in ml_dataset.columns:
                label_counts = ml_dataset["label"].value_counts()
                logger.info(f"Label distribution: {dict(label_counts)}")

            # Save dataset (use export_ml_dataset for format support)
            logger.info("")
            logger.info("Saving ML dataset...")
            from src.assembled_core.qa.dataset_builder import export_ml_dataset

            export_ml_dataset(ml_dataset, output_path, format=args.format)

        logger.info("")
        logger.info("=" * 60)
        logger.info("ML Dataset Build Complete")
        logger.info("=" * 60)
        logger.info(f"Output: {output_path}")
        logger.info(f"Records: {len(ml_dataset)}")
        logger.info(
            f"Features: {len([c for c in ml_dataset.columns if c not in ['label', 'open_time', 'symbol', 'open_price', 'close_time', 'pnl_pct', 'horizon_days']])}"
        )
        logger.info("=" * 60)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


def analyze_factors_subcommand(args: argparse.Namespace) -> int:
    """Run comprehensive factor analysis subcommand.

    Args:
        args: Parsed command-line arguments for analyze_factors

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Generate unique Run-ID for this execution
    run_id = generate_run_id(prefix="analyze_factors")
    setup_logging(run_id=run_id, level="INFO")

    logger.info("Comprehensive Factor Analysis (analyze_factors)")

    from scripts.run_factor_analysis import run_factor_analysis_from_args

    try:
        return run_factor_analysis_from_args(args)
    except Exception as e:
        logger.error(f"Factor analysis failed: {e}", exc_info=True)
        return 1


def batch_run_subcommand(args: argparse.Namespace) -> int:
    """Run batch backtests with resume support (MVP).

    This subcommand uses the MVP batch runner (scripts/batch_runner.py) which provides:
    - Deterministic run IDs (hash-based)
    - Resume support (skip successful runs)
    - Rerun failed runs option
    - Parallel execution support

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 if all successful, 1 otherwise)
    """
    from scripts.batch_runner import load_batch_config, run_batch, _setup_logging

    # Setup logging with verbosity
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


def leaderboard_subcommand(args: argparse.Namespace) -> int:
    """Rank and display best runs from batch backtest results.

    This subcommand uses the leaderboard tool (scripts/leaderboard.py) to:
    - Load summary.csv from batch output directory
    - Rank runs by specified metric (Sharpe, total return, final PF, etc.)
    - Display formatted table with top-k runs
    - Optionally export to JSON

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 on success, 1 on failure)
    """
    from scripts.leaderboard import (
        export_leaderboard_json,
        format_leaderboard_table,
        load_batch_summary,
        rank_runs,
    )

    # Validate batch output directory
    if not args.batch_output.exists():
        logger.error("Batch output directory does not exist: %s", args.batch_output)
        return 1

    if not args.batch_output.is_dir():
        logger.error("Batch output path is not a directory: %s", args.batch_output)
        return 1

    # Load summary
    try:
        df = load_batch_summary(args.batch_output)
    except FileNotFoundError as exc:
        logger.error("Failed to load batch summary: %s", exc)
        return 1
    except ValueError as exc:
        logger.error("Invalid batch summary: %s", exc)
        return 1

    # Rank runs
    try:
        ranked_df = rank_runs(df, sort_by=args.sort_by, top_k=args.top_k)
    except ValueError as exc:
        logger.error("Failed to rank runs: %s", exc)
        return 1

    # Print table
    table_str = format_leaderboard_table(ranked_df, args.sort_by)
    print(f"\nTop {len(ranked_df)} runs (sorted by {args.sort_by}):\n")
    print(table_str)

    # Export JSON if requested
    if args.json:
        try:
            export_leaderboard_json(ranked_df, args.json)
            logger.info("Leaderboard exported to %s", args.json)
        except Exception as exc:
            logger.error("Failed to export JSON: %s", exc, exc_info=True)
            return 1

    # Export best run config if requested
    if args.export_best:
        try:
            from scripts.leaderboard import export_best_run_config_yaml
            
            export_best_run_config_yaml(
                df,  # Use full df, not ranked_df, so we can filter to successful
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


def batch_backtest_subcommand(args: argparse.Namespace) -> int:
    """Run batch of strategy backtests from config file.

    This subcommand uses the new batch runner infrastructure (experiments module)
    with support for serial and parallel execution.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 on success, 1 on failure)
    """
    from src.assembled_core.experiments.batch_config import load_batch_config
    from src.assembled_core.experiments.batch_runner import (
        expand_run_specs,
        run_batch_parallel,
        run_batch_serial,
    )

    try:
        # Load config
        config = load_batch_config(args.config_file)

        # Override output_root if provided (handle both --output-root and --output-dir for compatibility)
        output_root = getattr(args, "output_root", None) or getattr(args, "output_dir", None)
        if output_root:
            config.output_root = Path(output_root).resolve()

        # Expand run specs (handles grid search)
        run_specs = expand_run_specs(config)

        # Handle dry-run: just print plan
        if args.dry_run:
            print("=" * 60)
            print(f"Dry-run: Batch '{config.batch_name}'")
            print("=" * 60)
            print(f"Description: {config.description}")
            print(f"Output root: {config.output_root / config.batch_name}")
            print(f"Total runs: {len(run_specs)}")
            print(
                f"Execution mode: {'serial' if args.serial else f'parallel (max_workers={args.max_workers})'}"
            )
            print(f"Fail fast: {args.fail_fast}")
            if config.seed:
                print(f"Random seed: {config.seed}")
            print("\nRun plan:")
            for idx, run_spec in enumerate(run_specs, 1):
                print(f"  {idx}. {run_spec.id}")
                print(f"     Bundle: {run_spec.bundle_path}")
                print(f"     Period: {run_spec.start_date} to {run_spec.end_date}")
                if run_spec.overrides:
                    print(f"     Overrides: {run_spec.overrides}")
            print("=" * 60)
            return 0

        # Handle rerun: check if batch already exists
        batch_output_dir = config.output_root / config.batch_name
        if args.rerun and batch_output_dir.exists():
            logger.info(f"Rerunning batch: {batch_output_dir}")
            # Could optionally clean previous results here, but we'll just overwrite

        # Execute batch
        if args.serial:
            result = run_batch_serial(
                run_specs=run_specs,
                batch_name=config.batch_name,
                output_root=config.output_root,
                base_args=config.base_args,
                repo_root=ROOT,
                fail_fast=args.fail_fast,
                seed=config.seed,
                config=config,  # Pass config for manifest generation
            )
        else:
            result = run_batch_parallel(
                run_specs=run_specs,
                batch_name=config.batch_name,
                output_root=config.output_root,
                base_args=config.base_args,
                max_workers=args.max_workers,
                repo_root=ROOT,
                fail_fast=args.fail_fast,
                timeout_per_run=None,  # Could be added as CLI arg later
                seed=config.seed,
                config=config,  # Pass config for manifest generation
            )

        # Determine exit code
        any_success = result.success_count > 0
        return 0 if any_success else 1

    except Exception as exc:
        logger.error("Batch backtest failed: %s", exc, exc_info=True)
        return 1


def tca_report_subcommand(args: argparse.Namespace) -> int:
    """Generate TCA report from backtest results subcommand.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    from scripts.generate_tca_report import generate_tca_report

    return generate_tca_report(
        backtest_dir=args.backtest_dir,
        output_dir=args.output_dir,
        method=args.method,
        commission_bps=args.commission_bps,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
    )


def check_health_subcommand(args: argparse.Namespace) -> int:
    """Check backend health status subcommand.

    Args:
        args: Parsed command-line arguments for check_health

    Returns:
        Exit code (0 for OK/SKIP, 1 for WARN, 2 for CRITICAL)
    """
    from scripts.check_health import run_health_checks_from_cli

    try:
        return run_health_checks_from_cli(args)
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return 1


def paper_track_subcommand(args: argparse.Namespace) -> int:
    """Run paper track subcommand.

    Args:
        args: Parsed command-line arguments for paper_track

    Returns:
        Exit code (0 = success, 1 = error)
    """
    from scripts.run_paper_track import (
        find_config_by_strategy_name,
        list_paper_track_configs,
        run_paper_track_from_cli,
    )

    # Handle --list flag
    if args.list:
        return list_paper_track_configs()

    # Determine config file
    config_file = None
    if args.strategy_name:
        # Auto-discover config from strategy name
        config_file = find_config_by_strategy_name(args.strategy_name)
        if config_file is None:
            logger.error(
                f"Config not found for strategy '{args.strategy_name}'. "
                "Run with --list to see available strategies."
            )
            return 1
        logger.info(f"Auto-discovered config for '{args.strategy_name}': {config_file}")
    elif args.config_file:
        config_file = args.config_file
    else:
        logger.error(
            "Either --config-file or --strategy-name must be provided. "
            "Run with --list to see available strategies."
        )
        return 1

    try:
        return run_paper_track_from_cli(
            config_file=config_file,
            as_of=args.as_of,
            start_date=args.start_date,
            end_date=args.end_date,
            catch_up=getattr(args, "catch_up", False),
            dry_run=args.dry_run,
            fail_fast=args.fail_fast,
            rerun=getattr(args, "rerun", False),
            generate_risk_report=getattr(args, "generate_risk_report", False),
            risk_report_frequency=getattr(args, "risk_report_frequency", "weekly"),
            benchmark_symbol=getattr(args, "benchmark_symbol", None),
            factor_returns_file=getattr(args, "factor_returns_file", None),
        )
    except Exception as e:
        logger.error(f"Paper track failed: {e}", exc_info=True)
        return 1


def walk_forward_subcommand(args: argparse.Namespace) -> int:
    """Run walk-forward analysis subcommand.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 = success, 1 = error)
    """
    from scripts.run_walk_forward_analysis import run_walk_forward_analysis

    return run_walk_forward_analysis(args)


def risk_report_subcommand(args: argparse.Namespace) -> int:
    """Generate risk report from backtest results subcommand.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 = success, 1 = error)
    """
    from scripts.generate_risk_report import generate_risk_report

    # Resolve paths
    backtest_dir = args.backtest_dir
    if not backtest_dir.is_absolute():
        backtest_dir = ROOT / backtest_dir
    backtest_dir = backtest_dir.resolve()

    regime_file = None
    if args.regime_file:
        regime_file = (
            args.regime_file
            if args.regime_file.is_absolute()
            else ROOT / args.regime_file
        )
        regime_file = regime_file.resolve()

    factor_panel_file = None
    if args.factor_panel_file:
        factor_panel_file = (
            args.factor_panel_file
            if args.factor_panel_file.is_absolute()
            else ROOT / args.factor_panel_file
        )
        factor_panel_file = factor_panel_file.resolve()

    output_dir = None
    if args.output_dir:
        output_dir = (
            args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
        )
        output_dir = output_dir.resolve()

    logger.info(f"Generating risk report for backtest: {backtest_dir}")
    if regime_file:
        logger.info(f"Using regime file: {regime_file}")
    if factor_panel_file:
        logger.info(f"Using factor panel file: {factor_panel_file}")

    benchmark_file = None
    if args.benchmark_file:
        benchmark_file = (
            args.benchmark_file
            if args.benchmark_file.is_absolute()
            else ROOT / args.benchmark_file
        )
        benchmark_file = benchmark_file.resolve()

    factor_returns_file = None
    if args.factor_returns_file:
        factor_returns_file = (
            args.factor_returns_file
            if args.factor_returns_file.is_absolute()
            else ROOT / args.factor_returns_file
        )
        factor_returns_file = factor_returns_file.resolve()

    return generate_risk_report(
        backtest_dir=backtest_dir,
        regime_file=regime_file,
        factor_panel_file=factor_panel_file,
        output_dir=output_dir,
        benchmark_symbol=args.benchmark_symbol,
        benchmark_file=benchmark_file,
        enable_factor_exposures=args.enable_factor_exposures,
        factor_returns_file=factor_returns_file,
        factor_exposures_window=args.factor_exposures_window,
        enable_regime_analysis=args.enable_regime_analysis,
    )


def ml_validate_factors_subcommand(args: argparse.Namespace) -> int:
    """Run ML validation on factor panels subcommand.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 = success, 1 = error)
    """
    from scripts.run_ml_factor_validation import run_ml_validation

    # Resolve paths
    factor_panel_file = args.factor_panel_file
    if not factor_panel_file.is_absolute():
        factor_panel_file = ROOT / factor_panel_file

    output_dir = None
    if args.output_dir:
        output_dir = (
            args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
        )

    # Parse model parameters (parse_model_params is a function inside run_ml_validation module)
    model_params = None
    if args.model_param:
        from scripts.run_ml_factor_validation import parse_model_params

        model_params = parse_model_params(args.model_param)

    logger.info(f"Running ML validation on factor panel: {factor_panel_file}")

    return run_ml_validation(
        factor_panel_file=factor_panel_file,
        label_col=args.label_col,
        model_type=args.model_type,
        model_params=model_params,
        n_splits=args.n_splits,
        test_start=args.test_start,
        test_end=args.test_end,
        output_dir=output_dir,
    )


def ml_model_zoo_subcommand(args: argparse.Namespace) -> int:
    """Run model zoo comparison on factor panels subcommand.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 = success, 1 = error)
    """
    from research.ml.model_zoo_factor_validation import (
        run_model_zoo_for_panel,
        write_model_zoo_summary,
    )

    # Generate Run-ID for this execution
    run_id = generate_run_id(prefix="ml_model_zoo")
    setup_logging(run_id=run_id, level="INFO")

    # Resolve paths
    factor_panel_file = args.factor_panel_file
    if not factor_panel_file.is_absolute():
        factor_panel_file = ROOT / factor_panel_file

    output_dir = None
    if args.output_dir:
        output_dir = (
            args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
        )
    else:
        output_dir = ROOT / "output" / "ml_model_zoo"

    logger.info(f"Running model zoo comparison on factor panel: {factor_panel_file}")

    # Build experiment_cfg_kwargs from args
    experiment_cfg_kwargs: dict[str, Any] = {}
    if args.n_splits is not None:
        experiment_cfg_kwargs["n_splits"] = args.n_splits
    if args.train_size is not None:
        experiment_cfg_kwargs["train_size"] = args.train_size
    if args.standardize is not None:
        experiment_cfg_kwargs["standardize"] = args.standardize
    if args.min_train_samples is not None:
        experiment_cfg_kwargs["min_train_samples"] = args.min_train_samples
    if args.test_start:
        experiment_cfg_kwargs["test_start"] = pd.to_datetime(args.test_start, utc=True)
    if args.test_end:
        experiment_cfg_kwargs["test_end"] = pd.to_datetime(args.test_end, utc=True)

    try:
        # Run model zoo
        summary_df = run_model_zoo_for_panel(
            factor_panel_path=factor_panel_file,
            label_col=args.label_col,
            output_dir=output_dir,
            experiment_cfg_kwargs=experiment_cfg_kwargs
            if experiment_cfg_kwargs
            else None,
        )

        # Write summary
        write_model_zoo_summary(
            summary_df=summary_df,
            output_dir=output_dir,
            write_markdown=not args.no_markdown,
        )

        logger.info("Model zoo comparison completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Model zoo comparison failed: {e}", exc_info=True)
        return 1


def factor_report_subcommand(args: argparse.Namespace) -> int:
    """Run factor analysis report subcommand.

    Args:
        args: Parsed command-line arguments for factor_report

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Generate Run-ID for this execution
    run_id = generate_run_id(prefix="factor_report")
    setup_logging(run_id=run_id, level="INFO")
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Factor Analysis Report")
    logger.info(f"Run-ID: {run_id}")
    logger.info("=" * 60)

    # Import here to avoid circular imports
    from scripts.cli_factor_report import run_factor_report_from_args

    try:
        return run_factor_report_from_args(args)
    except Exception as e:
        logger.error(f"Factor report failed: {e}", exc_info=True)
        return 1


def run_backtest_subcommand(args: argparse.Namespace) -> int:
    """Run strategy backtest subcommand.

    Args:
        args: Parsed command-line arguments for run_backtest

    Returns:
        Exit code (0 for success, 1 for failure)

    Note:
        Orders generated by this command are currently written to SAFE-CSV files.
        If orders are later routed to the Paper-Trading-API (Phase 10), the source
        field should be set to "CLI_BACKTEST" to identify the origin.

        This command automatically sets runtime profile to BACKTEST.
    """
    # Generate Run-ID for this execution
    run_id = generate_run_id(prefix="backtest")
    setup_logging(run_id=run_id, level="INFO")
    logger = logging.getLogger(__name__)

    # Backtest always uses BACKTEST profile
    profile = RuntimeProfile.BACKTEST

    logger.info("=" * 60)
    logger.info("Strategy Backtest (run_backtest)")
    logger.info(f"Run-ID: {run_id}")
    logger.info(f"Runtime Profile: {profile.value}")
    logger.info("=" * 60)

    # Import here to avoid circular imports
    from scripts.run_backtest_strategy import run_backtest_from_args

    # Initialize experiment_run to None (will be set if tracking is enabled)
    experiment_run = None

    try:
        # Setup experiment tracking if enabled
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
                "meta_model_path": str(args.meta_model_path)
                if args.meta_model_path
                else None,
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

        # Log meta-model status if enabled
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


def _run_backtest_for_ml_dataset(
    strategy: str,
    freq: str,
    price_file: Path | None = None,
    universe: Path | None = None,
    start_capital: float = 10000.0,
    with_costs: bool = True,
    commission_bps: float | None = None,
    spread_w: float | None = None,
    impact_w: float | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run backtest and return prices_with_features and trades for ML dataset building.

    This is a helper function that runs a backtest and extracts the prices_with_features
    and trades DataFrames needed for building ML datasets.

    Args:
        strategy: Strategy name ("trend_baseline" or "event_insider_shipping")
        freq: Trading frequency ("1d" or "5min")
        price_file: Optional explicit path to price file
        universe: Optional path to universe file
        start_capital: Starting capital
        with_costs: Whether to include transaction costs
        commission_bps: Optional commission override
        spread_w: Optional spread weight override
        impact_w: Optional market impact weight override
        output_dir: Optional output directory for price data loading

    Returns:
        Tuple of (prices_with_features, trades) DataFrames

    Raises:
        ValueError: If strategy is unknown or data loading fails
        FileNotFoundError: If price file or universe file not found
    """
    import pandas as pd
    from src.assembled_core.config import OUTPUT_DIR
    from src.assembled_core.data.prices_ingest import (
        load_eod_prices,
        load_eod_prices_for_universe,
    )
    from src.assembled_core.ema_config import get_default_ema_config
    from src.assembled_core.qa.backtest_engine import run_portfolio_backtest
    from scripts.run_backtest_strategy import (
        create_trend_baseline_signal_fn,
        create_position_sizing_fn,
        create_event_insider_shipping_signal_fn,
        create_event_position_sizing_fn,
        get_cost_model,
    )

    # Set output directory
    if output_dir is None:
        output_dir = OUTPUT_DIR

    # Load price data
    import logging

    logger = logging.getLogger(__name__)

    if price_file:
        logger.info(f"Loading prices from explicit file: {price_file}")
        prices = load_eod_prices(price_file=price_file, freq=freq)
    elif universe:
        logger.info(f"Loading prices for universe: {universe}")
        prices = load_eod_prices_for_universe(
            universe_file=universe, data_dir=output_dir, freq=freq
        )
    else:
        # Default: use watchlist.txt
        logger.info("Loading prices for default universe (watchlist.txt)")
        prices = load_eod_prices_for_universe(
            universe_file=None, data_dir=output_dir, freq=freq
        )

    if prices.empty:
        raise ValueError("No price data loaded")

    logger.info(f"Loaded {len(prices)} rows for {prices['symbol'].nunique()} symbols")

    # Get cost model
    class CostArgs:
        def __init__(self):
            self.commission_bps = commission_bps
            self.spread_w = spread_w
            self.impact_w = impact_w

    cost_args = CostArgs()
    cost_model = get_cost_model(cost_args)

    # Build prices_with_features by computing all features
    from src.assembled_core.features.ta_features import (
        add_all_features,
        add_log_returns,
        add_moving_averages,
    )

    # Add TA features
    has_ohlc = all(col in prices.columns for col in ["high", "low", "open"])
    if has_ohlc:
        prices_with_features = add_all_features(
            prices,
            ma_windows=(20, 50, 200),
            atr_window=14,
            rsi_window=14,
            include_rsi=True,
        )
    else:
        prices_with_features = add_log_returns(prices.copy())
        prices_with_features = add_moving_averages(
            prices_with_features, windows=(20, 50, 200)
        )

    # Add event features if event strategy
    if strategy == "event_insider_shipping":
        from src.assembled_core.features.insider_features import add_insider_features
        from src.assembled_core.features.shipping_features import add_shipping_features
        from src.assembled_core.data.insider_ingest import load_insider_sample
        from src.assembled_core.data.shipping_routes_ingest import load_shipping_sample
        from pathlib import Path as P

        ROOT = P(__file__).resolve().parents[1]
        EVENT_DIR = ROOT / "data" / "sample" / "events"

        insider_file = EVENT_DIR / "insider_sample.parquet"
        shipping_file = EVENT_DIR / "shipping_sample.parquet"

        if insider_file.exists():
            insider_events = pd.read_parquet(insider_file)
            if "timestamp" in insider_events.columns:
                insider_events["timestamp"] = pd.to_datetime(
                    insider_events["timestamp"], utc=True
                )
        else:
            insider_events = load_insider_sample()

        if shipping_file.exists():
            shipping_events = pd.read_parquet(shipping_file)
            if "timestamp" in shipping_events.columns:
                shipping_events["timestamp"] = pd.to_datetime(
                    shipping_events["timestamp"], utc=True
                )
        else:
            shipping_events = load_shipping_sample()

        prices_with_features = add_insider_features(
            prices_with_features, insider_events
        )
        prices_with_features = add_shipping_features(
            prices_with_features, shipping_events
        )

    # Create signal and position sizing functions
    if strategy == "trend_baseline":
        ema_config = get_default_ema_config(freq)
        signal_fn = create_trend_baseline_signal_fn(
            ma_fast=ema_config.fast, ma_slow=ema_config.slow
        )
        position_sizing_fn = create_position_sizing_fn()
    elif strategy == "event_insider_shipping":
        signal_fn = create_event_insider_shipping_signal_fn()
        position_sizing_fn = create_event_position_sizing_fn()
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. Supported: trend_baseline, event_insider_shipping"
        )

    # Run backtest to get trades
    result = run_portfolio_backtest(
        prices=prices_with_features,  # Use prices_with_features for signal generation
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        start_capital=start_capital,
        commission_bps=cost_model.commission_bps,
        spread_w=cost_model.spread_w,
        impact_w=cost_model.impact_w,
        include_costs=with_costs,
        include_trades=True,
        include_signals=False,
        include_targets=False,
        rebalance_freq=freq,
        compute_features=False,  # We already computed features above
    )

    if result.trades is None or result.trades.empty:
        logger.warning("No trades generated from backtest")
        trades = pd.DataFrame()
    else:
        trades = result.trades.copy()
        # Ensure trades have required columns for labeling
        if "open_time" not in trades.columns and "timestamp" in trades.columns:
            trades["open_time"] = trades["timestamp"]
        if "open_price" not in trades.columns and "price" in trades.columns:
            trades["open_price"] = trades["price"]

    return prices_with_features, trades


def train_meta_model_subcommand(args: argparse.Namespace) -> int:
    """Train meta-model subcommand.

    Args:
        args: Parsed command-line arguments for train_meta_model

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Generate Run-ID for this execution
    run_id = generate_run_id(prefix="meta_model")
    setup_logging(run_id=run_id, level="INFO")
    logger = logging.getLogger(__name__)

    # ML model training always uses BACKTEST profile
    profile = RuntimeProfile.BACKTEST

    logger.info("=" * 60)
    logger.info("Meta-Model Training (train_meta_model)")
    logger.info(f"Run-ID: {run_id}")
    logger.info(f"Runtime Profile: {profile.value}")
    logger.info("=" * 60)

    # Initialize experiment_run to None (will be set if tracking is enabled)
    experiment_run = None

    try:
        import pandas as pd
        from src.assembled_core.config.settings import get_settings
        from src.assembled_core.qa.dataset_builder import (
            build_ml_dataset_for_strategy,
        )
        from src.assembled_core.qa.ml_evaluation import (
            evaluate_meta_model,
            plot_calibration_curve,
        )
        from src.assembled_core.signals.meta_model import (
            save_meta_model,
            train_meta_model,
        )

        settings = get_settings()

        # Setup experiment tracking if enabled
        if args.track_experiment:
            if not args.experiment_name:
                logger.error(
                    "--experiment-name is required when --track-experiment is set"
                )
                return 1

            from src.assembled_core.qa.experiment_tracking import ExperimentTracker

            tracker = ExperimentTracker(settings.experiments_dir)
            tags = args.experiment_tags.split(",") if args.experiment_tags else []
            tags = [t.strip() for t in tags if t.strip()]

            config = {
                "model_type": args.model_type,
                "label_horizon_days": args.label_horizon_days,
                "success_threshold": args.success_threshold,
            }

            if args.dataset_path:
                config["dataset_path"] = str(args.dataset_path)
            else:
                config["strategy"] = args.strategy
                config["freq"] = args.freq
                config["start_date"] = args.start_date
                config["end_date"] = args.end_date
                if args.symbols:
                    config["symbols"] = args.symbols

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

        # Load or build dataset
        if args.dataset_path:
            logger.info(f"Loading dataset from: {args.dataset_path}")
            if args.dataset_path.suffix == ".parquet":
                df = pd.read_parquet(args.dataset_path)
            elif args.dataset_path.suffix == ".csv":
                df = pd.read_csv(args.dataset_path)
            else:
                logger.error(f"Unsupported file format: {args.dataset_path.suffix}")
                return 1
            logger.info(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
        else:
            # Build dataset on-the-fly
            if (
                not args.strategy
                or not args.freq
                or not args.start_date
                or not args.end_date
            ):
                logger.error(
                    "If --dataset-path is not provided, --strategy, --freq, --start-date, and --end-date are required"
                )
                return 1

            logger.info("Building ML dataset on-the-fly...")
            logger.info(f"Strategy: {args.strategy}")
            logger.info(f"Frequency: {args.freq}")
            logger.info(f"Date range: {args.start_date} to {args.end_date}")
            logger.info(f"Label Horizon: {args.label_horizon_days} days")
            logger.info(f"Success Threshold: {args.success_threshold:.2%}")

            df = build_ml_dataset_for_strategy(
                strategy_name=args.strategy,
                start_date=args.start_date,
                end_date=args.end_date,
                universe=args.symbols,
                label_params={
                    "horizon_days": args.label_horizon_days,
                    "threshold_pct": args.success_threshold,
                    "label_type": "binary_absolute",
                },
                freq=args.freq,
            )

            if df.empty:
                logger.error("ML dataset is empty")
                return 1

            logger.info(f"Built dataset: {len(df)} rows, {len(df.columns)} columns")

        # Validate dataset
        if "label" not in df.columns:
            logger.error("Dataset must contain 'label' column")
            return 1

        # Train model
        logger.info("")
        logger.info("Training meta-model...")
        logger.info(f"Model Type: {args.model_type}")

        meta_model = train_meta_model(
            df=df,
            feature_cols=None,  # Auto-detect
            label_col="label",
            model_type=args.model_type,
            random_state=42,
        )

        logger.info(f"Model trained with {len(meta_model.feature_names)} features")

        # Evaluate on training set
        logger.info("")
        logger.info("Evaluating model on training set...")
        X = df[meta_model.feature_names]
        y_prob = meta_model.predict_proba(X)
        y_true = df["label"]

        metrics = evaluate_meta_model(y_true, y_prob)

        logger.info("=" * 60)
        logger.info("Evaluation Metrics:")
        logger.info(
            f"  ROC-AUC: {metrics['roc_auc']:.4f}"
            if not pd.isna(metrics["roc_auc"])
            else "  ROC-AUC: N/A"
        )
        logger.info(f"  Brier Score: {metrics['brier_score']:.4f}")
        logger.info(
            f"  Log Loss: {metrics['log_loss']:.4f}"
            if not pd.isna(metrics["log_loss"])
            else "  Log Loss: N/A"
        )
        logger.info("=" * 60)

        # Save model
        if args.output_model_path:
            model_path = args.output_model_path
        else:
            # Default path based on strategy
            strategy_name = args.strategy if args.strategy else "unknown"
            model_dir = settings.models_dir / "meta"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{strategy_name}_meta_model.joblib"

        logger.info("")
        logger.info(f"Saving model to: {model_path}")
        save_meta_model(meta_model, model_path)

        # Plot calibration curve
        logger.info("")
        logger.info("Generating calibration curve...")
        calibration_dir = settings.output_dir / "reports" / "meta"
        calibration_dir.mkdir(parents=True, exist_ok=True)
        calibration_path = calibration_dir / f"{model_path.stem}_calibration.png"
        plot_calibration_curve(y_true, y_prob, calibration_path)
        logger.info(f"Calibration curve saved to: {calibration_path}")

        logger.info("")
        logger.info("=" * 60)
        logger.info("Meta-Model Training Complete")
        logger.info("=" * 60)
        logger.info(f"Model: {model_path}")
        logger.info(f"Features: {len(meta_model.feature_names)}")
        logger.info(f"Training Samples: {len(df)}")
        logger.info("=" * 60)

        # Log metrics to experiment tracking if enabled
        if experiment_run:
            from src.assembled_core.qa.experiment_tracking import ExperimentTracker

            settings = get_settings()
            tracker = ExperimentTracker(settings.experiments_dir)

            # Log evaluation metrics
            metrics_dict = {
                "brier_score": metrics["brier_score"],
            }
            if not pd.isna(metrics["roc_auc"]):
                metrics_dict["roc_auc"] = metrics["roc_auc"]
            if not pd.isna(metrics["log_loss"]):
                metrics_dict["log_loss"] = metrics["log_loss"]

            tracker.log_metrics(experiment_run, metrics_dict)

            # Log artifacts
            if model_path.exists():
                tracker.log_artifact(experiment_run, model_path, "meta_model.joblib")
            if calibration_path.exists():
                tracker.log_artifact(
                    experiment_run, calibration_path, "calibration_curve.png"
                )

            # Finish run
            tracker.finish_run(experiment_run, status="finished")
            logger.info("")
            logger.info(f"Experiment run completed: {experiment_run.run_id}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        if experiment_run:
            from src.assembled_core.qa.experiment_tracking import ExperimentTracker

            settings = get_settings()
            tracker = ExperimentTracker(settings.experiments_dir)
            tracker.finish_run(experiment_run, status="failed")
        return 1
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        if experiment_run:
            from src.assembled_core.qa.experiment_tracking import ExperimentTracker

            settings = get_settings()
            tracker = ExperimentTracker(settings.experiments_dir)
            tracker.finish_run(experiment_run, status="failed")
        return 1
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install scikit-learn with: pip install scikit-learn")
        if experiment_run:
            from src.assembled_core.qa.experiment_tracking import ExperimentTracker

            settings = get_settings()
            tracker = ExperimentTracker(settings.experiments_dir)
            tracker.finish_run(experiment_run, status="failed")
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        if experiment_run:
            from src.assembled_core.qa.experiment_tracking import ExperimentTracker

            settings = get_settings()
            tracker = ExperimentTracker(settings.experiments_dir)
            tracker.finish_run(experiment_run, status="failed")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        if experiment_run:
            from src.assembled_core.qa.experiment_tracking import ExperimentTracker

            settings = get_settings()
            tracker = ExperimentTracker(settings.experiments_dir)
            tracker.finish_run(experiment_run, status="failed")
        return 1


def run_phase4_tests_subcommand(args: argparse.Namespace) -> int:
    """Run Phase-4 test suite subcommand.

    Args:
        args: Parsed command-line arguments for run_phase4_tests

    Returns:
        Exit code from pytest (0 for success, non-zero for failure)
    """
    logger.info("=" * 60)
    logger.info("Phase-4 Test Suite (run_phase4_tests)")
    logger.info("=" * 60)

    # Build pytest command
    pytest_args = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "phase4",
        "-q",
        "--maxfail=1",
        "--tb=short",
    ]

    # Remove -q if verbose is requested
    if args.verbose:
        pytest_args = [arg for arg in pytest_args if arg != "-q"]
        pytest_args.append("-vv")

    if args.durations:
        pytest_args.append(f"--durations={args.durations}")

    logger.info(
        f"Running: {' '.join(pytest_args[2:])}"
    )  # Exclude python and -m pytest for cleaner log
    logger.info("")

    # Run pytest
    try:
        result = subprocess.run(
            pytest_args,
            cwd=str(ROOT),
            check=False,  # Don't raise on non-zero exit
        )
        return result.returncode
    except Exception as e:
        logger.error(f"Failed to run pytest: {e}", exc_info=True)
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Assembled Trading AI - Central CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run daily EOD pipeline
  python scripts/cli.py run_daily --freq 1d
  
  # Run strategy backtest
              python scripts/cli.py run_backtest --freq 1d --universe watchlist.txt --generate-report
              python scripts/cli.py run_backtest --freq 1d --strategy event_insider_shipping --generate-report
  
  # Run Phase-4 tests
  python scripts/cli.py run_phase4_tests
  
  # Run Phase-4 tests with verbose output and durations
  python scripts/cli.py run_phase4_tests --verbose --durations 5
  
  # Show project information
  python scripts/cli.py info
  
  # Show version
  python scripts/cli.py --version
        """,
    )

    # Global --version flag
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show version and exit",
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Subcommand to run", required=True
    )

    # run_daily subcommand
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
    # Experiment tracking arguments
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

    # run_backtest subcommand
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
        choices=["trend_baseline", "event_insider_shipping"],
        metavar="NAME",
        help="Strategy name: 'trend_baseline' (EMA crossover) or 'event_insider_shipping' (Phase 6 event-based)",
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
    # Meta-model ensemble arguments
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
    # Experiment tracking arguments
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
    backtest_parser.set_defaults(func=run_backtest_subcommand)

    # build_ml_dataset subcommand
    ml_dataset_parser = subparsers.add_parser(
        "build_ml_dataset",
        help="Build ML-ready dataset from backtest results",
        description="Runs a strategy backtest and builds an ML-ready dataset with features and labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""" 
Examples:
  python scripts/cli.py build_ml_dataset --strategy trend_baseline --freq 1d
  python scripts/cli.py build_ml_dataset --strategy event_insider_shipping --freq 1d --price-file data/sample/eod_sample.parquet
  python scripts/cli.py build_ml_dataset --strategy trend_baseline --freq 1d --label-horizon-days 5 --success-threshold 0.03
        """,
    )
    ml_dataset_parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["trend_baseline", "event_insider_shipping"],
        metavar="NAME",
        help="Strategy name: 'trend_baseline' (EMA crossover) or 'event_insider_shipping' (Phase 6 event-based)",
    )
    ml_dataset_parser.add_argument(
        "--freq",
        type=str,
        required=True,
        choices=["1d", "5min"],
        help="Trading frequency: '1d' for daily or '5min' for 5-minute bars",
    )
    ml_dataset_parser.add_argument(
        "--price-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Explicit path to price file (overrides default path)",
    )
    ml_dataset_parser.add_argument(
        "--universe",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to universe file (default: watchlist.txt in repo root)",
    )
    ml_dataset_parser.add_argument(
        "--start-capital",
        type=float,
        default=10000.0,
        metavar="AMOUNT",
        help="Starting capital in USD (default: 10000.0)",
    )
    ml_dataset_parser.add_argument(
        "--with-costs",
        action="store_true",
        default=True,
        help="Include transaction costs in backtest (default: True)",
    )
    ml_dataset_parser.add_argument(
        "--no-costs",
        action="store_false",
        dest="with_costs",
        help="Disable transaction costs (use cost-free simulation)",
    )
    ml_dataset_parser.add_argument(
        "--label-horizon-days",
        type=int,
        default=10,
        metavar="DAYS",
        help="Number of days to look forward for P&L calculation (default: 10)",
    )
    ml_dataset_parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.02,
        metavar="THRESHOLD",
        help="P&L percentage threshold for a successful trade (label=1) (default: 0.02 = 2%%)",
    )
    ml_dataset_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="FILE",
        help="Output path for ML dataset (default: output/ml_datasets/<strategy>_<freq>.parquet)",
    )
    ml_dataset_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Start date for dataset (default: use all available data)",
    )
    ml_dataset_parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="End date for dataset (default: use all available data)",
    )
    ml_dataset_parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        metavar="SYMBOL",
        help="List of symbols to include (e.g., --symbols AAPL MSFT GOOGL). Overrides --universe.",
    )
    ml_dataset_parser.add_argument(
        "--label-type",
        type=str,
        choices=["binary_absolute", "binary_outperformance", "multi_class"],
        default="binary_absolute",
        help="Label type: binary_absolute (default), binary_outperformance, or multi_class",
    )
    ml_dataset_parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format: parquet (default) or csv",
    )
    ml_dataset_parser.set_defaults(func=build_ml_dataset_subcommand)

    # train_meta_model subcommand
    train_meta_parser = subparsers.add_parser(
        "train_meta_model",
        help="Train meta-model for setup success prediction",
        description="Trains a meta-model that predicts the success probability (confidence_score) of trading setups.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from existing dataset
  python scripts/cli.py train_meta_model --dataset-path output/ml_datasets/trend_baseline_1d.parquet
  
  # Build dataset and train in one step
  python scripts/cli.py train_meta_model --strategy trend_baseline --freq 1d --start-date 2024-01-01 --end-date 2024-12-31
  
  # Train with custom model type
  python scripts/cli.py train_meta_model --dataset-path output/ml_datasets/trend_baseline_1d.parquet --model-type random_forest
        """,
    )
    train_meta_parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to existing ML dataset (Parquet or CSV). If not provided, dataset will be built on-the-fly.",
    )
    train_meta_parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["trend_baseline", "event_insider_shipping"],
        metavar="NAME",
        help="Strategy name (required if --dataset-path is not provided): 'trend_baseline' or 'event_insider_shipping'",
    )
    train_meta_parser.add_argument(
        "--freq",
        type=str,
        default=None,
        choices=["1d", "5min"],
        metavar="FREQ",
        help="Trading frequency (required if --dataset-path is not provided): '1d' or '5min'",
    )
    train_meta_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Start date for dataset (required if --dataset-path is not provided)",
    )
    train_meta_parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="End date for dataset (required if --dataset-path is not provided)",
    )
    train_meta_parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        metavar="SYMBOL",
        help="List of symbols to include (optional, for on-the-fly dataset building)",
    )
    train_meta_parser.add_argument(
        "--model-type",
        type=str,
        choices=["gradient_boosting", "random_forest"],
        default="gradient_boosting",
        help="Model type: 'gradient_boosting' (default) or 'random_forest'",
    )
    train_meta_parser.add_argument(
        "--output-model-path",
        type=Path,
        default=None,
        metavar="FILE",
        help="Output path for trained model (default: models/meta/<strategy>_meta_model.joblib)",
    )
    train_meta_parser.add_argument(
        "--label-horizon-days",
        type=int,
        default=10,
        metavar="DAYS",
        help="Label horizon in days (for on-the-fly dataset building, default: 10)",
    )
    train_meta_parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.05,
        metavar="THRESHOLD",
        help="Success threshold (for on-the-fly dataset building, default: 0.05 = 5%%)",
    )
    # Experiment tracking arguments
    train_meta_parser.add_argument(
        "--track-experiment",
        action="store_true",
        default=False,
        help="Enable experiment tracking (stores run config, metrics, and artifacts)",
    )
    train_meta_parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        metavar="NAME",
        help="Name for the experiment run (required if --track-experiment is set)",
    )
    train_meta_parser.add_argument(
        "--experiment-tags",
        type=str,
        default=None,
        metavar="TAGS",
        help="Comma-separated tags for the experiment (e.g., 'meta_model,gradient_boosting')",
    )
    train_meta_parser.set_defaults(func=train_meta_model_subcommand)

    # factor_report subcommand
    factor_report_parser = subparsers.add_parser(
        "factor_report",
        help="Run a factor analysis report on a given universe and date range",
        description="Generates a comprehensive factor analysis report with IC/IR statistics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Factor report for AI Tech universe (local alt-data)
  $env:ASSEMBLED_DATA_SOURCE = "local"
  $env:ASSEMBLED_LOCAL_DATA_ROOT = "F:\\Python_Projekt\\Aktiengerüst\\datensammlungen\\altdaten\\stand 3-12-2025"
  python scripts/cli.py factor_report --freq 1d --symbols-file config/universe_ai_tech_tickers.txt --start-date 2005-01-01 --end-date 2025-12-02 --factor-set core --fwd-horizon-days 5
  
  # Factor report with all factors and CSV output
  python scripts/cli.py factor_report --freq 1d --symbols-file config/universe_ai_tech_tickers.txt --start-date 2005-01-01 --end-date 2025-12-02 --factor-set all --fwd-horizon-days 21 --output-csv output/factor_reports/ai_tech_all_21d_ic.csv
        """,
    )
    factor_report_parser.add_argument(
        "--freq",
        type=str,
        required=True,
        choices=["1d", "5min"],
        help="Frequency, e.g. 1d",
    )
    factor_report_parser.add_argument(
        "--symbols-file",
        type=Path,
        required=True,
        metavar="FILE",
        help="Path to a text file with one symbol per line (e.g., config/universe_ai_tech_tickers.txt)",
    )
    factor_report_parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        metavar="YYYY-MM-DD",
        help="Start date for data loading (e.g., 2005-01-01)",
    )
    factor_report_parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        metavar="YYYY-MM-DD",
        help="End date for data loading (e.g., 2025-12-02)",
    )
    factor_report_parser.add_argument(
        "--factor-set",
        type=str,
        choices=["core", "vol_liquidity", "all"],
        default="core",
        help="Which factors to compute: 'core' (TA/Price factors, default), 'vol_liquidity' (volatility/liquidity), or 'all' (both)",
    )
    factor_report_parser.add_argument(
        "--fwd-horizon-days",
        type=int,
        default=5,
        metavar="DAYS",
        help="Forward return horizon in days (default: 5)",
    )
    factor_report_parser.add_argument(
        "--data-source",
        type=str,
        choices=["local", "yahoo"],
        default=None,
        help="Data source type: 'local' (Parquet files) or 'yahoo' (Yahoo Finance API). Default: from settings.data_source",
    )
    factor_report_parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        metavar="FILE",
        help="Optional path for summary CSV output (e.g., output/factor_reports/ai_tech_core_5d_ic.csv)",
    )
    factor_report_parser.set_defaults(func=factor_report_subcommand)

    # analyze_factors subcommand
    analyze_factors_parser = subparsers.add_parser(
        "analyze_factors",
        help="Comprehensive factor analysis (IC + Portfolio evaluation)",
        description="Run comprehensive factor analysis including IC-based evaluation (C1) and portfolio-based evaluation (C2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze factors for ETF universe
  python scripts/cli.py analyze_factors --freq 1d --symbols-file config/macro_world_etfs_tickers.txt --data-source local --start-date 2010-01-01 --end-date 2025-12-03 --factor-set core+vol_liquidity --horizon-days 20

  # Analyze with custom quantiles and output directory
  python scripts/cli.py analyze_factors --freq 1d --universe config/universe_ai_tech_tickers.txt --start-date 2005-01-01 --end-date 2025-12-02 --factor-set all --horizon-days 21 --quantiles 10 --output-dir output/custom_analysis
        """,
    )

    analyze_factors_parser.add_argument(
        "--freq",
        type=str,
        required=True,
        choices=["1d", "5min"],
        help="Frequency (1d or 5min)",
    )

    # Symbol input (priority: symbols > symbols-file > universe)
    symbol_group = analyze_factors_parser.add_mutually_exclusive_group(required=False)
    symbol_group.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="List of symbols (e.g., --symbols AAPL MSFT GOOG)",
    )
    symbol_group.add_argument(
        "--symbols-file", type=str, help="Path to file with symbols (one per line)"
    )
    symbol_group.add_argument(
        "--universe", type=str, help="Path to universe file (alias for --symbols-file)"
    )

    analyze_factors_parser.add_argument(
        "--data-source",
        type=str,
        default="local",
        choices=["local", "yahoo", "finnhub", "twelve_data"],
        help="Data source (default: local)",
    )

    analyze_factors_parser.add_argument(
        "--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)"
    )

    analyze_factors_parser.add_argument(
        "--end-date", type=str, required=True, help="End date (YYYY-MM-DD)"
    )

    # Import factor set utility function
    from scripts.run_factor_analysis import list_available_factor_sets

    analyze_factors_parser.add_argument(
        "--factor-set",
        type=str,
        default="core",
        choices=list_available_factor_sets(),
        help="Factor set: core (TA/Price), vol_liquidity (Volatility/Liquidity), core+vol_liquidity, all, alt_earnings_insider (Alt-Data B1 only), core+alt (Core + Alt-Data B1), alt_news_macro (Alt-Data B2 only), core+alt_news (Core + Alt-Data B2), or core+alt_full (Core + B1 + B2) (default: core)",
    )

    analyze_factors_parser.add_argument(
        "--horizon-days",
        type=int,
        default=20,
        help="Forward return horizon in days (default: 20)",
    )

    analyze_factors_parser.add_argument(
        "--quantiles",
        type=int,
        default=5,
        help="Number of quantiles for portfolio analysis (default: 5)",
    )

    analyze_factors_parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: output/factor_analysis)",
    )

    analyze_factors_parser.set_defaults(func=analyze_factors_subcommand)

    # ml_validate_factors subcommand
    ml_validate_parser = subparsers.add_parser(
        "ml_validate_factors",
        help="Run ML validation on factor panels",
        description="Trains ML models to predict forward returns from factor panels and evaluates them using time-series cross-validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation with Ridge model
  python scripts/cli.py ml_validate_factors \\
    --factor-panel-file output/factor_analysis/ai_tech_factors.parquet \\
    --label-col fwd_return_20d \\
    --model-type ridge
  
  # With custom parameters
  python scripts/cli.py ml_validate_factors \\
    --factor-panel-file output/factor_analysis/ai_tech_factors.parquet \\
    --label-col fwd_return_20d \\
    --model-type ridge \\
    --model-param alpha=0.1 \\
    --model-param max_iter=1000
  
  # Random Forest with time filter
  python scripts/cli.py ml_validate_factors \\
    --factor-panel-file output/factor_analysis/ai_tech_factors.parquet \\
    --label-col fwd_return_20d \\
    --model-type random_forest \\
    --n-splits 10 \\
    --test-start 2020-01-01 \\
    --test-end 2024-12-31
        """,
    )
    ml_validate_parser.add_argument(
        "--factor-panel-file",
        type=Path,
        required=True,
        metavar="FILE",
        help="Path to factor panel file (Parquet or CSV) with factors and forward returns",
    )
    ml_validate_parser.add_argument(
        "--label-col",
        type=str,
        required=True,
        metavar="COL",
        help="Name of label column (e.g., 'fwd_return_20d')",
    )
    ml_validate_parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["linear", "ridge", "lasso", "random_forest"],
        metavar="TYPE",
        help="Model type: 'linear', 'ridge', 'lasso', or 'random_forest'",
    )
    ml_validate_parser.add_argument(
        "--model-param",
        type=str,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Model hyperparameter in format 'key=value' (can be specified multiple times)",
    )
    ml_validate_parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        metavar="N",
        help="Number of time-series CV splits (default: 5)",
    )
    ml_validate_parser.add_argument(
        "--test-start",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Test start date (YYYY-MM-DD, optional)",
    )
    ml_validate_parser.add_argument(
        "--test-end",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Test end date (YYYY-MM-DD, optional)",
    )
    ml_validate_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory (default: output/ml_validation)",
    )
    ml_validate_parser.set_defaults(func=ml_validate_factors_subcommand)

    # ml_model_zoo subcommand
    ml_model_zoo_parser = subparsers.add_parser(
        "ml_model_zoo",
        help="Compare multiple ML models on factor panels (model zoo)",
        description="Runs a predefined set of ML models (linear, ridge, lasso, random forest) on a factor panel and compares their performance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic model zoo comparison
  python scripts/cli.py ml_model_zoo \\
    --factor-panel-file output/factor_panels/core_20d_factors.parquet \\
    --label-col fwd_return_20d
  
  # With custom CV splits
  python scripts/cli.py ml_model_zoo \\
    --factor-panel-file output/factor_panels/core_20d_factors.parquet \\
    --label-col fwd_return_20d \\
    --n-splits 10 \\
    --output-dir output/ml_validation/custom_zoo
        """,
    )
    ml_model_zoo_parser.add_argument(
        "--factor-panel-file",
        type=Path,
        required=True,
        metavar="FILE",
        help="Path to factor panel file (Parquet or CSV) with factors and forward returns",
    )
    ml_model_zoo_parser.add_argument(
        "--label-col",
        type=str,
        required=True,
        metavar="COL",
        help="Name of label column (e.g., 'fwd_return_20d')",
    )
    ml_model_zoo_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory (default: output/ml_model_zoo)",
    )
    ml_model_zoo_parser.add_argument(
        "--n-splits",
        type=int,
        default=None,
        metavar="N",
        help="Number of CV splits (default: 5)",
    )
    ml_model_zoo_parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        metavar="DAYS",
        help="Training window size in days (default: None = expanding window)",
    )
    ml_model_zoo_parser.add_argument(
        "--standardize",
        type=bool,
        default=None,
        help="Whether to standardize features (default: True)",
    )
    ml_model_zoo_parser.add_argument(
        "--min-train-samples",
        type=int,
        default=None,
        metavar="N",
        help="Minimum training samples (default: 252)",
    )
    ml_model_zoo_parser.add_argument(
        "--test-start",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Test start date (YYYY-MM-DD, optional)",
    )
    ml_model_zoo_parser.add_argument(
        "--test-end",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Test end date (YYYY-MM-DD, optional)",
    )
    ml_model_zoo_parser.add_argument(
        "--no-markdown", action="store_true", help="Skip Markdown report generation"
    )
    ml_model_zoo_parser.set_defaults(func=ml_model_zoo_subcommand)

    # batch_backtest subcommand
    batch_parser = subparsers.add_parser(
        "batch_backtest",
        help="Run batch of strategy backtests from config file",
        description="Runs multiple strategy backtests defined in a YAML/JSON config file using the optimized backtest engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load config and run (parallel, 4 workers)
  python scripts/cli.py batch_backtest --config-file configs/batch_backtests/ai_tech_core_vs_mlalpha.yaml

  # Serial execution
  python scripts/cli.py batch_backtest --config-file configs/batch_backtests/ai_tech_core_vs_mlalpha.yaml --serial

  # Parallel with custom workers and fail-fast
  python scripts/cli.py batch_backtest `
    --config-file configs/batch_backtests/ai_tech_core_vs_mlalpha.yaml `
    --max-workers 4 --fail-fast

  # Dry-run: show plan without executing
  python scripts/cli.py batch_backtest --config-file configs/batch_backtests/test.yaml --dry-run

  # Rerun existing batch (overwrites)
  python scripts/cli.py batch_backtest --config-file configs/batch_backtests/test.yaml --rerun

        """,
    )
    batch_parser.add_argument(
        "--config-file",
        type=Path,
        required=True,
        metavar="FILE",
        help="Path to batch config file (YAML or JSON).",
    )
    batch_parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        dest="output_root",
        metavar="DIR",
        help="Override output root directory for batch (default: from config.output_root).",
    )
    # Also support --output-dir for backward compatibility
    batch_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        dest="output_dir",
        metavar="DIR",
        help="(Deprecated: use --output-root) Override output root directory.",
    )
    batch_parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        metavar="N",
        help="Maximum number of parallel workers (default: 4, ignored if --serial).",
    )
    batch_parser.add_argument(
        "--serial",
        action="store_true",
        help="Run batch serially (no parallelization).",
    )
    batch_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort batch on first failed run.",
    )
    batch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show run plan without executing backtests.",
    )
    batch_parser.add_argument(
        "--rerun",
        action="store_true",
        help="Rerun batch even if output directory already exists (overwrites).",
    )
    batch_parser.set_defaults(func=batch_backtest_subcommand)

    # batch_run subcommand (MVP batch runner)
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

    # leaderboard subcommand
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
        choices=["sharpe", "total_return", "final_pf", "max_drawdown_pct", "cagr", "trades"],
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

    # risk_report subcommand
    risk_report_parser = subparsers.add_parser(
        "risk_report",
        help="Generate risk report from backtest results",
        description="Generates comprehensive risk reports from backtest outputs, including risk metrics, exposure analysis, regime segmentation, and factor attribution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic risk report from backtest directory
  python scripts/cli.py risk_report --backtest-dir output/backtests/experiment_123/
  
  # With regime data
  python scripts/cli.py risk_report --backtest-dir output/backtests/experiment_123/ --regime-file output/regime/regime_state.parquet
  
  # With factor attribution
  python scripts/cli.py risk_report --backtest-dir output/backtests/experiment_123/ --factor-panel-file output/factor_analysis/factors.parquet
  
  # Custom output directory
  python scripts/cli.py risk_report --backtest-dir output/backtests/experiment_123/ --output-dir output/risk_reports/
        """,
    )
    risk_report_parser.add_argument(
        "--backtest-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Path to backtest output directory (should contain equity_curve.csv/parquet, positions.csv/parquet, etc.)",
    )
    risk_report_parser.add_argument(
        "--regime-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Optional path to regime state file (parquet or csv) for regime-based risk analysis",
    )
    risk_report_parser.add_argument(
        "--factor-panel-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Optional path to factor panel file (parquet or csv) for factor attribution analysis",
    )
    risk_report_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory for risk report files (default: same as --backtest-dir)",
    )
    risk_report_parser.add_argument(
        "--benchmark-symbol",
        type=str,
        default=None,
        metavar="SYMBOL",
        help="Benchmark symbol (e.g., 'SPY', 'QQQ') for regime classification. Requires --enable-regime-analysis.",
    )
    risk_report_parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to benchmark file (CSV/Parquet) with timestamp and returns/close columns. Requires --enable-regime-analysis.",
    )
    risk_report_parser.add_argument(
        "--enable-regime-analysis",
        action="store_true",
        help="Enable extended regime analysis from benchmark/index. Classifies regimes and computes performance by regime.",
    )
    risk_report_parser.add_argument(
        "--enable-factor-exposures",
        action="store_true",
        help="Enable factor exposure analysis. Requires --factor-returns-file.",
    )
    risk_report_parser.add_argument(
        "--factor-returns-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to factor returns file (CSV/Parquet) with timestamp and factor columns. Required if --enable-factor-exposures is set.",
    )
    risk_report_parser.add_argument(
        "--factor-exposures-window",
        type=int,
        default=252,
        metavar="INT",
        help="Rolling window size for factor exposure regression (default: 252 periods)",
    )
    risk_report_parser.set_defaults(func=risk_report_subcommand)

    # tca_report subcommand
    tca_report_parser = subparsers.add_parser(
        "tca_report",
        help="Generate transaction cost analysis (TCA) report",
        description="Generates transaction cost analysis reports from backtest outputs, including cost estimation, aggregation, and cost-adjusted risk metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/cli.py tca_report --backtest-dir output/backtests/experiment_123/
  python scripts/cli.py tca_report --backtest-dir output/backtests/experiment_123/ --output-dir output/tca_reports/
  python scripts/cli.py tca_report --backtest-dir output/backtests/experiment_123/ --spread-bps 10.0 --slippage-bps 5.0
        """,
    )

    tca_report_parser.add_argument(
        "--backtest-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory containing backtest outputs (must contain trades.csv or trades.parquet)",
    )

    tca_report_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory (default: <backtest-dir>/tca)",
    )

    tca_report_parser.add_argument(
        "--method",
        type=str,
        default="simple",
        choices=["simple", "adaptive"],
        help="Cost estimation method (default: simple)",
    )

    tca_report_parser.add_argument(
        "--commission-bps",
        type=float,
        default=0.5,
        help="Commission in basis points (default: 0.5)",
    )

    tca_report_parser.add_argument(
        "--spread-bps",
        type=float,
        default=None,
        help="Spread in basis points (default: 5.0 if not specified)",
    )

    tca_report_parser.add_argument(
        "--slippage-bps",
        type=float,
        default=3.0,
        help="Slippage in basis points (default: 3.0)",
    )

    tca_report_parser.set_defaults(func=tca_report_subcommand)

    # check_health subcommand
    check_health_parser = subparsers.add_parser(
        "check_health",
        help="Check backend health status (read-only, operations monitoring)",
        description="Health checks for backend operations (existence, plausibility, status interpretation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic health check
  python scripts/cli.py check_health --backtests-root output/backtests/
  
  # With custom thresholds
  python scripts/cli.py check_health --backtests-root output/backtests/ --min-sharpe 0.5 --max-drawdown-min -0.3
  
  # JSON output format
  python scripts/cli.py check_health --backtests-root output/backtests/ --format json
  
  # With paper track checks
  python scripts/cli.py check_health --backtests-root output/backtests/ --paper-track-root output/paper_track/ --paper-track-days 3
  
  # Skip paper track if missing
  python scripts/cli.py check_health --backtests-root output/backtests/ --skip-paper-track-if-missing
        """,
    )
    check_health_parser.add_argument(
        "--backtests-root",
        type=Path,
        default=Path("output/backtests/"),
        help="Root directory containing backtest outputs (default: output/backtests/)",
    )
    check_health_parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Lookback window in days for historical comparison (default: 60)",
    )
    check_health_parser.add_argument(
        "--benchmark-symbol",
        type=str,
        default=None,
        help="Benchmark symbol (e.g., 'SPY') for correlation checks",
    )
    check_health_parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=None,
        help="Path to benchmark file (CSV/Parquet with timestamp, returns/close)",
    )
    check_health_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for health reports (default: output/health/)",
    )
    check_health_parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "both"],
        default="text",
        help="Output format: 'text' for human-readable, 'json' for machine-readable, 'both' for both (default: text)",
    )
    check_health_parser.add_argument(
        "--min-sharpe",
        type=float,
        default=0.0,
        help="Minimum acceptable Sharpe ratio (default: 0.0)",
    )
    check_health_parser.add_argument(
        "--max-drawdown-min",
        type=float,
        default=-0.40,
        help="Minimum acceptable max drawdown (more negative = worse, default: -0.40)",
    )
    check_health_parser.add_argument(
        "--max-drawdown-max",
        type=float,
        default=0.0,
        help="Maximum acceptable max drawdown (less negative = better, default: 0.0)",
    )
    check_health_parser.add_argument(
        "--max-turnover",
        type=float,
        default=10.0,
        help="Maximum acceptable turnover (default: 10.0)",
    )
    check_health_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    check_health_parser.add_argument(
        "--paper-track-root",
        type=Path,
        default=None,
        help="Root directory for paper track outputs (default: auto-detect under output/paper_track/)",
    )

    check_health_parser.add_argument(
        "--paper-track-days",
        type=int,
        default=3,
        help="Maximum allowed age in days for paper track runs (default: 3)",
    )

    check_health_parser.add_argument(
        "--skip-paper-track-if-missing",
        action="store_true",
        default=False,
        help="Skip paper track checks if paper track directory doesn't exist (default: False = WARN)",
    )

    check_health_parser.add_argument(
        "--paper-track-max-daily-pnl-pct",
        type=float,
        default=10.0,
        help="Maximum acceptable daily PnL percentage for plausibility check (default: 10.0%%)",
    )

    check_health_parser.add_argument(
        "--paper-track-max-drawdown-min",
        type=float,
        default=-0.25,
        help="Minimum acceptable max drawdown for paper track (default: -0.25 = -25%%)",
    )

    check_health_parser.set_defaults(func=check_health_subcommand)

    # paper_track subcommand
    paper_track_parser = subparsers.add_parser(
        "paper_track",
        help="Run paper track for a single day or date range",
        description="Runs paper track execution for trading strategies, executing the complete daily flow: load state -> compute signals -> size positions -> simulate fills -> update state -> write artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available strategies
  python scripts/cli.py paper_track --list
  
  # Run for single day (with strategy name)
  python scripts/cli.py paper_track --strategy-name trend_baseline_example --as-of 2025-01-15
  
  # Run for single day (with config file)
  python scripts/cli.py paper_track --config-file configs/paper_track/strategy.yaml --as-of 2025-01-15
  
  # Run for date range
  python scripts/cli.py paper_track --strategy-name trend_baseline_example --start-date 2025-01-15 --end-date 2025-01-20
  
  # Catch-up mode: automatically run from last_run_date+1 to today (or --as-of)
  python scripts/cli.py paper_track --strategy-name trend_baseline_example --catch-up
  python scripts/cli.py paper_track --strategy-name trend_baseline_example --catch-up --as-of 2025-01-20
  
  # Dry run (no files written)
  python scripts/cli.py paper_track --strategy-name trend_baseline_example --as-of 2025-01-15 --dry-run
  
  # Fail fast on errors
  python scripts/cli.py paper_track --strategy-name trend_baseline_example --start-date 2025-01-15 --end-date 2025-01-20 --fail-fast
        """,
    )

    paper_track_parser.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="List all available paper track configs and strategies (exits immediately)",
    )

    config_group = paper_track_parser.add_mutually_exclusive_group(required=False)
    config_group.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Path to paper track config file (YAML/JSON)",
    )
    config_group.add_argument(
        "--strategy-name",
        type=str,
        default=None,
        help="Strategy name (will auto-discover config from configs/paper_track/{name}.yaml or output/paper_track/{name}/config.yaml)",
    )

    paper_track_parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Single date to run (YYYY-MM-DD). Mutually exclusive with --start-date/--end-date",
    )

    paper_track_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for range (YYYY-MM-DD, inclusive). Requires --end-date",
    )

    paper_track_parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for range (YYYY-MM-DD, inclusive). Requires --start-date",
    )

    paper_track_parser.add_argument(
        "--catch-up",
        action="store_true",
        default=False,
        help=(
            "Catch-up mode: automatically compute date range from state last_run_date. "
            "If no --start-date/--end-date specified, starts from last_run_date+1 and ends at --as-of (or today). "
            "If no state exists, falls back to --as-of (single day) or errors."
        ),
    )

    paper_track_parser.add_argument(
        "--dry-run", action="store_true", help="Dry run mode: don't write any files"
    )

    paper_track_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error (default: continue and log errors)",
    )

    paper_track_parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-run days even if run directory already exists (default: skip existing days)",
    )

    paper_track_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    paper_track_parser.set_defaults(func=paper_track_subcommand)

    # run_phase4_tests subcommand
    tests_parser = subparsers.add_parser(
        "run_phase4_tests",
        help="Run Phase-4 test suite",
        description="Runs the Phase-4 regression test suite (~13s, 110 tests).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/cli.py run_phase4_tests
  python scripts/cli.py run_phase4_tests --verbose
  python scripts/cli.py run_phase4_tests --durations 5
  python scripts/cli.py run_phase4_tests --verbose --durations 10
        """,
    )
    tests_parser.add_argument(
        "--durations",
        type=int,
        default=None,
        metavar="N",
        help="Show N slowest tests (e.g., 5 for --durations=5)",
    )
    tests_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show verbose test output (-vv instead of -q)",
    )
    tests_parser.set_defaults(func=run_phase4_tests_subcommand)

    # info subcommand
    info_parser = subparsers.add_parser(
        "info",
        help="Show project information",
        description="Shows project information, available subcommands, and documentation links.",
    )
    info_parser.set_defaults(func=info_subcommand)

    return parser


def main() -> int:
    """Main entry point for central CLI.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = create_parser()
    args = parser.parse_args()

    # Handle --version flag (before subcommand routing)
    if hasattr(args, "version") and args.version:
        print_version()
        return 0

    # Route to appropriate subcommand
    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.error("No subcommand specified. Use --help for usage.")


if __name__ == "__main__":
    sys.exit(main())
