#!/usr/bin/env python
"""
Paper Track Runner - Daily execution script.

This script runs paper track for a single day or date range, executing the
complete daily flow: load state -> compute signals -> size positions ->
simulate fills -> update state -> write artifacts.

Design-Referenz:
- docs/PAPER_TRACK_RUNNER_A5_DESIGN.md
- docs/PAPER_TRACK_PLAYBOOK.md

Usage:
    python scripts/run_paper_track.py --config-file configs/paper_track/strategy.yaml --as-of 2025-01-15
    python scripts/run_paper_track.py --config-file configs/paper_track/strategy.yaml --start-date 2025-01-15 --end-date 2025-01-20
    python scripts/run_paper_track.py --config-file configs/paper_track/strategy.yaml --as-of 2025-01-15 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.paper.paper_track import (
    PaperTrackConfig,
    PaperTrackDayResult,
    run_paper_day,
    save_paper_state,
    write_paper_day_outputs,
)
from src.assembled_core.utils.random_state import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_paper_track_config(path: Path) -> PaperTrackConfig:
    """Load PaperTrackConfig from YAML/JSON file.

    Args:
        path: Path to config file

    Returns:
        PaperTrackConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    # Resolve config path
    config_path = path if path.is_absolute() else (ROOT / path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        raw = _load_yaml(config_path)
    elif suffix == ".json":
        raw = _load_json(config_path)
    else:
        raise ValueError(f"Unsupported config file extension: {config_path.suffix}")

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping/object")

    # Extract required fields
    strategy_name = str(raw.get("strategy_name") or "").strip()
    if not strategy_name:
        raise ValueError("strategy_name must be set in config")

    strategy_type = str(
        raw.get("strategy_type") or raw.get("strategy", {}).get("type", "")
    ).strip()
    if not strategy_type:
        raise ValueError("strategy_type must be set in config")

    # Universe file
    universe_raw = raw.get("universe", {})
    if isinstance(universe_raw, dict):
        universe_file = universe_raw.get("file")
        if universe_file:
            universe_file_path = Path(universe_file)
            if not universe_file_path.is_absolute():
                # Try relative to config file directory first, then ROOT
                config_dir = config_path.parent
                candidate = (config_dir / universe_file_path).resolve()
                if candidate.exists():
                    universe_file = candidate
                else:
                    universe_file = (ROOT / universe_file_path).resolve()
            else:
                universe_file = universe_file_path
        else:
            raise ValueError("universe.file must be set in config")
    else:
        universe_file_raw = str(universe_raw)
        universe_file_path = Path(universe_file_raw)
        if not universe_file_path.is_absolute():
            config_dir = config_path.parent
            candidate = (config_dir / universe_file_path).resolve()
            if candidate.exists():
                universe_file = candidate
            else:
                universe_file = (ROOT / universe_file_path).resolve()
        else:
            universe_file = universe_file_path

    if not universe_file.exists():
        raise FileNotFoundError(f"Universe file not found: {universe_file}")

    # Trading parameters
    trading = raw.get("trading", {})
    freq = str(trading.get("freq", "1d"))

    # Strategy params
    strategy_params = (
        raw.get("strategy", {}).get("params", {})
        if isinstance(raw.get("strategy"), dict)
        else {}
    )

    # Costs
    costs = raw.get("costs", {})
    commission_bps = float(costs.get("commission_bps", 0.5))
    spread_w = float(costs.get("spread_w", 0.25))
    impact_w = float(costs.get("impact_w", 0.5))

    # Portfolio
    portfolio = raw.get("portfolio", {})
    seed_capital = float(
        portfolio.get("seed_capital", portfolio.get("start_capital", 100000.0))
    )

    # Optional: random seed
    random_seed = raw.get("random_seed")
    if random_seed is not None:
        random_seed = int(random_seed)

    # Optional: enable_pit_checks
    integration = raw.get("integration", {})
    enable_pit_checks = bool(integration.get("enable_pit_checks", True))

    # Output root
    output = raw.get("output", {})
    output_root_raw = output.get("root", "output/paper_track")
    output_root = (
        (ROOT / output_root_raw).resolve()
        if not Path(output_root_raw).is_absolute()
        else Path(output_root_raw)
    )
    strategy_dir = output.get("strategy_dir", strategy_name)
    output_root = output_root / strategy_dir

    return PaperTrackConfig(
        strategy_name=strategy_name,
        strategy_type=strategy_type,  # type: ignore[arg-type]
        universe_file=universe_file,
        freq=freq,  # type: ignore[arg-type]
        seed_capital=seed_capital,
        commission_bps=commission_bps,
        spread_w=spread_w,
        impact_w=impact_w,
        strategy_params=strategy_params,
        enable_pit_checks=enable_pit_checks,
        random_seed=random_seed,
        output_root=output_root,
    )


def compute_date_list(
    as_of: str | None,
    start_date: str | None,
    end_date: str | None,
) -> list[pd.Timestamp]:
    """Compute list of dates to run.

    Args:
        as_of: Single date (YYYY-MM-DD) if provided
        start_date: Start date (YYYY-MM-DD) for range
        end_date: End date (YYYY-MM-DD) for range (inclusive)

    Returns:
        List of pd.Timestamp (UTC) dates

    Raises:
        ValueError: If arguments are inconsistent
    """
    if as_of:
        if start_date or end_date:
            raise ValueError("Cannot specify both --as-of and --start-date/--end-date")
        return [pd.Timestamp(as_of, tz="UTC").normalize()]

    if not start_date or not end_date:
        raise ValueError(
            "Must specify either --as-of or both --start-date and --end-date"
        )

    start = pd.Timestamp(start_date, tz="UTC").normalize()
    end = pd.Timestamp(end_date, tz="UTC").normalize()

    if start > end:
        raise ValueError(f"start_date ({start_date}) must be <= end_date ({end_date})")

    # Generate inclusive range
    dates = pd.date_range(start=start, end=end, freq="D")
    return [d.normalize() for d in dates]


def run_paper_track_from_cli(
    config_file: Path,
    as_of: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    dry_run: bool = False,
    fail_fast: bool = False,
) -> int:
    """Run paper track from CLI arguments.

    Args:
        config_file: Path to config file (YAML/JSON)
        as_of: Single date (YYYY-MM-DD) or None
        start_date: Start date (YYYY-MM-DD) or None
        end_date: End date (YYYY-MM-DD) or None
        dry_run: If True, don't write outputs
        fail_fast: If True, stop on first error

    Returns:
        Exit code (0 = success, 1 = error)
    """
    try:
        # Load config
        logger.info(f"Loading config from {config_file}")
        config = load_paper_track_config(config_file)

        # Set random seed if provided
        if config.random_seed is not None:
            set_global_seed(config.random_seed)
            logger.info(f"Random seed set to {config.random_seed}")

        # Compute date list
        dates = compute_date_list(as_of, start_date, end_date)
        logger.info(
            f"Running paper track for {len(dates)} day(s): {dates[0].date()} to {dates[-1].date()}"
        )

        if dry_run:
            logger.info("DRY RUN MODE: No files will be written")

        # Prepare state path
        output_root = config.output_root or (
            ROOT / "output" / "paper_track" / config.strategy_name
        )
        state_path = output_root / "state" / "state.json"

        # Results tracking
        results: list[PaperTrackDayResult] = []
        failed_days: list[tuple[pd.Timestamp, str]] = []

        # Run for each day
        for i, date in enumerate(dates):
            logger.info(f"Processing day {i + 1}/{len(dates)}: {date.date()}")

            try:
                # Run paper day
                result = run_paper_day(
                    config=config,
                    as_of=date,
                    state_path=state_path if not dry_run else None,
                )
                results.append(result)

                # Check status
                if result.status == "error":
                    error_msg = result.error_message or "Unknown error"
                    logger.error(f"Day {date.date()} failed: {error_msg}")
                    failed_days.append((date, error_msg))

                    if fail_fast:
                        logger.error("--fail-fast enabled, stopping")
                        return 1
                    continue

                # Write outputs (if not dry-run and success)
                if not dry_run and result.status == "success":
                    write_paper_day_outputs(result, output_root)
                    logger.info(f"Outputs written for {date.date()}")

                    # Update state file
                    save_paper_state(result.state_after, state_path)
                    logger.info(f"State updated for {date.date()}")
                else:
                    logger.info(
                        f"Skipping output write (dry_run={dry_run}, status={result.status})"
                    )

            except Exception as e:
                logger.error(
                    f"Day {date.date()} failed with exception: {e}", exc_info=True
                )
                failed_days.append((date, str(e)))

                if fail_fast:
                    logger.error("--fail-fast enabled, stopping")
                    return 1
                continue

        # Summary
        success_count = len([r for r in results if r.status == "success"])
        logger.info(f"Summary: {success_count}/{len(dates)} days succeeded")

        if failed_days:
            logger.warning(f"{len(failed_days)} day(s) failed:")
            for date, error in failed_days:
                logger.warning(f"  {date.date()}: {error}")

            if not fail_fast:
                logger.warning("Some days failed, but continuing due to --no-fail-fast")

        # Write run summary (if not dry-run)
        if not dry_run and results:
            summary_path = output_root / "paper_track_run_summary.csv"
            summary_data = []
            for result in results:
                summary_data.append(
                    {
                        "date": result.date.strftime("%Y-%m-%d"),
                        "status": result.status,
                        "equity": result.state_after.equity,
                        "cash": result.state_after.cash,
                        "daily_return_pct": result.daily_return_pct,
                        "daily_pnl": result.daily_pnl,
                        "trades_count": result.trades_count,
                        "buy_count": result.buy_count,
                        "sell_count": result.sell_count,
                        "error_message": result.error_message or "",
                    }
                )

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Run summary written: {summary_path}")

        # Return exit code
        if failed_days and fail_fast:
            return 1
        elif failed_days:
            return 1  # Warning exit code if some days failed
        else:
            return 0

    except Exception as e:
        logger.error(f"Paper track runner failed: {e}", exc_info=True)
        return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (for testing). If None, uses sys.argv.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Run paper track for a single day or date range",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run for single day
  python scripts/run_paper_track.py --config-file configs/paper_track/strategy.yaml --as-of 2025-01-15
  
  # Run for date range
  python scripts/run_paper_track.py --config-file configs/paper_track/strategy.yaml --start-date 2025-01-15 --end-date 2025-01-20
  
  # Dry run (no files written)
  python scripts/run_paper_track.py --config-file configs/paper_track/strategy.yaml --as-of 2025-01-15 --dry-run
  
  # Fail fast on errors
  python scripts/run_paper_track.py --config-file configs/paper_track/strategy.yaml --start-date 2025-01-15 --end-date 2025-01-20 --fail-fast
        """,
    )

    parser.add_argument(
        "--config-file",
        type=Path,
        required=True,
        help="Path to paper track config file (YAML/JSON)",
    )

    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Single date to run (YYYY-MM-DD). Mutually exclusive with --start-date/--end-date",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for range (YYYY-MM-DD, inclusive). Requires --end-date",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for range (YYYY-MM-DD, inclusive). Requires --start-date",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Dry run mode: don't write any files"
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error (default: continue and log errors)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        exit_code = run_paper_track_from_cli(
            config_file=args.config_file,
            as_of=args.as_of,
            start_date=args.start_date,
            end_date=args.end_date,
            dry_run=args.dry_run,
            fail_fast=args.fail_fast,
        )
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.error("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Paper track runner failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
