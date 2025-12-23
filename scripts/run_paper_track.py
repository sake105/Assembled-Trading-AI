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
from typing import Any, Literal

import pandas as pd
import yaml

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.logging_config import generate_run_id
from src.assembled_core.paper.paper_track import (
    PaperTrackConfig,
    PaperTrackDayResult,
    load_paper_state,
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


def validate_config_dict(config_dict: dict[str, Any], config_path: Path) -> None:
    """Validate config dictionary with clear error messages.

    Args:
        config_dict: Raw config dictionary
        config_path: Path to config file (for error messages)

    Raises:
        ValueError: If config is invalid with helpful error message
    """
    errors = []

    # Required fields
    strategy_name = config_dict.get("strategy_name", "").strip() if isinstance(config_dict.get("strategy_name"), str) else ""
    if not strategy_name:
        errors.append("strategy_name: Required field missing or empty")

    strategy_type = config_dict.get("strategy_type") or (config_dict.get("strategy", {}) or {}).get("type", "")
    if not strategy_type or strategy_type not in ("trend_baseline", "multifactor_long_short"):
        errors.append(
            f"strategy_type: Must be 'trend_baseline' or 'multifactor_long_short', got: {strategy_type}"
        )

    # Universe file validation
    universe_raw = config_dict.get("universe")
    if not universe_raw:
        errors.append("universe: Required field missing")
    elif isinstance(universe_raw, dict):
        universe_file = universe_raw.get("file")
        if not universe_file:
            errors.append("universe.file: Required field missing")
    elif not isinstance(universe_raw, str):
        errors.append("universe: Must be a string (file path) or dict with 'file' key")

    # Trading freq validation
    trading = config_dict.get("trading", {})
    freq = trading.get("freq", "1d") if isinstance(trading, dict) else "1d"
    if freq not in ("1d", "5min"):
        errors.append(f"trading.freq: Must be '1d' or '5min', got: {freq}")

    # Portfolio seed_capital validation
    portfolio = config_dict.get("portfolio", {})
    if isinstance(portfolio, dict):
        seed_capital = portfolio.get("seed_capital") or portfolio.get("start_capital")
        if seed_capital is not None:
            try:
                seed_capital_float = float(seed_capital)
                if seed_capital_float <= 0:
                    errors.append(f"portfolio.seed_capital: Must be > 0, got: {seed_capital_float}")
            except (ValueError, TypeError):
                errors.append(f"portfolio.seed_capital: Must be a number, got: {seed_capital}")

    # Costs validation
    costs = config_dict.get("costs", {})
    if isinstance(costs, dict):
        for cost_key in ("commission_bps", "spread_w", "impact_w"):
            if cost_key in costs:
                try:
                    cost_value = float(costs[cost_key])
                    if cost_value < 0:
                        errors.append(f"costs.{cost_key}: Must be >= 0, got: {cost_value}")
                except (ValueError, TypeError):
                    errors.append(f"costs.{cost_key}: Must be a number, got: {costs[cost_key]}")

    # Output format validation
    output = config_dict.get("output", {})
    if isinstance(output, dict):
        output_format = output.get("format", "csv")
        if output_format not in ("csv", "parquet"):
            errors.append(f"output.format: Must be 'csv' or 'parquet', got: {output_format}")

    # Output root validation (if provided, should be valid path format)
    if isinstance(output, dict) and "root" in output:
        output_root_raw = output["root"]
        if not isinstance(output_root_raw, (str, Path)):
            errors.append(f"output.root: Must be a string or Path, got: {type(output_root_raw)}")

    if errors:
        error_msg = f"Config validation failed for {config_path}:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)


def discover_paper_track_configs() -> list[tuple[str, Path]]:
    """Discover available paper track config files.

    Searches in:
    1. configs/paper_track/*.{yaml,yml,json}
    2. output/paper_track/*/config.{yaml,yml,json}

    Returns:
        List of tuples: (strategy_name, config_path)
    """
    configs = []

    # 1. Search in configs/paper_track/
    configs_dir = ROOT / "configs" / "paper_track"
    if configs_dir.exists():
        for ext in (".yaml", ".yml", ".json"):
            for config_path in configs_dir.glob(f"*{ext}"):
                if config_path.is_file():
                    # Extract strategy name from filename (without extension)
                    strategy_name = config_path.stem
                    configs.append((strategy_name, config_path))

    # 2. Search in output/paper_track/*/config.{yaml,yml,json}
    output_paper_track_dir = ROOT / "output" / "paper_track"
    if output_paper_track_dir.exists():
        for strategy_dir in output_paper_track_dir.iterdir():
            if not strategy_dir.is_dir():
                continue
            for ext in (".yaml", ".yml", ".json"):
                config_path = strategy_dir / f"config{ext}"
                if config_path.exists() and config_path.is_file():
                    strategy_name = strategy_dir.name
                    # Only add if not already in list (prioritize configs/paper_track/)
                    if not any(name == strategy_name for name, _ in configs):
                        configs.append((strategy_name, config_path))

    # Sort by strategy name
    configs.sort(key=lambda x: x[0])
    return configs


def find_config_by_strategy_name(strategy_name: str) -> Path | None:
    """Find config file by strategy name.

    Searches in:
    1. configs/paper_track/{strategy_name}.{yaml,yml,json}
    2. output/paper_track/{strategy_name}/config.{yaml,yml,json}

    Args:
        strategy_name: Name of the strategy

    Returns:
        Path to config file if found, None otherwise
    """
    # Try configs/paper_track/ first
    configs_dir = ROOT / "configs" / "paper_track"
    for ext in (".yaml", ".yml", ".json"):
        config_path = configs_dir / f"{strategy_name}{ext}"
        if config_path.exists() and config_path.is_file():
            return config_path

    # Try output/paper_track/{strategy_name}/config.*
    strategy_dir = ROOT / "output" / "paper_track" / strategy_name
    for ext in (".yaml", ".yml", ".json"):
        config_path = strategy_dir / f"config{ext}"
        if config_path.exists() and config_path.is_file():
            return config_path

    return None


def list_paper_track_configs() -> int:
    """List available paper track configs and strategies.

    Returns:
        Exit code (0 = success)
    """
    configs = discover_paper_track_configs()

    if not configs:
        print("No paper track configs found.")
        print("\nTo create a config:")
        print("  1. Create a YAML file in configs/paper_track/{strategy_name}.yaml")
        print("  2. Or create config.yaml in output/paper_track/{strategy_name}/")
        return 0

    print(f"Found {len(configs)} paper track config(s):\n")
    print(f"{'Strategy Name':<30} {'Config Path':<60}")
    print("-" * 90)

    for strategy_name, config_path in configs:
        # Make path relative to ROOT if possible
        try:
            rel_path = config_path.relative_to(ROOT)
            path_str = str(rel_path)
        except ValueError:
            path_str = str(config_path)

        print(f"{strategy_name:<30} {path_str:<60}")

    print("\nTo run a strategy:")
    print("  python scripts/cli.py paper_track --strategy-name <name> --as-of 2025-01-15")
    print("  python scripts/cli.py paper_track --config-file <path> --as-of 2025-01-15")
    return 0


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

    # Validate config structure before processing
    validate_config_dict(raw, config_path)

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

    # Output format (default: "csv")
    output_format_raw = output.get("format", "csv")
    if output_format_raw not in ("csv", "parquet"):
        raise ValueError(f"output.format must be 'csv' or 'parquet', got: {output_format_raw}")
    output_format = output_format_raw  # type: ignore[assignment]

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
        output_format=output_format,  # type: ignore[arg-type]
    )


def compute_date_list(
    as_of: str | None,
    start_date: str | None,
    end_date: str | None,
    catch_up: bool = False,
    state_path: Path | None = None,
    strategy_name: str | None = None,
) -> list[pd.Timestamp]:
    """Compute list of dates to run.

    Args:
        as_of: Single date (YYYY-MM-DD) if provided
        start_date: Start date (YYYY-MM-DD) for range
        end_date: End date (YYYY-MM-DD) for range (inclusive)
        catch_up: If True and no start/end specified, compute from state last_run_date
        state_path: Path to state file (required if catch_up=True and no start/end)
        strategy_name: Strategy name (required if catch_up=True and loading state)

    Returns:
        List of pd.Timestamp (UTC) dates

    Raises:
        ValueError: If arguments are inconsistent
    """
    # Catch-up mode: if enabled and no explicit start/end, compute from state
    if catch_up and not start_date and not end_date:
        if not state_path:
            raise ValueError(
                "catch_up=True requires state_path to determine last_run_date"
            )
        if not strategy_name:
            raise ValueError(
                "catch_up=True requires strategy_name to load state"
            )

        # Load state to get last_run_date
        from src.assembled_core.paper.paper_track import load_paper_state

        state = load_paper_state(state_path, strategy_name)
        if state and state.last_run_date is not None:
            # Start from day after last run
            start_ts = state.last_run_date + pd.Timedelta(days=1)
            start_str = start_ts.strftime("%Y-%m-%d")
            # End at as_of (if provided) or today
            # Note: as_of is used as end_date in catch-up mode, not as single date
            if as_of:
                end_str = as_of
            else:
                end_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

            logger.info(
                f"Catch-up mode: last_run_date={state.last_run_date.date()}, "
                f"computed range: {start_str} to {end_str}"
            )
            start_date = start_str
            end_date = end_str
        else:
            # No state exists: if as_of provided, use single day, otherwise error
            if as_of:
                logger.info(f"Catch-up mode: no state found, using single day: {as_of}")
                return [pd.Timestamp(as_of, tz="UTC").normalize()]
            else:
                raise ValueError(
                    "catch_up=True but no state found and --as-of not provided. "
                    "Either provide --as-of or ensure state file exists."
                )
    elif as_of and not catch_up:
        # Normal mode: as_of is single date (not catch-up)
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


def _is_friday(date: pd.Timestamp) -> bool:
    """Check if date is a Friday.

    Args:
        date: Date to check (pd.Timestamp)

    Returns:
        True if date is Friday (weekday == 4), False otherwise
    """
    return date.weekday() == 4


def _is_month_end(date: pd.Timestamp) -> bool:
    """Check if date is the last day of the month.

    Args:
        date: Date to check (pd.Timestamp)

    Returns:
        True if date is the last day of its month, False otherwise
    """
    # Get next day
    next_day = date + pd.Timedelta(days=1)
    # If next day's month is different, current day is month end
    return next_day.month != date.month


def _should_generate_risk_report(
    date: pd.Timestamp,
    frequency: Literal["daily", "weekly", "monthly"],
    is_last_day_in_range: bool = False,
) -> bool:
    """Determine if risk report should be generated for a given date.

    Args:
        date: Date to check (pd.Timestamp)
        frequency: Report frequency ("daily", "weekly", "monthly")
        is_last_day_in_range: If True, always generate if frequency matches (for end_date)

    Returns:
        True if risk report should be generated, False otherwise
    """
    if frequency == "daily":
        return True

    if frequency == "weekly":
        # Generate on Friday (or last day in range if it's a weekend)
        if is_last_day_in_range:
            # Check if we're at end of week (Friday or weekend)
            return date.weekday() >= 4
        return _is_friday(date)

    if frequency == "monthly":
        # Generate on month-end (or last day in range)
        if is_last_day_in_range:
            return True
        return _is_month_end(date)

    return False


def _generate_risk_report_for_paper_track(
    output_root: Path,
    strategy_name: str,
    date: pd.Timestamp,
    benchmark_symbol: str | None,
    factor_returns_file: Path | None,
    config: PaperTrackConfig,
) -> None:
    """Generate risk report for paper track strategy.

    Args:
        output_root: Output root directory (e.g., output/paper_track/{strategy_name})
        strategy_name: Strategy name
        date: Date for the report
        benchmark_symbol: Optional benchmark symbol (e.g., "SPY")
        factor_returns_file: Optional path to factor returns file
        config: PaperTrackConfig

    Side effects:
        Creates risk report files in output_root/risk_reports/YYYYMMDD/

    Raises:
        RuntimeError: If risk report generation fails
    """
    # Import here to avoid circular dependencies
    from scripts.generate_risk_report import generate_risk_report

    # Prepare report directory
    report_dir = output_root / "risk_reports" / date.strftime("%Y%m%d")
    report_dir.mkdir(parents=True, exist_ok=True)

    # For paper track, we use aggregates/equity_curve.csv
    aggregates_dir = output_root / "aggregates"
    equity_curve_path = aggregates_dir / "equity_curve.csv"

    if not equity_curve_path.exists():
        logger.warning(
            f"Equity curve not found at {equity_curve_path}. "
            "Risk report requires aggregated equity curve. Skipping."
        )
        return

    # Call generate_risk_report with aggregates_dir as backtest_dir
    # generate_risk_report will look for equity_curve.csv, trades_all.csv, etc.
    exit_code = generate_risk_report(
        backtest_dir=aggregates_dir,
        output_dir=report_dir,
        benchmark_symbol=benchmark_symbol,
        factor_returns_file=factor_returns_file,
        enable_regime_analysis=False,  # Can be enabled later if regime data available
        enable_factor_exposures=(factor_returns_file is not None),
    )

    if exit_code != 0:
        raise RuntimeError(f"Risk report generation failed with exit code {exit_code}")


def run_paper_track_from_cli(
    config_file: Path,
    as_of: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    catch_up: bool = False,
    dry_run: bool = False,
    fail_fast: bool = False,
    rerun: bool = False,
    generate_risk_report: bool = False,
    risk_report_frequency: Literal["daily", "weekly", "monthly"] = "weekly",
    benchmark_symbol: str | None = None,
    factor_returns_file: Path | None = None,
) -> int:
    """Run paper track from CLI arguments.

    Args:
        config_file: Path to config file (YAML/JSON)
        as_of: Single date (YYYY-MM-DD) or None
        start_date: Start date (YYYY-MM-DD) or None
        end_date: End date (YYYY-MM-DD) or None
        catch_up: If True, automatically compute date range from state
        dry_run: If True, don't write outputs
        fail_fast: If True, stop on first error
        rerun: If True, re-run days even if run directory already exists (default: skip)

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

        # Prepare state path (needed for catch-up)
        output_root = config.output_root or (
            ROOT / "output" / "paper_track" / config.strategy_name
        )
        state_path = output_root / "state" / "state.json"

        # Compute date list (with catch-up support)
        dates = compute_date_list(
            as_of=as_of,
            start_date=start_date,
            end_date=end_date,
            catch_up=catch_up,
            state_path=state_path if catch_up else None,
            strategy_name=config.strategy_name if catch_up else None,
        )
        logger.info(
            f"Running paper track for {len(dates)} day(s): {dates[0].date()} to {dates[-1].date()}"
        )

        if dry_run:
            logger.info("DRY RUN MODE: No files will be written")

        # Generate run_id for this execution
        run_id = generate_run_id(prefix="paper_track")
        logger.info(f"Run-ID: {run_id}")

        # Track start time
        start_time = pd.Timestamp.utcnow()

        # Results tracking
        results: list[PaperTrackDayResult] = []
        failed_days: list[tuple[pd.Timestamp, str]] = []
        skipped_days: list[pd.Timestamp] = []

        # Run for each day
        for i, date in enumerate(dates):
            logger.info(f"Processing day {i + 1}/{len(dates)}: {date.date()}")

            # Check if run directory already exists (skip logic)
            run_date_str = date.strftime("%Y%m%d")
            run_dir = output_root / "runs" / run_date_str
            if run_dir.exists() and not rerun:
                logger.info(
                    f"Run directory {run_dir} already exists, skipping day {date.date()} "
                    f"(use --rerun to force re-run)"
                )
                skipped_days.append(date)
                # Still load state to continue with next day
                try:
                    state = load_paper_state(state_path, config.strategy_name)
                    if state:
                        logger.debug(
                            f"State loaded: equity={state.equity:.2f}, last_run_date={state.last_run_date}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to load state for skip check: {e}")
                continue

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
                # If rerun is enabled and run_dir exists, backup or delete it first
                if rerun and run_dir.exists() and not dry_run:
                    logger.warning(f"Re-running day {date.date()}, existing run directory will be overwritten")
                    # Backup the old directory
                    import shutil
                    backup_dir = run_dir.with_suffix(f".backup.{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
                    try:
                        shutil.move(str(run_dir), str(backup_dir))
                        logger.info(f"Backed up existing run directory to {backup_dir.name}")
                    except Exception as e:
                        logger.warning(f"Failed to backup run directory: {e}, proceeding anyway")

                if not dry_run and result.status == "success":
                    write_paper_day_outputs(result, output_root, config=config, run_id=run_id)
                    logger.info(f"Outputs written for {date.date()}")

                    # Update state file
                    save_paper_state(result.state_after, state_path)
                    logger.info(f"State updated for {date.date()}")

                    # Generate risk report if enabled and trigger matches
                    if generate_risk_report:
                        is_last_day = (i == len(dates) - 1)
                        if _should_generate_risk_report(
                            date, risk_report_frequency, is_last_day
                        ):
                            try:
                                _generate_risk_report_for_paper_track(
                                    output_root=output_root,
                                    strategy_name=config.strategy_name,
                                    date=date,
                                    benchmark_symbol=benchmark_symbol,
                                    factor_returns_file=factor_returns_file,
                                    config=config,
                                )
                                logger.info(
                                    f"Risk report generated for {date.date()} ({risk_report_frequency})"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to generate risk report for {date.date()}: {e}",
                                    exc_info=True,
                                )
                                # Don't fail the entire run if risk report generation fails
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

        # Track end time
        end_time = pd.Timestamp.utcnow()

        # Write run summary CSV and JSON (if not dry-run)
        if not dry_run:
            # Write CSV summary (existing format)
            if results:
                summary_path_csv = output_root / "paper_track_run_summary.csv"
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
                summary_df.to_csv(summary_path_csv, index=False)
                logger.info(f"Run summary CSV written: {summary_path_csv}")

            # Write JSON summary (new format with run_id and per-day statuses)
            summary_path_json = output_root / "paper_track_run_summary.json"
            success_count = len([r for r in results if r.status == "success"])
            failure_count = len([r for r in results if r.status == "error"])
            skipped_count = len(skipped_days)

            # Build per-day statuses
            per_day_statuses = []
            for result in results:
                per_day_statuses.append(
                    {
                        "date": result.date.strftime("%Y-%m-%d"),
                        "status": result.status,
                        "equity": float(result.state_after.equity),
                        "cash": float(result.state_after.cash),
                        "daily_return_pct": float(result.daily_return_pct),
                        "daily_pnl": float(result.daily_pnl),
                        "trades_count": result.trades_count,
                        "buy_count": result.buy_count,
                        "sell_count": result.sell_count,
                        "error_message": result.error_message,
                    }
                )
            # Add skipped days
            for date in skipped_days:
                per_day_statuses.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "status": "skipped",
                        "equity": None,
                        "cash": None,
                        "daily_return_pct": None,
                        "daily_pnl": None,
                        "trades_count": None,
                        "buy_count": None,
                        "sell_count": None,
                        "error_message": None,
                    }
                )
            # Sort by date
            per_day_statuses.sort(key=lambda x: x["date"])

            run_summary = {
                "run_id": run_id,
                "strategy_name": config.strategy_name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": (end_time - start_time).total_seconds(),
                "dry_run": dry_run,
                "rerun": rerun,
                "days_attempted": len(dates),
                "days_successful": success_count,
                "days_failed": failure_count,
                "days_skipped": skipped_count,
                "per_day_statuses": per_day_statuses,
                "config_file": str(config_file),
                "date_range": {
                    "start": dates[0].strftime("%Y-%m-%d") if dates else None,
                    "end": dates[-1].strftime("%Y-%m-%d") if dates else None,
                },
            }

            with open(summary_path_json, "w", encoding="utf-8") as f:
                json.dump(run_summary, f, indent=2, ensure_ascii=True)
            logger.info(f"Run summary JSON written: {summary_path_json}")

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
        "--list",
        action="store_true",
        default=False,
        help="List all available paper track configs and strategies (exits immediately)",
    )

    config_group = parser.add_mutually_exclusive_group(required=False)
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
        "--catch-up",
        action="store_true",
        default=False,
        help=(
            "Catch-up mode: automatically compute date range from state last_run_date. "
            "If no --start-date/--end-date specified, starts from last_run_date+1 and ends at --as-of (or today). "
            "If no state exists, falls back to --as-of (single day) or errors."
        ),
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
        "--generate-risk-report",
        action="store_true",
        default=False,
        help="Generate risk reports (triggered based on --risk-report-frequency)",
    )

    parser.add_argument(
        "--risk-report-frequency",
        type=str,
        choices=["daily", "weekly", "monthly"],
        default="weekly",
        help=(
            "Risk report generation frequency. "
            "Weekly: on Friday (or end_date if weekend), "
            "Monthly: on month-end (or end_date), "
            "Daily: every day"
        ),
    )

    parser.add_argument(
        "--benchmark-symbol",
        type=str,
        default=None,
        help="Benchmark symbol for risk reports (e.g., 'SPY')",
    )

    parser.add_argument(
        "--factor-returns-file",
        type=Path,
        default=None,
        help="Path to factor returns file for risk reports (CSV/Parquet)",
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

    # Handle --list flag
    if args.list:
        exit_code = list_paper_track_configs()
        sys.exit(exit_code)

    # Determine config file
    config_file: Path | None = None
    if args.strategy_name:
        # Auto-discover config from strategy name
        config_file = find_config_by_strategy_name(args.strategy_name)
        if config_file is None:
            logger.error(
                f"Config not found for strategy '{args.strategy_name}'. "
                "Run with --list to see available strategies."
            )
            sys.exit(1)
        logger.info(f"Auto-discovered config for '{args.strategy_name}': {config_file}")
    elif args.config_file:
        config_file = args.config_file
    else:
        logger.error(
            "Either --config-file or --strategy-name must be provided. "
            "Run with --list to see available strategies."
        )
        sys.exit(1)

    try:
        exit_code = run_paper_track_from_cli(
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
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.error("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Paper track runner failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
