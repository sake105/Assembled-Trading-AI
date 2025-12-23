"""Health Check Script for Backend Operations.

This script performs read-only health checks on backend operations, including:
- Existence checks for backtest runs, risk reports, TCA reports
- Plausibility checks for performance metrics (drawdown, Sharpe, turnover, etc.)
- Benchmark correlation checks (optional)
- Paper track freshness, artifacts, and metrics checks (optional)

Usage:
    python scripts/check_health.py --backtests-root output/backtests/
    python scripts/check_health.py --backtests-root output/backtests/ --days 30 --format json
    python scripts/cli.py check_health --backtests-root output/backtests/ --min-sharpe 0.5
    python scripts/cli.py check_health --paper-track-root output/paper_track/ --paper-track-days 3
    python scripts/cli.py check_health --skip-paper-track-if-missing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.ops.health_check import (
    HealthCheck,
    HealthCheckResult,
    aggregate_overall_status,
    health_result_to_dict,
    render_health_summary_text,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Check backend health status (read-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic health check
  python scripts/check_health.py --backtests-root output/backtests/
  
  # With custom thresholds
  python scripts/check_health.py --backtests-root output/backtests/ --min-sharpe 0.5 --max-drawdown-min -0.3
  
  # JSON output format
  python scripts/check_health.py --backtests-root output/backtests/ --format json
        """,
    )

    parser.add_argument(
        "--backtests-root",
        type=Path,
        default=Path("output/backtests/"),
        help="Root directory containing backtest outputs (default: output/backtests/)",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Lookback window in days for historical comparison (default: 60)",
    )

    parser.add_argument(
        "--benchmark-symbol",
        type=str,
        default=None,
        help="Benchmark symbol (e.g., 'SPY') for correlation checks",
    )

    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=None,
        help="Path to benchmark file (CSV/Parquet with timestamp, returns/close)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for health reports (default: output/health/)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "both"],
        default="text",
        help="Output format: 'text' for human-readable, 'json' for machine-readable, 'both' for both (default: text)",
    )

    parser.add_argument(
        "--min-sharpe",
        type=float,
        default=0.0,
        help="Minimum acceptable Sharpe ratio (default: 0.0)",
    )

    parser.add_argument(
        "--max-drawdown-min",
        type=float,
        default=-0.40,
        help="Minimum acceptable max drawdown (more negative = worse, default: -0.40)",
    )

    parser.add_argument(
        "--max-drawdown-max",
        type=float,
        default=0.0,
        help="Maximum acceptable max drawdown (less negative = better, default: 0.0)",
    )

    parser.add_argument(
        "--max-turnover",
        type=float,
        default=10.0,
        help="Maximum acceptable turnover (default: 10.0)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--paper-track-root",
        type=Path,
        default=None,
        help="Root directory for paper track outputs (default: auto-detect under output/paper_track/)",
    )

    parser.add_argument(
        "--paper-track-days",
        type=int,
        default=3,
        help="Maximum allowed age in days for paper track runs (default: 3)",
    )

    parser.add_argument(
        "--skip-paper-track-if-missing",
        action="store_true",
        default=False,
        help="Skip paper track checks if paper track directory doesn't exist (default: False = WARN)",
    )

    parser.add_argument(
        "--paper-track-max-daily-pnl-pct",
        type=float,
        default=10.0,
        help="Maximum acceptable daily PnL percentage for plausibility check (default: 10.0%%)",
    )

    parser.add_argument(
        "--paper-track-max-drawdown-min",
        type=float,
        default=-0.25,
        help="Minimum acceptable max drawdown for paper track (default: -0.25 = -25%%)",
    )

    return parser.parse_args()


def find_latest_backtest_dir(backtests_root: Path) -> Path | None:
    """Find the latest backtest directory.

    Searches for directories containing equity_curve.parquet or equity_curve.csv,
    and returns the one with the most recent modification time.

    Args:
        backtests_root: Root directory containing backtest outputs

    Returns:
        Path to latest backtest directory, or None if none found
    """
    if not backtests_root.exists():
        logger.warning(f"Backtests root does not exist: {backtests_root}")
        return None

    latest_dir = None
    latest_mtime = 0.0

    # Search for directories containing equity curve files
    for item in backtests_root.iterdir():
        if not item.is_dir():
            continue

        # Check for equity curve files
        equity_parquet = item / "equity_curve.parquet"
        equity_csv = item / "equity_curve.csv"

        if equity_parquet.exists() or equity_csv.exists():
            # Use the most recent modification time
            mtime = 0.0
            if equity_parquet.exists():
                mtime = max(mtime, equity_parquet.stat().st_mtime)
            if equity_csv.exists():
                mtime = max(mtime, equity_csv.stat().st_mtime)

            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_dir = item

    if latest_dir is None:
        logger.warning(f"No backtest directories found in {backtests_root}")

    return latest_dir


def load_equity_curve(backtest_dir: Path) -> pd.DataFrame | None:
    """Load equity curve from backtest directory.

    Args:
        backtest_dir: Backtest directory path

    Returns:
        DataFrame with equity curve, or None if not found
    """
    equity_parquet = backtest_dir / "equity_curve.parquet"
    equity_csv = backtest_dir / "equity_curve.csv"

    try:
        if equity_parquet.exists():
            df = pd.read_parquet(equity_parquet)
            logger.debug(f"Loaded equity curve from {equity_parquet}")
            return df
        elif equity_csv.exists():
            df = pd.read_csv(equity_csv)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"], utc=True, errors="coerce"
                )
            logger.debug(f"Loaded equity curve from {equity_csv}")
            return df
        else:
            logger.warning(f"No equity curve file found in {backtest_dir}")
            return None
    except Exception as e:
        logger.warning(f"Failed to load equity curve from {backtest_dir}: {e}")
        return None


def load_risk_summary(backtest_dir: Path) -> pd.DataFrame | None:
    """Load risk summary CSV from backtest directory or risk_reports subdirectory.

    Args:
        backtest_dir: Backtest directory path

    Returns:
        DataFrame with risk summary, or None if not found
    """
    # Try backtest_dir first
    risk_summary_path = backtest_dir / "risk_summary.csv"
    if risk_summary_path.exists():
        try:
            df = pd.read_csv(risk_summary_path)
            logger.debug(f"Loaded risk summary from {risk_summary_path}")
            return df
        except Exception as e:
            logger.warning(f"Failed to load risk summary from {risk_summary_path}: {e}")

    # Try risk_reports subdirectory
    risk_reports_dir = backtest_dir.parent / "risk_reports" / backtest_dir.name
    risk_summary_path = risk_reports_dir / "risk_summary.csv"
    if risk_summary_path.exists():
        try:
            df = pd.read_csv(risk_summary_path)
            logger.debug(f"Loaded risk summary from {risk_summary_path}")
            return df
        except Exception as e:
            logger.warning(f"Failed to load risk summary from {risk_summary_path}: {e}")

    logger.warning(f"No risk summary file found for {backtest_dir}")
    return None


def find_paper_track_strategies(paper_track_root: Path) -> list[Path]:
    """Find all paper track strategy directories.

    Args:
        paper_track_root: Root directory containing paper track outputs

    Returns:
        List of strategy directory paths (each should contain state/ and runs/ subdirectories)
    """
    if not paper_track_root.exists():
        return []

    strategies = []
    for item in paper_track_root.iterdir():
        if not item.is_dir():
            continue

        # Check if this looks like a strategy directory (has state/ or runs/)
        state_dir = item / "state"
        runs_dir = item / "runs"

        if state_dir.exists() or runs_dir.exists():
            strategies.append(item)

    return strategies


def load_paper_track_state(state_dir: Path) -> dict[str, Any] | None:
    """Load paper track state from JSON file.

    Args:
        state_dir: Directory containing state.json file

    Returns:
        Dictionary with state data, or None if not found
    """
    state_path = state_dir / "state.json"
    if not state_path.exists():
        return None

    try:
        import json

        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load paper track state from {state_path}: {e}")
        return None


def find_latest_paper_track_run(strategy_dir: Path) -> Path | None:
    """Find the latest paper track run directory.

    Args:
        strategy_dir: Strategy directory (should contain runs/ subdirectory)

    Returns:
        Path to latest run directory (runs/{YYYYMMDD}/), or None if none found
    """
    runs_dir = strategy_dir / "runs"
    if not runs_dir.exists():
        return None

    latest_run = None
    latest_mtime = 0.0

    for item in runs_dir.iterdir():
        if not item.is_dir():
            continue

        # Check for daily_summary.json (indicates valid run)
        summary_json = item / "daily_summary.json"
        if summary_json.exists():
            mtime = summary_json.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_run = item

    return latest_run


def load_paper_track_equity_curve(strategy_dir: Path) -> pd.DataFrame | None:
    """Load aggregated equity curve from paper track strategy directory.

    Args:
        strategy_dir: Strategy directory (should contain equity_curve.csv)

    Returns:
        DataFrame with equity curve, or None if not found
    """
    equity_path = strategy_dir / "equity_curve.csv"
    if not equity_path.exists():
        return None

    try:
        df = pd.read_csv(equity_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        logger.debug(
            f"Loaded paper track equity curve from {equity_path}: {len(df)} rows"
        )
        return df
    except Exception as e:
        logger.warning(
            f"Failed to load paper track equity curve from {equity_path}: {e}"
        )
        return None


def check_paper_track_freshness(
    latest_run_dir: Path | None,
    days: int,
) -> HealthCheck:
    """Check if paper track is fresh enough (last run within days).

    Args:
        latest_run_dir: Path to latest run directory, or None if no runs found
        days: Maximum allowed age in days

    Returns:
        HealthCheck instance
    """
    if latest_run_dir is None:
        return HealthCheck(
            name="paper_track_freshness",
            status="WARN",
            value=None,
            expected=f"Last run within {days} days",
            details="No paper track runs found",
        )

    # Get run date from directory name (YYYYMMDD) or from daily_summary.json
    summary_json = latest_run_dir / "daily_summary.json"
    if summary_json.exists():
        try:
            import json

            with open(summary_json, "r", encoding="utf-8") as f:
                summary = json.load(f)
            if "date" in summary:
                run_date = pd.to_datetime(summary["date"], utc=True)
            else:
                # Fallback to directory name
                run_date = pd.to_datetime(
                    latest_run_dir.name, format="%Y%m%d", utc=True, errors="coerce"
                )
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
            # Fallback to directory name
            run_date = pd.to_datetime(
                latest_run_dir.name, format="%Y%m%d", utc=True, errors="coerce"
            )
    else:
        # Fallback to directory name
        run_date = pd.to_datetime(
            latest_run_dir.name, format="%Y%m%d", utc=True, errors="coerce"
        )

    if pd.isna(run_date):
        return HealthCheck(
            name="paper_track_freshness",
            status="WARN",
            value=None,
            expected=f"Last run within {days} days",
            details=f"Could not parse run date from directory {latest_run_dir.name}",
        )

    now = pd.Timestamp.now(tz="UTC")
    age_days = (now - run_date).days

    if age_days > days * 2:
        status = "CRITICAL"
        details = f"Last paper track run is {age_days} days old (threshold: {days * 2})"
    elif age_days > days:
        status = "WARN"
        details = f"Last paper track run is {age_days} days old (threshold: {days})"
    else:
        status = "OK"
        details = (
            f"Last paper track run is {age_days} days old (within {days} day threshold)"
        )

    return HealthCheck(
        name="paper_track_freshness",
        status=status,
        value=age_days,
        expected=f"<= {days} days",
        details=details,
        last_updated_at=run_date,
    )


def check_paper_track_artifacts(
    strategy_dir: Path,
    latest_run_dir: Path | None,
) -> list[HealthCheck]:
    """Check if paper track artifacts are present.

    Args:
        strategy_dir: Strategy directory
        latest_run_dir: Path to latest run directory, or None

    Returns:
        List of HealthCheck instances
    """
    checks = []

    # Check 1: State file exists
    state_dir = strategy_dir / "state"
    state_file = state_dir / "state.json"
    if state_file.exists():
        checks.append(
            HealthCheck(
                name="paper_track_state_exists",
                status="OK",
                value="exists",
                expected="exists",
                details=f"State file found: {state_file}",
            )
        )
    else:
        checks.append(
            HealthCheck(
                name="paper_track_state_exists",
                status="WARN",
                value="missing",
                expected="exists",
                details=f"State file not found: {state_file}",
            )
        )

    # Check 2: Latest run directory exists
    if latest_run_dir is None:
        checks.append(
            HealthCheck(
                name="paper_track_runs_exist",
                status="WARN",
                value="missing",
                expected="exists",
                details="No paper track runs found",
            )
        )
    else:
        checks.append(
            HealthCheck(
                name="paper_track_runs_exist",
                status="OK",
                value="exists",
                expected="exists",
                details=f"Latest run directory: {latest_run_dir.name}",
            )
        )

        # Check 3: Daily summary exists in latest run
        summary_json = latest_run_dir / "daily_summary.json"
        if summary_json.exists():
            checks.append(
                HealthCheck(
                    name="paper_track_daily_summary_exists",
                    status="OK",
                    value="exists",
                    expected="exists",
                    details=f"Daily summary found: {summary_json}",
                )
            )
        else:
            checks.append(
                HealthCheck(
                    name="paper_track_daily_summary_exists",
                    status="WARN",
                    value="missing",
                    expected="exists",
                    details=f"Daily summary not found in {latest_run_dir.name}",
                )
            )

    return checks


def check_paper_track_metrics_plausible(
    strategy_dir: Path,
    latest_run_dir: Path | None,
    max_daily_pnl_pct: float,
    max_drawdown_min: float,
) -> list[HealthCheck]:
    """Check if paper track metrics are plausible.

    Args:
        strategy_dir: Strategy directory
        latest_run_dir: Path to latest run directory, or None
        max_daily_pnl_pct: Maximum acceptable daily PnL percentage
        max_drawdown_min: Minimum acceptable max drawdown

    Returns:
        List of HealthCheck instances
    """
    checks = []

    if latest_run_dir is None:
        checks.append(
            HealthCheck(
                name="paper_track_metrics_plausible",
                status="SKIP",
                value=None,
                expected="N/A",
                details="No paper track runs available for metric checks",
            )
        )
        return checks

    # Load latest daily summary
    summary_json = latest_run_dir / "daily_summary.json"
    if not summary_json.exists():
        checks.append(
            HealthCheck(
                name="paper_track_metrics_plausible",
                status="SKIP",
                value=None,
                expected="N/A",
                details="Daily summary not available",
            )
        )
        return checks

    try:
        import json

        with open(summary_json, "r", encoding="utf-8") as f:
            summary = json.load(f)

        # Check daily PnL
        if "daily_return_pct" in summary and pd.notna(summary["daily_return_pct"]):
            daily_return_pct = float(summary["daily_return_pct"])
            abs_daily_return = abs(daily_return_pct)

            if abs_daily_return > max_daily_pnl_pct:
                checks.append(
                    HealthCheck(
                        name="paper_track_daily_pnl_spike",
                        status="WARN",
                        value=daily_return_pct,
                        expected=f"<= {max_daily_pnl_pct}%",
                        details=f"Daily return {daily_return_pct:.2f}% exceeds threshold {max_daily_pnl_pct}%",
                    )
                )
            else:
                checks.append(
                    HealthCheck(
                        name="paper_track_daily_pnl_spike",
                        status="OK",
                        value=daily_return_pct,
                        expected=f"<= {max_daily_pnl_pct}%",
                        details=f"Daily return {daily_return_pct:.2f}% within expected range",
                    )
                )

        # Check equity curve for drawdown (if available)
        equity_df = load_paper_track_equity_curve(strategy_dir)
        if equity_df is not None and not equity_df.empty:
            if "equity" in equity_df.columns:
                equity_series = equity_df["equity"]

                # Compute running maximum
                running_max = equity_series.expanding().max()
                drawdown_pct = (
                    (equity_series - running_max) / running_max * 100.0
                ).fillna(0.0)
                max_drawdown_pct = float(drawdown_pct.min())

                if max_drawdown_pct < max_drawdown_min * 100.0:
                    checks.append(
                        HealthCheck(
                            name="paper_track_max_drawdown",
                            status="CRITICAL",
                            value=max_drawdown_pct / 100.0,
                            expected=f">= {max_drawdown_min}",
                            details=f"Max drawdown {max_drawdown_pct:.2f}% is worse than threshold {max_drawdown_min * 100.0:.2f}%",
                        )
                    )
                else:
                    checks.append(
                        HealthCheck(
                            name="paper_track_max_drawdown",
                            status="OK",
                            value=max_drawdown_pct / 100.0,
                            expected=f">= {max_drawdown_min}",
                            details=f"Max drawdown {max_drawdown_pct:.2f}% within acceptable range",
                        )
                    )

    except Exception as e:
        logger.warning(f"Failed to check paper track metrics: {e}")
        checks.append(
            HealthCheck(
                name="paper_track_metrics_plausible",
                status="SKIP",
                value=None,
                expected="N/A",
                details=f"Error checking metrics: {e}",
            )
        )

    return checks


def run_paper_track_health_checks(
    paper_track_root: Path,
    args: argparse.Namespace,
) -> list[HealthCheck]:
    """Run health checks for paper track strategies.

    Args:
        paper_track_root: Root directory for paper track outputs
        args: Parsed command-line arguments

    Returns:
        List of HealthCheck instances
    """
    checks = []

    # Auto-detect paper track root if not provided
    if paper_track_root is None:
        paper_track_root = ROOT / "output" / "paper_track"

    if not paper_track_root.is_absolute():
        paper_track_root = ROOT / paper_track_root
    paper_track_root = paper_track_root.resolve()

    if not paper_track_root.exists():
        if args.skip_paper_track_if_missing:
            logger.info(
                f"Paper track root does not exist and --skip-paper-track-if-missing is set: {paper_track_root}"
            )
            checks.append(
                HealthCheck(
                    name="paper_track_root_exists",
                    status="SKIP",
                    value="missing",
                    expected="exists",
                    details=f"Paper track root not found: {paper_track_root} (skipped per flag)",
                )
            )
            return checks
        else:
            logger.warning(f"Paper track root does not exist: {paper_track_root}")
            checks.append(
                HealthCheck(
                    name="paper_track_root_exists",
                    status="WARN",
                    value="missing",
                    expected="exists",
                    details=f"Paper track root not found: {paper_track_root}",
                )
            )
            return checks

    # Find strategy directories
    strategies = find_paper_track_strategies(paper_track_root)

    if not strategies:
        checks.append(
            HealthCheck(
                name="paper_track_strategies_found",
                status="WARN",
                value=0,
                expected=">= 1",
                details=f"No paper track strategies found in {paper_track_root}",
            )
        )
        return checks

    checks.append(
        HealthCheck(
            name="paper_track_strategies_found",
            status="OK",
            value=len(strategies),
            expected=">= 1",
            details=f"Found {len(strategies)} paper track strategy(ies)",
        )
    )

    # Check each strategy
    for strategy_dir in strategies:
        strategy_name = strategy_dir.name

        # Find latest run
        latest_run = find_latest_paper_track_run(strategy_dir)

        # Freshness check
        freshness_check = check_paper_track_freshness(latest_run, args.paper_track_days)
        freshness_check.name = f"paper_track_freshness_{strategy_name}"
        checks.append(freshness_check)

        # Artifact checks
        artifact_checks = check_paper_track_artifacts(strategy_dir, latest_run)
        for check in artifact_checks:
            check.name = f"{check.name}_{strategy_name}"
        checks.extend(artifact_checks)

        # Metrics checks
        metrics_checks = check_paper_track_metrics_plausible(
            strategy_dir,
            latest_run,
            args.paper_track_max_daily_pnl_pct,
            args.paper_track_max_drawdown_min,
        )
        for check in metrics_checks:
            check.name = f"{check.name}_{strategy_name}"
        checks.extend(metrics_checks)

    return checks


def maybe_load_benchmark_returns(args: argparse.Namespace) -> pd.Series | None:
    """Load benchmark returns if benchmark symbol or file is provided.

    Args:
        args: Parsed command-line arguments

    Returns:
        Series with benchmark returns (index = timestamp), or None
    """
    if args.benchmark_file is not None and args.benchmark_file.exists():
        try:
            if args.benchmark_file.suffix == ".parquet":
                df = pd.read_parquet(args.benchmark_file)
            elif args.benchmark_file.suffix == ".csv":
                df = pd.read_csv(args.benchmark_file)
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"], utc=True, errors="coerce"
                    )
            else:
                logger.warning(
                    f"Unknown benchmark file format: {args.benchmark_file.suffix}"
                )
                return None

            # Extract returns
            if "returns" in df.columns:
                if "timestamp" in df.columns:
                    series = pd.Series(
                        df["returns"].values,
                        index=pd.to_datetime(df["timestamp"], utc=True),
                    )
                else:
                    series = pd.Series(df["returns"].values)
                logger.debug(f"Loaded benchmark returns from {args.benchmark_file}")
                return series
            elif "close" in df.columns:
                # Compute returns from prices
                if "timestamp" in df.columns:
                    prices = pd.Series(
                        df["close"].values,
                        index=pd.to_datetime(df["timestamp"], utc=True),
                    )
                else:
                    prices = pd.Series(df["close"].values)
                returns = prices.pct_change().dropna()
                logger.debug(f"Computed benchmark returns from {args.benchmark_file}")
                return returns
            else:
                logger.warning("Benchmark file missing 'returns' or 'close' column")
                return None
        except Exception as e:
            logger.warning(f"Failed to load benchmark from {args.benchmark_file}: {e}")
            return None

    # TODO: Load from benchmark symbol (if data source available)
    # For now, just log that it's not implemented
    if args.benchmark_symbol:
        logger.warning(
            f"Loading benchmark from symbol not yet implemented: {args.benchmark_symbol}"
        )

    return None


def check_backtest_freshness(
    equity_df: pd.DataFrame,
    days: int,
) -> HealthCheck:
    """Check if backtest is fresh enough (last equity timestamp within days).

    Args:
        equity_df: Equity curve DataFrame (must have timestamp column)
        days: Maximum allowed age in days

    Returns:
        HealthCheck instance
    """
    if equity_df is None or equity_df.empty:
        return HealthCheck(
            name="backtest_freshness",
            status="CRITICAL",
            value=None,
            expected=f"Last equity timestamp within {days} days",
            details="Equity curve is empty or missing",
        )

    # Get last timestamp
    if "timestamp" in equity_df.columns:
        last_timestamp = pd.to_datetime(equity_df["timestamp"].max(), utc=True)
    else:
        # Assume index is timestamp
        last_timestamp = pd.to_datetime(equity_df.index.max(), utc=True)

    now = pd.Timestamp.now(tz="UTC")
    age_days = (now - last_timestamp).days

    if age_days > days * 2:  # More than 2x the lookback window
        status = "CRITICAL"
        details = (
            f"Last equity timestamp is {age_days} days old (threshold: {days * 2})"
        )
    elif age_days > days:
        status = "WARN"
        details = f"Last equity timestamp is {age_days} days old (threshold: {days})"
    else:
        status = "OK"
        details = f"Last equity timestamp is {age_days} days old (within {days} day threshold)"

    return HealthCheck(
        name="backtest_freshness",
        status=status,
        value=age_days,
        expected=f"<= {days} days",
        details=details,
        last_updated_at=last_timestamp,
    )


def check_risk_report_exists(backtest_dir: Path) -> HealthCheck:
    """Check if risk report exists.

    Args:
        backtest_dir: Backtest directory path

    Returns:
        HealthCheck instance
    """
    risk_report_md = backtest_dir / "risk_report.md"
    risk_summary_csv = backtest_dir / "risk_summary.csv"

    # Also check risk_reports subdirectory
    risk_reports_dir = backtest_dir.parent / "risk_reports" / backtest_dir.name
    risk_report_md_alt = risk_reports_dir / "risk_report.md"
    risk_summary_csv_alt = risk_reports_dir / "risk_summary.csv"

    exists = (risk_report_md.exists() and risk_summary_csv.exists()) or (
        risk_report_md_alt.exists() and risk_summary_csv_alt.exists()
    )

    if exists:
        return HealthCheck(
            name="risk_report_exists",
            status="OK",
            value="exists",
            expected="exists",
            details="Risk report and risk_summary.csv found",
        )
    else:
        return HealthCheck(
            name="risk_report_exists",
            status="WARN",
            value="missing",
            expected="exists",
            details="Risk report or risk_summary.csv not found",
        )


def check_metrics_in_range(
    risk_summary_df: pd.DataFrame,
    min_sharpe: float,
    max_drawdown_min: float,
    max_drawdown_max: float,
    max_turnover: float,
) -> list[HealthCheck]:
    """Check if performance metrics are within expected ranges.

    Args:
        risk_summary_df: Risk summary DataFrame (single row expected)
        min_sharpe: Minimum acceptable Sharpe ratio
        max_drawdown_min: Minimum acceptable max drawdown (more negative = worse)
        max_drawdown_max: Maximum acceptable max drawdown (less negative = better)
        max_turnover: Maximum acceptable turnover

    Returns:
        List of HealthCheck instances
    """
    checks = []

    if risk_summary_df is None or risk_summary_df.empty:
        checks.append(
            HealthCheck(
                name="metrics_in_range",
                status="SKIP",
                value=None,
                expected="N/A",
                details="Risk summary not available",
            )
        )
        return checks

    # Get first row (should be only row)
    row = risk_summary_df.iloc[0]

    # Check Sharpe Ratio
    if "sharpe" in row and pd.notna(row["sharpe"]):
        sharpe = float(row["sharpe"])
        if sharpe < min_sharpe:
            checks.append(
                HealthCheck(
                    name="sharpe_ratio",
                    status="WARN" if sharpe >= min_sharpe - 0.5 else "CRITICAL",
                    value=sharpe,
                    expected=f">= {min_sharpe}",
                    details=f"Sharpe ratio {sharpe:.4f} below threshold {min_sharpe}",
                )
            )
        else:
            checks.append(
                HealthCheck(
                    name="sharpe_ratio",
                    status="OK",
                    value=sharpe,
                    expected=f">= {min_sharpe}",
                    details=f"Sharpe ratio {sharpe:.4f} within expected range",
                )
            )
    else:
        checks.append(
            HealthCheck(
                name="sharpe_ratio",
                status="SKIP",
                value=None,
                expected=f">= {min_sharpe}",
                details="Sharpe ratio not available in risk summary",
            )
        )

    # Check Max Drawdown
    if "max_drawdown" in row and pd.notna(row["max_drawdown"]):
        max_dd = float(row["max_drawdown"])
        # Note: max_drawdown is typically negative (e.g., -0.12 = -12%)
        # max_drawdown_min is also negative (e.g., -0.40 = -40%), more negative = worse
        # max_drawdown_max is less negative or positive (e.g., 0.0 = 0%), less negative = better
        # So: max_dd should be >= max_drawdown_min (less negative = better) AND <= max_drawdown_max
        if max_dd < max_drawdown_min:
            # Drawdown is worse (more negative) than minimum threshold
            checks.append(
                HealthCheck(
                    name="max_drawdown",
                    status="CRITICAL",
                    value=max_dd,
                    expected=f"[{max_drawdown_min}, {max_drawdown_max}]",
                    details=f"Max drawdown {max_dd:.4f} is worse (more negative) than threshold {max_drawdown_min}",
                )
            )
        elif max_drawdown_max is not None and max_dd > max_drawdown_max:
            # Drawdown is better (less negative) than maximum threshold (unusual but OK)
            checks.append(
                HealthCheck(
                    name="max_drawdown",
                    status="OK",  # Changed from WARN to OK - better drawdown is good
                    value=max_dd,
                    expected=f"[{max_drawdown_min}, {max_drawdown_max}]",
                    details=f"Max drawdown {max_dd:.4f} is better (less negative) than threshold {max_drawdown_max}",
                )
            )
        else:
            checks.append(
                HealthCheck(
                    name="max_drawdown",
                    status="OK",
                    value=max_dd,
                    expected=f"[{max_drawdown_min}, {max_drawdown_max}]",
                    details=f"Max drawdown {max_dd:.4f} within expected range",
                )
            )
    else:
        checks.append(
            HealthCheck(
                name="max_drawdown",
                status="SKIP",
                value=None,
                expected=f"[{max_drawdown_min}, {max_drawdown_max}]",
                details="Max drawdown not available in risk summary",
            )
        )

    # Check Turnover (optional)
    if "turnover" in row and pd.notna(row["turnover"]):
        turnover = float(row["turnover"])
        if turnover > max_turnover:
            checks.append(
                HealthCheck(
                    name="turnover",
                    status="WARN",
                    value=turnover,
                    expected=f"<= {max_turnover}",
                    details=f"Turnover {turnover:.4f} above threshold {max_turnover}",
                )
            )
        else:
            checks.append(
                HealthCheck(
                    name="turnover",
                    status="OK",
                    value=turnover,
                    expected=f"<= {max_turnover}",
                    details=f"Turnover {turnover:.4f} within expected range",
                )
            )
    # Don't add SKIP for turnover if it's not present (it's optional)

    return checks


def check_benchmark_correlation(
    equity_df: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> HealthCheck:
    """Check correlation between equity returns and benchmark returns.

    Args:
        equity_df: Equity curve DataFrame
        benchmark_returns: Benchmark returns Series (index = timestamp)

    Returns:
        HealthCheck instance
    """
    if equity_df is None or equity_df.empty:
        return HealthCheck(
            name="benchmark_correlation",
            status="SKIP",
            value=None,
            expected="[-0.2, 0.99]",
            details="Equity curve not available",
        )

    if benchmark_returns is None or benchmark_returns.empty:
        return HealthCheck(
            name="benchmark_correlation",
            status="SKIP",
            value=None,
            expected="[-0.2, 0.99]",
            details="Benchmark returns not available",
        )

    try:
        # Compute equity returns
        if "equity" in equity_df.columns:
            equity_series = equity_df["equity"]
        elif "close" in equity_df.columns:
            equity_series = equity_df["close"]
        else:
            return HealthCheck(
                name="benchmark_correlation",
                status="SKIP",
                value=None,
                expected="[-0.2, 0.99]",
                details="Equity column not found in equity curve",
            )

        # Get timestamp index
        if "timestamp" in equity_df.columns:
            equity_series.index = pd.to_datetime(equity_df["timestamp"], utc=True)

        equity_returns = equity_series.pct_change().dropna()

        # Align equity returns with benchmark returns
        common_index = equity_returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 10:  # Need at least 10 overlapping points
            return HealthCheck(
                name="benchmark_correlation",
                status="SKIP",
                value=None,
                expected="[-0.2, 0.99]",
                details=f"Insufficient overlapping data points ({len(common_index)})",
            )

        equity_aligned = equity_returns.loc[common_index]
        benchmark_aligned = benchmark_returns.loc[common_index]

        # Compute correlation
        correlation = equity_aligned.corr(benchmark_aligned)

        if pd.isna(correlation):
            return HealthCheck(
                name="benchmark_correlation",
                status="SKIP",
                value=None,
                expected="[-0.2, 0.99]",
                details="Correlation computation resulted in NaN",
            )

        # Check if correlation is in reasonable range
        if correlation < -0.2 or correlation > 0.99:
            status = "WARN"
            details = (
                f"Correlation {correlation:.4f} outside expected range [-0.2, 0.99]"
            )
        else:
            status = "OK"
            details = f"Correlation {correlation:.4f} within expected range"

        return HealthCheck(
            name="benchmark_correlation",
            status=status,
            value=correlation,
            expected="[-0.2, 0.99]",
            details=details,
        )
    except Exception as e:
        logger.warning(f"Failed to compute benchmark correlation: {e}")
        return HealthCheck(
            name="benchmark_correlation",
            status="SKIP",
            value=None,
            expected="[-0.2, 0.99]",
            details=f"Error computing correlation: {e}",
        )


def run_health_checks_from_cli(args: argparse.Namespace) -> int:
    """Run health checks from CLI arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code: 0 for OK/SKIP, 1 for WARN, 2 for CRITICAL
    """
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve paths
    backtests_root = args.backtests_root
    if not backtests_root.is_absolute():
        backtests_root = ROOT / backtests_root
    backtests_root = backtests_root.resolve()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ROOT / "output" / "health"
    elif not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Health check starting: backtests_root={backtests_root}, output_dir={output_dir}"
    )

    # Find latest backtest directory
    latest_backtest_dir = find_latest_backtest_dir(backtests_root)

    checks = []
    notes = []

    if latest_backtest_dir is None:
        # No backtest directory found
        checks.append(
            HealthCheck(
                name="backtest_exists",
                status="CRITICAL",
                value="missing",
                expected="exists",
                details=f"No backtest directories found in {backtests_root}",
            )
        )
    else:
        logger.info(f"Found latest backtest directory: {latest_backtest_dir}")
        notes.append(f"Latest backtest: {latest_backtest_dir.name}")

        # Check 1: Backtest exists
        checks.append(
            HealthCheck(
                name="backtest_exists",
                status="OK",
                value="exists",
                expected="exists",
                details=f"Backtest directory found: {latest_backtest_dir.name}",
            )
        )

        # Load equity curve
        equity_df = load_equity_curve(latest_backtest_dir)

        # Check 2: Equity curve exists
        if equity_df is None or equity_df.empty:
            checks.append(
                HealthCheck(
                    name="equity_curve_exists",
                    status="CRITICAL",
                    value="missing",
                    expected="exists",
                    details="Equity curve file not found or empty",
                )
            )
        else:
            checks.append(
                HealthCheck(
                    name="equity_curve_exists",
                    status="OK",
                    value="exists",
                    expected="exists",
                    details=f"Equity curve loaded: {len(equity_df)} rows",
                )
            )

            # Check 3: Backtest freshness
            freshness_check = check_backtest_freshness(equity_df, args.days)
            checks.append(freshness_check)

        # Check 4: Risk report exists
        risk_report_check = check_risk_report_exists(latest_backtest_dir)
        checks.append(risk_report_check)

        # Load risk summary
        risk_summary_df = load_risk_summary(latest_backtest_dir)

        # Check 5: Metrics in range
        if risk_summary_df is not None:
            metric_checks = check_metrics_in_range(
                risk_summary_df,
                args.min_sharpe,
                args.max_drawdown_min,
                args.max_drawdown_max,
                args.max_turnover,
            )
            checks.extend(metric_checks)

        # Check 6: Benchmark correlation (optional)
        benchmark_returns = maybe_load_benchmark_returns(args)
        if benchmark_returns is not None and equity_df is not None:
            correlation_check = check_benchmark_correlation(
                equity_df, benchmark_returns
            )
            checks.append(correlation_check)

    # Check 7: Paper Track health checks (optional)
    try:
        paper_track_checks = run_paper_track_health_checks(args.paper_track_root, args)
        checks.extend(paper_track_checks)
    except Exception as e:
        logger.warning(f"Paper track health checks failed: {e}", exc_info=args.verbose)
        # Don't crash - add SKIP check instead
        checks.append(
            HealthCheck(
                name="paper_track_checks",
                status="SKIP",
                value=None,
                expected="N/A",
                details=f"Paper track checks failed: {e}",
            )
        )

    # Aggregate overall status
    overall_status = aggregate_overall_status(checks)

    # Create result
    result = HealthCheckResult(
        overall_status=overall_status,
        timestamp=pd.Timestamp.now(tz="UTC"),
        checks=checks,
        notes=notes if notes else None,
        meta={
            "backtests_root": str(backtests_root),
            "days": args.days,
            "min_sharpe": args.min_sharpe,
            "max_drawdown_min": args.max_drawdown_min,
            "max_drawdown_max": args.max_drawdown_max,
            "max_turnover": args.max_turnover,
            "paper_track_root": str(args.paper_track_root)
            if args.paper_track_root
            else "auto-detected",
            "paper_track_days": args.paper_track_days,
        },
    )

    # Write outputs
    summary_json_path = output_dir / "health_summary.json"
    summary_md_path = output_dir / "health_summary.md"

    # Write JSON
    result_dict = health_result_to_dict(result)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, default=str)
    logger.info(f"Health summary JSON written: {summary_json_path}")

    # Write Markdown
    summary_text = render_health_summary_text(result)
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    logger.info(f"Health summary Markdown written: {summary_md_path}")

    # Print to stdout if requested
    if args.format in ["text", "both"]:
        print(summary_text)

    if args.format == "json":
        print(json.dumps(result_dict, indent=2, default=str))

    # Return exit code
    if overall_status == "CRITICAL":
        return 2
    elif overall_status == "WARN":
        return 1
    else:  # OK or SKIP
        return 0


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    try:
        exit_code = run_health_checks_from_cli(args)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.error("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
