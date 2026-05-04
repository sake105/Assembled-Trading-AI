# scripts/commands/reports.py
"""Report subcommands: risk_report, tca_report."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def risk_report_subcommand(args: argparse.Namespace) -> int:
    """Generate risk report from backtest results subcommand."""
    from scripts.generate_risk_report import generate_risk_report

    logger = logging.getLogger(__name__)

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


def tca_report_subcommand(args: argparse.Namespace) -> int:
    """Generate TCA report from backtest results subcommand."""
    from scripts.generate_tca_report import generate_tca_report

    return generate_tca_report(
        backtest_dir=args.backtest_dir,
        output_dir=args.output_dir,
        method=args.method,
        commission_bps=args.commission_bps,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
    )


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register report subcommands."""
    # risk_report
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

    # tca_report
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
