"""Generate Transaction Cost Analysis (TCA) Report from Backtest Results.

This script generates transaction cost analysis reports from backtest outputs, including:
- Per-trade cost estimation (commission, spread, slippage)
- Cost aggregation and summary statistics
- Cost-adjusted risk metrics (net returns, net Sharpe, etc.)
- Comparison of gross vs. net performance

Usage:
    python scripts/generate_tca_report.py --backtest-dir output/backtests/experiment_123/
    python scripts/cli.py tca_report --backtest-dir output/backtests/experiment_123/ --output-dir output/tca_reports/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.qa.metrics import _compute_returns
from src.assembled_core.risk.transaction_costs import (
    compute_cost_adjusted_risk_metrics,
    compute_tca_for_trades,
    estimate_per_trade_cost,
    summarize_tca,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_dataframe(file_path: Path, expected_columns: list[str] | None = None) -> pd.DataFrame | None:
    """Load DataFrame from CSV or Parquet file.
    
    Args:
        file_path: Path to file (can be .csv or .parquet)
        expected_columns: Optional list of expected column names (for validation)
    
    Returns:
        DataFrame if file exists and can be loaded, None otherwise
    """
    if not file_path.exists():
        logger.debug(f"File not found: {file_path}")
        return None
    
    try:
        if file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        elif file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
            # Try to parse timestamp column if present
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        else:
            logger.warning(f"Unknown file format: {file_path.suffix}. Skipping.")
            return None
        
        if expected_columns:
            missing = [col for col in expected_columns if col not in df.columns]
            if missing:
                logger.warning(f"Missing expected columns in {file_path.name}: {', '.join(missing)}")
                return None
        
        logger.info(f"Loaded {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None


def find_trades_file(backtest_dir: Path) -> Path | None:
    """Find trades file in backtest directory.
    
    Args:
        backtest_dir: Directory containing backtest outputs
    
    Returns:
        Path to trades file if found, None otherwise
    """
    trades_patterns = [
        "trades.parquet",
        "trades.csv",
    ]
    
    for pattern in trades_patterns:
        trades_path = backtest_dir / pattern
        if trades_path.exists():
            return trades_path
    
    return None


def find_equity_file(backtest_dir: Path) -> Path | None:
    """Find equity curve file in backtest directory.
    
    Args:
        backtest_dir: Directory containing backtest outputs
    
    Returns:
        Path to equity file if found, None otherwise
    """
    equity_patterns = [
        "equity_curve.parquet",
        "equity_curve.csv",
        "portfolio_equity_curve.parquet",
        "portfolio_equity_curve.csv",
    ]
    
    for pattern in equity_patterns:
        equity_path = backtest_dir / pattern
        if equity_path.exists():
            return equity_path
    
    return None


def generate_tca_report(
    backtest_dir: Path,
    output_dir: Path | None = None,
    method: str = "simple",
    commission_bps: float = 0.5,
    spread_bps: float | None = None,
    slippage_bps: float = 3.0,
) -> int:
    """Generate TCA report from backtest results.
    
    Args:
        backtest_dir: Directory containing backtest outputs
        output_dir: Output directory (default: backtest_dir / "tca")
        method: Cost estimation method ("simple" or "adaptive")
        commission_bps: Commission in basis points (default: 0.5)
        spread_bps: Spread in basis points (None = use default 5 bps)
        slippage_bps: Slippage in basis points (default: 3.0)
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info("=" * 80)
    logger.info("Transaction Cost Analysis (TCA) Report Generation")
    logger.info("=" * 80)
    
    # Set output directory
    if output_dir is None:
        output_dir = backtest_dir / "tca"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Backtest directory: {backtest_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Find and load trades file
    trades_file = find_trades_file(backtest_dir)
    if trades_file is None:
        logger.error(f"No trades file found in {backtest_dir}")
        logger.error("Expected files: trades.parquet or trades.csv")
        return 1
    
    logger.info(f"Loading trades from {trades_file.name}...")
    trades_df = load_dataframe(trades_file, expected_columns=["timestamp", "symbol", "side", "qty", "price"])
    if trades_df is None or trades_df.empty:
        logger.error("Failed to load trades or trades DataFrame is empty")
        return 1
    
    logger.info(f"Loaded {len(trades_df)} trades")
    
    # Estimate costs per trade
    logger.info("Estimating costs per trade...")
    cost_per_trade = estimate_per_trade_cost(
        trades=trades_df,
        method=method,
        commission_bps=commission_bps,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
    )
    
    logger.info(f"Estimated costs: mean={cost_per_trade.mean():.4f}, total={cost_per_trade.sum():.4f}")
    
    # Compute TCA for trades
    logger.info("Computing TCA for trades...")
    tca_trades_df = compute_tca_for_trades(trades_df, cost_per_trade)
    
    # Save TCA trades
    tca_trades_path = output_dir / "tca_trades.csv"
    tca_trades_df.to_csv(tca_trades_path, index=False)
    logger.info(f"Saved TCA trades to {tca_trades_path.name}")
    
    # Summarize TCA
    logger.info("Summarizing TCA metrics...")
    tca_summary_df = summarize_tca(tca_trades_df, freq="D")
    
    # Save summary
    tca_summary_path = output_dir / "tca_summary.csv"
    tca_summary_df.to_csv(tca_summary_path, index=False)
    logger.info(f"Saved TCA summary to {tca_summary_path.name}")
    
    # Try to load equity curve for cost-adjusted risk metrics
    equity_file = find_equity_file(backtest_dir)
    equity_df = None
    if equity_file:
        logger.info(f"Loading equity curve from {equity_file.name}...")
        equity_df = load_dataframe(equity_file, expected_columns=["timestamp", "equity"])
    
    risk_metrics_dict = None
    if equity_df is not None and not equity_df.empty:
        logger.info("Computing cost-adjusted risk metrics...")
        
        # Compute returns from equity
        if "daily_return" not in equity_df.columns:
            equity_series = equity_df.set_index("timestamp")["equity"].sort_index()
            returns = _compute_returns(equity_series)
        else:
            equity_df = equity_df.set_index("timestamp").sort_index()
            returns = equity_df["daily_return"]
        
        # Aggregate costs by day (to match returns)
        if "timestamp" not in tca_trades_df.columns:
            tca_trades_df = tca_trades_df.reset_index()
        
        tca_trades_with_ts = tca_trades_df.set_index("timestamp").sort_index()
        daily_costs = tca_trades_with_ts["cost_total"].resample("D").sum()
        
        # Align returns and costs
        common_index = returns.index.intersection(daily_costs.index)
        if len(common_index) > 0:
            returns_aligned = returns.loc[common_index]
            costs_aligned = daily_costs.loc[common_index]
            
            # Determine frequency from data (approximate)
            freq: Literal["1d", "5min"] = "1d"  # Default to daily
            
            risk_metrics_dict = compute_cost_adjusted_risk_metrics(
                returns=returns_aligned,
                costs=costs_aligned,
                freq=freq,
            )
            
            logger.info(f"Cost-adjusted metrics computed: net_sharpe={risk_metrics_dict.get('net_sharpe'):.4f}")
            
            # Save risk summary
            risk_summary_path = output_dir / "tca_risk_summary.csv"
            risk_summary_df = pd.DataFrame([risk_metrics_dict])
            risk_summary_df.to_csv(risk_summary_path, index=False)
            logger.info(f"Saved TCA risk summary to {risk_summary_path.name}")
    
    # Generate Markdown report
    logger.info("Generating TCA report...")
    report_path = output_dir / "tca_report.md"
    write_tca_report_markdown(
        report_path=report_path,
        tca_summary_df=tca_summary_df,
        tca_trades_df=tca_trades_df,
        risk_metrics_dict=risk_metrics_dict,
        backtest_dir=backtest_dir,
    )
    logger.info(f"Saved TCA report to {report_path.name}")
    
    logger.info("=" * 80)
    logger.info("TCA Report Generation Completed")
    logger.info("=" * 80)
    
    return 0


def write_tca_report_markdown(
    report_path: Path,
    tca_summary_df: pd.DataFrame,
    tca_trades_df: pd.DataFrame,
    risk_metrics_dict: dict | None,
    backtest_dir: Path,
) -> None:
    """Write TCA report as Markdown.
    
    Args:
        report_path: Path to output Markdown file
        tca_summary_df: TCA summary DataFrame
        tca_trades_df: TCA trades DataFrame
        risk_metrics_dict: Optional risk metrics dictionary
        backtest_dir: Backtest directory path
    """
    lines = []
    
    # Header
    lines.append("# Transaction Cost Analysis (TCA) Report")
    lines.append("")
    lines.append(f"**Generated:** {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"**Backtest Directory:** `{backtest_dir}`")
    lines.append("")
    
    # Summary Statistics
    lines.append("## Summary Statistics")
    lines.append("")
    total_cost = tca_trades_df["cost_total"].sum()
    total_trades = len(tca_trades_df)
    avg_cost_per_trade = tca_trades_df["cost_total"].mean()
    
    lines.append(f"- **Total Trades:** {total_trades}")
    lines.append(f"- **Total Costs:** {total_cost:.4f}")
    lines.append(f"- **Average Cost per Trade:** {avg_cost_per_trade:.4f}")
    lines.append("")
    
    # TCA Summary Table
    if not tca_summary_df.empty:
        lines.append("## Daily TCA Summary")
        lines.append("")
        lines.append("| Date | Total Cost | N Trades | Avg Cost/Trade |")
        lines.append("|------|------------|----------|----------------|")
        for _, row in tca_summary_df.head(20).iterrows():
            date_str = row["timestamp"].strftime("%Y-%m-%d") if pd.notna(row["timestamp"]) else "N/A"
            lines.append(f"| {date_str} | {row['total_cost']:.4f} | {row['n_trades']} | {row['avg_cost_per_trade']:.4f} |")
        
        if len(tca_summary_df) > 20:
            lines.append(f"| ... | ... | ... | ... | ({len(tca_summary_df) - 20} more rows) |")
        lines.append("")
    
    # Risk Metrics Comparison (if available)
    if risk_metrics_dict:
        lines.append("## Cost-Adjusted Risk Metrics")
        lines.append("")
        lines.append("### Gross vs. Net Performance")
        lines.append("")
        lines.append("| Metric | Gross | Net | Difference |")
        lines.append("|--------|-------|-----|------------|")
        
        gross_sharpe = risk_metrics_dict.get("gross_sharpe")
        net_sharpe = risk_metrics_dict.get("net_sharpe")
        if gross_sharpe is not None and net_sharpe is not None:
            diff_sharpe = gross_sharpe - net_sharpe
            lines.append(f"| Sharpe Ratio | {gross_sharpe:.4f} | {net_sharpe:.4f} | {diff_sharpe:.4f} |")
        
        gross_sortino = risk_metrics_dict.get("gross_sortino")
        net_sortino = risk_metrics_dict.get("net_sortino")
        if gross_sortino is not None and net_sortino is not None:
            diff_sortino = gross_sortino - net_sortino
            lines.append(f"| Sortino Ratio | {gross_sortino:.4f} | {net_sortino:.4f} | {diff_sortino:.4f} |")
        
        total_cost_metric = risk_metrics_dict.get("total_cost")
        cost_ratio = risk_metrics_dict.get("cost_ratio")
        if total_cost_metric is not None:
            lines.append(f"| Total Costs | {total_cost_metric:.4f} | - | - |")
        if cost_ratio is not None:
            lines.append(f"| Cost Ratio | {cost_ratio:.4%} | - | - |")
        
        lines.append("")
        
        lines.append("### Net Performance Metrics")
        lines.append("")
        lines.append(f"- **Net Annualized Return:** {risk_metrics_dict.get('net_mean_return_annualized', 'N/A')}")
        lines.append(f"- **Net Volatility:** {risk_metrics_dict.get('net_vol_annualized', 'N/A')}")
        lines.append(f"- **Net Sharpe Ratio:** {risk_metrics_dict.get('net_sharpe', 'N/A')}")
        lines.append(f"- **Net Sortino Ratio:** {risk_metrics_dict.get('net_sortino', 'N/A')}")
        lines.append(f"- **Net Max Drawdown:** {risk_metrics_dict.get('net_max_drawdown', 'N/A')}")
        lines.append("")
    
    # Files
    lines.append("## Output Files")
    lines.append("")
    lines.append(f"- `tca_trades.csv`: Detailed TCA for each trade ({len(tca_trades_df)} trades)")
    lines.append(f"- `tca_summary.csv`: Daily aggregated TCA metrics ({len(tca_summary_df)} days)")
    if risk_metrics_dict:
        lines.append("- `tca_risk_summary.csv`: Cost-adjusted risk metrics")
    lines.append("")
    
    # Write file
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Transaction Cost Analysis (TCA) Report from Backtest Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_tca_report.py --backtest-dir output/backtests/experiment_123/
  python scripts/generate_tca_report.py --backtest-dir output/backtests/experiment_123/ --output-dir output/tca_reports/
  python scripts/generate_tca_report.py --backtest-dir output/backtests/experiment_123/ --spread-bps 10.0 --slippage-bps 5.0
        """
    )
    
    parser.add_argument(
        "--backtest-dir",
        type=Path,
        required=True,
        help="Directory containing backtest outputs (must contain trades.csv or trades.parquet)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <backtest-dir>/tca)"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="simple",
        choices=["simple", "adaptive"],
        help="Cost estimation method (default: simple)"
    )
    
    parser.add_argument(
        "--commission-bps",
        type=float,
        default=0.5,
        help="Commission in basis points (default: 0.5)"
    )
    
    parser.add_argument(
        "--spread-bps",
        type=float,
        default=None,
        help="Spread in basis points (default: 5.0 if not specified)"
    )
    
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=3.0,
        help="Slippage in basis points (default: 3.0)"
    )
    
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    
    return generate_tca_report(
        backtest_dir=args.backtest_dir,
        output_dir=args.output_dir,
        method=args.method,
        commission_bps=args.commission_bps,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
    )


if __name__ == "__main__":
    sys.exit(main())

