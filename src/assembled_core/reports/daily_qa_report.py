"""Daily QA Report generation.

This module provides functions to generate QA reports that summarize:
- Performance metrics (from qa.metrics)
- QA gate results (from qa.qa_gates)
- Equity curve visualization/link
- Data status and configuration
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.assembled_core.config import OUTPUT_DIR
from src.assembled_core.qa.metrics import PerformanceMetrics
from src.assembled_core.qa.qa_gates import QAGatesSummary
from src.assembled_core.qa.risk_metrics import compute_portfolio_risk_metrics
from src.assembled_core.qa.shipping_risk import (
    compute_shipping_exposure,
    compute_systemic_risk_flags,
)


def generate_qa_report(
    metrics: PerformanceMetrics,
    gate_result: QAGatesSummary | None = None,
    strategy_name: str = "default",
    freq: str = "1d",
    equity_curve_path: Path | str | None = None,
    data_start_date: pd.Timestamp | None = None,
    data_end_date: pd.Timestamp | None = None,
    config_info: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    equity: pd.DataFrame | None = None,
    portfolio_positions: pd.DataFrame | None = None,
    shipping_features: pd.DataFrame | None = None,
) -> Path:
    """Generate a daily QA report in Markdown format.

    Args:
        metrics: PerformanceMetrics from qa.metrics
        gate_result: Optional QAGatesSummary from qa.qa_gates
        strategy_name: Name of the strategy (for filename)
        freq: Trading frequency ("1d" or "5min")
        equity_curve_path: Optional path to equity curve CSV file
        data_start_date: Optional start date of data used
        data_end_date: Optional end date of data used
        config_info: Optional dict with configuration info (e.g., EMA params, cost model)
        output_dir: Output directory (default: OUTPUT_DIR / "reports")

    Returns:
        Path to generated report file

    Example:
        ```python
        from src.assembled_core.reports.daily_qa_report import generate_qa_report
        from src.assembled_core.qa.metrics import compute_all_metrics
        from src.assembled_core.qa.qa_gates import evaluate_all_gates

        metrics = compute_all_metrics(equity_df, trades_df, start_capital=10000.0, freq="1d")
        gate_result = evaluate_all_gates(metrics)

        report_path = generate_qa_report(
            metrics=metrics,
            gate_result=gate_result,
            strategy_name="ema_trend",
            freq="1d",
            equity_curve_path=Path("output/portfolio_equity_1d.csv")
        )
        ```
    """
    base = output_dir if output_dir else OUTPUT_DIR / "reports"
    base.mkdir(parents=True, exist_ok=True)

    # Generate filename
    date_str = datetime.now().strftime("%Y%m%d")
    report_filename = f"qa_report_{strategy_name}_{freq}_{date_str}.md"
    report_path = base / report_filename

    # Generate report content
    content = _build_report_content(
        metrics=metrics,
        gate_result=gate_result,
        strategy_name=strategy_name,
        freq=freq,
        equity_curve_path=equity_curve_path,
        data_start_date=data_start_date,
        data_end_date=data_end_date,
        config_info=config_info,
        equity=equity,
        portfolio_positions=portfolio_positions,
        shipping_features=shipping_features,
    )

    # Write report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    return report_path


def _build_report_content(
    metrics: PerformanceMetrics,
    gate_result: QAGatesSummary | None,
    strategy_name: str,
    freq: str,
    equity_curve_path: Path | str | None,
    data_start_date: pd.Timestamp | None,
    data_end_date: pd.Timestamp | None,
    config_info: dict[str, Any] | None,
    equity: pd.DataFrame | None = None,
    portfolio_positions: pd.DataFrame | None = None,
    shipping_features: pd.DataFrame | None = None,
) -> str:
    """Build Markdown report content.

    Args:
        metrics: PerformanceMetrics
        gate_result: Optional QAGatesSummary
        strategy_name: Strategy name
        freq: Trading frequency
        equity_curve_path: Optional equity curve path
        data_start_date: Optional data start date
        data_end_date: Optional data end date
        config_info: Optional config info

    Returns:
        Markdown content as string
    """
    lines = []

    # Header
    lines.append(f"# QA Report: {strategy_name} ({freq})")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Performance Metrics Section
    lines.append("## Performance Metrics")
    lines.append("")
    lines.append("### Returns")
    lines.append(f"- **Final Performance Factor:** {metrics.final_pf:.4f}")
    lines.append(f"- **Total Return:** {metrics.total_return:.2%}")
    if metrics.cagr is not None:
        lines.append(f"- **CAGR:** {metrics.cagr:.2%}")
    else:
        lines.append("- **CAGR:** N/A (less than 1 year of data)")
    lines.append("")

    lines.append("### Risk-Adjusted Returns")
    if metrics.sharpe_ratio is not None:
        lines.append(f"- **Sharpe Ratio:** {metrics.sharpe_ratio:.4f}")
    else:
        lines.append("- **Sharpe Ratio:** N/A (insufficient data)")
    if metrics.sortino_ratio is not None:
        lines.append(f"- **Sortino Ratio:** {metrics.sortino_ratio:.4f}")
    if metrics.calmar_ratio is not None:
        lines.append(f"- **Calmar Ratio:** {metrics.calmar_ratio:.4f}")
    lines.append("")

    lines.append("### Risk Metrics")
    lines.append(
        f"- **Max Drawdown:** {metrics.max_drawdown:.2f} ({metrics.max_drawdown_pct:.2f}%)"
    )
    lines.append(f"- **Current Drawdown:** {metrics.current_drawdown:.2f}")
    if metrics.volatility is not None:
        lines.append(f"- **Volatility (annualized):** {metrics.volatility:.2%}")
    if metrics.var_95 is not None:
        lines.append(f"- **VaR (95%):** {metrics.var_95:.2f}")

    # Additional portfolio risk metrics from risk_metrics module (if equity provided)
    if equity is not None and not equity.empty:
        try:
            risk_metrics = compute_portfolio_risk_metrics(equity, freq=freq)

            # Add daily volatility if available
            if risk_metrics["daily_vol"] is not None:
                lines.append(f"- **Daily Volatility:** {risk_metrics['daily_vol']:.4f}")

            # Add Expected Shortfall if available
            if risk_metrics["es_95"] is not None:
                lines.append(
                    f"- **Expected Shortfall (95%):** {risk_metrics['es_95']:.2f}"
                )
        except Exception:
            # Silently fail if risk metrics computation fails (backward compatibility)
            pass

    lines.append("")

    # Portfolio Risk Metrics Section (additional risk metrics)
    # Note: This requires the equity DataFrame, which we don't have directly in metrics
    # We'll add this section if/when we have access to the equity DataFrame
    # For now, we keep the existing risk metrics section above

    # Trade Metrics (if available)
    if metrics.total_trades is not None and metrics.total_trades > 0:
        lines.append("### Trade Metrics")
        lines.append(f"- **Total Trades:** {metrics.total_trades}")
        if metrics.hit_rate is not None:
            lines.append(f"- **Hit Rate:** {metrics.hit_rate:.2%}")
        if metrics.profit_factor is not None:
            lines.append(f"- **Profit Factor:** {metrics.profit_factor:.2f}")
        if metrics.turnover is not None:
            lines.append(f"- **Turnover (annualized):** {metrics.turnover:.2f}x")
        lines.append("")

    # Metadata
    lines.append("### Period Information")
    lines.append(f"- **Start Date:** {metrics.start_date.strftime('%Y-%m-%d')}")
    lines.append(f"- **End Date:** {metrics.end_date.strftime('%Y-%m-%d')}")
    lines.append(f"- **Periods:** {metrics.periods}")
    lines.append(f"- **Start Capital:** {metrics.start_capital:,.2f}")
    lines.append(f"- **End Equity:** {metrics.end_equity:,.2f}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Shipping Risk Section (optional, if portfolio and shipping features provided)
    if portfolio_positions is not None and shipping_features is not None:
        try:
            shipping_exposure = compute_shipping_exposure(
                portfolio_positions, shipping_features
            )
            risk_flags = compute_systemic_risk_flags(shipping_exposure)

            lines.append("## Shipping Risk")
            lines.append("")
            lines.append(
                f"- **Average Shipping Congestion:** {shipping_exposure['avg_shipping_congestion']:.2f}"
            )
            lines.append(
                f"- **High Congestion Weight:** {shipping_exposure['high_congestion_weight']:.2%}"
            )
            if shipping_exposure["exposed_symbols"]:
                lines.append(
                    f"- **Exposed Symbols:** {', '.join(shipping_exposure['exposed_symbols'])}"
                )
            if shipping_exposure["top_routes"]:
                lines.append(
                    f"- **Top Routes:** {', '.join(shipping_exposure['top_routes'][:5])}"
                )
            lines.append("")
            lines.append("### Systemic Risk Flags")
            lines.append(
                f"- **High Shipping Risk:** {risk_flags['high_shipping_risk']}"
            )
            lines.append(
                f"- **Exposed to Blockade Routes:** {risk_flags['exposed_to_blockade_routes']}"
            )
            lines.append(f"- **Risk Level:** {risk_flags['risk_level']}")
            lines.append(f"- **Risk Reason:** {risk_flags['risk_reason']}")
            lines.append("")
        except Exception:
            # Silently fail if shipping risk computation fails (backward compatibility)
            pass

    # QA Gates Section
    if gate_result is not None:
        lines.append("## QA Gates")
        lines.append("")

        # Overall status
        overall_status = gate_result.overall_result.value.upper()
        status_emoji = {"OK": "✅", "WARNING": "⚠️", "BLOCK": "❌"}
        emoji = status_emoji.get(overall_status, "❓")
        lines.append(f"### Overall Status: {emoji} **{overall_status}**")
        lines.append("")

        # Gate counts (use attributes if available, otherwise compute)
        if (
            hasattr(gate_result, "passed_gates")
            and hasattr(gate_result, "warning_gates")
            and hasattr(gate_result, "blocked_gates")
        ):
            passed = gate_result.passed_gates
            warnings = gate_result.warning_gates
            blocked = gate_result.blocked_gates
        else:
            passed = sum(1 for r in gate_result.gate_results if r.result.value == "ok")
            warnings = sum(
                1 for r in gate_result.gate_results if r.result.value == "warning"
            )
            blocked = sum(
                1 for r in gate_result.gate_results if r.result.value == "block"
            )

        lines.append(f"- **Passed:** {passed}")
        lines.append(f"- **Warnings:** {warnings}")
        lines.append(f"- **Blocked:** {blocked}")
        lines.append("")

        # Individual gate results
        lines.append("### Gate Details")
        lines.append("")
        lines.append("| Gate | Status | Reason |")
        lines.append("|------|--------|--------|")

        for gate in gate_result.gate_results:
            status_emoji_gate = {"ok": "✅", "warning": "⚠️", "block": "❌"}
            emoji_gate = status_emoji_gate.get(gate.result.value, "❓")
            status_str = f"{emoji_gate} {gate.result.value.upper()}"
            reason_short = (
                gate.reason[:60] + "..." if len(gate.reason) > 60 else gate.reason
            )
            lines.append(f"| {gate.gate_name} | {status_str} | {reason_short} |")

        lines.append("")
        lines.append("---")
        lines.append("")

    # Equity Curve Section
    if equity_curve_path is not None:
        lines.append("## Equity Curve")
        lines.append("")
        if isinstance(equity_curve_path, str):
            equity_path = equity_curve_path
        else:
            equity_path = str(equity_curve_path)

        # Check if file exists and is relative to output dir
        if Path(equity_path).exists():
            # Use relative path if possible
            try:
                rel_path = Path(equity_path).relative_to(OUTPUT_DIR)
                lines.append(f"**Equity Curve File:** `{rel_path}`")
            except ValueError:
                lines.append(f"**Equity Curve File:** `{equity_path}`")
        else:
            lines.append(f"**Equity Curve File:** `{equity_path}` (file not found)")

        lines.append("")
        lines.append(
            "> **Note:** To visualize the equity curve, load the CSV file in your preferred tool (Excel, Python, etc.)"
        )
        lines.append("")
        lines.append("---")
        lines.append("")

    # Data Status Section
    lines.append("## Data Status")
    lines.append("")
    if data_start_date is not None:
        lines.append(f"- **Data Start Date:** {data_start_date.strftime('%Y-%m-%d')}")
    if data_end_date is not None:
        lines.append(f"- **Data End Date:** {data_end_date.strftime('%Y-%m-%d')}")
    if data_start_date is not None and data_end_date is not None:
        days = (data_end_date - data_start_date).days
        lines.append(f"- **Data Range:** {days} days")
    lines.append(f"- **Frequency:** {freq}")
    lines.append("")

    # Configuration Section
    if config_info:
        lines.append("## Configuration")
        lines.append("")
        for key, value in config_info.items():
            if isinstance(value, (int, float)):
                lines.append(f"- **{key}:** {value}")
            elif isinstance(value, bool):
                lines.append(f"- **{key}:** {value}")
            else:
                lines.append(f"- **{key}:** {value}")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by assembled_core.reports.daily_qa_report*")

    return "\n".join(lines)


def generate_qa_report_from_files(
    freq: str,
    strategy_name: str = "default",
    equity_file: str | Path | None = None,
    trades_file: str | Path | None = None,
    start_capital: float = 10000.0,
    config_info: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Generate QA report by loading metrics from files.

    This is a convenience function that loads equity/trades data, computes metrics,
    evaluates gates, and generates a report.

    Args:
        freq: Trading frequency ("1d" or "5min")
        strategy_name: Strategy name
        equity_file: Path to equity curve CSV (default: portfolio_equity_{freq}.csv in OUTPUT_DIR)
        trades_file: Optional path to trades CSV
        start_capital: Starting capital
        config_info: Optional config info dict
        output_dir: Output directory for reports (default: OUTPUT_DIR / "reports")

    Returns:
        Path to generated report file
    """
    from src.assembled_core.config import OUTPUT_DIR
    from src.assembled_core.qa.metrics import compute_all_metrics
    from src.assembled_core.qa.qa_gates import evaluate_all_gates

    # Base directory for loading equity/trades files
    base = OUTPUT_DIR

    # Reports directory (can be overridden)
    reports_dir = output_dir if output_dir else OUTPUT_DIR / "reports"

    # Load equity data
    if equity_file is None:
        equity_file = base / f"portfolio_equity_{freq}.csv"

    if isinstance(equity_file, str):
        equity_file = Path(equity_file)

    if not equity_file.exists():
        raise FileNotFoundError(f"Equity file not found: {equity_file}")

    equity_df = pd.read_csv(equity_file)
    equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)

    # Load trades if available
    trades_df = None
    if trades_file is not None:
        if isinstance(trades_file, str):
            trades_file = Path(trades_file)
        if trades_file.exists():
            trades_df = pd.read_csv(trades_file)
            trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"], utc=True)

    # Compute metrics
    metrics = compute_all_metrics(
        equity=equity_df,
        trades=trades_df,
        start_capital=start_capital,
        freq=freq,
        risk_free_rate=0.0,
    )

    # Evaluate gates
    gate_result = evaluate_all_gates(metrics)

    # Generate report
    return generate_qa_report(
        metrics=metrics,
        gate_result=gate_result,
        strategy_name=strategy_name,
        freq=freq,
        equity_curve_path=equity_file,
        data_start_date=metrics.start_date,
        data_end_date=metrics.end_date,
        config_info=config_info,
        output_dir=reports_dir,
    )
