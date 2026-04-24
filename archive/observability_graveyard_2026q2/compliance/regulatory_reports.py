"""Regulatory Report Generator.

Generates compliance-ready reports:
- Best Execution Report (RTS 28 / MiFID II)
- Transaction Cost Report
- Risk Report (VaR, Stress Tests, Exposure)
- Model Inventory (all ML models with version, training date, performance)

Output: structured dicts suitable for JSON/PDF rendering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BestExecutionReport:
    """RTS 28 Best Execution Report."""
    report_date: str
    period_start: str
    period_end: str
    total_orders: int
    total_fills: int
    venues: list[dict[str, Any]]  # [{venue, pct_volume, avg_fill_time_ms}]
    avg_slippage_bps: float
    avg_market_impact_bps: float
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TransactionCostReport:
    """Transaction cost analysis report."""
    report_date: str
    period_start: str
    period_end: str
    total_trades: int
    total_cost_bps: float
    cost_breakdown: dict[str, float]  # {commission, spread, impact, fees}
    by_symbol: dict[str, float]  # symbol -> total cost in bps
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RiskReport:
    """Risk metrics report."""
    report_date: str
    var_95: float  # 95% VaR as fraction
    var_99: float
    cvar_95: float
    max_drawdown: float
    current_drawdown: float
    gross_exposure: float
    net_exposure: float
    sector_exposures: dict[str, float]
    stress_test_results: dict[str, float]  # scenario -> PnL impact
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelInventoryEntry:
    """Single model entry in the inventory."""
    model_name: str
    model_type: str  # e.g., "XGBoost", "Ridge", "BNN"
    version: str
    training_date: str
    feature_count: int
    performance_metrics: dict[str, float]  # {sharpe, ic, hit_rate, etc.}
    status: str  # "active", "retired", "testing"


@dataclass
class ModelInventoryReport:
    """Complete model inventory report."""
    report_date: str
    models: list[ModelInventoryEntry]
    total_active: int
    total_retired: int
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_best_execution_report(
    fills: pd.DataFrame,
    period_start: str,
    period_end: str,
) -> BestExecutionReport:
    """Generate RTS 28 Best Execution Report.

    Args:
        fills: DataFrame with columns [symbol, venue, fill_price, arrival_price,
               fill_time_ms, volume, timestamp].
        period_start: Start date ISO-8601.
        period_end: End date ISO-8601.

    Returns:
        BestExecutionReport.
    """
    if fills.empty:
        return BestExecutionReport(
            report_date=datetime.now(timezone.utc).isoformat(),
            period_start=period_start,
            period_end=period_end,
            total_orders=0,
            total_fills=0,
            venues=[],
            avg_slippage_bps=0.0,
            avg_market_impact_bps=0.0,
            summary="No fills in period.",
        )

    total = len(fills)

    # Venue breakdown
    venues = []
    if "venue" in fills.columns:
        for venue, grp in fills.groupby("venue"):
            venues.append({
                "venue": str(venue),
                "pct_volume": round(len(grp) / total * 100, 2),
                "avg_fill_time_ms": round(float(grp["fill_time_ms"].mean()), 1)
                if "fill_time_ms" in grp.columns else 0.0,
            })

    # Slippage: fill_price vs arrival_price
    slippage_bps = 0.0
    if "fill_price" in fills.columns and "arrival_price" in fills.columns:
        mask = fills["arrival_price"] > 0
        if mask.any():
            slip = (fills.loc[mask, "fill_price"] - fills.loc[mask, "arrival_price"]) / fills.loc[mask, "arrival_price"]
            slippage_bps = round(float(slip.mean()) * 10000, 2)

    return BestExecutionReport(
        report_date=datetime.now(timezone.utc).isoformat(),
        period_start=period_start,
        period_end=period_end,
        total_orders=total,
        total_fills=total,
        venues=venues,
        avg_slippage_bps=slippage_bps,
        avg_market_impact_bps=round(abs(slippage_bps) * 0.6, 2),  # rough estimate
        summary=f"Executed {total} fills across {len(venues)} venues. Avg slippage: {slippage_bps:.1f}bps.",
    )


def generate_transaction_cost_report(
    trades: pd.DataFrame,
    period_start: str,
    period_end: str,
) -> TransactionCostReport:
    """Generate transaction cost analysis report.

    Args:
        trades: DataFrame with columns [symbol, cost_bps, commission_bps,
                spread_bps, impact_bps].
        period_start: Start date.
        period_end: End date.

    Returns:
        TransactionCostReport.
    """
    if trades.empty:
        return TransactionCostReport(
            report_date=datetime.now(timezone.utc).isoformat(),
            period_start=period_start,
            period_end=period_end,
            total_trades=0,
            total_cost_bps=0.0,
            cost_breakdown={},
            by_symbol={},
            summary="No trades in period.",
        )

    total = len(trades)
    cost_cols = ["commission_bps", "spread_bps", "impact_bps"]
    breakdown = {}
    for col in cost_cols:
        if col in trades.columns:
            breakdown[col.replace("_bps", "")] = round(float(trades[col].mean()), 2)

    total_cost = sum(breakdown.values())

    by_symbol = {}
    if "symbol" in trades.columns and "cost_bps" in trades.columns:
        for sym, grp in trades.groupby("symbol"):
            by_symbol[str(sym)] = round(float(grp["cost_bps"].mean()), 2)

    return TransactionCostReport(
        report_date=datetime.now(timezone.utc).isoformat(),
        period_start=period_start,
        period_end=period_end,
        total_trades=total,
        total_cost_bps=round(total_cost, 2),
        cost_breakdown=breakdown,
        by_symbol=by_symbol,
        summary=f"{total} trades, avg total cost: {total_cost:.1f}bps.",
    )


def generate_risk_report(
    returns: pd.Series,
    positions: dict[str, float] | None = None,
    sector_mapping: dict[str, str] | None = None,
    stress_scenarios: dict[str, float] | None = None,
) -> RiskReport:
    """Generate risk metrics report.

    Args:
        returns: Portfolio daily returns series.
        positions: Current symbol → weight.
        sector_mapping: Symbol → sector.
        stress_scenarios: Scenario name → portfolio impact (fraction).

    Returns:
        RiskReport.
    """
    returns = pd.Series(returns, dtype=float).dropna()

    if len(returns) < 2:
        return RiskReport(
            report_date=datetime.now(timezone.utc).isoformat(),
            var_95=0.0, var_99=0.0, cvar_95=0.0,
            max_drawdown=0.0, current_drawdown=0.0,
            gross_exposure=0.0, net_exposure=0.0,
            sector_exposures={}, stress_test_results={},
            summary="Insufficient data.",
        )

    var_95 = float(np.percentile(returns, 5))
    var_99 = float(np.percentile(returns, 1))
    cvar_95 = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else var_95

    cumret = (1 + returns).cumprod()
    peak = cumret.cummax()
    dd = (cumret - peak) / peak
    max_dd = float(dd.min())
    curr_dd = float(dd.iloc[-1])

    # Exposure
    positions = positions or {}
    gross = sum(abs(v) for v in positions.values())
    net = sum(positions.values())

    # Sector exposure
    sector_exp = {}
    if sector_mapping and positions:
        for sym, w in positions.items():
            sec = sector_mapping.get(sym, "Unknown")
            sector_exp[sec] = sector_exp.get(sec, 0.0) + w

    return RiskReport(
        report_date=datetime.now(timezone.utc).isoformat(),
        var_95=round(var_95, 6),
        var_99=round(var_99, 6),
        cvar_95=round(cvar_95, 6),
        max_drawdown=round(max_dd, 6),
        current_drawdown=round(curr_dd, 6),
        gross_exposure=round(gross, 4),
        net_exposure=round(net, 4),
        sector_exposures={k: round(v, 4) for k, v in sector_exp.items()},
        stress_test_results=stress_scenarios or {},
        summary=f"VaR95={var_95:.4f}, MaxDD={max_dd:.4f}, GrossExp={gross:.2f}",
    )


def generate_model_inventory(
    models: list[dict[str, Any]],
) -> ModelInventoryReport:
    """Generate model inventory report.

    Args:
        models: List of model dicts with keys:
            name, type, version, training_date, feature_count,
            metrics (dict), status.

    Returns:
        ModelInventoryReport.
    """
    entries = []
    for m in models:
        entries.append(ModelInventoryEntry(
            model_name=m.get("name", "unknown"),
            model_type=m.get("type", "unknown"),
            version=m.get("version", "0.0.1"),
            training_date=m.get("training_date", "unknown"),
            feature_count=m.get("feature_count", 0),
            performance_metrics=m.get("metrics", {}),
            status=m.get("status", "active"),
        ))

    active = sum(1 for e in entries if e.status == "active")
    retired = sum(1 for e in entries if e.status == "retired")

    return ModelInventoryReport(
        report_date=datetime.now(timezone.utc).isoformat(),
        models=entries,
        total_active=active,
        total_retired=retired,
        summary=f"{len(entries)} models: {active} active, {retired} retired.",
    )


__all__ = [
    "BestExecutionReport",
    "TransactionCostReport",
    "RiskReport",
    "ModelInventoryReport",
    "generate_best_execution_report",
    "generate_transaction_cost_report",
    "generate_risk_report",
    "generate_model_inventory",
]
