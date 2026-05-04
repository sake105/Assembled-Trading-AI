"""Benchmark-relative performance attribution (V7).

Computes alpha, beta, information ratio, tracking error vs a benchmark,
and Brinson-Fachler sector attribution decomposition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

PERIODS_PER_YEAR = 252


@dataclass
class BenchmarkMetrics:
    """Benchmark-relative performance metrics."""

    alpha: float | None  # Annualized Jensen's alpha
    beta: float | None  # Portfolio beta to benchmark
    information_ratio: (
        float | None
    )  # IR = mean(active return) / std(active return) * sqrt(252)
    tracking_error: float | None  # Annualized std of active returns
    active_return: float | None  # Annualized mean active return
    r_squared: float | None  # R² of portfolio vs benchmark
    up_capture: float | None  # Up-market capture ratio
    down_capture: float | None  # Down-market capture ratio


def compute_benchmark_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> BenchmarkMetrics:
    """Compute benchmark-relative metrics.

    Args:
        portfolio_returns: Daily portfolio returns.
        benchmark_returns: Daily benchmark returns (aligned by index).
        risk_free_rate: Annualized risk-free rate.

    Returns:
        BenchmarkMetrics dataclass.
    """
    # Align series
    aligned = pd.DataFrame(
        {
            "port": portfolio_returns,
            "bench": benchmark_returns,
        }
    ).dropna()

    if len(aligned) < 10:
        return BenchmarkMetrics(
            alpha=None,
            beta=None,
            information_ratio=None,
            tracking_error=None,
            active_return=None,
            r_squared=None,
            up_capture=None,
            down_capture=None,
        )

    port = aligned["port"]
    bench = aligned["bench"]
    daily_rf = risk_free_rate / PERIODS_PER_YEAR

    # Beta and Alpha (CAPM regression)
    cov_pb = port.cov(bench)
    var_b = bench.var()
    beta = float(cov_pb / var_b) if var_b > 1e-12 else None

    if beta is not None:
        alpha_daily = float(port.mean() - daily_rf - beta * (bench.mean() - daily_rf))
        alpha = alpha_daily * PERIODS_PER_YEAR
    else:
        alpha = None

    # Active returns
    active = port - bench
    active_mean = float(active.mean())
    active_std = float(active.std())

    tracking_error = (
        active_std * np.sqrt(PERIODS_PER_YEAR) if active_std > 1e-12 else None
    )
    active_return = active_mean * PERIODS_PER_YEAR
    information_ratio = (
        float(active_mean / active_std * np.sqrt(PERIODS_PER_YEAR))
        if active_std > 1e-12
        else None
    )

    # R-squared
    ss_res = (
        ((port - (daily_rf + beta * (bench - daily_rf))) ** 2).sum() if beta else None
    )
    ss_tot = ((port - port.mean()) ** 2).sum()
    r_squared = (
        float(1 - ss_res / ss_tot) if ss_res is not None and ss_tot > 1e-12 else None
    )

    # Capture ratios
    up_days = bench > 0
    down_days = bench < 0

    up_capture = None
    if up_days.sum() > 5:
        up_port = port[up_days].mean()
        up_bench = bench[up_days].mean()
        if abs(up_bench) > 1e-12:
            up_capture = float(up_port / up_bench)

    down_capture = None
    if down_days.sum() > 5:
        down_port = port[down_days].mean()
        down_bench = bench[down_days].mean()
        if abs(down_bench) > 1e-12:
            down_capture = float(down_port / down_bench)

    return BenchmarkMetrics(
        alpha=round(alpha, 4) if alpha is not None else None,
        beta=round(beta, 4) if beta is not None else None,
        information_ratio=(
            round(information_ratio, 4) if information_ratio is not None else None
        ),
        tracking_error=round(tracking_error, 4) if tracking_error is not None else None,
        active_return=round(active_return, 4),
        r_squared=round(r_squared, 4) if r_squared is not None else None,
        up_capture=round(up_capture, 4) if up_capture is not None else None,
        down_capture=round(down_capture, 4) if down_capture is not None else None,
    )


@dataclass
class BrinsonAttribution:
    """Brinson-Fachler sector attribution."""

    allocation_effect: dict[str, float]  # Sector -> allocation contribution
    selection_effect: dict[str, float]  # Sector -> stock selection contribution
    interaction_effect: dict[str, float]  # Sector -> interaction term
    total_allocation: float
    total_selection: float
    total_interaction: float
    total_active_return: float


def brinson_fachler_attribution(
    portfolio_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    sector_mapping: dict[str, str] | None = None,
) -> BrinsonAttribution:
    """Compute Brinson-Fachler sector attribution.

    Args:
        portfolio_weights: DataFrame with symbol, weight, sector.
        benchmark_weights: DataFrame with symbol, weight, sector.
        portfolio_returns: DataFrame with symbol, return.
        benchmark_returns: DataFrame with symbol, return.
        sector_mapping: Optional symbol -> sector mapping.

    Returns:
        BrinsonAttribution with per-sector decomposition.
    """

    # Build sector-level aggregates
    def aggregate_by_sector(weights_df, returns_df, sector_map):
        merged = weights_df.merge(
            returns_df, on="symbol", how="inner", suffixes=("", "_ret")
        )
        if "sector" not in merged.columns and sector_map:
            merged["sector"] = merged["symbol"].map(sector_map).fillna("Other")
        elif "sector" not in merged.columns:
            merged["sector"] = "Other"

        ret_col = [
            c for c in merged.columns if "return" in c.lower() or c.endswith("_ret")
        ]
        ret_col = ret_col[0] if ret_col else "return"

        sector_agg = merged.groupby("sector").agg(
            weight=("weight", "sum"),
            weighted_return=(
                ret_col,
                lambda x: (x * merged.loc[x.index, "weight"]).sum()
                / max(merged.loc[x.index, "weight"].sum(), 1e-12),
            ),
        )
        return sector_agg

    try:
        port_sectors = aggregate_by_sector(
            portfolio_weights, portfolio_returns, sector_mapping
        )
        bench_sectors = aggregate_by_sector(
            benchmark_weights, benchmark_returns, sector_mapping
        )
    except Exception as e:
        _log.warning("Brinson-Fachler attribution failed: %s", e)
        return BrinsonAttribution(
            allocation_effect={},
            selection_effect={},
            interaction_effect={},
            total_allocation=0.0,
            total_selection=0.0,
            total_interaction=0.0,
            total_active_return=0.0,
        )

    all_sectors = set(port_sectors.index) | set(bench_sectors.index)
    bench_total_return = float(
        (bench_sectors["weight"] * bench_sectors["weighted_return"]).sum()
    )

    allocation: dict[str, float] = {}
    selection: dict[str, float] = {}
    interaction: dict[str, float] = {}

    for sector in all_sectors:
        wp = port_sectors.loc[sector, "weight"] if sector in port_sectors.index else 0.0
        wb = (
            bench_sectors.loc[sector, "weight"]
            if sector in bench_sectors.index
            else 0.0
        )
        rp = (
            port_sectors.loc[sector, "weighted_return"]
            if sector in port_sectors.index
            else 0.0
        )
        rb = (
            bench_sectors.loc[sector, "weighted_return"]
            if sector in bench_sectors.index
            else 0.0
        )

        allocation[sector] = round(float((wp - wb) * (rb - bench_total_return)), 6)
        selection[sector] = round(float(wb * (rp - rb)), 6)
        interaction[sector] = round(float((wp - wb) * (rp - rb)), 6)

    return BrinsonAttribution(
        allocation_effect=allocation,
        selection_effect=selection,
        interaction_effect=interaction,
        total_allocation=round(sum(allocation.values()), 6),
        total_selection=round(sum(selection.values()), 6),
        total_interaction=round(sum(interaction.values()), 6),
        total_active_return=round(
            sum(allocation.values())
            + sum(selection.values())
            + sum(interaction.values()),
            6,
        ),
    )


__all__ = [
    "BenchmarkMetrics",
    "BrinsonAttribution",
    "compute_benchmark_metrics",
    "brinson_fachler_attribution",
]
