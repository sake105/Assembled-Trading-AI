"""Portfolio Analyzer — comprehensive portfolio performance and risk analytics.

Covers 7 analysis dimensions:
1. Performance: CAGR, Sharpe, Sortino, Calmar, MaxDD, Profit Factor, Win Rate
2. Portfolio structure: weights, core/satellite split, sector/region exposure
3. Monte Carlo: (delegates to qa.monte_carlo)
4. Stress tests: (delegates to qa.scenario_engine)
5. Regime analysis: performance by regime
6. Attribution: per-symbol/sector/strategy contribution
7. Sensitivity: parameter stability across variations
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Performance Profile
# ---------------------------------------------------------------------------


@dataclass
class PerformanceProfile:
    """Comprehensive performance metrics derived from daily returns."""

    cagr: float
    total_return: float
    annualized_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_duration_days: int
    profit_factor: float
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy: float
    turnover_annual: float
    avg_exposure: float
    trading_days: int


def _to_array(returns: np.ndarray | pd.Series) -> np.ndarray:
    """Coerce input to a clean 1-D float array, dropping NaNs."""
    if isinstance(returns, pd.Series):
        arr = returns.to_numpy(dtype=float)
    else:
        arr = np.asarray(returns, dtype=float)
    return arr[~np.isnan(arr)]


def compute_performance_profile(
    daily_returns: np.ndarray | pd.Series,
    trading_days_per_year: int = 252,
    turnover_annual: float = 0.0,
    avg_exposure: float = 1.0,
) -> PerformanceProfile:
    """Compute all performance metrics from daily returns.

    Args:
        daily_returns: Array or Series of daily returns (e.g. 0.01 = 1%).
        trading_days_per_year: Annualization constant (default 252).
        turnover_annual: Optional annualised turnover figure.
        avg_exposure: Optional average portfolio exposure.

    Returns:
        PerformanceProfile dataclass.

    Raises:
        ValueError: If daily_returns is empty.
    """
    arr = _to_array(daily_returns)
    if len(arr) == 0:
        raise ValueError("daily_returns must not be empty.")

    n = len(arr)
    trading_days = n

    # Total return (compound)
    total_return = float(np.prod(1.0 + arr) - 1.0)

    # CAGR (geometric)
    years = n / trading_days_per_year
    if years > 0 and (1.0 + total_return) > 0:
        cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    else:
        cagr = 0.0

    # Volatility (annualised)
    vol = float(np.std(arr, ddof=1)) if n >= 2 else 0.0
    annualized_vol = vol * np.sqrt(trading_days_per_year)

    # Sharpe (rf = 0)
    mean_r = float(np.mean(arr))
    sharpe = (mean_r / vol * np.sqrt(trading_days_per_year)) if vol > 0 else 0.0

    # Sortino — downside std of negative returns only
    neg = arr[arr < 0]
    if len(neg) >= 2:
        downside_std = float(np.std(neg, ddof=1))
    elif len(neg) == 1:
        downside_std = float(abs(neg[0]))
    else:
        downside_std = 0.0
    ann_downside = downside_std * np.sqrt(trading_days_per_year)
    sortino = (mean_r * trading_days_per_year / ann_downside) if ann_downside > 0 else 0.0

    # MaxDrawdown (negative value) and duration
    equity_curve = np.cumprod(1.0 + arr)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown_series = equity_curve / running_max - 1.0
    max_drawdown = float(np.min(drawdown_series))

    # Drawdown duration: longest consecutive period below peak
    in_dd = drawdown_series < 0
    max_dur = 0
    cur_dur = 0
    for v in in_dd:
        if v:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0
    max_drawdown_duration_days = max_dur

    # Calmar
    calmar = (cagr / abs(max_drawdown)) if max_drawdown < 0 else 0.0

    # Win / loss stats
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    win_rate = float(len(wins) / n)
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    profit_factor = (
        float(np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else float("inf")
    )
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    return PerformanceProfile(
        cagr=cagr,
        total_return=total_return,
        annualized_vol=annualized_vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_drawdown,
        max_drawdown_duration_days=max_drawdown_duration_days,
        profit_factor=profit_factor,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy=expectancy,
        turnover_annual=turnover_annual,
        avg_exposure=avg_exposure,
        trading_days=trading_days,
    )


# ---------------------------------------------------------------------------
# Portfolio Structure
# ---------------------------------------------------------------------------


@dataclass
class PortfolioStructure:
    """Portfolio composition and concentration metrics."""

    weights: dict[str, float]
    total_invested: float
    cash_pct: float
    n_positions: int
    top_5_concentration: float
    herfindahl_index: float
    sector_weights: dict[str, float]
    region_weights: dict[str, float]


def compute_portfolio_structure(
    weights: dict[str, float],
    symbol_metadata: dict[str, dict] | None = None,
) -> PortfolioStructure:
    """Compute portfolio structure metrics.

    Args:
        weights: symbol -> portfolio weight (e.g. 0.05 = 5%).
        symbol_metadata: Optional mapping symbol -> {sector: str, region: str}.

    Returns:
        PortfolioStructure dataclass.
    """
    if not weights:
        return PortfolioStructure(
            weights={},
            total_invested=0.0,
            cash_pct=1.0,
            n_positions=0,
            top_5_concentration=0.0,
            herfindahl_index=0.0,
            sector_weights={},
            region_weights={},
        )

    total_invested = float(sum(weights.values()))
    cash_pct = max(0.0, 1.0 - total_invested)
    n_positions = len(weights)

    sorted_weights = sorted(weights.values(), reverse=True)
    top_5_concentration = float(sum(sorted_weights[:5]))
    herfindahl_index = float(sum(w ** 2 for w in weights.values()))

    sector_weights: dict[str, float] = {}
    region_weights: dict[str, float] = {}

    if symbol_metadata:
        for sym, w in weights.items():
            meta = symbol_metadata.get(sym, {})
            sector = meta.get("sector", "Unknown")
            region = meta.get("region", "Unknown")
            sector_weights[sector] = sector_weights.get(sector, 0.0) + w
            region_weights[region] = region_weights.get(region, 0.0) + w

    return PortfolioStructure(
        weights=dict(weights),
        total_invested=total_invested,
        cash_pct=cash_pct,
        n_positions=n_positions,
        top_5_concentration=top_5_concentration,
        herfindahl_index=herfindahl_index,
        sector_weights=sector_weights,
        region_weights=region_weights,
    )


# ---------------------------------------------------------------------------
# Regime Performance
# ---------------------------------------------------------------------------


@dataclass
class RegimePerformance:
    """Performance statistics for a single market regime."""

    regime_name: str
    n_days: int
    mean_daily_return: float
    hit_rate: float
    sharpe: float
    avg_drawdown: float


def analyze_regime_performance(
    daily_returns: pd.Series,
    regime_labels: pd.Series,
) -> dict[str, RegimePerformance]:
    """Compute performance metrics broken down by regime label.

    Args:
        daily_returns: pd.Series of daily returns, datetime-indexed.
        regime_labels: pd.Series of string regime names with the same index.

    Returns:
        Dict mapping regime_name -> RegimePerformance.
    """
    result: dict[str, RegimePerformance] = {}

    aligned = pd.DataFrame({"return": daily_returns, "regime": regime_labels}).dropna()

    for regime_name, regime_data in aligned.groupby("regime", sort=False):
        sub = regime_data["return"]
        n = len(sub)
        if n < 2:
            continue

        arr = sub.to_numpy(dtype=float)
        mean_r = float(np.mean(arr))
        hit_rate = float(np.sum(arr > 0) / n)
        vol = float(np.std(arr, ddof=1))
        sharpe = (mean_r / vol * np.sqrt(252)) if vol > 0 else 0.0

        # Avg drawdown within this regime
        eq = np.cumprod(1.0 + arr)
        rm = np.maximum.accumulate(eq)
        dd = eq / rm - 1.0
        avg_drawdown = float(np.mean(dd))

        result[regime_name] = RegimePerformance(
            regime_name=regime_name,
            n_days=n,
            mean_daily_return=mean_r,
            hit_rate=hit_rate,
            sharpe=sharpe,
            avg_drawdown=avg_drawdown,
        )

    return result


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


@dataclass
class AttributionReport:
    """Return attribution across symbols and sectors."""

    symbol_contributions: dict[str, float]
    sector_contributions: dict[str, float]
    top_contributors: list[tuple[str, float]]
    bottom_contributors: list[tuple[str, float]]
    total_return: float


def compute_attribution(
    weights: dict[str, float],
    returns: dict[str, float],
    symbol_metadata: dict[str, dict] | None = None,
) -> AttributionReport:
    """Compute return attribution per symbol and sector.

    Args:
        weights: symbol -> weight.
        returns: symbol -> period return.
        symbol_metadata: Optional symbol -> {sector: str, region: str}.

    Returns:
        AttributionReport dataclass.
    """
    symbol_contributions: dict[str, float] = {}
    for sym, w in weights.items():
        r = returns.get(sym, 0.0)
        symbol_contributions[sym] = w * r

    total_return = float(sum(symbol_contributions.values()))

    sector_contributions: dict[str, float] = {}
    if symbol_metadata:
        for sym, contrib in symbol_contributions.items():
            sector = symbol_metadata.get(sym, {}).get("sector", "Unknown")
            sector_contributions[sector] = sector_contributions.get(sector, 0.0) + contrib

    sorted_all = sorted(symbol_contributions.items(), key=lambda x: x[1], reverse=True)
    top_contributors = sorted_all[:5]
    bottom_contributors = list(reversed(sorted_all[-5:]))

    return AttributionReport(
        symbol_contributions=symbol_contributions,
        sector_contributions=sector_contributions,
        top_contributors=top_contributors,
        bottom_contributors=bottom_contributors,
        total_return=total_return,
    )


# ---------------------------------------------------------------------------
# Top-level Result
# ---------------------------------------------------------------------------


@dataclass
class PortfolioAnalysisResult:
    """Aggregated result from analyze_portfolio."""

    performance: PerformanceProfile
    structure: PortfolioStructure | None
    regime_performance: dict[str, RegimePerformance] | None
    attribution: AttributionReport | None
    generated_at: datetime


def analyze_portfolio(
    daily_returns: np.ndarray | pd.Series,
    weights: dict[str, float] | None = None,
    symbol_returns: dict[str, pd.Series] | None = None,
    regime_labels: pd.Series | None = None,
    symbol_metadata: dict[str, dict] | None = None,
) -> PortfolioAnalysisResult:
    """Top-level portfolio analysis function.

    Args:
        daily_returns: Portfolio-level daily returns.
        weights: Optional symbol weights for structure and attribution.
        symbol_returns: Optional per-symbol return series.
        regime_labels: Optional regime label series (same index as daily_returns).
        symbol_metadata: Optional symbol -> {sector, region} mapping.

    Returns:
        PortfolioAnalysisResult with all available analysis layers populated.
    """
    performance = compute_performance_profile(daily_returns)

    structure = None
    if weights is not None:
        structure = compute_portfolio_structure(weights, symbol_metadata)

    regime_performance = None
    if regime_labels is not None:
        returns_series = (
            daily_returns
            if isinstance(daily_returns, pd.Series)
            else pd.Series(_to_array(daily_returns))
        )
        regime_performance = analyze_regime_performance(returns_series, regime_labels)

    attribution = None
    if weights is not None and symbol_returns is not None:
        period_returns = {sym: float(s.mean()) for sym, s in symbol_returns.items()}
        attribution = compute_attribution(weights, period_returns, symbol_metadata)

    return PortfolioAnalysisResult(
        performance=performance,
        structure=structure,
        regime_performance=regime_performance,
        attribution=attribution,
        generated_at=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Human-Readable Report
# ---------------------------------------------------------------------------


def format_portfolio_report(result: PortfolioAnalysisResult) -> str:
    """Format a PortfolioAnalysisResult as a human-readable text report."""
    p = result.performance
    lines: list[str] = [
        "=" * 60,
        "Portfolio Analysis Report",
        f"Generated: {result.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
        "=" * 60,
        "",
        "--- Performance ---",
        f"  Total Return    : {p.total_return:.2%}",
        f"  CAGR            : {p.cagr:.2%}",
        f"  Annualized Vol  : {p.annualized_vol:.2%}",
        f"  Sharpe          : {p.sharpe:.3f}",
        f"  Sortino         : {p.sortino:.3f}",
        f"  Calmar          : {p.calmar:.3f}",
        f"  Max Drawdown    : {p.max_drawdown:.2%}",
        f"  DD Duration     : {p.max_drawdown_duration_days} days",
        f"  Win Rate        : {p.win_rate:.2%}",
        f"  Profit Factor   : {p.profit_factor:.3f}",
        f"  Expectancy      : {p.expectancy:.4f}",
        f"  Trading Days    : {p.trading_days}",
        "",
    ]

    if result.structure is not None:
        s = result.structure
        lines += [
            "--- Structure ---",
            f"  Positions       : {s.n_positions}",
            f"  Total Invested  : {s.total_invested:.2%}",
            f"  Cash            : {s.cash_pct:.2%}",
            f"  Top-5 Conc.     : {s.top_5_concentration:.2%}",
            f"  Herfindahl      : {s.herfindahl_index:.4f}",
            "",
        ]
        if s.sector_weights:
            lines.append("  Sector Weights:")
            for sec, w in sorted(s.sector_weights.items(), key=lambda x: -x[1]):
                lines.append(f"    {sec:<20}: {w:.2%}")
            lines.append("")

    if result.regime_performance is not None:
        lines.append("--- Regime Performance ---")
        for name, rp in result.regime_performance.items():
            lines.append(
                f"  {name:<15}: n={rp.n_days:>4}, sharpe={rp.sharpe:+.2f}, hit={rp.hit_rate:.0%}"
            )
        lines.append("")

    if result.attribution is not None:
        a = result.attribution
        lines += [
            "--- Attribution ---",
            f"  Total Return    : {a.total_return:.4f}",
            "  Top Contributors:",
        ]
        for sym, contrib in a.top_contributors:
            lines.append(f"    {sym:<10}: {contrib:+.4f}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
