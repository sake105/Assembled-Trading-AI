"""Drawdown-Decomposition — zerlegt Drawdown-Perioden nach Faktor-Contribution.

Kern-Frage: In unserem schlimmsten Drawdown — welche Faktoren haben am meisten
gezogen? Ist das systematic Beta (Marktkrise) oder idiosynkratisch (schlechte
Strategy in diesem Regime)?

Workflow:
1. Finde Worst-Drawdown-Periode in portfolio_returns
2. Filter factor_returns auf dieselbe Periode
3. Attribution-Regression auf DD-Teilsample
4. Attribution-Breiten pro Faktor + α-Komponent

Verwendet `compute_attribution` aus Round 6.

PIT-Invariante: Decomposition auf realisierten historischen Returns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DrawdownPeriod:
    """Eine identifizierte Drawdown-Phase."""

    start_idx: int
    end_idx: int
    peak_value: float
    trough_value: float
    max_drawdown: float
    duration: int
    start_timestamp: str | None = None
    end_timestamp: str | None = None


@dataclass
class DrawdownDecompositionReport:
    drawdown: DrawdownPeriod
    factor_betas: dict[str, float] = field(default_factory=dict)
    factor_contributions: dict[str, float] = field(default_factory=dict)
    alpha_during_dd: float = 0.0
    alpha_t_stat: float = 0.0
    r_squared: float = 0.0
    idiosyncratic_return: float = 0.0
    """Nicht-durch-Faktoren-erklärter Return während DD (= residual mean)."""

    def summary(self) -> dict:
        return {
            "max_drawdown": round(self.drawdown.max_drawdown, 4),
            "duration": self.drawdown.duration,
            "alpha_during_dd": round(self.alpha_during_dd, 6),
            "alpha_t_stat": round(self.alpha_t_stat, 3),
            "r_squared": round(self.r_squared, 4),
            "idiosyncratic": round(self.idiosyncratic_return, 6),
            "factor_betas": {k: round(v, 4) for k, v in self.factor_betas.items()},
            "factor_contributions": {k: round(v, 6) for k, v in self.factor_contributions.items()},
        }


def find_worst_drawdown(returns: pd.Series) -> DrawdownPeriod:
    """Findet die längste/tiefste zusammenhängende Drawdown-Periode."""
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0

    # Worst point
    trough_idx = int(dd.values.argmin())
    # Peak before trough
    peak_idx = int(equity.iloc[:trough_idx + 1].values.argmax())

    max_dd = float(dd.iloc[trough_idx])
    duration = trough_idx - peak_idx

    return DrawdownPeriod(
        start_idx=peak_idx,
        end_idx=trough_idx,
        peak_value=float(equity.iloc[peak_idx]),
        trough_value=float(equity.iloc[trough_idx]),
        max_drawdown=max_dd,
        duration=duration,
        start_timestamp=str(returns.index[peak_idx]) if hasattr(returns.index, "date") else None,
        end_timestamp=str(returns.index[trough_idx]) if hasattr(returns.index, "date") else None,
    )


def decompose_drawdown(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> DrawdownDecompositionReport:
    """Zerlegt schlimmsten Drawdown via Attribution-Regression.

    Args:
        portfolio_returns: pd.Series mit Portfolio-Returns
        factor_returns: DataFrame mit Factor-Returns (selbe Index-Art)

    Returns:
        DrawdownDecompositionReport mit β/α pro Faktor während DD-Periode.
    """
    dd = find_worst_drawdown(portfolio_returns)

    # Slice auf DD-Periode
    port_slice = portfolio_returns.iloc[dd.start_idx:dd.end_idx + 1]
    factor_slice = factor_returns.loc[port_slice.index.intersection(factor_returns.index)]
    port_slice = port_slice.loc[factor_slice.index]

    if len(port_slice) < 10:
        logger.warning("[DDDecomp] Nur %d Perioden im DD — zu wenig für Regression", len(port_slice))
        return DrawdownDecompositionReport(drawdown=dd)

    try:
        from src.assembled_core.qa.performance_attribution import compute_attribution

        attr = compute_attribution(port_slice, factor_slice, min_obs=5)
        return DrawdownDecompositionReport(
            drawdown=dd,
            factor_betas=attr.factor_betas,
            factor_contributions=attr.factor_contributions,
            alpha_during_dd=attr.alpha,
            alpha_t_stat=attr.alpha_t_stat,
            r_squared=attr.r_squared,
            idiosyncratic_return=float(port_slice.mean() - sum(attr.factor_contributions.values())),
        )
    except Exception as exc:
        logger.warning("[DDDecomp] Attribution fehlgeschlagen: %s", exc)
        return DrawdownDecompositionReport(drawdown=dd)


def find_all_drawdowns(
    returns: pd.Series,
    min_depth: float = 0.05,
    min_duration: int = 5,
) -> list[DrawdownPeriod]:
    """Findet alle Drawdowns > min_depth mit min_duration Tagen."""
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0

    drawdowns: list[DrawdownPeriod] = []
    dd_arr = dd.values
    equity_arr = equity.values
    in_dd = False
    current_peak_idx = 0
    running_max = equity_arr[0] if len(equity_arr) > 0 else 0.0
    running_max_idx = 0

    for i in range(len(dd_arr)):
        if equity_arr[i] > running_max:
            running_max = equity_arr[i]
            running_max_idx = i
        if dd_arr[i] <= -min_depth and not in_dd:
            in_dd = True
            current_peak_idx = running_max_idx
        elif dd_arr[i] >= -0.001 and in_dd:
            trough_range = dd_arr[current_peak_idx:i]
            trough_idx_rel = int(trough_range.argmin())
            trough_idx = current_peak_idx + trough_idx_rel
            duration = trough_idx - current_peak_idx
            if duration >= min_duration:
                drawdowns.append(DrawdownPeriod(
                    start_idx=current_peak_idx,
                    end_idx=trough_idx,
                    peak_value=float(equity_arr[current_peak_idx]),
                    trough_value=float(equity_arr[trough_idx]),
                    max_drawdown=float(dd_arr[trough_idx]),
                    duration=duration,
                ))
            in_dd = False

    return drawdowns


__all__ = [
    "DrawdownPeriod",
    "DrawdownDecompositionReport",
    "find_worst_drawdown",
    "decompose_drawdown",
    "find_all_drawdowns",
]
