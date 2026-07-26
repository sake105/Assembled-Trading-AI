"""Backtest-Comparison-Framework für Multi-Strategy-Vergleich.

Verglichen wird N > 2 Backtests gegeneinander:
- Kennzahlen: Sharpe, IC, MaxDD, Turnover, Calmar, Sortino, HitRate
- Paarweise signifikanz via Diebold-Mariano + Bonferroni-Korrektur
- Ranking mit Tie-Breakern

Ergänzt `deflated_sharpe.py` (einzelner Strategy-Test) und
`scripts/analysis/compare_models.py` (2-Modell-Vergleich).

PIT-Invariante: nur realisierte historische Returns, keine Future-Daten.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Kennzahlen einer einzelnen Strategie."""

    name: str
    sharpe: float
    ic: float
    max_drawdown: float
    turnover: float
    calmar: float
    sortino: float
    hit_rate: float
    total_return: float
    n_periods: int


@dataclass
class PairwiseComparison:
    strategy_a: str
    strategy_b: str
    sharpe_diff: float
    dm_statistic: float
    dm_pvalue: float
    bonferroni_pvalue: float


@dataclass
class BacktestComparisonReport:
    """Output eines Multi-Strategy-Vergleichs."""

    strategies: list[StrategyMetrics] = field(default_factory=list)
    pairwise: list[PairwiseComparison] = field(default_factory=list)
    ranking: list[tuple[str, float]] = field(default_factory=list)
    """Sortiert nach Sharpe desc: [(name, sharpe), ...]"""

    def to_dict(self) -> dict:
        return {
            "strategies": [vars(s) for s in self.strategies],
            "pairwise": [vars(p) for p in self.pairwise],
            "ranking": self.ranking,
        }

    def summary_df(self) -> pd.DataFrame:
        """Tabellarische Zusammenfassung aller Strategien."""
        return pd.DataFrame([vars(s) for s in self.strategies]).set_index("name")


def _compute_metrics(
    name: str,
    returns: pd.Series,
    predictions: pd.Series | None = None,
    actuals: pd.Series | None = None,
    turnover: float = 0.0,
) -> StrategyMetrics:
    if returns.std() > 1e-9:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    equity = (1.0 + returns).cumprod()
    max_dd = float((equity / equity.cummax() - 1.0).min())

    # Calmar: annualized return / max_dd
    n_years = len(returns) / 252.0
    if n_years > 0 and abs(max_dd) > 1e-9:
        ann_ret = (equity.iloc[-1]) ** (1.0 / n_years) - 1.0
        calmar = float(ann_ret / abs(max_dd))
    else:
        calmar = 0.0

    # Sortino: only downside deviation
    downside = returns[returns < 0]
    sortino = (
        float(returns.mean() / downside.std() * np.sqrt(252))
        if len(downside) > 1 and downside.std() > 1e-9
        else 0.0
    )

    hit_rate = float((returns > 0).mean())

    ic = 0.0
    if predictions is not None and actuals is not None:
        common = predictions.index.intersection(actuals.index)
        if len(common) > 10:
            p, a = predictions.loc[common].values, actuals.loc[common].values
            if np.std(p) > 1e-9 and np.std(a) > 1e-9:
                corr = np.corrcoef(p, a)[0, 1]
                if not np.isnan(corr):
                    ic = float(corr)

    return StrategyMetrics(
        name=name,
        sharpe=round(sharpe, 3),
        ic=round(ic, 4),
        max_drawdown=round(max_dd, 4),
        turnover=round(turnover, 4),
        calmar=round(calmar, 3),
        sortino=round(sortino, 3),
        hit_rate=round(hit_rate, 4),
        total_return=round(float(equity.iloc[-1] - 1.0), 4),
        n_periods=len(returns),
    )


def _diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray) -> tuple[float, float]:
    """DM-Test — copy aus scripts/analysis/compare_models.py für Self-Containment."""
    d = errors_a**2 - errors_b**2
    n = len(d)
    if n < 10 or d.std() < 1e-12:
        return 0.0, 1.0
    se = np.sqrt(d.var(ddof=1) / n)
    stat = float(d.mean() / se) if se > 0 else 0.0
    try:
        from scipy.stats import norm

        p = 2.0 * (1.0 - norm.cdf(abs(stat)))
    except ImportError:
        p = float(2.0 * np.exp(-0.5 * stat**2) / np.sqrt(2 * np.pi))
        p = min(1.0, max(0.0, p))
    return stat, p


def compare_backtests(
    strategies: dict[str, pd.Series],
    predictions_by_strategy: dict[str, pd.Series] | None = None,
    actuals: pd.Series | None = None,
    turnovers: dict[str, float] | None = None,
) -> BacktestComparisonReport:
    """Vergleicht mehrere Strategien gegeneinander.

    Args:
        strategies: {name: pd.Series(returns)}
        predictions_by_strategy: optional {name: pd.Series(predictions)} für IC
        actuals: optional pd.Series(realized) für IC
        turnovers: optional {name: avg_turnover_value}

    Returns:
        BacktestComparisonReport
    """
    if len(strategies) < 2:
        raise ValueError("Mindestens 2 Strategien erforderlich")

    # Metriken pro Strategie
    metrics: list[StrategyMetrics] = []
    for name, rets in strategies.items():
        pred = (predictions_by_strategy or {}).get(name)
        to = (turnovers or {}).get(name, 0.0)
        metrics.append(_compute_metrics(name, rets, pred, actuals, turnover=to))

    # Pairwise DM + Bonferroni
    n_pairs = len(list(combinations(strategies.keys(), 2)))
    pairwise: list[PairwiseComparison] = []
    for a, b in combinations(strategies.keys(), 2):
        ret_a = strategies[a]
        ret_b = strategies[b]
        common = ret_a.index.intersection(ret_b.index)
        if len(common) < 10:
            continue
        errs_a = -ret_a.loc[
            common
        ].values  # negative return = error against "alpha = 0"
        errs_b = -ret_b.loc[common].values
        stat, p = _diebold_mariano(errs_a, errs_b)
        bonf_p = min(1.0, p * max(1, n_pairs))
        sharpe_diff = round(
            {m.name: m.sharpe for m in metrics}[b]
            - {m.name: m.sharpe for m in metrics}[a],
            3,
        )
        pairwise.append(
            PairwiseComparison(
                strategy_a=a,
                strategy_b=b,
                sharpe_diff=sharpe_diff,
                dm_statistic=round(stat, 3),
                dm_pvalue=round(p, 4),
                bonferroni_pvalue=round(bonf_p, 4),
            )
        )

    # Ranking by Sharpe desc, tie-break by Calmar
    ranked = sorted(metrics, key=lambda m: (-m.sharpe, -m.calmar))
    ranking = [(m.name, m.sharpe) for m in ranked]

    report = BacktestComparisonReport(
        strategies=metrics,
        pairwise=pairwise,
        ranking=ranking,
    )
    logger.info(
        "[BTCompare] %d Strategien verglichen, Top: %s (Sharpe=%.2f)",
        len(metrics),
        ranking[0][0],
        ranking[0][1],
    )
    return report


def rank_strategies(
    strategies: dict[str, pd.Series],
    primary_metric: str = "sharpe",
    tiebreaker: str = "calmar",
) -> list[tuple[str, float]]:
    """Einfaches Ranking ohne Statistik-Tests."""
    metrics = [_compute_metrics(name, rets) for name, rets in strategies.items()]
    ranked = sorted(
        metrics,
        key=lambda m: (-getattr(m, primary_metric), -getattr(m, tiebreaker)),
    )
    return [(m.name, getattr(m, primary_metric)) for m in ranked]


__all__ = [
    "StrategyMetrics",
    "PairwiseComparison",
    "BacktestComparisonReport",
    "compare_backtests",
    "rank_strategies",
]
