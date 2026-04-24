"""Probability of Backtest Overfitting (PBO) — Bailey & Lopez de Prado.

Testet, ob eine Strategie-Performance wahrscheinlich durch Backtesting-
Overfitting entstanden ist. Ergänzt deflated_sharpe.py.

Algorithmus — Combinatorially Symmetric Cross-Validation (CSCV):
1. Matrix M aus N Strategien × T Perioden
2. Für jede 50/50-Aufteilung der Perioden in J/J':
   - Beste Strategie in J identifizieren
   - Ihren Rang in J' prüfen
3. PBO = Anteil der Splits, bei denen die J-beste Strategie
         in J' unterdurchschnittlich abschneidet

PBO > 0.5 → Strategie ist vermutlich overfittet
PBO < 0.1 → Strategie ist robust

PIT-Invariante: Operiert auf bereits realisierten Perioden-Returns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PBOResult:
    """Probability of Backtest Overfitting (PBO) Ergebnis."""

    pbo: float
    """Wahrscheinlichkeit des Overfittings in [0, 1]."""

    n_strategies: int
    n_periods: int
    n_splits: int
    """Anzahl ausgewerteter 50/50-Splits."""

    median_logit: float
    """Median der Logit-Transformation — zusätzlicher Robustheits-Indikator."""

    def interpret(self) -> str:
        if self.pbo < 0.1:
            return "ROBUST (PBO < 0.1)"
        if self.pbo < 0.3:
            return "AKZEPTABEL (PBO < 0.3)"
        if self.pbo < 0.5:
            return "VERDACHT (PBO < 0.5)"
        return "STARK OVERFITTET (PBO ≥ 0.5)"


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Einfache Rank-Funktion (1..n, Ties → average)."""
    try:
        from scipy.stats import rankdata
        return rankdata(x)
    except ImportError:
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(x) + 1)
        return ranks


def compute_pbo(
    strategy_returns: pd.DataFrame,
    performance_metric: callable | None = None,
    n_splits: int | None = None,
) -> PBOResult:
    """Berechnet Probability of Backtest Overfitting via CSCV.

    Args:
        strategy_returns: DataFrame mit (Perioden × Strategien). Zeilen = Zeit, Spalten = Strategien.
        performance_metric: Callable(Series) → float. Default: Sharpe Ratio.
        n_splits: Max. Anzahl Splits (None = alle C(T, T/2), was schnell explodiert).

    Returns:
        PBOResult
    """
    if performance_metric is None:
        def performance_metric(returns: pd.Series) -> float:
            mu = returns.mean()
            sigma = returns.std()
            return float(mu / sigma) if sigma > 1e-9 else 0.0

    M = strategy_returns.values
    T, N = M.shape

    if N < 2:
        raise ValueError("Mindestens 2 Strategien erforderlich für PBO")
    if T < 4:
        raise ValueError("Mindestens 4 Perioden erforderlich für 50/50-Splits")

    half = T // 2
    all_period_idx = list(range(T))

    # Max splits: hard cap bei 256 um Explosion zu verhindern
    from math import comb
    total_combos = comb(T, half)
    use_splits = min(n_splits or total_combos, 256, total_combos)

    if total_combos <= 256:
        split_iter = combinations(all_period_idx, half)
    else:
        rng = np.random.default_rng(42)
        split_iter = iter([
            tuple(sorted(rng.choice(all_period_idx, size=half, replace=False).tolist()))
            for _ in range(use_splits)
        ])

    logit_values: list[float] = []
    count = 0

    for J_idx in split_iter:
        if count >= use_splits:
            break
        J = np.array(J_idx)
        J_mask = np.zeros(T, dtype=bool)
        J_mask[J] = True
        J_out = np.where(~J_mask)[0]

        # In-Sample: Metric über Strategien auf J
        in_perf = np.array([
            performance_metric(pd.Series(M[J, s])) for s in range(N)
        ])
        out_perf = np.array([
            performance_metric(pd.Series(M[J_out, s])) for s in range(N)
        ])

        # Beste In-Sample Strategie
        best_in = int(np.argmax(in_perf))
        # Rank der besten in Out-of-Sample (normalisiert 0..1)
        ranks_out = _rankdata(out_perf)
        rank_best = (ranks_out[best_in] - 1) / (N - 1)  # 0 = worst, 1 = best

        # Logit-Transformation (robuster für Tail-Verhalten)
        eps = 1e-4
        rank_best = np.clip(rank_best, eps, 1 - eps)
        logit = float(np.log(rank_best / (1 - rank_best)))
        logit_values.append(logit)
        count += 1

    logit_arr = np.array(logit_values)
    pbo = float(np.mean(logit_arr < 0))
    median_logit = float(np.median(logit_arr))

    result = PBOResult(
        pbo=pbo,
        n_strategies=N,
        n_periods=T,
        n_splits=count,
        median_logit=median_logit,
    )
    logger.info(
        "[PBO] %d Strategien × %d Perioden, %d Splits → PBO=%.3f (%s)",
        N, T, count, pbo, result.interpret(),
    )
    return result


def performance_degradation(
    strategy_returns: pd.DataFrame,
    is_period: tuple[int, int],
    oos_period: tuple[int, int],
    performance_metric: callable | None = None,
) -> dict:
    """Misst Performance-Degradation: IS vs OOS (für "echte" Out-of-Sample-Tests).

    Args:
        strategy_returns: DataFrame (Perioden × Strategien)
        is_period: (start, end) Index für In-Sample
        oos_period: (start, end) Index für Out-of-Sample

    Returns:
        dict mit is_metric, oos_metric, degradation (oos/is ratio).
    """
    if performance_metric is None:
        def performance_metric(r: pd.Series) -> float:
            return float(r.mean() / r.std()) if r.std() > 1e-9 else 0.0

    is_slice = strategy_returns.iloc[is_period[0]:is_period[1]]
    oos_slice = strategy_returns.iloc[oos_period[0]:oos_period[1]]

    is_vals = is_slice.apply(performance_metric, axis=0)
    oos_vals = oos_slice.apply(performance_metric, axis=0)

    best_strat = int(is_vals.values.argmax())
    is_best = float(is_vals.iloc[best_strat])
    oos_best = float(oos_vals.iloc[best_strat])

    return {
        "best_strategy_index": best_strat,
        "is_metric": is_best,
        "oos_metric": oos_best,
        "degradation_ratio": oos_best / is_best if abs(is_best) > 1e-9 else 0.0,
    }


__all__ = [
    "PBOResult",
    "compute_pbo",
    "performance_degradation",
]
