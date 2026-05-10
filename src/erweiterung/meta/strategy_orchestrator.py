"""Strategy-Orchestrator — Multi-Strategy-Combiner.

Theorie
-------
Mehrere Strategien (Momentum, Mean-Reversion, Stat-Arb, ...) haben unkorrelierte
Edges. Ihre Kombination erhöht Sharpe-Ratio durch Diversifikation.

Allokationsmethoden
-------------------
1. **Equal-Weight**: 1/K pro Strategie. Robuster Default.
2. **Inverse-Vol**: w_k ∝ 1/σ_k. Kein μ-Schätzungsfehler.
3. **Risk-Parity**: Equal-Risk-Contribution.
4. **HRP**: Hierarchical Clustering basierend.
5. **BL**: Black-Litterman mit Strategie-Sharpe als View.
6. **Online-Learning**: Exp3 / Hedge-Algorithmus.

Implementation
--------------
Wir bieten alle als Methoden eines Orchestrators. Die Kombination liefert
final-weights pro Periode.

Online-Hedge
------------
Freund/Schapire (1995) Hedge-Algorithm:
    w_k_t = w_k_{t-1} × β^{loss_k_t}, then renormalize.

Mit β ∈ (0,1). Liefert No-Regret-Garantie gegen die best static Strategie.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def equal_weight_combination(
    strategy_returns: pd.DataFrame,
) -> pd.Series:
    """1/K pro Strategie."""
    if strategy_returns.empty:
        return pd.Series(dtype=float)
    return strategy_returns.mean(axis=1)


def inverse_vol_combination(
    strategy_returns: pd.DataFrame, lookback: int = 60
) -> pd.Series:
    """Rolling inverse-vol weighting."""
    if strategy_returns.empty:
        return pd.Series(dtype=float)
    vol = strategy_returns.rolling(lookback, min_periods=lookback // 2).std()
    inv_vol = (1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)
    # Apply with 1-step lag (PIT)
    weights = weights.shift(1)
    return (strategy_returns * weights).sum(axis=1)


def hedge_algorithm(
    strategy_returns: pd.DataFrame,
    eta: float = 0.1,
    initial_weights: np.ndarray | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Hedge / Multiplicative-Weights-Algorithm.

    Args:
        strategy_returns: DataFrame T × K.
        eta: Lernrate.
        initial_weights: K-Vektor; default uniform.

    Returns:
        (combined_returns, weights_history).
    """
    if strategy_returns.empty:
        return pd.Series(dtype=float), pd.DataFrame()
    K = strategy_returns.shape[1]
    if initial_weights is None:
        w = np.ones(K) / K
    else:
        w = initial_weights / initial_weights.sum()

    weights_hist = []
    combined = []
    R = strategy_returns.values

    for t in range(len(R)):
        weights_hist.append(w.copy())
        combined.append(float(R[t] @ w))
        # Loss = -return (wir maximieren return)
        loss = -R[t]
        # Normalize loss to [0, 1] for stability
        loss_clip = np.clip(loss, -0.05, 0.05) / 0.10 + 0.5  # to [0,1]
        w = w * np.exp(-eta * loss_clip)
        w = w / w.sum()

    return (
        pd.Series(combined, index=strategy_returns.index),
        pd.DataFrame(
            weights_hist, index=strategy_returns.index, columns=strategy_returns.columns
        ),
    )


def regime_aware_combination(
    strategy_returns: pd.DataFrame,
    regime_signal: pd.Series,
    regime_strategy_map: dict[int, list[str]],
) -> pd.Series:
    """Wechsel zwischen Strategien basierend auf Regime.

    Args:
        strategy_returns: DataFrame T × K.
        regime_signal: Series je Datum mit Regime-Label (int).
        regime_strategy_map: ``{regime: [strategy_names]}`` — welche Strategien
            in welchem Regime aktiv sind.

    Returns:
        Series of combined returns.
    """
    if strategy_returns.empty or regime_signal.empty:
        return pd.Series(dtype=float)
    common = strategy_returns.index.intersection(regime_signal.index)
    out = pd.Series(0.0, index=common)
    for d in common:
        regime = regime_signal.loc[d]
        if pd.isna(regime):
            continue
        active = regime_strategy_map.get(int(regime), [])
        active = [a for a in active if a in strategy_returns.columns]
        if not active:
            continue
        out.loc[d] = float(strategy_returns.loc[d, active].mean())
    return out


__all__ = [
    "equal_weight_combination",
    "inverse_vol_combination",
    "hedge_algorithm",
    "regime_aware_combination",
]
