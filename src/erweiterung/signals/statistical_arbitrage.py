"""Statistical Arbitrage — Cointegration-basiertes Pairs Trading.

Theorie
-------
Zwei Aktien sind **cointegriert**, wenn eine Linearkombination ihrer Preise
stationär (mean-reverting) ist:
    p_i_t − β · p_j_t = ε_t,   ε_t stationär

Wenn ε_t signifikant von seinem Mittelwert abweicht (z. B. ±2σ), eröffne man
eine Long-Short-Position und schließt bei Mean-Reversion.

Methodik
--------
1. **Engle-Granger-2-Step**: OLS p_i ~ p_j; teste Residuum auf Stationarität (ADF).
2. **Half-life of mean-reversion** = -ln(2) / ln(1+λ) wobei λ aus AR(1)
   geschätzt wird; gibt Hold-Periode-Erwartung.
3. **Hedge-Ratio**: β aus OLS.
4. **Trading-Regel**: Z-Score des Spreads, |Z| > entry_z -> Position, |Z| < exit_z -> close.

Robustheit
----------
- Rolling-Cointegration alle X Tage neu schätzen, da Beziehungen brechen.
- Multi-Pair-Test: nur Pairs mit |β| ∈ [0.3, 3.0] und half-life ∈ [3, 30] Tage.
- Industrie-/Sektor-Filter: nur cointegrieren innerhalb gleichen Sektors.

Code-Notiz
----------
ADF-Test via statsmodels (optional). Bei Nicht-Verfügbarkeit fallback auf
KPSS-Approximation via Variance-Ratio.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CointPair:
    sym1: str
    sym2: str
    beta: float
    intercept: float
    spread_mean: float
    spread_std: float
    adf_p_value: float
    half_life: float
    n_obs: int


def _adf_test(series: pd.Series) -> float:
    """ADF-Test p-value. Falls statsmodels nicht installiert: variance-ratio
    Approximation."""
    s = pd.Series(series).dropna()
    if len(s) < 30:
        return 1.0
    try:
        from statsmodels.tsa.stattools import adfuller  # type: ignore

        res = adfuller(s.values, regression="c", autolag="AIC")
        return float(res[1])
    except ImportError:
        # Fallback: variance ratio test (Lo-MacKinlay 1988)
        # VR(2) = Var(r_t + r_{t-1}) / (2*Var(r_t))
        # Stationarity ~> VR < 1
        d = s.diff().dropna()
        if d.std() == 0:
            return 1.0
        vr = ((d + d.shift(1)).dropna()).var() / (2 * d.var())
        # heuristic: VR < 0.7 ~> p ~ 0.05; VR > 0.95 ~> p ~ 0.5
        if vr < 0.7:
            return 0.04
        if vr < 0.85:
            return 0.10
        if vr < 1.0:
            return 0.30
        return 0.60


def _half_life(spread: pd.Series) -> float:
    """Half-life of mean-reversion via AR(1) fit."""
    s = pd.Series(spread).dropna()
    if len(s) < 30:
        return float("nan")
    s_lag = s.shift(1).dropna()
    delta = (s - s_lag).dropna()
    s_lag = s_lag.loc[delta.index]
    if len(delta) < 10:
        return float("nan")
    X = np.column_stack([np.ones(len(s_lag)), s_lag.values])
    beta, *_ = np.linalg.lstsq(X, delta.values, rcond=None)
    lam = beta[1]  # ΔS = α + λ * S_{t-1}
    if lam >= 0:
        return float("inf")
    hl = -np.log(2) / lam
    return float(hl)


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    sym_universe: Sequence[str] | None = None,
    sector_map: Optional[dict[str, str]] = None,
    p_threshold: float = 0.05,
    beta_range: tuple[float, float] = (0.3, 3.0),
    half_life_range: tuple[float, float] = (3.0, 30.0),
    min_obs: int = 252,
) -> list[CointPair]:
    """Finde alle cointegrierten Paare in einem Preis-Panel.

    Args:
        prices: DataFrame mit Index = date, Spalten = Symbole, Werte = log-prices.
            (Bei nominalen Preisen vorher ``np.log`` anwenden.)
        sym_universe: Beschränkung auf Untermenge.
        sector_map: Wenn gegeben, nur intra-sector pairs.
        p_threshold: ADF-p-value-Schwelle (Spread muss stationär sein).
        beta_range: zulässiger Hedge-Ratio-Bereich.
        half_life_range: zulässiger Mean-Reversion-Horizont (Tage).
        min_obs: Mindestbeobachtungen pro Paar.

    Returns:
        Liste ``CointPair`` (sortiert nach ADF-p-value asc).
    """
    cols = sym_universe if sym_universe else list(prices.columns)
    cols = [c for c in cols if c in prices.columns]
    pairs = list(itertools.combinations(cols, 2))
    if sector_map:
        pairs = [(a, b) for (a, b) in pairs if sector_map.get(a) == sector_map.get(b)]

    out: list[CointPair] = []
    for a, b in pairs:
        s = prices[[a, b]].dropna()
        if len(s) < min_obs:
            continue
        # OLS: a = α + β*b + ε
        X = np.column_stack([np.ones(len(s)), s[b].values])
        beta, *_ = np.linalg.lstsq(X, s[a].values, rcond=None)
        intercept, hedge = float(beta[0]), float(beta[1])
        if hedge < beta_range[0] or hedge > beta_range[1]:
            continue
        spread = s[a] - hedge * s[b]
        p = _adf_test(spread)
        if not np.isfinite(p) or p >= p_threshold:
            continue
        hl = _half_life(spread)
        if not np.isfinite(hl) or hl < half_life_range[0] or hl > half_life_range[1]:
            continue
        out.append(
            CointPair(
                sym1=a,
                sym2=b,
                beta=hedge,
                intercept=intercept,
                spread_mean=float(spread.mean()),
                spread_std=float(spread.std(ddof=0)),
                adf_p_value=float(p),
                half_life=float(hl),
                n_obs=int(len(s)),
            )
        )
    return sorted(out, key=lambda p: p.adf_p_value)


def pair_zscore_signal(
    prices: pd.DataFrame, pair: CointPair, lookback: int = 60
) -> pd.DataFrame:
    """Erzeuge Z-Score-Signal für ein cointegriertes Paar (rolling-stats).

    Returns:
        DataFrame [date, spread, z, signal] mit signal ∈ {-1, 0, +1}
        (positiv: long sym1 / short sym2 (β-skaliert)).
    """
    if pair.sym1 not in prices.columns or pair.sym2 not in prices.columns:
        return pd.DataFrame()
    p1 = prices[pair.sym1]
    p2 = prices[pair.sym2]
    spread = p1 - pair.beta * p2
    mean = spread.rolling(lookback, min_periods=lookback // 2).mean()
    std = spread.rolling(lookback, min_periods=lookback // 2).std()
    z = (spread - mean) / std

    sig = pd.Series(0, index=z.index)
    # entry rules
    sig[z > 2.0] = -1  # spread overshoot -> short spread = short sym1, long sym2
    sig[z < -2.0] = +1
    # exit rules: we'll let downstream handle |z| < 0.5 closing
    return pd.DataFrame(
        {
            "date": z.index,
            "spread": spread.values,
            "z": z.values,
            "signal": sig.values,
        }
    )


__all__ = ["CointPair", "find_cointegrated_pairs", "pair_zscore_signal"]
