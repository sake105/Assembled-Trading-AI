"""Regime-Conditional Metrics — Strategy-Performance je Regime-Bucket.

Idee
----
Gegeben (a) Strategy-Returns, (b) Regime-Indikator (vol-regime, trend-regime, etc.),
zerlege Performance in Buckets.

Beispiel
--------
- High-Vol-Days: was bringt die Strategy?
- Trending-Days vs Mean-Reverting-Days?
- Crisis-Days (crisis_score > 0.7)?
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def returns_by_regime(
    returns: pd.Series, regime: pd.Series, annual_factor: float = 252
) -> pd.DataFrame:
    """Performance-Metriken aggregiert nach Regime-Label.

    Args:
        returns: pd.Series of daily returns.
        regime: pd.Series of regime-labels (categorical or int).

    Returns:
        DataFrame [regime, n_obs, mean_ret, std_ret, sharpe, win_rate, frequency].
    """
    df = pd.concat([returns.rename("r"), regime.rename("reg")], axis=1).dropna()
    if df.empty:
        return pd.DataFrame()
    total = len(df)
    rows = []
    for reg, g in df.groupby("reg"):
        n = len(g)
        mean = float(g["r"].mean())
        std = float(g["r"].std(ddof=0))
        sharpe = mean / std * np.sqrt(annual_factor) if std > 0 else float("nan")
        win = float((g["r"] > 0).mean())
        rows.append(
            {
                "regime": reg,
                "n_obs": n,
                "frequency": n / total,
                "mean_return": mean,
                "std_return": std,
                "sharpe_ann": sharpe,
                "win_rate": win,
                "total_return_in_regime": float((1 + g["r"]).prod() - 1),
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe_ann", ascending=False)


def regime_transition_matrix(regime: pd.Series) -> pd.DataFrame:
    """P(reg_t+1 | reg_t) — Markov-Übergangs-Matrix.

    Returns:
        DataFrame N×N — rows: from, columns: to.
    """
    r = pd.Series(regime).dropna()
    if len(r) < 2:
        return pd.DataFrame()
    pairs = list(zip(r.iloc[:-1], r.iloc[1:]))
    counts = pd.crosstab(
        pd.Series([p[0] for p in pairs], name="from"),
        pd.Series([p[1] for p in pairs], name="to"),
    )
    row_sums = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(row_sums, axis=0).fillna(0)


def regime_expected_duration(regime: pd.Series) -> pd.Series:
    """Average duration per regime (in periods).

    Computed via maximum-likelihood: 1 / (1 − P(reg→reg)).
    """
    trans = regime_transition_matrix(regime)
    if trans.empty:
        return pd.Series(dtype=float)
    diag = pd.Series(np.diag(trans.values), index=trans.index)
    return 1.0 / (1.0 - diag.clip(upper=0.9999))


def conditional_sharpe_breakdown(returns: pd.Series, regime: pd.Series) -> dict:
    """Compact summary: regime mit höchster vs niedrigster Sharpe."""
    df = returns_by_regime(returns, regime)
    if df.empty:
        return {"error": "no data"}
    best = df.iloc[0]
    worst = df.iloc[-1]
    return {
        "best_regime": str(best["regime"]),
        "best_sharpe": float(best["sharpe_ann"]),
        "best_frequency": float(best["frequency"]),
        "worst_regime": str(worst["regime"]),
        "worst_sharpe": float(worst["sharpe_ann"]),
        "worst_frequency": float(worst["frequency"]),
        "sharpe_spread": float(best["sharpe_ann"] - worst["sharpe_ann"]),
    }


__all__ = [
    "returns_by_regime",
    "regime_transition_matrix",
    "regime_expected_duration",
    "conditional_sharpe_breakdown",
]
