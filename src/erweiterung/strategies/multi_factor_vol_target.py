"""Multi-Faktor-Vol-Targeting — kombiniert Vol-skalierte Long-Only-Faktoren.

Idee
----
Vol-Targeting auf Pure-Mom-12/1-LO lieferte OOS-Sharpe 1.462. Die Hypothese
hier ist, dass eine **Multi-Faktor-Kombination** mit jeweils eigenem
Vol-Targeting noch besser sein sollte, weil:

1. Jeder Faktor hat seine eigene Vol-Charakteristik (Mom hoch-vol,
   LowVol niedrig-vol, ResMom mittel) → einzelnes Targeting korrekter.
2. Faktor-Korrelationen sind < 1 → Diversifikation reduziert Vol weiter.
3. Vol-Target-Portfolio-Mix hat höhere effektive Sharpe als Single-Factor
   (Markowitz-Argument).

Pipeline
--------
1. Lade Faktor-Equity-Returns.
2. Vol-Targete jeden Faktor einzeln auf gemeinsame Ziel-Vol.
3. Kombiniere via Equal-Weight, Inverse-Vol oder HRP.

Wichtig
-------
Faktor-Pre-Vol-Target hat oft sehr unterschiedliche Vol-Niveaus (Mom ≈ 18 %,
LowVol ≈ 8 %). Ohne Vol-Targeting würde Mom die Kombination dominieren.
Mit Vol-Targeting ist jeder Faktor auf das gleiche Risiko-Budget skaliert
→ saubere Diversifikation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from erweiterung.strategies.volatility_targeting import (
    VolTargetConfig,
    apply_vol_targeting,
)


@dataclass
class MultiFactorVolTargetConfig:
    target_vol_annual: float = 0.12
    vol_window: int = 60
    max_leverage: float = 2.0
    combiner: str = "equal_weight"  # 'equal_weight' | 'inverse_vol' | 'hrp'
    hrp_lookback: int = 252
    weights: dict[str, float] | None = None  # custom weights override (sums to 1)
    smoothing_window: int = 5


def _vol_target_one(
    returns: pd.Series, config: MultiFactorVolTargetConfig
) -> pd.Series:
    """Wendet Vol-Targeting an, ausgehend von MultiFactor-Config."""
    cfg = VolTargetConfig(
        target_vol_annual=config.target_vol_annual,
        vol_window=config.vol_window,
        max_leverage=config.max_leverage,
        smoothing_window=config.smoothing_window,
    )
    out = apply_vol_targeting(returns, cfg)
    return out["scaled_return"]


def _inverse_vol_weights(panel: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Trailing-Inverse-Vol-Weights. Returns DataFrame (Index x Factor)."""
    rv = panel.rolling(lookback, min_periods=10).std()
    inv = 1.0 / rv.replace(0, np.nan)
    weights = inv.div(inv.sum(axis=1), axis=0)
    return weights.fillna(1.0 / len(panel.columns))


def _hrp_weights(panel: pd.DataFrame) -> pd.Series:
    """Static HRP-Weights über das gesamte Panel."""
    from erweiterung.portfolio.hierarchical_risk_parity import hrp_weights

    return hrp_weights(panel.dropna())


def combine_factors(
    factor_returns: dict[str, pd.Series],
    config: MultiFactorVolTargetConfig | None = None,
) -> pd.DataFrame:
    """Kombiniere Vol-skalierte Faktor-Returns.

    Args:
        factor_returns: Map factor-name -> Tagesreturn-Series.
        config: MultiFactorVolTargetConfig.

    Returns:
        DataFrame mit Spalten je Faktor (scaled), 'combined', 'leverage_sum'.
    """
    cfg = config or MultiFactorVolTargetConfig()

    # Step 1: Vol-Targete jeden Faktor
    scaled = {}
    for name, ret in factor_returns.items():
        scaled[name] = _vol_target_one(ret, cfg)
    scaled_df = pd.DataFrame(scaled).dropna(how="all")

    # Step 2: Kombiniere
    if cfg.weights is not None:
        # Custom weights
        wkeys = list(cfg.weights.keys())
        ws = np.array([cfg.weights[k] for k in wkeys])
        ws = ws / ws.sum()
        combined = (scaled_df[wkeys] * ws[None, :]).sum(axis=1)
    elif cfg.combiner == "equal_weight":
        combined = scaled_df.mean(axis=1)
    elif cfg.combiner == "inverse_vol":
        weights = _inverse_vol_weights(scaled_df, lookback=cfg.vol_window)
        combined = (scaled_df * weights).sum(axis=1)
    elif cfg.combiner == "hrp":
        try:
            ws = _hrp_weights(scaled_df)
            combined = (scaled_df * ws).sum(axis=1)
        except Exception:
            combined = scaled_df.mean(axis=1)
    else:
        raise ValueError(f"Unknown combiner: {cfg.combiner}")

    scaled_df = scaled_df.copy()
    scaled_df["combined"] = combined
    return scaled_df


__all__ = [
    "MultiFactorVolTargetConfig",
    "combine_factors",
]
