"""Volatility-Targeting — dynamische Position-Skalierung auf Ziel-Vol.

Idee
----
Walk-Forward hat gezeigt, dass binäres Regime-Switching out-of-sample nicht
robust ist (`docs/erweiterung/WALK_FORWARD_VALIDATION.md`). Vol-Targeting
ist eine methodisch sauberere Alternative:

Position-Größe(t) = Ziel-Vol / Realized-Vol(t-1)

Damit wird das **Risiko-Budget** konstant gehalten, statt das Modell zu
zwingen, zwischen "0 % Faktor" und "100 % Faktor" zu wählen.

Theorie
-------
Moreira & Muir (2017), "Volatility-Managed Portfolios":
- Vol-skalierte Faktor-Returns liefern höhere Sharpe-Ratios als Unskalierte
- Effekt ist robust über Faktoren (Market, Value, Momentum, ...)
- Funktioniert auch out-of-sample und über mehrere Asset-Klassen

Hier konkret: skalieren wir `momentum_12_1_LongOnly` so, dass die Annualised-
Vol auf `target_vol_annual` (Default 12 %) gezielt wird. Position-Cap = 200 %
um pathologische Leverage-Spikes zu vermeiden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VolTargetConfig:
    target_vol_annual: float = 0.12
    vol_window: int = 60  # trailing window für realized-vol
    max_leverage: float = 2.0
    min_leverage: float = 0.0
    smoothing_window: int = 5  # smooth leverage signal


def realized_vol(
    returns: pd.Series, window: int = 60, annualize: bool = True
) -> pd.Series:
    """Trailing realized-vol (annualized by default)."""
    vol = returns.rolling(window, min_periods=10).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def vol_target_leverage(
    returns: pd.Series, config: VolTargetConfig | None = None
) -> pd.Series:
    """Berechne die dynamische Leverage-Series für Vol-Targeting.

    Args:
        returns: Strategie-Tagesreturns.
        config: VolTargetConfig.

    Returns:
        Series mit Leverage in [min_leverage, max_leverage].
        Lag t-1 automatisch eingebaut.
    """
    cfg = config or VolTargetConfig()
    rv = realized_vol(returns, cfg.vol_window)
    leverage = (cfg.target_vol_annual / rv.replace(0, np.nan)).clip(
        cfg.min_leverage, cfg.max_leverage
    )
    if cfg.smoothing_window > 1:
        leverage = leverage.rolling(cfg.smoothing_window, min_periods=1).mean()
    # t-1 lag: heute kann ich nur leverage anwenden, der auf gestern bekannt war
    return leverage.shift(1)


def apply_vol_targeting(
    returns: pd.Series, config: VolTargetConfig | None = None
) -> pd.DataFrame:
    """Wende Vol-Targeting auf Strategie-Returns an.

    Args:
        returns: Strategie-Tagesreturns.
        config: VolTargetConfig.

    Returns:
        DataFrame [raw_return, leverage, scaled_return, realized_vol].
    """
    cfg = config or VolTargetConfig()
    rv = realized_vol(returns, cfg.vol_window)
    lev = vol_target_leverage(returns, cfg)
    scaled = returns * lev.fillna(0)
    return pd.DataFrame(
        {
            "raw_return": returns,
            "realized_vol": rv,
            "leverage": lev,
            "scaled_return": scaled,
        }
    )


__all__ = [
    "VolTargetConfig",
    "realized_vol",
    "vol_target_leverage",
    "apply_vol_targeting",
]
