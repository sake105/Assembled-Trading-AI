"""Soft-Profit-Lock-Overlay (portiert aus Mainline risk/profit_lock).

Idee
----
Aus dem Mainline ``src/assembled_core/risk/profit_lock.py``:
Reduziere Exposure nach starken Gains, um Profite zu schützen.

Logik
-----
Wenn lookback-Return >= trigger_return:
  exposure_multiplier = multiplier_on_trigger (floored)
  cooldown_days bleibt diese Reduktion
Sonst: full exposure (1.0).

Default
-------
- lookback_days: 20
- trigger_return: 8 %
- multiplier_on_trigger: 0.8
- floor: 0.5
- cooldown_days: 10
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ProfitLockConfig:
    enabled: bool = True
    lookback_days: int = 20
    trigger_return: float = 0.08
    multiplier_on_trigger: float = 0.80
    floor: float = 0.50
    cooldown_days: int = 10


def compute_profit_lock_series(
    portfolio_returns: pd.Series, config: ProfitLockConfig | None = None
) -> pd.Series:
    """Berechne Profit-Lock-Multiplier-Series (Daily).

    Returns:
        Series mit Multiplier in [floor, 1.0] pro Tag.
    """
    cfg = config or ProfitLockConfig()
    if not cfg.enabled or portfolio_returns.empty:
        return pd.Series(1.0, index=portfolio_returns.index)

    eq = (1 + portfolio_returns.fillna(0)).cumprod()
    lookback_ret = eq.pct_change(cfg.lookback_days)
    # Trigger-Index
    trigger_idx = lookback_ret >= cfg.trigger_return
    multipliers = np.ones(len(eq))
    cooldown_remaining = 0
    for i in range(len(eq)):
        if cooldown_remaining > 0:
            multipliers[i] = max(cfg.multiplier_on_trigger, cfg.floor)
            cooldown_remaining -= 1
        elif trigger_idx.iloc[i] if i < len(trigger_idx) else False:
            multipliers[i] = max(cfg.multiplier_on_trigger, cfg.floor)
            cooldown_remaining = cfg.cooldown_days - 1
    return pd.Series(multipliers, index=portfolio_returns.index)


def apply_profit_lock(
    portfolio_returns: pd.Series, config: ProfitLockConfig | None = None
) -> pd.DataFrame:
    """Wende Profit-Lock auf Portfolio-Returns an.

    Returns:
        DataFrame [raw_return, multiplier, locked_return].
    """
    cfg = config or ProfitLockConfig()
    multiplier = compute_profit_lock_series(portfolio_returns, cfg)
    # t-1 lag: heutige Multiplier basiert auf Returns-Calc bis gestern
    multiplier_lag = multiplier.shift(1).fillna(1.0)
    locked = portfolio_returns * multiplier_lag
    return pd.DataFrame(
        {
            "raw_return": portfolio_returns,
            "multiplier": multiplier_lag,
            "locked_return": locked,
        }
    )


__all__ = [
    "ProfitLockConfig",
    "compute_profit_lock_series",
    "apply_profit_lock",
]
