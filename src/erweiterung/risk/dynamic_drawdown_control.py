"""Dynamic Drawdown Control — Online-Adjustierung der Exposure.

Theorie
-------
Static Drawdown-Stops (z. B. "Equity unter -10 % => 0 % Risk") sind brutal
und ignorieren statistische Information.  Eine **dynamische Vorschrift**
adjustiert Exposure kontinuierlich basierend auf:
- Current Drawdown vs Max-Allowed-Drawdown
- Volatility-Regime
- Tail-Risk-Estimation

Closed-Form-Strategy
--------------------
Cesari/Cremonini (2003), Grinold (2004), CPPI-Generalisierung:
    f_t = max(0, multiplier × (DD_max − DD_current))

Wir kombinieren mit Vol-Targeting:
    leverage_t = target_vol / realized_vol_t × DD_factor_t
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_running_drawdown(equity_curve: pd.Series) -> pd.Series:
    """Drawdown von rolling-Hoch."""
    if equity_curve.empty:
        return equity_curve
    cummax = equity_curve.cummax()
    return equity_curve / cummax - 1.0


def cppi_floor_factor(
    current_dd: float,
    max_dd: float = 0.20,
    cushion_multiplier: float = 5.0,
) -> float:
    """CPPI-style Allocation Factor: ``max(0, m × (max_dd − |dd|))``.

    Args:
        current_dd: aktueller DD (negativ, z. B. -0.07).
        max_dd: erlaubter Max-DD (z. B. 0.20 -> Floor bei -20%).
        cushion_multiplier: m im CPPI; höher = aggressiver.

    Returns:
        Allocation-Factor ∈ [0, 1+] (kann >1 für Hebel werden).
    """
    cushion = max_dd - abs(current_dd)
    if cushion <= 0:
        return 0.0
    return float(np.clip(cushion_multiplier * cushion, 0.0, 1.0 + cushion_multiplier))


def vol_targeted_leverage(
    realized_vol: float,
    target_vol: float = 0.15,
    max_leverage: float = 1.5,
) -> float:
    """Vol-Targeting: ``min(max_lev, target_vol / realized_vol)``.

    Args:
        realized_vol: trailing 30-day annualized vol.
        target_vol: target portfolio vol.
        max_leverage: Cap.

    Returns:
        Leverage-Faktor.
    """
    if realized_vol <= 0:
        return 1.0
    return float(min(max_leverage, target_vol / realized_vol))


def combined_dd_vol_control(
    equity_curve: pd.Series,
    returns: pd.Series,
    vol_window: int = 30,
    target_vol: float = 0.15,
    max_dd: float = 0.20,
    cushion_multiplier: float = 5.0,
    max_leverage: float = 1.5,
) -> pd.Series:
    """Kombiniere CPPI-DD-Floor mit Vol-Targeting.

    Returns:
        Series ``allocation_factor`` ∈ [0, max_leverage]; 1.0 = baseline.
    """
    dd = compute_running_drawdown(equity_curve)
    rv = returns.rolling(vol_window, min_periods=vol_window // 2).std() * np.sqrt(252)
    out = pd.Series(np.nan, index=equity_curve.index)
    for t, dd_t in dd.items():
        rv_t = rv.get(t, np.nan)
        if pd.isna(rv_t):
            continue
        cppi = cppi_floor_factor(
            dd_t, max_dd=max_dd, cushion_multiplier=cushion_multiplier
        )
        vol_lev = vol_targeted_leverage(
            rv_t, target_vol=target_vol, max_leverage=max_leverage
        )
        out.loc[t] = cppi * vol_lev
    return out.ffill().fillna(0.0)


__all__ = [
    "compute_running_drawdown",
    "cppi_floor_factor",
    "vol_targeted_leverage",
    "combined_dd_vol_control",
]
