"""Macro-Regime 4-Quadrant Classifier (portiert aus Mainline macro_regime_quadrant).

Idee
----
Bridgewater-/Dalio-inspired:

Quadranten (Growth × Inflation):
- growth_up_infl_up   → Commodities, Emerging, Value
- growth_up_infl_down → Growth, Large-Cap-Tech
- growth_down_infl_up → Gold, Defensive-Value (Stagflation)
- growth_down_infl_down → Treasuries, Cash

Daten
-----
- Growth-Proxy: ISM-PMI oder NFP-3m-Change oder GDP-Trend
  Hier (offline): use a proxy via realized macro: yield-curve-slope steepening = growth.
- Inflation-Proxy: CPI-YoY oder breakeven inflation (T10YIE)
  Hier: T10YIE oder CPI-Change.

Anwendung
---------
Quadrant-Label kann als kategorisches Feature für ML genutzt werden ODER
als Trigger für Asset-Klassen-Allocation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


QUADRANT_ALLOCATIONS = {
    "growth_up_infl_up": ["commodities", "emerging", "value"],
    "growth_up_infl_down": ["growth", "large_cap_tech"],
    "growth_down_infl_up": ["gold", "defensive_value"],
    "growth_down_infl_down": ["treasuries", "cash"],
}

ETF_PROXIES_PER_QUADRANT = {
    "growth_up_infl_up": ["DBC", "EEM", "SLV"],  # commodities + EM + silver
    "growth_up_infl_down": ["QQQ", "SPY"],  # growth + LC tech
    "growth_down_infl_up": ["GLD", "TLT"],  # gold + long bonds (defensive)
    "growth_down_infl_down": ["AGG", "TLT"],  # treasuries
}


@dataclass
class MacroQuadrantConfig:
    growth_window: int = 60  # rolling-z für growth
    inflation_window: int = 60
    smoothing_days: int = 5


def _zscore(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window, min_periods=max(window // 3, 10)).mean()
    std = s.rolling(window, min_periods=max(window // 3, 10)).std()
    return (s - mean) / std.replace(0, np.nan)


def classify_macro_quadrant(
    growth_proxy: pd.Series,
    inflation_proxy: pd.Series,
    config: MacroQuadrantConfig | None = None,
) -> pd.Series:
    """Klassifiziere jeden Tag in einen der 4 Quadranten.

    Args:
        growth_proxy: Series, z. B. yield-curve-slope oder GDP-Trend.
        inflation_proxy: Series, z. B. CPI-YoY oder T10YIE.

    Returns:
        Series mit Labels {growth_up_infl_up, growth_up_infl_down,
        growth_down_infl_up, growth_down_infl_down}.
    """
    cfg = config or MacroQuadrantConfig()
    g_z = _zscore(growth_proxy, cfg.growth_window)
    i_z = _zscore(inflation_proxy, cfg.inflation_window)
    aligned = pd.concat({"g": g_z, "i": i_z}, axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=str)
    growth_up = aligned["g"] > 0
    infl_up = aligned["i"] > 0
    quadrants = np.where(
        growth_up & infl_up,
        "growth_up_infl_up",
        np.where(
            growth_up & ~infl_up,
            "growth_up_infl_down",
            np.where(
                ~growth_up & infl_up,
                "growth_down_infl_up",
                "growth_down_infl_down",
            ),
        ),
    )
    out = pd.Series(quadrants, index=aligned.index, name="quadrant")
    # Smoothing — Single-Day-Flips unterdrücken (manual loop, kein rolling.apply für strings)
    if cfg.smoothing_days > 1:
        smoothed = out.copy()
        for i in range(cfg.smoothing_days, len(out)):
            window = out.iloc[i - cfg.smoothing_days + 1 : i + 1]
            modes = window.mode()
            if not modes.empty:
                smoothed.iloc[i] = modes.iloc[0]
        out = smoothed
    return out


def regime_returns_summary(
    portfolio_returns: pd.Series, regime_labels: pd.Series
) -> pd.DataFrame:
    """Aggregiere Returns pro Regime."""
    aligned = pd.concat({"r": portfolio_returns, "q": regime_labels}, axis=1).dropna()
    if aligned.empty:
        return pd.DataFrame()
    rows = []
    for quad, sub in aligned.groupby("q"):
        if sub.empty or sub["r"].std() == 0:
            continue
        ann = (1 + sub["r"]).prod() ** (252 / len(sub)) - 1
        vol = sub["r"].std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        rows.append(
            {
                "quadrant": quad,
                "n_days": int(len(sub)),
                "ann_return": float(ann),
                "ann_vol": float(vol),
                "sharpe": float(sharpe),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "MacroQuadrantConfig",
    "QUADRANT_ALLOCATIONS",
    "ETF_PROXIES_PER_QUADRANT",
    "classify_macro_quadrant",
    "regime_returns_summary",
]
