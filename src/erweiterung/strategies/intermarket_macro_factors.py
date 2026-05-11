"""Intermarket-Cross-Asset-Macro-Faktoren (portiert aus Mainline intermarket_factors).

Idee
----
Aus dem Mainline ``src/assembled_core/features/intermarket_factors.py``:
Universal-Macro-Faktoren aus Cross-Asset-ETFs. Anwendung in der Erweiterung:
diese Faktoren als Regime-Detector oder Macro-Score nutzen.

Faktoren
--------
- bond_equity_ratio_20d: TLT/SPY 20d-Ratio (Risk-Off-Signal)
- dollar_trend_20d: UUP 20d-Mom (USD-Stärke; bei uns DBC als Proxy)
- credit_spread_change: HYG/AGG (Credit-Stress-Proxy)
- gold_equity_divergence: GLD - SPY 20d-Return-Difference
- yield_curve_slope: über FRED-Daten
- hy_ig_ratio: HYG vs AGG (Credit-Risk-Proxy)

Anwendung in der Erweiterung
----------------------------
Kann als Composite-Macro-Stress-Score genutzt werden ODER als Faktor in
einem Multi-Faktor-Ensemble.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bond_equity_ratio(
    tlt_close: pd.Series, spy_close: pd.Series, window: int = 20
) -> pd.Series:
    """TLT/SPY-Ratio mit Rolling-Mean (>1 = bonds outperform, risk-off)."""
    aligned = pd.concat({"tlt": tlt_close, "spy": spy_close}, axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    ratio = aligned["tlt"] / aligned["spy"].replace(0, np.nan)
    return ratio.rolling(window, min_periods=5).mean()


def dollar_trend(dollar_close: pd.Series, window: int = 20) -> pd.Series:
    """USD-Strength via Trend (z. B. UUP oder DXY-Proxy)."""
    return dollar_close.pct_change(window)


def credit_spread_proxy(hyg_close: pd.Series, agg_close: pd.Series) -> pd.Series:
    """HYG/AGG-Ratio als Credit-Stress-Proxy.

    HYG = High-Yield, AGG = Aggregate. HYG-Underperformance = Credit-Stress.
    """
    aligned = pd.concat({"h": hyg_close, "a": agg_close}, axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    ratio = aligned["h"] / aligned["a"].replace(0, np.nan)
    # Lower ratio = HY underperform = stress -> negate so high = stress
    return -ratio.diff(5)


def gold_equity_divergence(
    gld_close: pd.Series, spy_close: pd.Series, window: int = 20
) -> pd.Series:
    """GLD 20d-Return minus SPY 20d-Return (positive = risk-off)."""
    gld_ret = gld_close.pct_change(window)
    spy_ret = spy_close.pct_change(window)
    return gld_ret - spy_ret


def build_intermarket_panel(
    panel_closes: pd.DataFrame,
) -> pd.DataFrame:
    """Berechne alle Intermarket-Faktoren aus Cross-Asset-Panel.

    Args:
        panel_closes: DataFrame mit Spalten SPY, TLT, GLD, HYG, AGG, DBC (oder USD-Proxy).

    Returns:
        DataFrame mit Intermarket-Faktoren als Spalten.
    """
    out = pd.DataFrame(index=panel_closes.index)
    if "TLT" in panel_closes.columns and "SPY" in panel_closes.columns:
        out["bond_equity_ratio_20d"] = bond_equity_ratio(
            panel_closes["TLT"], panel_closes["SPY"]
        )
    if "DBC" in panel_closes.columns:
        # DBC als USD-Inverse-Proxy (commodity = anti-USD)
        out["dbc_trend_20d"] = dollar_trend(panel_closes["DBC"])
    if "HYG" in panel_closes.columns and "AGG" in panel_closes.columns:
        out["credit_spread_proxy"] = credit_spread_proxy(
            panel_closes["HYG"], panel_closes["AGG"]
        )
    if "GLD" in panel_closes.columns and "SPY" in panel_closes.columns:
        out["gold_equity_divergence_20d"] = gold_equity_divergence(
            panel_closes["GLD"], panel_closes["SPY"]
        )
    return out


def macro_stress_composite_score(
    intermarket_panel: pd.DataFrame, percentile_window: int = 252
) -> pd.Series:
    """Composite-Macro-Stress-Score aus Intermarket-Faktoren.

    Höhere Werte = mehr Risk-Off-Druck.

    Args:
        intermarket_panel: Output von build_intermarket_panel().
        percentile_window: Window für rolling-percentile-rank.

    Returns:
        Series in [0, 1]: 0 = niedriger Stress, 1 = hoher Stress.
    """
    if intermarket_panel.empty:
        return pd.Series(dtype=float)

    components = []
    if "bond_equity_ratio_20d" in intermarket_panel.columns:
        # Hoher Ratio = bonds outperform = stress
        components.append(intermarket_panel["bond_equity_ratio_20d"])
    if "credit_spread_proxy" in intermarket_panel.columns:
        # Hoch = stress
        components.append(intermarket_panel["credit_spread_proxy"])
    if "gold_equity_divergence_20d" in intermarket_panel.columns:
        # Hoch = stress
        components.append(intermarket_panel["gold_equity_divergence_20d"])
    if not components:
        return pd.Series(dtype=float)

    # Rolling percentile-rank per component
    def _pct_rank(s):
        return s.rolling(percentile_window, min_periods=20).apply(
            lambda x: (x <= x.iloc[-1]).mean() if len(x) > 0 else np.nan,
            raw=False,
        )

    ranked = pd.concat([_pct_rank(c) for c in components], axis=1)
    return ranked.mean(axis=1)


__all__ = [
    "bond_equity_ratio",
    "dollar_trend",
    "credit_spread_proxy",
    "gold_equity_divergence",
    "build_intermarket_panel",
    "macro_stress_composite_score",
]
