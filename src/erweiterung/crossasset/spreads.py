"""Cross-Asset Spread-Signale.

Spreads
-------
1. **Gold-Silver-Ratio** (GSR = GLD/SLV): historisch ≈ 60-80 in stable times,
   spike auf 100+ in crisis. Mean-Reverting.
2. **Dollar-Equity-Spread** (DXY vs SP500): inverser Korrelations-Indikator.
3. **Oil-Energy-Spread** (USO vs XLE): Marktstruktur-Indikator.
4. **TED-Spread Proxy** (3M-T-Bill vs 3M-LIBOR/SOFR): Stress-Indikator.
5. **HYG-LQD-Ratio**: High-Yield- vs IG-Credit-Spread.
6. **VIX/VVIX Term-Slope**: Vol-of-Vol-Indikator.

Anwendung
---------
- Risk-On/Off-Score
- Macro-Beta-Faktoren
- Crisis-Detection (Spike-Erkennung)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def gold_silver_ratio(gld: pd.Series, slv: pd.Series) -> pd.Series:
    """GLD / SLV ratio."""
    return (gld / slv).dropna()


def dollar_equity_spread(
    dxy: pd.Series, spy: pd.Series, lookback: int = 60
) -> pd.Series:
    """Rolling-Z-Score des log-Spreads ln(DXY) − ln(SPY)."""
    log_dxy = np.log(dxy)
    log_spy = np.log(spy)
    spread = log_dxy - log_spy
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()
    return (spread - mean) / std


def hyg_lqd_ratio(hyg: pd.Series, lqd: pd.Series) -> pd.Series:
    """HYG / LQD — High-Yield vs Investment-Grade Credit-Spread-Proxy."""
    return (hyg / lqd).dropna()


def vix_term_structure(vix: pd.Series, vix3m: pd.Series) -> pd.Series:
    """Term-structure: VIX / VIX3M.  >1 = inverted = stress."""
    return (vix / vix3m).dropna()


def cross_asset_risk_off_score(
    spy: pd.Series,
    gld: pd.Series,
    tlt: pd.Series,
    vix: pd.Series,
    lookback: int = 60,
) -> pd.Series:
    """Composite Risk-Off-Score aus mehreren Cross-Asset-Streams.

    Bullish-Risk-Off-Komponenten:
    - Gold rally vs Equity sell-off
    - Bonds (TLT) up
    - VIX spike
    """
    rets = pd.DataFrame(
        {
            "spy": spy.pct_change(),
            "gld": gld.pct_change(),
            "tlt": tlt.pct_change(),
            "vix": vix.pct_change(),
        }
    ).dropna()

    def z(s: pd.Series) -> pd.Series:
        return (s - s.rolling(lookback).mean()) / s.rolling(lookback).std()

    score = (
        -z(rets["spy"]).fillna(0)
        + 0.5 * z(rets["gld"]).fillna(0)
        + 0.5 * z(rets["tlt"]).fillna(0)
        + 0.5 * z(rets["vix"]).fillna(0)
    )
    return score / 2.5  # normalize range


__all__ = [
    "gold_silver_ratio",
    "dollar_equity_spread",
    "hyg_lqd_ratio",
    "vix_term_structure",
    "cross_asset_risk_off_score",
]
