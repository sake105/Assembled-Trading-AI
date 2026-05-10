"""Crisis-Composite-Indicator — Multi-Signal Stress-Index.

Idee
----
Kein einzelner Indikator alleine fängt Krisen zuverlässig. Composite aus:
1. Equity-Volatility-Spike (VIX-Level oder rolling-vol-z)
2. Average-Pairwise-Correlation (APC) — alle korrelierter in Stress
3. Yield-Curve-Slope (Inversion = Recession-Signal)
4. Credit-Spread (HYG/LQD-Ratio)
5. Dollar-Index-Rally (DXY-z)
6. Crypto-Risk-Off-Score (BTC drop + Stablecoin growth)
7. Realized-Drawdown-Magnitude

Composite ∈ [0, 1], 1 = full crisis.

Anwendung
---------
- De-Risking-Trigger
- Strategy-Switching (Trend → Defensive bei score > 0.7)
- Position-Sizing-Reduction-Multiplier
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore_to_unit(z: pd.Series, k: float = 2.0) -> pd.Series:
    """Convert z-score zu [0, 1] via sigmoid mit Skalierung k."""
    return 1 / (1 + np.exp(-z / k))


def composite_crisis_index(
    market_returns: pd.Series,
    correlation_apc: pd.Series | None = None,
    yield_slope: pd.Series | None = None,
    credit_spread: pd.Series | None = None,
    dollar_index: pd.Series | None = None,
    crypto_risk_off: pd.Series | None = None,
    vol_window: int = 21,
) -> pd.DataFrame:
    """Composite Crisis-Index aus mehreren Streams.

    Args:
        market_returns: Series of broad-market daily returns (e.g. SPY).
        correlation_apc: Series of avg-pairwise-correlation, optional.
        yield_slope: 10Y-3M slope, optional.
        credit_spread: HYG/LQD ratio (or BAA-AAA spread), optional.
        dollar_index: DXY series, optional.
        crypto_risk_off: pre-computed crypto risk-off score, optional.
        vol_window: window for vol-spike z-score.

    Returns:
        DataFrame [date, component_z_scores..., crisis_score].
        crisis_score ∈ [0, 1].
    """
    components = {}
    base_index = pd.Series(market_returns).dropna().index

    # 1. Vol-Spike
    r = pd.Series(market_returns).dropna()
    vol = r.rolling(vol_window).std() * np.sqrt(252)
    vol_z = (vol - vol.rolling(252, min_periods=126).mean()) / vol.rolling(
        252, min_periods=126
    ).std()
    components["vol_z"] = vol_z

    # 2. APC
    if correlation_apc is not None:
        apc = pd.Series(correlation_apc).reindex(base_index)
        apc_z = (apc - apc.rolling(252, min_periods=126).mean()) / apc.rolling(
            252, min_periods=126
        ).std()
        components["apc_z"] = apc_z

    # 3. Yield-Curve (inversion = high crisis signal => negate slope)
    if yield_slope is not None:
        ys = pd.Series(yield_slope).reindex(base_index, method="ffill")
        ys_z = (
            -(ys - ys.rolling(252, min_periods=126).mean())
            / ys.rolling(252, min_periods=126).std()
        )
        components["yc_z"] = ys_z

    # 4. Credit-Spread (low HYG/LQD = stress => negate)
    if credit_spread is not None:
        cs = pd.Series(credit_spread).reindex(base_index, method="ffill")
        # high credit spread => high crisis
        cs_z = (cs - cs.rolling(252, min_periods=126).mean()) / cs.rolling(
            252, min_periods=126
        ).std()
        components["credit_z"] = cs_z

    # 5. Dollar-Index Rally (USD-Rally = risk-off)
    if dollar_index is not None:
        dx = pd.Series(dollar_index).reindex(base_index, method="ffill")
        dx_ret = dx.pct_change()
        dx_z = dx_ret.rolling(21).sum()  # 1-month rally
        dx_z = (dx_z - dx_z.rolling(252).mean()) / dx_z.rolling(252).std()
        components["dxy_z"] = dx_z

    # 6. Crypto Risk-Off (already a score)
    if crypto_risk_off is not None:
        co = pd.Series(crypto_risk_off).reindex(base_index, method="ffill")
        components["crypto_z"] = co

    # 7. Realized-Drawdown
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1  # negative
    dd_score = -dd  # 0 to ~max-dd
    dd_z = (dd_score - dd_score.rolling(252).mean()) / dd_score.rolling(252).std()
    components["dd_z"] = dd_z

    # Combine via mean of available components, transformed to [0, 1]
    df = pd.DataFrame(components)
    if df.empty:
        return pd.DataFrame()

    # Map each z to [0, 1] via sigmoid (k=2 stretch)
    score_components = pd.DataFrame(
        {c: _zscore_to_unit(df[c], k=2.0) for c in df.columns}
    )
    df["crisis_score"] = score_components.mean(axis=1)
    return df.dropna(subset=["crisis_score"])


def crisis_state(
    score: pd.Series, threshold_high: float = 0.7, threshold_low: float = 0.3
) -> pd.Series:
    """Map crisis-score zu State-Labels ``normal | warning | crisis``.

    Hysterese-Behandlung: in crisis bleiben bis score < threshold_low.
    """
    s = pd.Series(score).dropna()
    out = pd.Series("normal", index=s.index, dtype=object)
    state = "normal"
    for d, v in s.items():
        if state == "normal":
            if v > threshold_high:
                state = "crisis"
            elif v > (threshold_high + threshold_low) / 2:
                state = "warning"
        elif state == "warning":
            if v > threshold_high:
                state = "crisis"
            elif v < threshold_low:
                state = "normal"
        elif state == "crisis":
            if v < threshold_low:
                state = "normal"
            elif v < (threshold_high + threshold_low) / 2:
                state = "warning"
        out.loc[d] = state
    return out


def exposure_multiplier_from_crisis(
    state: pd.Series, multipliers: dict | None = None
) -> pd.Series:
    """Map crisis-state zu Exposure-Multiplier ∈ [0, 1]."""
    multipliers = multipliers or {"normal": 1.0, "warning": 0.5, "crisis": 0.0}
    return state.map(multipliers).fillna(1.0)


__all__ = [
    "composite_crisis_index",
    "crisis_state",
    "exposure_multiplier_from_crisis",
]
