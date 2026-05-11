"""FX Carry-Trade Signals (Lustig/Verdelhan 2007).

Theorie
-------
Carry-Trade: borrow low-interest currency, invest in high-interest. Historisch
positive Sharpe (~0.4-0.6) — aber mit periodischen Crashes ("carry-unwind").

Annualized Carry = i_foreign − i_domestic.

Signale
-------
1. **Carry-Score**: rank G10 currencies by interest-rate.
2. **HML-Carry**: long top-3, short bottom-3.
3. **Carry-with-Filter**: only when global volatility-regime allows.

Reference
---------
- Lustig, H. & Verdelhan, A. (2007). The Cross-Section of Foreign Currency Risk
  Premia and Consumption Growth Risk. *AER* 97.
- Brunnermeier, M., Nagel, S. & Pedersen, L. (2009). Carry Trades and Currency
  Crashes. *NBER Macro Annual*.
"""

from __future__ import annotations

import pandas as pd


def carry_ranking(
    interest_rates: pd.DataFrame, n_long: int = 3, n_short: int = 3
) -> pd.DataFrame:
    """Cross-section carry-trade portfolio.

    Args:
        interest_rates: DataFrame [date, currency, interest_rate].
        n_long, n_short: long/short legs.

    Returns:
        DataFrame [date, currency, position] mit positions ∈ {-1, 0, +1} /
        renormalisiert auf Equal-Weight.
    """
    df = interest_rates.copy()
    df["rank"] = df.groupby("date")["interest_rate"].rank(
        ascending=False, method="first"
    )
    df["position"] = 0.0
    df.loc[df["rank"] <= n_long, "position"] = 1.0 / n_long
    n_total = df.groupby("date")["rank"].transform("max")
    df.loc[df["rank"] > n_total - n_short, "position"] = -1.0 / n_short
    return df[["date", "currency", "position", "interest_rate"]]


def carry_returns(interest_rates: pd.DataFrame, fx_returns: pd.DataFrame) -> pd.Series:
    """Realized carry-portfolio returns.

    Args:
        interest_rates: from carry_ranking.
        fx_returns: DataFrame [date, currency, return].

    Returns:
        Series of daily portfolio returns.
    """
    positions = carry_ranking(interest_rates)
    df = positions.merge(fx_returns, on=["date", "currency"], how="left")
    df["pnl"] = df["position"] * df["return"]
    return df.groupby("date")["pnl"].sum()


def carry_crash_indicator(
    portfolio_returns: pd.Series, vol_window: int = 60, threshold: float = 2.5
) -> pd.Series:
    """Crash-Indicator for carry-trades.

    Brunnermeier et al. (2009): carry-crashes co-occur with VIX-spikes.
    Hier: rolling-vol-z > threshold = crash-mode = de-risk.
    """
    vol = portfolio_returns.rolling(vol_window, min_periods=vol_window // 2).std()
    z = (vol - vol.rolling(252, min_periods=126).mean()) / vol.rolling(
        252, min_periods=126
    ).std()
    return (z > threshold).astype(float)


def downside_volatility_carry(
    portfolio_returns: pd.Series, window: int = 60
) -> pd.Series:
    """Downside-Vola only — sensitiver für Carry-Crashes."""
    downside = portfolio_returns.clip(upper=0).abs()
    return downside.rolling(window, min_periods=window // 2).std()


def carry_score_with_volatility_filter(
    interest_rates: pd.DataFrame,
    fx_returns: pd.DataFrame,
    vix_proxy: pd.Series,
    vix_threshold_pct: float = 0.8,
) -> pd.DataFrame:
    """Vol-Filtered Carry: only trade when global vol below 80th-percentile.

    Args:
        interest_rates, fx_returns: standard panels.
        vix_proxy: e.g. SPX-VIX or DXY-Vol — proxy für globalen Risk-Appetit.

    Returns:
        Filtered positions.
    """
    positions = carry_ranking(interest_rates)
    vix_threshold = vix_proxy.rolling(252, min_periods=126).quantile(vix_threshold_pct)
    is_low_vol = (vix_proxy < vix_threshold.shift(1)).astype(float)
    df_dates = pd.DataFrame({"date": is_low_vol.index, "in_regime": is_low_vol.values})
    df = positions.merge(df_dates, on="date", how="left")
    df["position"] = df["position"] * df["in_regime"].fillna(0)
    return df[["date", "currency", "position"]]


__all__ = [
    "carry_ranking",
    "carry_returns",
    "carry_crash_indicator",
    "downside_volatility_carry",
    "carry_score_with_volatility_filter",
]
