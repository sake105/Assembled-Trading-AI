"""Time-Series-Momentum + Donchian-Breakout Trend-Following Templates.

Klassiker
---------
- **Moskowitz/Ooi/Pedersen (2012)**: Time-Series Momentum — sign of past 12M return.
- **Donchian-Channel**: long if close > 20-day-high, short if < 20-day-low.
- **Dual-MA-Crossover**: long if fast-MA > slow-MA.

Reference
---------
- Moskowitz, T., Ooi, Y. & Pedersen, L. (2012). Time series momentum.
  *J. Financial Economics* 104.
- Hurst, B., Ooi, Y. & Pedersen, L. (2017). A Century of Evidence on
  Trend-Following Investing. *J. Portfolio Management* 44.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_series_momentum_signal(
    prices: pd.Series, lookback: int = 252, skip: int = 0
) -> pd.Series:
    """TSM: sign of (lookback-skip)-month return.

    Args:
        prices: Series.
        lookback: 252 = 12 months.
        skip: skip-period (Carhart-style skip 1 month).

    Returns:
        Series of {-1, 0, +1}.
    """
    if skip > 0:
        ret = prices.shift(skip) / prices.shift(lookback) - 1
    else:
        ret = prices / prices.shift(lookback) - 1
    return np.sign(ret).fillna(0).shift(1).fillna(0)  # PIT-shift


def donchian_breakout(prices: pd.Series, lookback: int = 20) -> pd.Series:
    """Donchian-Channel: long if close > high(lookback), short if < low(lookback)."""
    hi = prices.rolling(lookback, min_periods=lookback // 2).max()
    lo = prices.rolling(lookback, min_periods=lookback // 2).min()
    sig = pd.Series(0, index=prices.index, dtype=float)
    sig[prices >= hi.shift(1)] = 1.0
    sig[prices <= lo.shift(1)] = -1.0
    # ffill positions while no new signal
    sig = sig.replace(0, np.nan).ffill().fillna(0)
    return sig.shift(1).fillna(0)  # PIT


def dual_ma_crossover(prices: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """Dual-MA-Crossover: 1 if MA_fast > MA_slow else 0."""
    ma_fast = prices.rolling(fast, min_periods=fast // 2).mean()
    ma_slow = prices.rolling(slow, min_periods=slow // 2).mean()
    sig = (ma_fast > ma_slow).astype(float)
    return sig.shift(1).fillna(0)


def cta_multi_asset_strategy(
    panel_prices: pd.DataFrame,
    method: str = "tsm",
    vol_target: float = 0.15,
    vol_lookback: int = 60,
) -> pd.DataFrame:
    """Multi-Asset CTA mit Vol-Targeting je Asset.

    Args:
        panel_prices: DataFrame (T, N) of prices per asset.
        method: 'tsm' | 'donchian' | 'dual_ma'.
        vol_target: target annualized vol per asset.
        vol_lookback: rolling vol window.

    Returns:
        DataFrame of position weights per asset.
    """
    signal_fn = {
        "tsm": time_series_momentum_signal,
        "donchian": donchian_breakout,
        "dual_ma": dual_ma_crossover,
    }
    if method not in signal_fn:
        raise ValueError(f"unknown method: {method}")
    fn = signal_fn[method]

    out = pd.DataFrame(0.0, index=panel_prices.index, columns=panel_prices.columns)
    for col in panel_prices.columns:
        prices = panel_prices[col].dropna()
        sig = fn(prices)
        # vol scale
        ret = prices.pct_change()
        vol = ret.rolling(vol_lookback, min_periods=vol_lookback // 2).std() * np.sqrt(
            252
        )
        scale = (vol_target / vol).clip(upper=3.0).fillna(0)  # cap leverage
        out[col] = (sig * scale).reindex(panel_prices.index, fill_value=0)
    # Equal weight across active assets
    n_active = (out.abs() > 0).sum(axis=1).clip(lower=1)
    return out.div(n_active, axis=0)


__all__ = [
    "time_series_momentum_signal",
    "donchian_breakout",
    "dual_ma_crossover",
    "cta_multi_asset_strategy",
]
