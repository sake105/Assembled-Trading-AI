"""Mean-Reversion Signal Layer (Plan 1.8).

Generates reversion signals from:
- RSI extremes (RSI < 20 → long, RSI > 80 → short)
- Bollinger Band reversion (price below lower band + momentum reversal)
- Z-score reversion (5d return z-score > 2.5 → short, < -2.5 → long)

Only active in bull/sideways regimes — mean reversion is dangerous in crisis/bear.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI for a price series."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_mean_reversion_signals(
    prices_df: pd.DataFrame,
    close_col: str = "close",
    symbol_col: str = "symbol",
    regime: str = "bull",
    rsi_period: int = 14,
    rsi_long: float = 20.0,
    rsi_short: float = 80.0,
    bb_window: int = 20,
    bb_std: float = 2.0,
    zscore_window: int = 5,
    zscore_threshold: float = 2.5,
) -> pd.DataFrame:
    """Compute mean-reversion signals for a panel of symbols.

    Args:
        prices_df: Panel with close prices, symbol column.
        close_col: Close price column.
        symbol_col: Symbol column.
        regime: Current regime. Signals only active in bull/sideways.
        rsi_period: RSI calculation period.
        rsi_long: RSI threshold for long reversion signal.
        rsi_short: RSI threshold for short reversion signal.
        bb_window: Bollinger Band window.
        bb_std: Bollinger Band standard deviations.
        zscore_window: Return z-score lookback.
        zscore_threshold: Z-score threshold for signals.

    Returns:
        DataFrame with symbol, reversion_signal, reversion_type columns.
    """
    # Regime gate: only active in bull or sideways
    if regime.lower() not in ("bull", "sideways"):
        logger.debug("[MeanReversion] Inactive in regime '%s'", regime)
        return pd.DataFrame(columns=[symbol_col, "reversion_signal", "reversion_type"])

    results = []
    for sym, group in prices_df.groupby(symbol_col):
        close = group[close_col].sort_index()
        if len(close) < max(rsi_period, bb_window, zscore_window) + 5:
            continue

        # RSI signal
        # A NaN RSI (e.g. all-flat price series, where compute_rsi divides
        # gain/loss by zero loss and propagates NaN) used to be silently
        # imputed to 50.0 — a fabricated "neutral" that still enters the
        # composite as weighted zero and pollutes portfolio aggregates when
        # many symbols flatline. Abstain instead so the caller sees the
        # symbol contribute nothing rather than a fake mid-reading.
        rsi = compute_rsi(close, rsi_period)
        if rsi.empty or pd.isna(rsi.iloc[-1]):
            continue
        latest_rsi = float(rsi.iloc[-1])

        rsi_sig = 0.0
        if latest_rsi < rsi_long:
            rsi_sig = (rsi_long - latest_rsi) / rsi_long  # stronger as RSI drops
        elif latest_rsi > rsi_short:
            rsi_sig = -(latest_rsi - rsi_short) / (100 - rsi_short)

        # Bollinger Band signal
        sma = close.rolling(bb_window).mean()
        std = close.rolling(bb_window).std()
        lower = sma - bb_std * std
        upper = sma + bb_std * std

        bb_sig = 0.0
        if pd.notna(lower.iloc[-1]) and close.iloc[-1] < lower.iloc[-1]:
            # Below lower band — check for momentum reversal
            if len(close) >= 3 and close.iloc[-1] > close.iloc[-2]:
                bb_sig = 0.5  # reversal underway
        elif pd.notna(upper.iloc[-1]) and close.iloc[-1] > upper.iloc[-1]:
            if len(close) >= 3 and close.iloc[-1] < close.iloc[-2]:
                bb_sig = -0.5

        # Z-score signal
        returns = close.pct_change(zscore_window)
        ret_mean = returns.rolling(60, min_periods=20).mean()
        ret_std = returns.rolling(60, min_periods=20).std()
        zscore = (returns - ret_mean) / ret_std.replace(0, np.nan)

        z_sig = 0.0
        if pd.notna(zscore.iloc[-1]):
            z = float(zscore.iloc[-1])
            if z < -zscore_threshold:
                z_sig = min(1.0, abs(z) / 4.0)
            elif z > zscore_threshold:
                z_sig = -min(1.0, abs(z) / 4.0)

        # Combine
        combined = (rsi_sig * 0.4 + bb_sig * 0.3 + z_sig * 0.3)
        if abs(combined) > 0.05:
            sig_type = "long_reversion" if combined > 0 else "short_reversion"
            results.append({
                symbol_col: sym,
                "reversion_signal": round(combined, 4),
                "reversion_type": sig_type,
                "rsi_component": round(rsi_sig, 4),
                "bb_component": round(bb_sig, 4),
                "zscore_component": round(z_sig, 4),
            })

    if not results:
        return pd.DataFrame(columns=[symbol_col, "reversion_signal", "reversion_type"])

    return pd.DataFrame(results)


__all__ = ["compute_mean_reversion_signals", "compute_rsi"]
