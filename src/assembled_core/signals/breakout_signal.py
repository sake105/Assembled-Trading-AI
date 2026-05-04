"""Donchian-channel breakout signal.

Generates a long/short breakout signal based on N-day Donchian channels:
- Long signal (score > 0): close > rolling N-day high (price breaking out above range)
- Short signal (score < 0): close < rolling N-day low (price breaking down below range)
- Neutral (score = 0): price within the channel

The raw signal is further smoothed by a confirmation window and z-scored
cross-sectionally to make it comparable across symbols.

PIT-safe: only uses past OHLC data — no lookahead.
"""

from __future__ import annotations

import logging

import pandas as pd

_log = logging.getLogger(__name__)

_DEFAULT_CHANNEL_DAYS = 20
_DEFAULT_CONFIRM_DAYS = 3
_DEFAULT_ATR_DAYS = 14


def _compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int
) -> pd.Series:
    """Average True Range over n days."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=max(1, n // 2)).mean()


def compute_breakout_signal(
    prices: pd.DataFrame,
    *,
    channel_days: int = _DEFAULT_CHANNEL_DAYS,
    confirm_days: int = _DEFAULT_CONFIRM_DAYS,
    atr_days: int = _DEFAULT_ATR_DAYS,
    use_atr_filter: bool = True,
    min_atr_expansion: float = 1.0,
) -> pd.DataFrame:
    """Compute Donchian breakout signal for a single-symbol OHLCV DataFrame.

    Args:
        prices: DataFrame with columns [open, high, low, close, volume] indexed by date.
        channel_days: Lookback window for Donchian channel (default 20).
        confirm_days: Days close must stay above/below channel to confirm (default 3).
        atr_days: ATR window for volatility filter (default 14).
        use_atr_filter: If True, only fire when ATR is expanding (breakout with volatility).
        min_atr_expansion: Minimum ratio of current ATR to prior ATR to pass filter.

    Returns:
        DataFrame with columns: [breakout_score, breakout_direction, channel_high,
        channel_low, atr] indexed same as prices.
    """
    required = {"high", "low", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices missing columns: {missing}")

    hi = prices["high"]
    lo = prices["low"]
    close = prices["close"]

    # Donchian channel: shift by 1 to avoid lookahead
    channel_high = (
        hi.shift(1).rolling(channel_days, min_periods=channel_days // 2).max()
    )
    channel_low = lo.shift(1).rolling(channel_days, min_periods=channel_days // 2).min()

    # Raw breakout flags
    above_channel = (close > channel_high).astype(float)
    below_channel = (close < channel_low).astype(float)

    # Confirmation: require confirm_days consecutive closes above/below
    if confirm_days > 1:
        above_confirmed = (
            above_channel.rolling(confirm_days, min_periods=confirm_days).sum()
            >= confirm_days
        ).astype(float)
        below_confirmed = (
            below_channel.rolling(confirm_days, min_periods=confirm_days).sum()
            >= confirm_days
        ).astype(float)
    else:
        above_confirmed = above_channel
        below_confirmed = below_channel

    # ATR filter: only fire breakouts when volatility is expanding
    atr = _compute_atr(hi, lo, close, atr_days)
    atr_ratio = atr / atr.shift(atr_days).clip(lower=1e-8)
    atr_expanding = (
        (atr_ratio >= min_atr_expansion).astype(float) if use_atr_filter else 1.0
    )

    # Breakout score: +1 for long breakout, -1 for short breakout
    raw_score = (above_confirmed - below_confirmed) * atr_expanding

    # Smooth score over confirmation window to reduce noise
    breakout_score = raw_score.rolling(confirm_days, min_periods=1).mean()

    direction = pd.Series("neutral", index=prices.index)
    direction[breakout_score > 0] = "long"
    direction[breakout_score < 0] = "short"

    result = pd.DataFrame(
        {
            "breakout_score": breakout_score,
            "breakout_direction": direction,
            "channel_high": channel_high,
            "channel_low": channel_low,
            "atr": atr,
        },
        index=prices.index,
    )
    return result


def compute_breakout_signals_panel(
    panel: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
    channel_days: int = _DEFAULT_CHANNEL_DAYS,
    confirm_days: int = _DEFAULT_CONFIRM_DAYS,
    atr_days: int = _DEFAULT_ATR_DAYS,
    use_atr_filter: bool = True,
    min_atr_expansion: float = 1.0,
    cross_sectional_zscore: bool = True,
) -> pd.DataFrame:
    """Compute breakout signals for a full symbol panel.

    Args:
        panel: Long-format DataFrame with [symbol, date, high, low, close] columns.
        symbol_col: Name of the symbol column.
        date_col: Name of the date column.
        channel_days: Donchian channel lookback.
        confirm_days: Confirmation window.
        atr_days: ATR window.
        use_atr_filter: Enable ATR expansion filter.
        min_atr_expansion: Minimum ATR expansion ratio.
        cross_sectional_zscore: If True, z-score breakout_score cross-sectionally per date.

    Returns:
        Panel with added [breakout_score, breakout_direction] columns.
    """
    required = {symbol_col, date_col, "high", "low", "close"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {missing}")

    results: list[pd.DataFrame] = []
    symbols = panel[symbol_col].unique()
    n_ok = 0

    for sym in symbols:
        sym_df = panel[panel[symbol_col] == sym].copy()
        sym_df = sym_df.sort_values(date_col).set_index(date_col)

        try:
            sig = compute_breakout_signal(
                sym_df,
                channel_days=channel_days,
                confirm_days=confirm_days,
                atr_days=atr_days,
                use_atr_filter=use_atr_filter,
                min_atr_expansion=min_atr_expansion,
            )
        except Exception as exc:
            _log.warning("[Breakout] %s failed: %s", sym, exc)
            continue

        sym_df["breakout_score"] = sig["breakout_score"]
        sym_df["breakout_direction"] = sig["breakout_direction"]
        sym_df[symbol_col] = sym
        sym_df = sym_df.reset_index().rename(columns={"index": date_col})
        results.append(sym_df)
        n_ok += 1

    if not results:
        _log.warning("[Breakout] No symbols produced signals")
        return panel.copy()

    out = pd.concat(results, ignore_index=True)

    if cross_sectional_zscore:
        # Z-score breakout_score per date for cross-sectional comparability
        def _zscore(s: pd.Series) -> pd.Series:
            std = s.std()
            if std < 1e-10:
                return s - s.mean()
            return (s - s.mean()) / std

        out["breakout_score"] = out.groupby(date_col)["breakout_score"].transform(
            _zscore
        )

    _log.info("[Breakout] Computed signals for %d/%d symbols", n_ok, len(symbols))
    return out


__all__ = [
    "compute_breakout_signal",
    "compute_breakout_signals_panel",
]
