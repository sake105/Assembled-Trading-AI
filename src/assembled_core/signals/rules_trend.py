"""Trend-following signal rules module.

This module provides trend-following signal generation based on technical indicators.
It extends the basic EMA crossover functionality from pipeline.signals.

Supports an optional news/intel overlay via ``intel_overlay`` parameter:
pass an IntelOverlay from intel_signal_adapter.adapt_intel_signal to blend
news-derived scores with the technical signal.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from src.assembled_core.features.ta_features import add_moving_averages

logger = logging.getLogger(__name__)


def generate_trend_signals(
    df: pd.DataFrame,
    ma_fast: int = 20,
    ma_slow: int = 50,
    volume_threshold: float | None = None,
    min_volume_multiplier: float = 1.0,
    require_weekly_alignment: bool = False,
    intel_overlay: "object | None" = None,
    news_alpha: float = 0.20,
) -> pd.DataFrame:
    """Generate trend-following signals based on moving average crossover.

    Signal logic:
    - LONG: ma_fast > ma_slow AND (volume > threshold OR no volume filter)
    - FLAT: otherwise

    Args:
        df: DataFrame with columns: timestamp, symbol, close, and optionally volume
            Should have moving averages already computed (ma_fast, ma_slow)
            If not present, they will be computed automatically
        ma_fast: Fast moving average window (default: 20)
        ma_slow: Slow moving average window (default: 50)
        volume_threshold: Optional volume threshold. If None, uses min_volume_multiplier * mean(volume)
        min_volume_multiplier: Multiplier for mean volume to compute threshold (default: 1.0)
        intel_overlay: Optional IntelOverlay from intel_signal_adapter.adapt_intel_signal.
            If provided and actionable, news scores are blended with trend scores.
        news_alpha: Blend weight for news scores (default 0.20 = 20% news, 80% trend).

    Returns:
        DataFrame with columns: timestamp, symbol, direction, score
        direction: "LONG" or "FLAT"
        score: Signal strength (0.0 to 1.0), based on MA spread and volume
        Sorted by symbol, then timestamp

    Raises:
        KeyError: If required columns are missing
    """
    df = df.copy()

    # Ensure required columns
    required = ["timestamp", "symbol", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}. Available: {list(df.columns)}"
        )

    # Add moving averages if not present
    ma_fast_col = f"ma_{ma_fast}"
    ma_slow_col = f"ma_{ma_slow}"

    if ma_fast_col not in df.columns or ma_slow_col not in df.columns:
        df = add_moving_averages(df, windows=(ma_fast, ma_slow))

    # Sort by symbol and timestamp
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Compute volume threshold if needed
    if volume_threshold is None and "volume" in df.columns:
        # Use rolling mean volume per symbol (same window as slow MA) to avoid
        # distortion from unadjusted historical volume (e.g. pre-split data).
        # A full-history mean would be dominated by old high-volume data and
        # block all signals for recent dates.
        df["volume_threshold"] = (
            df.groupby("symbol")["volume"]
            .transform(lambda x: x.rolling(ma_slow, min_periods=1).mean())
            * min_volume_multiplier
        )
    elif volume_threshold is not None:
        df["volume_threshold"] = volume_threshold
    else:
        # No volume filter
        df["volume_threshold"] = 0.0

    # Generate signals
    # LONG: ma_fast > ma_slow AND (volume > threshold OR no volume column)
    has_volume = "volume" in df.columns
    if has_volume:
        long_condition = (df[ma_fast_col] > df[ma_slow_col]) & (
            df["volume"] >= df["volume_threshold"]
        )
    else:
        long_condition = df[ma_fast_col] > df[ma_slow_col]

    # F3 — weekly-alignment gate (opt-in). When enabled, a LONG signal is
    # only kept if the weekly EMA slope of the symbol's close series is also
    # positive on that timestamp. Prevents daily whipsaw in chop regimes.
    if require_weekly_alignment:
        from src.assembled_core.features.weekly_alignment import add_weekly_alignment

        alignment_frames = []
        for sym, grp in df.sort_values("timestamp").groupby("symbol", sort=False):
            g = grp.set_index("timestamp").copy()
            g["daily_trend"] = np.where(g[ma_fast_col] > g[ma_slow_col], 1.0, -1.0)
            aligned = add_weekly_alignment(g, price_col="close")
            alignment_frames.append(
                aligned[["weekly_alignment_ok"]].assign(
                    symbol=sym, timestamp=aligned.index
                )
            )
        if not alignment_frames:
            long_condition = long_condition & False
        else:
            align_df = pd.concat(alignment_frames).reset_index(drop=True)
            df = df.merge(align_df, on=["timestamp", "symbol"], how="left")
            long_condition = long_condition & df["weekly_alignment_ok"].fillna(False)

    df["direction"] = np.where(long_condition, "LONG", "FLAT")

    # Compute signal score (0.0 to 1.0)
    # Score based on:
    # - MA spread: (ma_fast - ma_slow) / ma_slow (normalized)
    # - Volume strength: volume / threshold (if volume available)
    ma_spread = (df[ma_fast_col] - df[ma_slow_col]) / (df[ma_slow_col] + 1e-10)
    ma_score = pd.Series(ma_spread).clip(lower=0.0, upper=1.0)  # Normalize to [0, 1]

    if has_volume and (df["volume_threshold"] > 0).any():
        volume_score = (df["volume"] / (df["volume_threshold"] + 1e-10)).clip(
            lower=0.0, upper=1.0
        )
        df["score"] = (ma_score * 0.7 + volume_score * 0.3).fillna(0.0)
    else:
        df["score"] = ma_score.fillna(0.0)

    # Only set score for LONG signals, FLAT = 0.0
    df["score"] = np.where(df["direction"] == "LONG", df["score"], 0.0)

    # Select output columns
    result = df[["timestamp", "symbol", "direction", "score"]].copy()

    # Apply news intel overlay if provided
    if intel_overlay is not None:
        try:
            from src.assembled_core.signals.news_signal_bridge import blend_with_news
            result = blend_with_news(result, intel_overlay, news_alpha=news_alpha)
        except Exception as exc:
            logger.warning("[rules_trend] news blend failed, using pure trend: %s", exc)

    return result


def generate_trend_signals_from_prices(
    prices: pd.DataFrame,
    ma_fast: int = 20,
    ma_slow: int = 50,
    volume_threshold: float | None = None,
    min_volume_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Generate trend signals directly from price DataFrame.

    Convenience function that computes moving averages and generates signals in one step.

    Args:
        prices: DataFrame with columns: timestamp, symbol, close, and optionally volume
        ma_fast: Fast moving average window (default: 20)
        ma_slow: Slow moving average window (default: 50)
        volume_threshold: Optional volume threshold. If None, uses min_volume_multiplier * mean(volume)
        min_volume_multiplier: Multiplier for mean volume to compute threshold (default: 1.0)

    Returns:
        DataFrame with columns: timestamp, symbol, direction, score
        direction: "LONG" or "FLAT"
        score: Signal strength (0.0 to 1.0)
    """
    return generate_trend_signals(
        prices,
        ma_fast=ma_fast,
        ma_slow=ma_slow,
        volume_threshold=volume_threshold,
        min_volume_multiplier=min_volume_multiplier,
    )


# ---------------------------------------------------------------------------
# Sector Rotation Signal
# ---------------------------------------------------------------------------

#: Sector ETF → sector label mapping
SECTOR_ETF_MAP: dict[str, str] = {
    "XLK": "technology",
    "XLF": "financials",
    "XLE": "energy",
    "XLV": "healthcare",
    "XLI": "industrials",
    "XLY": "consumer_discretionary",
    "XLP": "consumer_staples",
    "XLU": "utilities",
    "XLB": "materials",
    "XLRE": "real_estate",
    "XLC": "communication_services",
}

#: Benchmark for relative strength calculation
_SECTOR_BENCHMARK = "SPY"


def compute_sector_rotation_signal(
    prices: pd.DataFrame,
    momentum_window: int = 20,
    long_window: int = 60,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    close_col: str = "close",
) -> pd.DataFrame:
    """Compute sector rotation signal from ETF relative strength.

    Ranks sectors by momentum relative to SPY. Returns signal scores for each
    sector ETF on each date.

    Args:
        prices: Daily OHLCV panel containing sector ETFs (XLK, XLF, etc.)
                and SPY as benchmark. Missing sectors are skipped.
        momentum_window: Rolling window for momentum calculation (default: 20).
        long_window: Longer-term window for trend confirmation (default: 60).
        symbol_col: Symbol column name.
        timestamp_col: Timestamp column name.
        close_col: Close price column name.

    Returns:
        DataFrame with columns: timestamp, symbol, sector, relative_strength,
        rs_rank (1=strongest), sector_signal (1.0=strong, 0.0=weak, -1.0=short),
        sector_score (normalized 0–1).
    """
    df = prices.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values([symbol_col, timestamp_col])

    available_sectors = [s for s in SECTOR_ETF_MAP if s in df[symbol_col].unique()]
    has_benchmark = _SECTOR_BENCHMARK in df[symbol_col].unique()

    if not available_sectors:
        return pd.DataFrame(columns=[timestamp_col, symbol_col, "sector", "sector_signal"])

    # Pivot to wide format
    pivot = df.pivot_table(index=timestamp_col, columns=symbol_col, values=close_col, aggfunc="last")
    pivot = pivot.sort_index()

    results = []
    timestamps = pivot.index

    for sym in available_sectors:
        if sym not in pivot.columns:
            continue

        sym_ret = pivot[sym].pct_change(momentum_window, fill_method=None)
        sym_ret_long = pivot[sym].pct_change(long_window, fill_method=None)

        if has_benchmark and _SECTOR_BENCHMARK in pivot.columns:
            bench_ret = pivot[_SECTOR_BENCHMARK].pct_change(momentum_window, fill_method=None)
            rs = sym_ret - bench_ret
        else:
            rs = sym_ret

        for ts in timestamps:
            results.append({
                timestamp_col: ts,
                symbol_col: sym,
                "sector": SECTOR_ETF_MAP[sym],
                "relative_strength": float(rs.get(ts, float("nan"))),
                "momentum_short": float(sym_ret.get(ts, float("nan"))),
                "momentum_long": float(sym_ret_long.get(ts, float("nan"))),
            })

    if not results:
        return pd.DataFrame(columns=[timestamp_col, symbol_col, "sector", "sector_signal"])

    out = pd.DataFrame(results)

    # Cross-sectional rank per date
    def _rank_group(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.copy()
        rs_vals = grp["relative_strength"].values
        valid = ~np.isnan(rs_vals)
        if valid.sum() < 2:
            grp["rs_rank"] = float("nan")
            grp["sector_signal"] = 0.0
            grp["sector_score"] = 0.5
            return grp
        ranks = np.argsort(np.argsort(-rs_vals[valid]))  # 0 = best
        full_ranks = np.full(len(rs_vals), float("nan"))
        full_ranks[valid] = ranks
        grp["rs_rank"] = full_ranks
        n_valid = valid.sum()
        # Top third → signal=1, bottom third → signal=-1, middle → 0
        top_cutoff = n_valid // 3
        bot_cutoff = n_valid - n_valid // 3
        grp["sector_signal"] = np.where(
            full_ranks <= top_cutoff, 1.0,
            np.where(full_ranks >= bot_cutoff, -1.0, 0.0),
        )
        # Normalized score 0–1 (1=strongest)
        grp["sector_score"] = 1.0 - (full_ranks / (n_valid - 1)).clip(0, 1)
        return grp

    frames = [_rank_group(grp) for _, grp in out.groupby(timestamp_col, sort=False)]
    out = pd.concat(frames) if frames else out
    return out.sort_values([timestamp_col, symbol_col]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Multi-Timeframe Trend Confirmation Signal
# ---------------------------------------------------------------------------

try:
    from src.assembled_core.data.resample import (
        align_higher_tf_to_daily,
        resample_to_monthly,
        resample_to_weekly,
    )

    _HAS_RESAMPLE = True
except ImportError:  # pragma: no cover
    _HAS_RESAMPLE = False


def compute_multi_timeframe_signal(
    prices: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    close_col: str = "close",
    daily_fast: int = 50,
    daily_slow: int = 200,
    weekly_fast: int = 10,
    weekly_slow: int = 40,
    monthly_fast: int = 3,
    monthly_slow: int = 12,
) -> pd.DataFrame:
    """Compute multi-timeframe trend confirmation signal.

    Combines daily, weekly and monthly SMA crossover signals into a single
    consensus score.  The signal is PIT-safe: weekly/monthly resampling
    respects ``as_of`` and higher-timeframe signals are forward-filled onto
    the daily grid via ``merge_asof`` (backward direction only).

    Signal logic per symbol per day:
        - daily_trend:   1.0 if SMA(daily_fast) > SMA(daily_slow), else -1.0
        - weekly_trend:  1.0 if SMA(weekly_fast) > SMA(weekly_slow), else -1.0
        - monthly_trend: 1.0 if SMA(monthly_fast) > SMA(monthly_slow), else -1.0
        - mtf_signal:    1.0 (LONG) if all three bullish,
                        -1.0 (SHORT) if all three bearish,
                         0.0 (NEUTRAL) otherwise.
        - mtf_score:     Normalized consensus in [0, 1]  (0 = all bearish, 1 = all bullish).

    Args:
        prices: Daily panel with at least ``symbol``, ``timestamp`` and
            ``close`` columns.  ``open``, ``high``, ``low``, ``volume`` are
            used for resampling if present.
        as_of: PIT cutoff.  Rows after ``as_of`` are excluded before any
            computation.  Partial weeks/months are dropped by the resample
            functions.
        symbol_col: Name of the symbol column.
        timestamp_col: Name of the timestamp column.
        close_col: Name of the close-price column.
        daily_fast: Fast SMA window for the daily timeframe (default: 50).
        daily_slow: Slow SMA window for the daily timeframe (default: 200).
        weekly_fast: Fast SMA window for the weekly timeframe (default: 10).
        weekly_slow: Slow SMA window for the weekly timeframe (default: 40).
        monthly_fast: Fast SMA window for the monthly timeframe (default: 3).
        monthly_slow: Slow SMA window for the monthly timeframe (default: 12).

    Returns:
        DataFrame with columns: ``timestamp``, ``symbol``, ``daily_trend``,
        ``weekly_trend``, ``monthly_trend``, ``mtf_signal``, ``mtf_score``.

    Raises:
        RuntimeError: If the resample module is not available.
        KeyError: If required columns are missing from *prices*.
    """
    if not _HAS_RESAMPLE:
        raise RuntimeError(
            "resample module not available — cannot compute multi-timeframe signal"
        )

    # --- validate inputs ---------------------------------------------------
    required = [symbol_col, timestamp_col, close_col]
    missing = [c for c in required if c not in prices.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = prices.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    if as_of is not None:
        df = df[df[timestamp_col] <= as_of]
    df = df.sort_values([symbol_col, timestamp_col]).reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(
            columns=[
                timestamp_col, symbol_col, "daily_trend", "weekly_trend",
                "monthly_trend", "mtf_signal", "mtf_score",
            ]
        )

    # --- helper: SMA crossover trend per symbol ----------------------------
    def _sma_trend(
        group: pd.DataFrame,
        ts_col: str,
        price_col: str,
        fast: int,
        slow: int,
        col_name: str,
    ) -> pd.DataFrame:
        """Return group with an added *col_name* column (+1 / -1 / NaN)."""
        group = group.sort_values(ts_col).copy()
        sma_f = group[price_col].rolling(window=fast, min_periods=fast).mean()
        sma_s = group[price_col].rolling(window=slow, min_periods=slow).mean()
        group[col_name] = np.where(
            sma_f.isna() | sma_s.isna(),
            np.nan,
            np.where(sma_f > sma_s, 1.0, -1.0),
        )
        return group

    # --- daily trend -------------------------------------------------------
    daily = (
        df.groupby(symbol_col, group_keys=False)
        .apply(
            lambda g: _sma_trend(g, timestamp_col, close_col, daily_fast, daily_slow, "daily_trend"),
            include_groups=False,
        )
    )

    # --- weekly trend ------------------------------------------------------
    weekly = resample_to_weekly(df, symbol_col=symbol_col, timestamp_col=timestamp_col, as_of=as_of)
    if not weekly.empty and close_col in weekly.columns:
        weekly = (
            weekly.groupby(symbol_col, group_keys=False)
            .apply(
                lambda g: _sma_trend(g, timestamp_col, close_col, weekly_fast, weekly_slow, "weekly_trend"),
                include_groups=False,
            )
        )
        daily = align_higher_tf_to_daily(
            daily,
            weekly[[symbol_col, timestamp_col, "weekly_trend"]],
            symbol_col=symbol_col,
            timestamp_col=timestamp_col,
            feature_cols=["weekly_trend"],
        )
    else:
        daily["weekly_trend"] = np.nan

    # --- monthly trend -----------------------------------------------------
    monthly = resample_to_monthly(df, symbol_col=symbol_col, timestamp_col=timestamp_col, as_of=as_of)
    if not monthly.empty and close_col in monthly.columns:
        monthly = (
            monthly.groupby(symbol_col, group_keys=False)
            .apply(
                lambda g: _sma_trend(g, timestamp_col, close_col, monthly_fast, monthly_slow, "monthly_trend"),
                include_groups=False,
            )
        )
        daily = align_higher_tf_to_daily(
            daily,
            monthly[[symbol_col, timestamp_col, "monthly_trend"]],
            symbol_col=symbol_col,
            timestamp_col=timestamp_col,
            feature_cols=["monthly_trend"],
        )
    else:
        daily["monthly_trend"] = np.nan

    # --- consensus ---------------------------------------------------------
    dt = daily["daily_trend"]
    wt = daily["weekly_trend"]
    mt = daily["monthly_trend"]

    all_bullish = (dt == 1.0) & (wt == 1.0) & (mt == 1.0)
    all_bearish = (dt == -1.0) & (wt == -1.0) & (mt == -1.0)

    daily["mtf_signal"] = np.where(
        all_bullish, 1.0,
        np.where(all_bearish, -1.0, 0.0),
    )
    # Normalized score: map sum of trends (-3..+3) to (0..1).
    # Only defined when all three timeframes report — fillna(0.0) on a
    # missing weekly/monthly value would silently lift an early-series
    # daily-only observation from "no data" to "0.67 bullish-ish",
    # producing phantom multi-timeframe agreement.
    complete = dt.notna() & wt.notna() & mt.notna()
    trend_sum = dt + wt + mt
    daily["mtf_score"] = np.where(
        complete, ((trend_sum + 3.0) / 6.0).clip(0.0, 1.0), np.nan
    )

    # --- output columns ----------------------------------------------------
    out_cols = [
        timestamp_col, symbol_col, "daily_trend", "weekly_trend",
        "monthly_trend", "mtf_signal", "mtf_score",
    ]
    return daily[out_cols].sort_values([symbol_col, timestamp_col]).reset_index(drop=True)
