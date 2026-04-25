"""9-Dimension Composite Score with regime-dependent weighting.

From 31_COMPOSITE_SCORE.md.

Dimensions:
  1. MTF Alignment (multi-timeframe trend direction)
  2. Classical TA (RSI, MACD, BB with regime-dependent params)
  3. Microstructure (Amihud, OFI proxy, volume-price correlation)
  4. Volume/Market Profile (POC, AVWAP deviation)
  5. Chart Pattern ML (placeholder until ML training complete)
  6. Volatility Surface (IV-Rank, Skew, VIX term, VRP)
  7. Breadth/Intermarket (McClellan, Risk-On/Off, HYG/TLT)
  8. Seasonality (overnight gap, turn-of-month, day-of-week)
  9. News (news_z_score / 3, from 30_NEWS_TA_FUSION)

Regime-dependent weights are central to the design: news weight goes from
5 % in calm to 20 % in crisis; breadth from 10 % to 20 %.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime weight matrix
# ---------------------------------------------------------------------------

COMPOSITE_WEIGHTS_BY_REGIME: dict[str, dict[str, float]] = {
    "calm": {
        "mtf": 0.15, "classical_ta": 0.20, "microstructure": 0.05,
        "volume_profile": 0.15, "chart_pattern": 0.10,
        "vol_surface": 0.05, "breadth": 0.10, "seasonality": 0.15,
        "news": 0.05,
    },
    "normal": {
        "mtf": 0.15, "classical_ta": 0.15, "microstructure": 0.05,
        "volume_profile": 0.15, "chart_pattern": 0.10,
        "vol_surface": 0.10, "breadth": 0.10, "seasonality": 0.10,
        "news": 0.10,
    },
    "elevated": {
        "mtf": 0.15, "classical_ta": 0.10, "microstructure": 0.05,
        "volume_profile": 0.10, "chart_pattern": 0.10,
        "vol_surface": 0.15, "breadth": 0.15, "seasonality": 0.05,
        "news": 0.15,
    },
    "crisis": {
        "mtf": 0.10, "classical_ta": 0.05, "microstructure": 0.05,
        "volume_profile": 0.05, "chart_pattern": 0.10,
        "vol_surface": 0.20, "breadth": 0.20, "seasonality": 0.05,
        "news": 0.20,
    },
}

TA_PARAMS_BY_REGIME: dict[str, dict[str, Any]] = {
    "bull_trend": {"rsi_ob": 75, "rsi_os": 35, "bb_std": 2.0, "use_bb_mr": False},
    "bear_trend": {"rsi_ob": 65, "rsi_os": 25, "bb_std": 2.0, "use_bb_mr": False},
    "ranging":    {"rsi_ob": 70, "rsi_os": 30, "bb_std": 1.5, "use_bb_mr": True},
    "high_vol":   {"rsi_ob": 80, "rsi_os": 20, "bb_std": 2.5, "use_bb_mr": True},
    "normal":     {"rsi_ob": 70, "rsi_os": 30, "bb_std": 2.0, "use_bb_mr": True},
    "calm":       {"rsi_ob": 70, "rsi_os": 30, "bb_std": 2.0, "use_bb_mr": True},
    "elevated":   {"rsi_ob": 72, "rsi_os": 28, "bb_std": 2.0, "use_bb_mr": True},
    "crisis":     {"rsi_ob": 80, "rsi_os": 20, "bb_std": 2.5, "use_bb_mr": False},
}


# ---------------------------------------------------------------------------
# Dimension 1: Multi-Timeframe Alignment
# ---------------------------------------------------------------------------


def mtf_alignment_score(
    close_daily: pd.Series,
    macd_hist_15m: float = 0.0,
    rsi_5m: float = 50.0,
    adx_daily: float = 20.0,
) -> float:
    """Top-down alignment: Daily trend × 15m MACD × 5m RSI.

    Returns score in [-1, +1].
    """
    if len(close_daily) < 200:
        return 0.0

    sma50 = close_daily.iloc[-50:].mean() if len(close_daily) >= 50 else close_daily.mean()
    sma200 = close_daily.mean()
    d_trend = np.sign(sma50 - sma200)
    d_strength = min(adx_daily / 25.0, 2.0)

    mtf_macd = np.sign(macd_hist_15m)
    entry_bias = (rsi_5m - 50.0) / 50.0

    aligned_signs = [d_trend, mtf_macd, np.sign(entry_bias)]
    n_aligned = sum(s > 0 for s in aligned_signs) - sum(s < 0 for s in aligned_signs)

    score = (n_aligned / 3.0) * d_strength
    return float(np.clip(score, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Dimension 2: Classical TA
# ---------------------------------------------------------------------------


def classical_ta_score(
    rsi: float,
    macd_hist: float,
    bb_percent: float,
    regime: str = "normal",
) -> float:
    """RSI + MACD + Bollinger Band score with regime-dependent parameters.

    Args:
        rsi: Current RSI value (0-100).
        macd_hist: MACD histogram value.
        bb_percent: BB %B position (0=lower band, 1=upper band).
        regime: Regime label.

    Returns:
        Score in [-1, +1].
    """
    params = TA_PARAMS_BY_REGIME.get(regime, TA_PARAMS_BY_REGIME["normal"])
    ob, os_ = params["rsi_ob"], params["rsi_os"]

    rsi_score = 0.0
    if rsi > ob:
        rsi_score = -1.0 * (rsi - ob) / max(100 - ob, 1)
    elif rsi < os_:
        rsi_score = (os_ - rsi) / max(os_, 1)

    macd_score = float(np.sign(macd_hist) * min(abs(macd_hist) * 10, 1.0))

    bb_score = 0.0
    if params["use_bb_mr"]:
        bb_score = -1.0 * (bb_percent - 0.5) * 2.0

    return float(np.clip((rsi_score + macd_score + bb_score) / 3.0, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Dimension 3: Microstructure
# ---------------------------------------------------------------------------


def microstructure_score(
    returns: pd.Series,
    dollar_volume: pd.Series,
    up_vol: float = 0.0,
    down_vol: float = 0.0,
    vp_corr: float = 0.0,
) -> float:
    """Amihud illiquidity + OFI proxy + volume-price correlation.

    Returns score in [-1, +1].  Zero when insufficient data.
    """
    if len(returns) < 5 or returns.isna().all():
        return 0.0

    amihud = (returns.abs() / (dollar_volume + 1e-6)).rolling(20).mean()
    if amihud.empty or amihud.isna().all():
        amihud_z = 0.0
    else:
        mu, sigma = amihud.mean(), amihud.std()
        amihud_z = float((amihud.iloc[-1] - mu) / (sigma + 1e-9)) if sigma > 0 else 0.0
    illiquidity_penalty = -min(amihud_z, 2.0)

    total_vol = up_vol + down_vol + 1e-6
    ofi_proxy = float((up_vol - down_vol) / total_vol)

    return float(np.clip(0.3 * illiquidity_penalty + 0.5 * ofi_proxy + 0.2 * vp_corr, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Dimension 4: Volume / Market Profile
# ---------------------------------------------------------------------------


def volume_profile_score(close: pd.Series, volume: pd.Series) -> float:
    """POC mean-reversion + AVWAP deviation score.

    Returns score in [-1, +1].
    """
    if len(close) < 10 or close.isna().any():
        return 0.0

    try:
        price_bins = np.linspace(close.min(), close.max(), 50)
        profile = np.zeros(49)
        for i in range(len(close)):
            idx = int(np.searchsorted(price_bins, close.iloc[i]))
            if 0 <= idx < 49:
                profile[idx] += float(volume.iloc[i]) if len(volume) > i else 1.0
        poc_idx = int(np.argmax(profile))
        poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2.0
        dist_pct = (float(close.iloc[-1]) - poc_price) / max(poc_price, 1e-6)
        mr_score = float(-np.tanh(dist_pct * 10.0))

        vwap = (close * volume).cumsum() / volume.cumsum()
        avwap_dev = (float(close.iloc[-1]) - float(vwap.iloc[-1])) / max(float(vwap.iloc[-1]), 1e-6)
        avwap_score = float(-np.tanh(avwap_dev * 10.0))

        return float(np.clip(0.6 * mr_score + 0.4 * avwap_score, -1.0, 1.0))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Dimension 5: Chart Pattern ML (placeholder)
# ---------------------------------------------------------------------------


def chart_pattern_score(close: pd.Series) -> float:
    """Placeholder: returns 0 until ML model is trained.

    Phase 3: replace with stumpy Matrix-Profile similarity to winning patterns.
    """
    return 0.0


# ---------------------------------------------------------------------------
# Dimension 6: Volatility Surface
# ---------------------------------------------------------------------------


def vol_surface_score(
    iv_rank: float = 50.0,
    skew: float = 0.0,
    vix_9d: float = 20.0,
    vix_30d: float = 20.0,
    vrp: float = 0.0,
) -> float:
    """IV-Rank + skew + VIX term structure + VRP.

    Args:
        iv_rank: IV rank in [0, 100] (low = cheap vol, buy signal).
        skew: 25-delta put IV − call IV (high = panic, contrarian long).
        vix_9d / vix_30d: Short/long VIX for term structure.
        vrp: Variance risk premium (IV − realized vol).

    Returns:
        Score in [-1, +1].
    """
    iv_rank_score = 1.0 - 2.0 * (iv_rank / 100.0)  # low IV = bullish
    skew_score = float(np.tanh(-skew * 10.0))
    vix_term_score = float(-np.sign(vix_9d - vix_30d))
    vrp_score = float(np.tanh(-vrp * 5.0))

    return float(np.clip(
        0.4 * iv_rank_score + 0.2 * skew_score + 0.2 * vix_term_score + 0.2 * vrp_score,
        -1.0, 1.0,
    ))


# ---------------------------------------------------------------------------
# Dimension 7: Breadth / Intermarket
# ---------------------------------------------------------------------------


def breadth_intermarket_score(
    mcclellan: float = 0.0,
    xly_xlp_ratio_change: float = 0.0,
    hyg_tlt_change: float = 0.0,
    dxy_change_20d: float = 0.0,
) -> float:
    """McClellan oscillator + risk-on ratios + dollar trend.

    All inputs are pre-computed before calling this function.

    Returns:
        Score in [-1, +1].
    """
    mc_score = float(np.tanh(mcclellan / 50.0))
    risk_on_score = float(np.tanh((xly_xlp_ratio_change - 1.0) * 5.0))
    credit_score = float(np.tanh((hyg_tlt_change - 1.0) * 5.0))
    dxy_score = float(-np.tanh(dxy_change_20d * 10.0))  # strong $ = equity headwind

    return float(np.clip(
        0.3 * mc_score + 0.3 * risk_on_score + 0.2 * credit_score + 0.2 * dxy_score,
        -1.0, 1.0,
    ))


# ---------------------------------------------------------------------------
# Dimension 8: Seasonality
# ---------------------------------------------------------------------------


def seasonality_score(
    today: date,
    overnight_gap: float = 0.0,
    gap_30d_mean: float = 0.0,
    gap_30d_std: float = 0.01,
) -> float:
    """Overnight gap fade + turn-of-month + day-of-week effects.

    Returns:
        Score in [-1, +1].
    """
    gap_z = (overnight_gap - gap_30d_mean) / max(gap_30d_std, 1e-6)
    gap_score = float(-np.tanh(gap_z * 2.0))  # gap-fade hypothesis

    days_in_month = (date(today.year + (today.month // 12), today.month % 12 + 1, 1) -
                     date(today.year, today.month, 1)).days
    is_turn = today.day <= 2 or today.day >= (days_in_month - 1)
    tom_score = 0.3 if is_turn else 0.0

    dow = today.weekday()
    dow_score = -0.1 if dow == 0 else (0.1 if dow == 4 else 0.0)

    return float(np.clip(gap_score + tom_score + dow_score, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Dimension 9: News (delegated to news_fusion)
# ---------------------------------------------------------------------------


def news_score(news_features: dict[str, float]) -> float:
    """Thin wrapper around news_fusion.news_score_normalized."""
    from src.assembled_core.signals.news_fusion import news_score_normalized
    return news_score_normalized(news_features)


# ---------------------------------------------------------------------------
# Master composite
# ---------------------------------------------------------------------------


def composite_score(
    regime: str,
    mtf: float,
    classical_ta: float,
    microstructure: float,
    volume_profile: float,
    chart_pattern: float,
    vol_surface: float,
    breadth: float,
    seasonality: float,
    news: float,
) -> tuple[float, dict[str, float]]:
    """Weighted composite of all 9 dimensions.

    Args:
        regime: One of 'calm', 'normal', 'elevated', 'crisis'.
        mtf .. news: Individual dimension scores, each in [-1, +1].

    Returns:
        (composite_score in [-1, +1], per-dimension dict for attribution).
    """
    weights = COMPOSITE_WEIGHTS_BY_REGIME.get(regime, COMPOSITE_WEIGHTS_BY_REGIME["normal"])
    scores = {
        "mtf": mtf,
        "classical_ta": classical_ta,
        "microstructure": microstructure,
        "volume_profile": volume_profile,
        "chart_pattern": chart_pattern,
        "vol_surface": vol_surface,
        "breadth": breadth,
        "seasonality": seasonality,
        "news": news,
    }
    raw = sum(weights[k] * scores[k] for k in weights)
    return float(np.clip(raw, -1.0, 1.0)), scores


__all__ = [
    "COMPOSITE_WEIGHTS_BY_REGIME",
    "TA_PARAMS_BY_REGIME",
    "mtf_alignment_score",
    "classical_ta_score",
    "microstructure_score",
    "volume_profile_score",
    "chart_pattern_score",
    "vol_surface_score",
    "breadth_intermarket_score",
    "seasonality_score",
    "news_score",
    "composite_score",
]
