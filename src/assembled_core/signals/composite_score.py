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


def compute_news_dim_with_edcl(
    base_news_score: float,
    edcl_basket: Any | None,
    conviction: float,
) -> float:
    """Blend base news score with EDCL trigger-basket score.

    When an EDCL basket is active, the news dimension is conviction-weighted
    toward the basket's composite score, amplifying the geo-event signal.
    Falls back to base_news_score when basket is None or conviction is 0.

    Args:
        base_news_score: Raw news dimension score in [-1, +1].
        edcl_basket: TriggerBasket from intel.trigger_basket (Phase B), or None.
        conviction: EDCL conviction score in [0, 1] from conviction_engine.

    Returns:
        Blended news dimension score in [-1, +1].
    """
    if edcl_basket is None or conviction <= 0.0:
        return base_news_score
    try:
        from src.assembled_core.intel.trigger_basket import compute_basket_score
        edcl_score = compute_basket_score(edcl_basket)
        # Map [0,1] basket score to [-1,+1]: positive geo-risk events are bearish
        # (reduce exposure toward affected assets) unless conviction is directional.
        # Default: high basket score → bearish news signal (risk-off).
        edcl_news = edcl_score * 2.0 - 1.0  # [0,1] → [-1,+1], so 0 becomes -1
        # Actually: geo-risk means bearish, so basket_score=1 → news=-1
        edcl_news = -(edcl_score)  # high geo-risk → bearish news dimension
    except Exception:
        return base_news_score

    blended = (1.0 - conviction) * base_news_score + conviction * edcl_news
    return float(np.clip(blended, -1.0, 1.0))


def compute_edcl_conviction_multiplier(
    edcl_conviction: float,
    composite_regime: str,
    options_iv_skew_z: float = 0.0,
    policy: dict[str, Any] | None = None,
) -> float:
    """Phase H — Triple-Confirmation sizing multiplier.

    Rewards confluence of three independent signals:
      1. EDCL trigger conviction > threshold (Phase B/C)
      2. Composite score regime == 'crisis' or 'elevated'
      3. Options IV skew Z-score > 2.0 (tail-risk skew confirmation)

    Args:
        edcl_conviction: EDCL conviction score [0, 1] from conviction_engine.
        composite_regime: Current composite regime label ('calm'/'normal'/'elevated'/'crisis').
        options_iv_skew_z: Options IV skew Z-score (0.0 = not available).
        policy: Policy dict — reads edcl_conviction_overlay sub-dict.

    Returns:
        Sizing multiplier [1.0, 2.0].
        - Triple confirmation (all three): 2.0
        - Double (EDCL + regime, no IV): 1.5
        - EDCL only above threshold: 1.2
        - No confirmation: 1.0
    """
    cfg = (policy or {}).get("edcl_conviction_overlay") or {}
    threshold = float(cfg.get("conviction_threshold", 0.70))
    max_mult = float(cfg.get("max_multiplier", 2.0))

    if edcl_conviction < threshold:
        return 1.0

    crisis_regime = composite_regime in ("crisis", "elevated")
    iv_spike = options_iv_skew_z > 2.0

    if crisis_regime and iv_spike:
        multiplier = min(2.0, max_mult)   # triple confirmation
    elif crisis_regime:
        multiplier = 1.5                  # double: EDCL + regime
    else:
        multiplier = 1.2                  # EDCL only

    logger.debug(
        "[EDCL-H] triple_confirm: conviction=%.3f regime=%s iv_z=%.2f → mult=%.2f",
        edcl_conviction, composite_regime, options_iv_skew_z, multiplier,
    )
    return multiplier


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
    edcl_basket: Any | None = None,
    edcl_conviction: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """Weighted composite of all 9 dimensions.

    Args:
        regime: One of 'calm', 'normal', 'elevated', 'crisis'.
        mtf .. news: Individual dimension scores, each in [-1, +1].
        edcl_basket: Optional TriggerBasket from EDCL Phase B (enriches news dim).
        edcl_conviction: EDCL conviction score [0,1]; only used when edcl_basket set.

    Returns:
        (composite_score in [-1, +1], per-dimension dict for attribution).
    """
    weights = COMPOSITE_WEIGHTS_BY_REGIME.get(regime, COMPOSITE_WEIGHTS_BY_REGIME["normal"])
    # Enrich news dimension with EDCL basket if provided
    if edcl_basket is not None and edcl_conviction > 0.0:
        news = compute_news_dim_with_edcl(news, edcl_basket, edcl_conviction)
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


def _col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return first candidate column name that exists in df."""
    return next((c for c in candidates if c in df.columns), None)


def generate_composite_score_signals(
    panel: pd.DataFrame,
    regime: str = "normal",
    signal_threshold: float = 0.10,
    as_of_date: "date | None" = None,
    min_history_bars: int = 30,
) -> pd.DataFrame:
    """Generate buy/sell signals for all symbols using the 9-dimension composite score.

    Computes each dimension from panel columns where available.
    Dimensions that require external data (IV, intraday) default to 0.0 (neutral).

    Panel column mappings (accepts both legacy and ta_*_v1 names):
        RSI: ta_rsi_14_v1 / rsi_14
        MACD histogram: ta_macd_hist_v1 / macd_hist
        BB %B: ta_bb_pctb_v1 / bb_pos / bb_pctb
        ADX: ta_adx_v1 / adx_14
        Log returns: ta_log_return_v1 / log_return
        MA20: ta_ma_20_v1 / ma_20

    Args:
        panel: Panel DataFrame with columns: symbol, timestamp, close, volume + TA features.
        regime: Composite weight regime (calm/normal/elevated/crisis).
                "bull"/"bear"/"sideways" are mapped to composite regime labels.
        signal_threshold: Minimum absolute composite score to generate a non-NEUTRAL signal.
        as_of_date: If provided, only use rows with timestamp <= as_of_date.
        min_history_bars: Minimum rows per symbol required for computation.

    Returns:
        DataFrame with columns: symbol, direction, score.
        direction is "BUY" / "SELL" / "NEUTRAL".
    """
    if panel is None or panel.empty:
        return pd.DataFrame(columns=["symbol", "direction", "score"])

    # Map pipeline regime labels to composite weight regime labels
    _regime_map = {
        "bull": "normal", "sideways": "elevated",
        "bear": "elevated", "crisis": "crisis",
    }
    composite_regime = _regime_map.get(regime, regime)
    if composite_regime not in COMPOSITE_WEIGHTS_BY_REGIME:
        composite_regime = "normal"

    df = panel.copy()
    if as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date, tz="UTC") if pd.Timestamp(as_of_date).tzinfo is None else pd.Timestamp(as_of_date)
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], utc=True)
            df = df[ts <= cutoff]

    if df.empty:
        return pd.DataFrame(columns=["symbol", "direction", "score"])

    # Column aliases
    rsi_col   = _col(df, "ta_rsi_14_v1", "rsi_14", "rsi")
    macd_col  = _col(df, "ta_macd_hist_v1", "macd_hist")
    bb_col    = _col(df, "ta_bb_pctb_v1", "bb_pos", "bb_pctb")
    adx_col   = _col(df, "ta_adx_v1", "adx_14", "adx")
    ret_col   = _col(df, "ta_log_return_v1", "log_return")
    ma20_col  = _col(df, "ta_ma_20_v1", "ma_20")

    # Cross-sectional breadth: % of symbols trading above their MA20
    breadth_ratio = 0.0
    if ma20_col and "close" in df.columns and "symbol" in df.columns:
        last = (
            df.sort_values("timestamp").groupby("symbol").last().reset_index()
            if "timestamp" in df.columns
            else df.groupby("symbol").last().reset_index()
        )
        n_above = (last["close"] > last[ma20_col]).sum()
        n_total = len(last)
        breadth_ratio = float(n_above / n_total) if n_total > 0 else 0.5
    mcclellan_proxy = (breadth_ratio - 0.5) * 200.0  # map [0,1] → [-100, +100]

    results = []
    for symbol, grp in df.groupby("symbol"):
        grp = grp.sort_values("timestamp") if "timestamp" in grp.columns else grp
        if len(grp) < min_history_bars:
            continue

        close = grp["close"]
        volume = grp["volume"] if "volume" in grp.columns else pd.Series(1.0, index=grp.index)

        # Dim 1: MTF alignment (use daily close + ADX; intraday signals = 0)
        adx_val = float(grp[adx_col].iloc[-1]) if adx_col else 20.0
        dim1 = mtf_alignment_score(close, macd_hist_15m=0.0, rsi_5m=50.0, adx_daily=adx_val)

        # Dim 2: Classical TA
        rsi_val  = float(grp[rsi_col].iloc[-1])  if rsi_col  else 50.0
        macd_val = float(grp[macd_col].iloc[-1]) if macd_col else 0.0
        bb_val   = float(grp[bb_col].iloc[-1])   if bb_col   else 0.5
        dim2 = classical_ta_score(rsi_val, macd_val, bb_val, composite_regime)

        # Dim 3: Microstructure (returns + dollar volume)
        if ret_col and not grp[ret_col].isna().all():
            returns_s = grp[ret_col].fillna(0.0)
            dv = close * volume
            dim3 = microstructure_score(returns_s, dv)
        else:
            dim3 = 0.0

        # Dim 4: Volume profile
        dim4 = volume_profile_score(close, volume) if len(close) >= 10 else 0.0

        # Dim 5: Chart pattern (placeholder)
        dim5 = chart_pattern_score(close)

        # Dim 6: Vol surface — no IV data in panel; use realized vol ratio as proxy
        dim6 = 0.0
        if "rv_20" in grp.columns and "rv_60" in grp.columns:
            rv20 = float(grp["rv_20"].iloc[-1])
            rv60 = float(grp["rv_60"].iloc[-1])
            if rv60 > 0:
                vrp_proxy = (rv20 - rv60) / rv60
                dim6 = float(np.tanh(-vrp_proxy * 2.0))

        # Dim 7: Breadth / intermarket (cross-sectional breadth proxy)
        dim7 = breadth_intermarket_score(mcclellan=mcclellan_proxy)

        # Dim 8: Seasonality
        if "timestamp" in grp.columns:
            _ts = pd.Timestamp(grp["timestamp"].iloc[-1])
            _d = _ts.date() if not isinstance(_ts, type(None)) else (as_of_date or date.today())
        else:
            _d = as_of_date or date.today()
        dim8 = seasonality_score(_d)

        # Dim 9: News (0.0 when no news_features available in panel)
        dim9 = 0.0
        if "news_sentiment" in grp.columns:
            dim9 = float(np.clip(grp["news_sentiment"].iloc[-1], -1.0, 1.0))

        score, _ = composite_score(
            composite_regime, dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, dim9
        )

        if score > signal_threshold:
            direction = "BUY"
        elif score < -signal_threshold:
            direction = "SELL"
        else:
            direction = "NEUTRAL"

        results.append({"symbol": symbol, "direction": direction, "score": score})

    if not results:
        return pd.DataFrame(columns=["symbol", "direction", "score"])

    out = pd.DataFrame(results)
    logger.debug(
        "[composite_score] generated %d signals (%d BUY, %d SELL, %d NEUTRAL) regime=%s",
        len(out),
        (out["direction"] == "BUY").sum(),
        (out["direction"] == "SELL").sum(),
        (out["direction"] == "NEUTRAL").sum(),
        composite_regime,
    )
    return out


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
    "compute_news_dim_with_edcl",
    "compute_edcl_conviction_multiplier",
    "composite_score",
    "generate_composite_score_signals",
]
