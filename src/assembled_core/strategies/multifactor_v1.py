"""STRATEGY-V1: Production multi-factor strategy.

Combines 7 signal dimensions into a unified score per symbol:
  1. Trend (EMA crossover, multi-timeframe alignment)
  2. Momentum (12m returns excl. last month, trend strength)
  3. Mean-Reversion (RSI extremes, Bollinger %B, short-term reversal z-score)
  4. Volume/Liquidity (OBV trend, abnormal volume, tick imbalance)
  5. Volatility Regime (realized vol rank, vol-of-vol)
  6. Market Breadth (advance/decline, McClellan, new highs-lows)
  7. Regime Filter (bull/bear/crisis classification → exposure multiplier)

Scoring: Cross-sectional z-score per factor → weighted sum → top-N selection.
Exit management: Stop-loss, trailing-stop, take-profit (same as ema_trend_v0).

The strategy receives `prices_with_features` from trading_cycle which already
contains all ta_* columns computed by add_all_features().
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Factor weights — tuned for daily EOD trend-following with risk control
# ---------------------------------------------------------------------------

DEFAULT_FACTOR_WEIGHTS = {
    # Trend factors (40%) — primary alpha source
    "trend_ema_spread": 0.15,  # EMA20/EMA60 normalized spread
    "trend_ma200_position": 0.10,  # Price vs MA200 (long-term trend)
    "trend_adx_strength": 0.08,  # ADX trend strength
    "trend_macd_hist": 0.07,  # MACD histogram momentum
    # Momentum factors (20%) — confirmation + timing
    "mom_rsi_centered": 0.08,  # RSI centered around 50 (not overbought/oversold)
    "mom_volume_weighted": 0.07,  # Volume-weighted momentum
    "mom_obv_trend": 0.05,  # OBV slope (smart money)
    # Mean-reversion guard (10%) — avoid buying tops
    "mr_bollinger_pctb": 0.05,  # Bollinger %B (penalize >0.95)
    "mr_stoch_oversold": 0.05,  # Stochastic K (reward oversold in uptrend)
    # Volume/Liquidity (10%) — confirmation
    "vol_abnormal": 0.05,  # Abnormal volume (breakout confirmation)
    "vol_tick_imbalance": 0.05,  # Buy/sell pressure
    # Volatility regime (10%) — risk adjustment
    "vola_regime_score": 0.05,  # Low vol = positive, high vol = negative
    "vola_vov_penalty": 0.05,  # Vol-of-vol penalty (uncertainty)
    # Market breadth (10%) — universe-level filter
    "breadth_above_ma": 0.05,  # Fraction above MA50
    "breadth_ad_line": 0.05,  # Advance/decline momentum
}

# ---------------------------------------------------------------------------
# Regime multipliers — scale exposure by market regime
# ---------------------------------------------------------------------------

REGIME_EXPOSURE = {
    "bull": 1.0,
    "reflation": 0.90,
    "sideways": 0.70,
    "neutral": 0.70,
    "bear": 0.30,
    "crisis": 0.10,
}


def compute_signals(
    prices_with_features: pd.DataFrame,
    strategy_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Generate multi-factor signals from pre-computed feature DataFrame.

    Args:
        prices_with_features: Panel DataFrame with timestamp, symbol, close,
            and all ta_* feature columns from add_all_features().
        strategy_cfg: Optional config overrides for factor weights.

    Returns:
        DataFrame with columns: timestamp, symbol, direction, score, reason.
        Only LONG signals returned (score > 0 after all filters).
    """
    cfg = strategy_cfg or {}
    weights = cfg.get("factor_weights", DEFAULT_FACTOR_WEIGHTS)
    min_score = float(cfg.get("min_signal_score", 0.0))
    ema_fast = int(cfg.get("ema_fast") or 20)
    ema_slow = int(cfg.get("ema_slow") or 60)

    # D5 — signal-decay read-path. When enabled, stale factors are muted via
    # the weekly decay report at ``output/qa/signal_decay/latest.json``. When
    # disabled (default), weights pass through unchanged; callers (e.g. the
    # paper cycle) can still snapshot the hypothetical multipliers to
    # ``output/shadow/signal_decay_<date>.json`` for the A/B review.
    from src.assembled_core.strategies.signal_decay_gate import apply_multipliers

    decay_cfg = cfg.get("signal_decay", {}) or {}
    weights, _decay_multipliers = apply_multipliers(
        dict(weights),
        enabled=bool(decay_cfg.get("enabled", False)),
        stale_multiplier=float(decay_cfg.get("stale_multiplier", 0.0)),
        report_path=(
            None
            if decay_cfg.get("report_path") is None
            else __import__("pathlib").Path(decay_cfg["report_path"])
        ),
    )

    # F1 — IC-decay-weighted factor combination. Default off; enable via
    # ``strategy_cfg["ic_decay"] = {"enabled": True, "ic_snapshot": {...},
    # "lags": {...}, "half_lives": {...}}``. When enabled, the base weights
    # are replaced by decay-adjusted IC weights; when disabled or when no
    # IC snapshot is provided, the existing weights pass through.
    ic_cfg = cfg.get("ic_decay", {}) or {}
    if ic_cfg.get("enabled") and ic_cfg.get("ic_snapshot"):
        from src.assembled_core.strategies.ic_decay_weights import (
            DEFAULT_MAX_W_PER_FACTOR,
            compute_ic_decay_weights,
        )

        ic_result = compute_ic_decay_weights(
            ic_cfg["ic_snapshot"],
            lags=ic_cfg.get("lags") or {},
            half_lives=ic_cfg.get("half_lives") or {},
            max_w_per_factor=float(
                ic_cfg.get("max_w_per_factor", DEFAULT_MAX_W_PER_FACTOR)
            ),
            fallback_weights=weights,
        )
        weights = ic_result.weights

    if prices_with_features.empty or "symbol" not in prices_with_features.columns:
        return _empty_signals()

    df = prices_with_features.copy()

    # Ensure we have timestamps
    if "timestamp" not in df.columns:
        return _empty_signals()

    # Work on latest bar per symbol only (EOD signal)
    latest = (
        df.sort_values("timestamp").groupby("symbol", group_keys=False).tail(1).copy()
    )

    if latest.empty:
        return _empty_signals()

    # --- Compute raw factor values per symbol ---
    scores = pd.DataFrame({"symbol": latest["symbol"].values})
    scores.index = latest.index

    # 1. TREND: EMA spread
    close = pd.to_numeric(latest.get("close", pd.Series(dtype=float)), errors="coerce")
    scores["trend_ema_spread"] = _compute_ema_spread(df, ema_fast, ema_slow)
    scores["trend_ma200_position"] = _ma_position(
        latest, close, "ta_ma_200_v1", fallback_window=200
    )
    scores["trend_adx_strength"] = (
        _safe_col(latest, "ta_adx_v1", default=0.0) / 100.0
    )  # normalize to 0-1
    scores["trend_macd_hist"] = _safe_col(latest, "ta_macd_hist_v1", default=0.0)

    # 2. MOMENTUM
    scores["mom_rsi_centered"] = _rsi_score(latest)
    scores["mom_volume_weighted"] = _safe_col(
        latest, "ta_vol_weighted_mom_20d_v1", default=0.0
    )
    scores["mom_obv_trend"] = _obv_trend(df)

    # 3. MEAN-REVERSION GUARD
    scores["mr_bollinger_pctb"] = _bollinger_score(latest)
    scores["mr_stoch_oversold"] = _stochastic_score(latest)

    # 4. VOLUME/LIQUIDITY
    scores["vol_abnormal"] = _abnormal_volume_score(latest)
    scores["vol_tick_imbalance"] = _safe_col(latest, "tick_imbalance_20d", default=0.5)

    # 5. VOLATILITY REGIME
    scores["vola_regime_score"] = _volatility_regime_score(latest)
    scores["vola_vov_penalty"] = _vov_penalty(latest)

    # 6. MARKET BREADTH (universe-level, same for all symbols)
    breadth_score = _compute_breadth_score(df)
    scores["breadth_above_ma"] = breadth_score
    scores["breadth_ad_line"] = breadth_score  # simplified: same breadth context

    # --- Cross-sectional z-score per factor (vectorized across all columns) ---
    factor_cols = [c for c in scores.columns if c != "symbol"]
    factor_df = scores[factor_cols].astype(float)
    means = factor_df.mean()
    stds = factor_df.std()
    # safe_stds: NaN where std ≈ 0 → normalized becomes NaN → fillna(0.0).
    # Avoid .where(valid, 0.0) with a column-indexed boolean Series: pandas 2.x
    # aligns it against the row index (not column axis), zeroing every cell.
    safe_stds = stds.copy()
    safe_stds[stds <= 1e-10] = np.nan
    normalized = (factor_df - means) / safe_stds
    scores[factor_cols] = normalized.fillna(0.0).clip(-3.0, 3.0)

    # --- Weighted composite score ---
    # Mirror the v2 guard (multifactor_v2.py:694): only count a factor into
    # total_weight if it actually contributed non-zero values. A column that is
    # present but entirely NaN/zero (e.g. insufficient history, upstream feature
    # outage) previously took its share of the renormalization budget and
    # silently diluted every other factor — turning a missing feature into a
    # systematic downweight rather than a visible gap.
    composite = pd.Series(0.0, index=scores.index)
    total_weight = 0.0
    used_factors = []
    for factor_name, weight in weights.items():
        if factor_name in scores.columns:
            factor_vals = scores[factor_name].fillna(0.0)
            if factor_vals.abs().sum() > 1e-10:
                composite += weight * factor_vals
                total_weight += weight
                used_factors.append(factor_name)

    if total_weight > 0:
        composite = composite / total_weight  # Renormalize

    # --- Regime filter ---
    regime_mult = _compute_regime_multiplier(df, cfg)
    composite = composite * regime_mult

    # --- Build output signals (LONG only where composite > min_score) ---
    composite_arr = composite.to_numpy(dtype=float)
    ema_spread_arr = (
        scores["trend_ema_spread"].to_numpy(dtype=float)
        if "trend_ema_spread" in scores.columns
        else np.zeros(len(composite_arr))
    )
    sym_list = latest["symbol"].tolist()
    ts_list = (
        latest["timestamp"].tolist()
        if "timestamp" in latest.columns
        else [None] * len(latest)
    )
    top_factors = used_factors[:3]
    factor_arrs = {
        f: scores[f].to_numpy(dtype=float) for f in top_factors if f in scores.columns
    }

    out = []
    for i in range(len(latest)):
        score = composite_arr[i]
        ema_spread = ema_spread_arr[i]
        if (
            score > min_score and ema_spread > -0.5
        ):  # Allow slight negative EMA if other factors strong
            reasons = [
                f"{f}={factor_arrs[f][i]:.2f}" for f in top_factors if f in factor_arrs
            ]
            out.append(
                {
                    "timestamp": ts_list[i],
                    "symbol": sym_list[i],
                    "direction": "LONG",
                    "score": float(score),
                    "reason": "; ".join(reasons),
                }
            )

    if not out:
        return _empty_signals()

    result = pd.DataFrame(out).sort_values("score", ascending=False)
    logger.info(
        "[MF-V1] %d LONG signals from %d symbols (regime_mult=%.2f, factors=%d)",
        len(result),
        len(latest),
        regime_mult,
        len(used_factors),
    )
    return result


def compute_target_positions(
    signals: pd.DataFrame,
    total_capital: float,
    equal_weight: bool = False,
    prices_latest: pd.DataFrame | None = None,
    max_positions: int = 10,
    min_position_weight: float = 0.03,
    target_invested_pct: float = 0.80,
) -> pd.DataFrame:
    """Compute target positions from multi-factor signals.

    Score-proportional sizing by default. Higher multi-factor score = larger position.

    Args:
        signals: DataFrame with columns symbol, score.
        total_capital: Total capital to allocate.
        equal_weight: If True, 1/N weight. If False, score-proportional.
        max_positions: Limit to top-N by score.
        min_position_weight: Minimum weight per position.
        target_invested_pct: Fraction of capital to invest.

    Returns:
        DataFrame with columns: symbol, target_weight, target_qty (NOTIONAL).
    """
    empty = pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
    if signals is None or signals.empty:
        return empty
    if "symbol" not in signals.columns:
        return empty

    sig = signals.copy()
    if "score" in sig.columns:
        sig = sig.sort_values("score", ascending=False)
    else:
        sig["score"] = 1.0

    # Top-N selection
    if max_positions > 0 and len(sig) > max_positions:
        sig = sig.head(max_positions)

    syms = sig["symbol"].drop_duplicates().tolist()
    if not syms:
        return empty

    scores_map = sig.set_index("symbol")["score"].to_dict()
    n = len(syms)
    available_capital = total_capital * min(target_invested_pct, 1.0)

    if equal_weight:
        weights = {sym: 1.0 / n for sym in syms}
    else:
        # Score-proportional: shift scores to be positive
        min_score = min(scores_map.get(s, 0.0) for s in syms)
        shifted = {s: scores_map.get(s, 0.0) - min_score + 0.01 for s in syms}
        total_score = sum(shifted.values())
        if total_score > 0:
            weights = {s: shifted[s] / total_score for s in syms}
        else:
            weights = {s: 1.0 / n for s in syms}

    # Apply min_position_weight filter
    if min_position_weight > 0:
        filtered = [s for s in syms if weights[s] >= min_position_weight]
        if not filtered:
            max_possible = int(1.0 / min_position_weight)
            filtered = syms[:max_possible] if max_possible > 0 else syms[:1]
        syms = filtered
        n = len(syms)
        if equal_weight:
            weights = {s: 1.0 / n for s in syms}
        else:
            total_score = sum(scores_map.get(s, 0.01) - min_score + 0.01 for s in syms)
            if total_score > 0:
                weights = {
                    s: (scores_map.get(s, 0.01) - min_score + 0.01) / total_score
                    for s in syms
                }
            else:
                weights = {s: 1.0 / n for s in syms}

    # Cap maximum weight per position to prevent extreme concentration
    max_weight = 1.0 / max(n, 1) * 2.5  # At most 2.5x equal-weight share
    max_weight = min(max_weight, 0.20)  # Hard cap at 20% per position
    capped = False
    for s in syms:
        if weights[s] > max_weight:
            weights[s] = max_weight
            capped = True
    if capped:
        total_w = sum(weights[s] for s in syms)
        if total_w > 0:
            weights = {s: weights[s] / total_w for s in syms}

    rows = []
    for sym in syms:
        w = weights[sym] * min(target_invested_pct, 1.0)
        rows.append(
            {
                "symbol": sym,
                "target_weight": w,
                "target_qty": available_capital * w,  # NOTIONAL (post-cap weight)
            }
        )

    return pd.DataFrame(rows)


def check_exit_signals(
    current_positions: dict,
    prices_latest: pd.DataFrame,
    strategy_cfg: dict | None = None,
) -> pd.DataFrame:
    """Check exit conditions for current positions.

    Uses same logic as ema_trend_v0 exits: stop-loss, trailing-stop, take-profit.

    Returns:
        DataFrame with columns: symbol, direction, exit_reason, exit_qty_pct.
    """
    cfg = strategy_cfg or {}
    stop_loss_pct = float(cfg.get("stop_loss_pct", 0.08))
    trailing_stop_pct = float(cfg.get("trailing_stop_pct", 0.10))
    take_profit_pct = float(cfg.get("take_profit_pct", 0.15))

    empty = pd.DataFrame(columns=["symbol", "direction", "exit_reason", "exit_qty_pct"])

    if not current_positions or prices_latest is None or prices_latest.empty:
        return empty

    price_map = {}
    if "symbol" in prices_latest.columns and "close" in prices_latest.columns:
        price_map = dict(
            zip(prices_latest["symbol"].values, prices_latest["close"].values)
        )

    exits = []
    for sym, pos in current_positions.items():
        qty = float(pos.get("qty", 0))
        if qty <= 0:
            continue
        avg_price = float(pos.get("avg_price", 0))
        hwm = float(pos.get("hwm", avg_price))
        if avg_price <= 0:
            continue

        current_price = float(price_map.get(sym, 0))
        if current_price <= 0:
            continue

        if current_price > hwm:
            hwm = current_price

        # Stop-loss
        if stop_loss_pct > 0:
            stop_price = avg_price * (1 - stop_loss_pct)
            if current_price <= stop_price:
                exits.append(
                    {
                        "symbol": sym,
                        "direction": "FLAT",
                        "exit_reason": f"stop_loss ({current_price:.2f} <= {stop_price:.2f})",
                        "exit_qty_pct": 1.0,
                    }
                )
                continue

        # Trailing stop
        if trailing_stop_pct > 0 and hwm > avg_price:
            trail_price = hwm * (1 - trailing_stop_pct)
            if current_price <= trail_price:
                exits.append(
                    {
                        "symbol": sym,
                        "direction": "FLAT",
                        "exit_reason": f"trailing_stop ({current_price:.2f} <= {trail_price:.2f}, hwm={hwm:.2f})",
                        "exit_qty_pct": 1.0,
                    }
                )
                continue

        # Take-profit
        if take_profit_pct > 0:
            tp_price = avg_price * (1 + take_profit_pct)
            if current_price >= tp_price:
                exits.append(
                    {
                        "symbol": sym,
                        "direction": "FLAT",
                        "exit_reason": f"take_profit ({current_price:.2f} >= {tp_price:.2f})",
                        "exit_qty_pct": 0.5,
                    }
                )
                continue

    if not exits:
        return empty
    return pd.DataFrame(exits)


# ---------------------------------------------------------------------------
# Internal factor computation helpers
# ---------------------------------------------------------------------------


def _empty_signals() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score", "reason"])


def _safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """Safely get a column, returning default if missing."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def _compute_ema_spread(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """Compute normalized EMA spread per symbol (latest bar)."""
    results = {}
    for sym, grp in df.groupby("symbol", group_keys=False):
        g = grp.sort_values("timestamp")
        if len(g) < slow:
            results[sym] = 0.0
            continue
        close = pd.to_numeric(g["close"], errors="coerce").ffill()
        ema_f = close.ewm(span=fast, adjust=False).mean().iloc[-1]
        ema_s = close.ewm(span=slow, adjust=False).mean().iloc[-1]
        if ema_s > 0:
            results[sym] = (ema_f - ema_s) / ema_s
        else:
            results[sym] = 0.0

    latest = df.sort_values("timestamp").groupby("symbol", group_keys=False).tail(1)
    return latest["symbol"].map(results).fillna(0.0)


def _ma_position(
    latest: pd.DataFrame, close: pd.Series, ma_col: str, fallback_window: int = 200
) -> pd.Series:
    """Price position relative to long-term MA. >0 = above MA (bullish)."""
    if ma_col in latest.columns:
        ma = pd.to_numeric(latest[ma_col], errors="coerce")
        valid = ma > 0
        result = pd.Series(0.0, index=latest.index)
        result[valid] = (close[valid] - ma[valid]) / ma[valid]
        return result.clip(-1.0, 1.0)
    return pd.Series(0.0, index=latest.index)


def _rsi_score(latest: pd.DataFrame) -> pd.Series:
    """RSI-based score: penalize extremes, reward mid-range in uptrend.

    Score mapping:
      RSI 30-50: positive (oversold → buying opportunity in uptrend)
      RSI 50-70: neutral to slightly positive (healthy trend)
      RSI 70-90: negative (overbought → mean reversion risk)
      RSI <30 or >90: strong negative (extreme)
    """
    rsi = _safe_col(latest, "ta_rsi_14_v1", default=50.0)
    # Center around 50, penalize extremes
    # Optimal zone: 40-65 (strongest uptrend without being overbought)
    score = pd.Series(0.0, index=latest.index)
    score = -(((rsi - 52.5) / 25.0) ** 2) + 1.0  # Inverted parabola centered at 52.5
    return score.clip(-1.0, 1.0)


def _bollinger_score(latest: pd.DataFrame) -> pd.Series:
    """Bollinger %B score: reward mid-to-upper band, penalize extremes."""
    pctb = _safe_col(latest, "ta_bb_pctb_v1", default=0.5)
    # Ideal: 0.4-0.8 (in uptrend but not hitting upper band)
    score = -(((pctb - 0.6) / 0.4) ** 2) + 1.0
    return score.clip(-1.0, 1.0)


def _stochastic_score(latest: pd.DataFrame) -> pd.Series:
    """Stochastic score: reward oversold conditions in uptrend."""
    stoch_k = _safe_col(latest, "ta_stoch_k_v1", default=50.0)
    # Lower stochastic = more oversold = better entry (in uptrend context)
    score = (50.0 - stoch_k) / 50.0
    return score.clip(-1.0, 1.0)


def _obv_trend(df: pd.DataFrame) -> pd.Series:
    """OBV slope direction: positive slope = accumulation."""
    results = {}
    for sym, grp in df.groupby("symbol", group_keys=False):
        g = grp.sort_values("timestamp")
        obv = _safe_col(g, "ta_obv_v1", default=0.0)
        if len(obv) >= 20:
            obv_ma_short = obv.iloc[-5:].mean()
            obv_ma_long = obv.iloc[-20:].mean()
            if abs(obv_ma_long) > 1e-10:
                results[sym] = (obv_ma_short - obv_ma_long) / abs(obv_ma_long)
            else:
                results[sym] = 0.0
        else:
            results[sym] = 0.0

    latest = df.sort_values("timestamp").groupby("symbol", group_keys=False).tail(1)
    return latest["symbol"].map(results).fillna(0.0).clip(-1.0, 1.0)


def _abnormal_volume_score(latest: pd.DataFrame) -> pd.Series:
    """Abnormal volume: >1.5x average = breakout confirmation, <0.5x = low conviction."""
    abnorm = _safe_col(latest, "abnormal_vol_20d", default=1.0)
    # Score: log scale centered at 1.0
    score = np.log(abnorm.clip(0.1, 10.0))  # log(1) = 0, log(2) = 0.69
    return score.clip(-1.0, 1.0)


def _volatility_regime_score(latest: pd.DataFrame) -> pd.Series:
    """Low volatility = positive (calm market), high = negative (risky)."""
    rv = _safe_col(latest, "rv_20", default=0.15)
    # Typical annualized vol: 0.10 (calm) to 0.40 (stressed)
    # Score: invert — low vol is good
    score = (0.20 - rv) / 0.15  # vol=0.10 → score=0.67, vol=0.30 → score=-0.67
    return score.clip(-1.0, 1.0)


def _vov_penalty(latest: pd.DataFrame) -> pd.Series:
    """Vol-of-vol: high uncertainty in vol itself is bad."""
    vov = _safe_col(latest, "vov_20_60", default=0.0)
    # Invert: higher vov = more penalty
    score = -vov / 0.10  # vov=0 → score=0, vov=0.05 → score=-0.5
    return score.clip(-1.0, 0.0)


def _compute_breadth_score(df: pd.DataFrame) -> float:
    """Market breadth: what fraction of stocks are above their MA."""
    # Callers may pass a per-bar slice without a timestamp column (e.g. the
    # regime detector in check_exit_signals). Treat missing timestamp as
    # "already latest", and missing symbol as "single symbol" so the sort
    # is a no-op rather than a KeyError.
    if "symbol" not in df.columns:
        return 0.0
    if "timestamp" in df.columns:
        latest = df.sort_values("timestamp").groupby("symbol", group_keys=False).tail(1)
    else:
        latest = df.groupby("symbol", group_keys=False).tail(1)
    close = pd.to_numeric(latest.get("close", pd.Series(dtype=float)), errors="coerce")

    # Use MA50 if available, otherwise compute from close
    if "ta_ma_50_v1" in latest.columns:
        ma50 = pd.to_numeric(latest["ta_ma_50_v1"], errors="coerce")
        valid = (close > 0) & (ma50 > 0)
        if valid.sum() > 0:
            fraction_above = (close[valid] > ma50[valid]).mean()
            # Score: 0.5 = neutral, >0.7 = bullish, <0.3 = bearish
            return float((fraction_above - 0.5) * 2.0)
    return 0.0


def _crash_prediction_multiplier(df: pd.DataFrame, cfg: dict) -> float:
    """Sprint 1 / W6 — Crash-prediction exposure multiplier.

    Maps the CrashPredictionEngine composite probability to an exposure
    scaler as specified in the plan:
      crash_prob < 0.30 -> 1.0
      0.30..0.50       -> 0.8
      0.50..0.70       -> 0.5
      0.70..0.85       -> 0.2
      >= 0.85          -> 0.0

    Gated by cfg['crash_prediction']['enabled'] (default False). Any
    exception is swallowed and the multiplier defaults to 1.0 so the
    strategy remains inert when the engine cannot run.
    """
    try:
        cp_cfg = (cfg or {}).get("crash_prediction") or {}
        if not cp_cfg.get("enabled", False):
            return 1.0

        ref_symbol = str(cp_cfg.get("reference_symbol", "SPY")).upper()
        if "symbol" not in df.columns or "close" not in df.columns:
            return 1.0

        sym_df = df[df["symbol"].astype(str).str.upper() == ref_symbol]
        if sym_df.empty:
            return 1.0
        if "timestamp" in sym_df.columns:
            sym_df = sym_df.sort_values("timestamp")

        from src.assembled_core.signals.crash_prediction import CrashPredictionEngine

        engine = CrashPredictionEngine()
        signal = engine.predict(market_data=sym_df)
        prob = float(signal.crash_probability)

        if prob >= 0.85:
            mult = 0.0
        elif prob >= 0.70:
            mult = 0.2
        elif prob >= 0.50:
            mult = 0.5
        elif prob >= 0.30:
            mult = 0.8
        else:
            mult = 1.0

        logger.info("[MF-V1] CrashPrediction prob=%.3f -> mult=%.2f", prob, mult)
        return mult
    except Exception as exc:
        logger.debug("[MF-V1] crash-prediction unavailable: %s", exc)
        return 1.0


def _compute_regime_multiplier(df: pd.DataFrame, cfg: dict) -> float:
    """Compute regime-based exposure multiplier.

    Tries to detect current market regime from available features.
    Falls back to 1.0 (no adjustment) if insufficient data.
    """
    base_mult: float
    # F2 — posterior-blended path (soft multiplier over regime distribution).
    reg_cfg = (cfg or {}).get("regime_posterior") or {}
    if reg_cfg.get("enabled") and reg_cfg.get("posterior"):
        from src.assembled_core.signals.regime.hmm_posterior import (
            DEFAULT_HALF_LIFE_DAYS,
            smooth_posterior,
        )

        try:
            smoothed = smooth_posterior(
                reg_cfg["posterior"],
                reg_cfg.get("prev_posterior"),
                half_life_days=float(
                    reg_cfg.get("half_life_days", DEFAULT_HALF_LIFE_DAYS)
                ),
            )
            base_mult = float(
                sum(
                    float(prob) * float(REGIME_EXPOSURE.get(str(k).lower(), 0.70))
                    for k, prob in smoothed.items()
                )
            )
            logger.info(
                "[MF-V1] Regime posterior-blended -> exposure_mult=%.3f (keys=%s)",
                base_mult,
                sorted(smoothed.keys()),
            )
            crash_mult = _crash_prediction_multiplier(df, cfg)
            return base_mult * crash_mult
        except Exception as exc:  # noqa: BLE001
            logger.debug("[MF-V1] Posterior-blend failed, falling back: %s", exc)

    try:
        from src.assembled_core.risk.regime_models import build_regime_state

        regime_df = build_regime_state(df)
        if regime_df is not None and not regime_df.empty:
            latest_regime = regime_df.iloc[-1]
            raw_label = latest_regime.get("regime_label")
            # A missing regime_label column (upstream schema drift) previously
            # silently defaulted to "neutral" which then maps to 0.70 —
            # indistinguishable from a correctly detected neutral regime. Log
            # the fallback so the distinction is observable.
            if raw_label is None or (
                isinstance(raw_label, float) and raw_label != raw_label
            ):
                logger.warning(
                    "[MF-V1] regime_label missing/NaN in regime_df — using 'neutral' fallback"
                )
                label = "neutral"
            else:
                label = str(raw_label).lower()
            if label not in REGIME_EXPOSURE:
                logger.warning(
                    "[MF-V1] regime_label=%r not in REGIME_EXPOSURE keys %s — "
                    "using 0.70 fallback",
                    label,
                    list(REGIME_EXPOSURE.keys()),
                )
            base_mult = REGIME_EXPOSURE.get(label, 0.70)
            logger.info("[MF-V1] Regime=%s -> exposure_mult=%.2f", label, base_mult)
        else:
            raise RuntimeError("empty regime_df")
    except Exception as exc:
        logger.debug("[MF-V1] Regime detection unavailable: %s", exc)
        # Fallback: simple breadth-based regime
        breadth = _compute_breadth_score(df)
        if breadth < -0.4:
            base_mult = 0.40  # Bearish market
        elif breadth < -0.1:
            base_mult = 0.70  # Cautious
        elif breadth > 0.3:
            base_mult = 1.0  # Bullish
        else:
            base_mult = 0.85  # Neutral

    crash_mult = _crash_prediction_multiplier(df, cfg)
    return base_mult * crash_mult


__all__ = ["compute_signals", "compute_target_positions", "check_exit_signals"]
