"""Per-position trailing stops with regime-adaptive ATR multipliers (V16 + M16 E3/E4).

Implements:
- ATR-based trailing stops: stop = high_watermark - multiplier * ATR
- Regime-dependent multipliers: tighter in bear (1.5x), wider in bull (3x)
- E4 (M16): VIX-scaled multipliers — wider stops in high-vol environments
- E3 (M16): Time-stop — exit stale positions that show no progress
- Gradual de-risking: reduce position by 25%/50%/75% at -1sig/-2sig/-3sig
- Integration point for kill-switch and order generation

Reference: Katz & McCormick "The Encyclopedia of Trading Strategies" (Ch. 8-9).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

# Regime-specific ATR multipliers
_REGIME_MULTIPLIERS: dict[str, float] = {
    "bull": 3.0,
    "bear": 1.5,
    "sideways": 2.0,
    "crisis": 1.0,
    "unknown": 2.0,
}


def _vix_multiplier_factor(vix: float | None) -> float:
    """Scale factor applied on top of regime multiplier based on VIX level.

    Low VIX (calm market) → tighter stops.
    High VIX (stress) → wider stops to avoid being shaken out.
    """
    if vix is None or vix != vix:  # None or NaN
        return 1.0
    if vix < 15:
        return 0.8  # tight — calm market, precise stops
    if vix < 20:
        return 1.0  # normal
    if vix < 30:
        return 1.3  # elevated — widen stops
    return 1.7  # crisis — very wide stops


@dataclass
class TrailingStopState:
    """Per-position trailing stop state."""

    symbol: str
    entry_price: float
    high_watermark: float
    current_atr: float
    stop_price: float
    regime: str = "unknown"
    multiplier: float = 2.0
    triggered: bool = False
    reduction_pct: float = 0.0  # 0.0 = no reduction, 0.25/0.50/0.75 = gradual
    # E3: time-stop tracking
    entry_bar: int = 0  # bar index when position was opened
    bars_held: int = 0  # bars held so far (updated externally)


@dataclass
class TrailingStopResult:
    """Result of trailing stop check across all positions."""

    stops: list[TrailingStopState] = field(default_factory=list)
    triggered_symbols: list[str] = field(default_factory=list)
    reduction_symbols: dict[str, float] = field(default_factory=dict)
    # symbol -> reduction fraction (0.25, 0.50, 0.75)


def compute_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 14,
) -> float:
    """Compute latest ATR value."""
    if len(close) < window + 1:
        if len(close) >= 2:
            return float(np.mean(high - low))
        return 0.0

    prev_close = close[:-1]
    curr_high = high[1:]
    curr_low = low[1:]

    tr = np.maximum(
        curr_high - curr_low,
        np.maximum(
            np.abs(curr_high - prev_close),
            np.abs(curr_low - prev_close),
        ),
    )
    # EMA-style ATR
    atr_series = pd.Series(tr).ewm(span=window, adjust=False).mean()
    return float(atr_series.iloc[-1])


def compute_trailing_stops(
    positions: dict[str, dict],
    prices_df: pd.DataFrame,
    regime: str = "unknown",
    atr_window: int = 14,
    custom_multipliers: dict[str, float] | None = None,
    prior_states: dict[str, TrailingStopState] | None = None,
    vix_level: float | None = None,
    time_stop_warn_bars: int = 30,
    time_stop_close_bars: int = 60,
    current_bar: int = 0,
) -> TrailingStopResult:
    """Compute trailing stop levels for all positions.

    Args:
        positions: Symbol -> dict with keys: entry_price, qty, weight
        prices_df: OHLCV panel with symbol, timestamp, high, low, close
        regime: Current market regime label
        atr_window: ATR computation window
        custom_multipliers: Override regime multipliers
        prior_states: Previous stop states (for high watermark tracking)
        vix_level: Current VIX level for E4 stop-width scaling (None = no scaling)
        time_stop_warn_bars: Bars held before reducing stale position by 50% (E3)
        time_stop_close_bars: Bars held before closing stale position fully (E3)
        current_bar: Current bar index (used for time-stop tracking)

    Returns:
        TrailingStopResult with per-position stop states.
    """
    multipliers = custom_multipliers or _REGIME_MULTIPLIERS
    base_mult = multipliers.get(regime, 2.0)
    # E4: Scale by VIX level
    vix_factor = _vix_multiplier_factor(vix_level)
    mult = base_mult * vix_factor
    prior = prior_states or {}
    result = TrailingStopResult()

    for sym, pos in positions.items():
        entry = float(pos.get("entry_price", 0.0))
        if entry <= 0:
            continue

        # Get price data for symbol
        sym_data = (
            prices_df[prices_df["symbol"] == sym].sort_values("timestamp")
            if "symbol" in prices_df.columns
            else pd.DataFrame()
        )

        if sym_data.empty or "close" not in sym_data.columns:
            continue

        close_vals = sym_data["close"].values.astype(float)
        high_vals = (
            sym_data["high"].values.astype(float)
            if "high" in sym_data.columns
            else close_vals
        )
        low_vals = (
            sym_data["low"].values.astype(float)
            if "low" in sym_data.columns
            else close_vals
        )

        current_price = float(close_vals[-1])
        atr = compute_atr(high_vals, low_vals, close_vals, atr_window)

        # High watermark: max of prior HWM and current price
        prev_hwm = prior[sym].high_watermark if sym in prior else entry
        hwm = max(prev_hwm, current_price)

        # Stop level
        stop = hwm - mult * atr if atr > 0 else hwm * 0.90  # 10% fallback

        # Check trigger — grace period: no stops for first 5 bars after entry
        grace_period = 5
        bars_since_entry = current_bar - (
            prior[sym].entry_bar if sym in prior else current_bar
        )
        triggered = (current_price <= stop) and (bars_since_entry >= grace_period)

        # Gradual de-risking: based on distance from HWM in sigma units
        reduction = 0.0
        if atr > 0 and hwm > 0:
            drawdown_sigma = (hwm - current_price) / atr
            if drawdown_sigma >= 3.0:
                reduction = 0.75
            elif drawdown_sigma >= 2.0:
                reduction = 0.50
            elif drawdown_sigma >= 1.0:
                reduction = 0.25

        # E3: Time-stop — track bars held
        prior_state = prior.get(sym)
        entry_bar = prior_state.entry_bar if prior_state else current_bar
        bars_held = current_bar - entry_bar

        # Time-stop logic: stale position with no meaningful gain
        unrealized_pnl = (current_price / entry - 1.0) if entry > 0 else 0.0
        if not triggered:
            if bars_held >= time_stop_close_bars and unrealized_pnl < 0.0:
                triggered = True
                _log.info(
                    "TIME-STOP CLOSE: %s held %d bars with pnl=%.1f%%",
                    sym,
                    bars_held,
                    unrealized_pnl * 100,
                )
            elif bars_held >= time_stop_warn_bars and unrealized_pnl < 0.05:
                # Reduce by 50% but don't close
                reduction = max(reduction, 0.50)
                _log.info(
                    "TIME-STOP WARN: %s held %d bars with pnl=%.1f%% — reducing 50%%",
                    sym,
                    bars_held,
                    unrealized_pnl * 100,
                )

        state = TrailingStopState(
            symbol=sym,
            entry_price=entry,
            high_watermark=hwm,
            current_atr=round(atr, 4),
            stop_price=round(stop, 4),
            regime=regime,
            multiplier=round(mult, 3),
            triggered=triggered,
            reduction_pct=reduction,
            entry_bar=entry_bar,
            bars_held=bars_held,
        )
        result.stops.append(state)

        if triggered:
            result.triggered_symbols.append(sym)
        if reduction > 0:
            result.reduction_symbols[sym] = reduction

    if result.triggered_symbols:
        _log.warning(
            "TRAILING STOPS TRIGGERED: %s (regime=%s, mult=%.1f)",
            result.triggered_symbols,
            regime,
            mult,
        )
    if result.reduction_symbols:
        _log.info(
            "GRADUAL DE-RISK: %s",
            {s: f"{r:.0%}" for s, r in result.reduction_symbols.items()},
        )

    return result


def apply_stop_reductions_to_weights(
    target_weights: dict[str, float],
    stop_result: TrailingStopResult,
) -> dict[str, float]:
    """Apply trailing stop reductions to target weights.

    Triggered symbols are zeroed. Partially reduced symbols get weight scaled down.
    """
    adjusted = dict(target_weights)

    for sym in stop_result.triggered_symbols:
        if sym in adjusted:
            adjusted[sym] = 0.0

    for sym, reduction in stop_result.reduction_symbols.items():
        if sym in adjusted and sym not in stop_result.triggered_symbols:
            adjusted[sym] *= 1.0 - reduction

    return adjusted


__all__ = [
    "TrailingStopState",
    "TrailingStopResult",
    "compute_atr",
    "compute_trailing_stops",
    "apply_stop_reductions_to_weights",
    "_vix_multiplier_factor",
]
