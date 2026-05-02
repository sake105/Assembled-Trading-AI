"""Regime-adaptive multi-asset allocator (M16).

Detects market regime (bull/sideways/bear/crisis) from VIX + breadth + SPY trend.
Allocates across equity, bonds (TLT/IEF), gold (GLD), and inverse ETFs (SH).

Regime detection rules:
  bull:      VIX < 20 AND breadth > 0.60 AND SPY > 200d MA
  sideways:  VIX 20-28 OR breadth 0.40-0.60 OR SPY within 5% of 200d MA
  bear:      VIX 28-38 OR breadth < 0.40 OR SPY < 200d MA * 0.95
  crisis:    VIX > 38 OR breadth < 0.25

Hysteresis: regime must persist for 3 consecutive bars before switching.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

_log = logging.getLogger(__name__)


@dataclass
class RegimeAllocation:
    """Target allocation fractions by regime."""
    # equity bucket weight (applied to equity signal weights)
    equity: float
    # individual macro instrument targets
    tlt: float = 0.0
    ief: float = 0.0
    gld: float = 0.0
    sh: float = 0.0  # SH (S&P 500 inverse) — bear/crisis hedge

    def as_dict(self) -> dict[str, float]:
        return {
            "equity": self.equity,
            "TLT": self.tlt,
            "IEF": self.ief,
            "GLD": self.gld,
            "SH": self.sh,
        }


REGIME_ALLOCATIONS: dict[str, RegimeAllocation] = {
    "bull":      RegimeAllocation(equity=0.85, tlt=0.08, ief=0.02, gld=0.05, sh=0.00),
    "sideways":  RegimeAllocation(equity=0.60, tlt=0.20, ief=0.05, gld=0.15, sh=0.00),
    "bear":      RegimeAllocation(equity=0.30, tlt=0.35, ief=0.05, gld=0.20, sh=0.10),
    "crisis":    RegimeAllocation(equity=0.10, tlt=0.35, ief=0.05, gld=0.30, sh=0.20),
}


@dataclass
class RegimeDetectorConfig:
    vix_bull_threshold: float = 20.0
    vix_sideways_threshold: float = 28.0
    vix_bear_threshold: float = 38.0
    breadth_bull_threshold: float = 0.60
    breadth_bear_threshold: float = 0.40
    breadth_crisis_threshold: float = 0.25
    spy_ma_window: int = 200
    spy_bear_pct: float = 0.95   # SPY < 200d MA * this → bear signal
    hysteresis_bars: int = 3     # bars regime must persist before switching


class RegimeDetector:
    """Stateful regime detector with hysteresis."""

    def __init__(self, config: RegimeDetectorConfig | None = None) -> None:
        self.config = config or RegimeDetectorConfig()
        self._current_regime: str = "bull"
        self._candidate_regime: str = "bull"
        self._candidate_count: int = 0
        self._spy_prices: list[float] = []

    def update(
        self,
        vix: float | None,
        breadth: float | None,
        spy_close: float | None,
    ) -> str:
        """Update detector with today's readings. Returns current regime."""
        cfg = self.config

        # Track SPY for MA computation
        if spy_close is not None and not np.isnan(spy_close):
            self._spy_prices.append(spy_close)

        # Compute SPY 200d MA position
        spy_above_ma = True  # default: assume bull
        if len(self._spy_prices) >= cfg.spy_ma_window:
            ma200 = np.mean(self._spy_prices[-cfg.spy_ma_window:])
            spy_above_ma = self._spy_prices[-1] >= ma200 * cfg.spy_bear_pct

        # Determine raw regime
        raw = self._classify(vix, breadth, spy_above_ma)

        # Apply hysteresis
        if raw == self._candidate_regime:
            self._candidate_count += 1
        else:
            self._candidate_regime = raw
            self._candidate_count = 1

        if self._candidate_count >= cfg.hysteresis_bars:
            if self._current_regime != self._candidate_regime:
                _log.info(
                    "REGIME CHANGE: %s → %s (VIX=%.1f, breadth=%.2f)",
                    self._current_regime, self._candidate_regime,
                    vix or 0, breadth or 0,
                )
            self._current_regime = self._candidate_regime

        return self._current_regime

    def _classify(self, vix: float | None, breadth: float | None, spy_above_ma: bool) -> str:
        cfg = self.config

        # Safe defaults
        v = vix if (vix is not None and not np.isnan(vix)) else 20.0
        b = breadth if (breadth is not None and not np.isnan(breadth)) else 0.55

        # Crisis: VIX is PRIMARY driver (extreme fear required); breadth alone is insufficient
        # Requires both: extremely high VIX AND very low breadth
        if v > cfg.vix_bear_threshold and b < cfg.breadth_crisis_threshold:
            return "crisis"
        # VIX alone can trigger crisis at extreme levels (>45)
        if v > 45:
            return "crisis"

        # Bear: high VIX (>28) OR (weak breadth AND SPY below 200d MA)
        # Note: breadth alone (without VIX confirmation) does NOT trigger bear
        if v > cfg.vix_sideways_threshold:
            return "bear"
        if b < cfg.breadth_bear_threshold and not spy_above_ma:
            return "bear"

        # Bull: low VIX AND decent breadth AND SPY above MA
        if v < cfg.vix_bull_threshold and b > cfg.breadth_bull_threshold and spy_above_ma:
            return "bull"

        # Default: sideways
        return "sideways"

    @property
    def current_regime(self) -> str:
        return self._current_regime


def allocate_by_regime(
    regime: str,
    equity_weights: dict[str, float],
    sector_weights: dict[str, float] | None = None,
    custom_allocations: dict[str, RegimeAllocation] | None = None,
) -> dict[str, float]:
    """Build final portfolio weights given regime and sub-strategy weights.

    Args:
        regime: Current market regime string.
        equity_weights: Individual stock weights (sum should be ~1.0).
        sector_weights: Sector ETF weights (optional, sum should be ~1.0).
        custom_allocations: Override default REGIME_ALLOCATIONS if provided.

    Returns:
        Combined weight dict: {symbol: final_weight} normalized to sum ≤ 1.0.
    """
    alloc_map = custom_allocations or REGIME_ALLOCATIONS
    allocation = alloc_map.get(regime, REGIME_ALLOCATIONS["bull"])

    final: dict[str, float] = {}

    # Scale equity positions by regime equity bucket
    total_equity_raw = sum(abs(w) for w in equity_weights.values()) or 1.0
    for sym, w in equity_weights.items():
        final[sym] = w / total_equity_raw * allocation.equity

    # Sector ETF overlay (if provided, use as fraction of equity bucket)
    if sector_weights:
        total_sector_raw = sum(abs(w) for w in sector_weights.values()) or 1.0
        # Sector ETFs get 20% of equity bucket in all regimes
        sector_fraction = min(0.20, allocation.equity * 0.25)
        for sym, w in sector_weights.items():
            final[sym] = final.get(sym, 0.0) + w / total_sector_raw * sector_fraction

    # Macro instruments — fixed targets from regime allocation
    macro_map = {
        "TLT": allocation.tlt,
        "IEF": allocation.ief,
        "GLD": allocation.gld,
        "SH":  allocation.sh,
    }
    for sym, target_w in macro_map.items():
        if target_w > 0:
            final[sym] = final.get(sym, 0.0) + target_w

    # Normalize: cap gross exposure at 1.3 (allow modest leverage for L/S)
    gross = sum(abs(w) for w in final.values())
    if gross > 1.3:
        scale = 1.3 / gross
        final = {s: w * scale for s, w in final.items()}

    return final


def allocate_by_regime_with_strategy_weights(
    regime: str,
    strategy_returns: dict[str, list[float]],
    equity_weights: dict[str, float],
    sector_weights: dict[str, float] | None = None,
    custom_allocations: dict[str, "RegimeAllocation"] | None = None,
    target_vol: float = 0.15,
) -> dict[str, float]:
    """Allocate using regime-conditional strategy weights as the equity sub-layer.

    Combines the macro regime allocation (equity/bonds/gold buckets) with
    Sharpe-proportional strategy weights from RegimeAllocator. Falls back
    to flat equity_weights if regime_conditional_allocator is unavailable or
    has insufficient history.

    Args:
        regime: Current regime label (bull/sideways/bear/crisis).
        strategy_returns: {strategy_name: [daily_returns]} used to compute regime Sharpes.
        equity_weights: Base per-symbol weights (used as fallback).
        sector_weights: Optional sector ETF overlay.
        custom_allocations: Override REGIME_ALLOCATIONS if provided.
        target_vol: Target portfolio vol for vol-scaling inside regime allocator.

    Returns:
        Combined weight dict {symbol: final_weight}.
    """
    scaled_equity: dict[str, float] = equity_weights
    try:
        if strategy_returns:
            from src.assembled_core.portfolio.regime_conditional_allocator import (
                build_regime_allocator,
            )
            # Build a synthetic regime series from the regime label (all bars = current regime)
            n_bars = max(len(v) for v in strategy_returns.values()) if strategy_returns else 1
            import pandas as _pd
            regime_series = _pd.Series([regime] * n_bars)
            allocator = build_regime_allocator(
                strategy_returns=strategy_returns,
                regime_series=regime_series,
                target_vol=target_vol,
            )
            result = allocator.allocate(regime)
            if result.weights:
                # Use Sharpe-proportional weights as the equity bucket weights
                scaled_equity = dict(result.weights)
    except Exception as _exc:
        _log.debug("[allocate_by_regime_with_strategy_weights] regime allocator failed: %s", _exc)

    return allocate_by_regime(
        regime=regime,
        equity_weights=scaled_equity,
        sector_weights=sector_weights,
        custom_allocations=custom_allocations,
    )


def allocate_with_hrp(
    returns: "pd.DataFrame",
    regime: str = "bull",
    sector_weights: dict[str, float] | None = None,
    custom_allocations: dict[str, RegimeAllocation] | None = None,
    max_weight: float = 0.20,
    current_weights: dict[str, float] | None = None,
    max_turnover: float | None = None,
) -> dict[str, float]:
    """HRP equity weights + regime macro overlay.

    Replaces the rule-based equity weighting inside ``allocate_by_regime``
    with Lopez de Prado Hierarchical Risk Parity.  The macro instruments
    (TLT/IEF/GLD/SH) are still allocated by regime as usual.

    Falls back to equal-weight equity allocation if scipy is unavailable or
    returns has insufficient history.

    Args:
        returns: Wide-format DataFrame (dates × symbols) of daily equity returns.
            Must have at least 30 rows after dropping NaNs.
        regime: Current market regime (bull/sideways/bear/crisis).
        sector_weights: Optional sector ETF overlay (passed through to regime allocator).
        custom_allocations: Override REGIME_ALLOCATIONS if provided.
        max_weight: Maximum weight cap per equity asset before normalization.
        current_weights: If provided, applies turnover control overlay.
        max_turnover: Maximum allowed one-period weight change (turnover control).
            Only applies when ``current_weights`` is also provided.

    Returns:
        Combined weight dict {symbol: final_weight} with equity + macro instruments.
    """
    try:
        import pandas as pd
        from src.assembled_core.portfolio.hierarchical_risk_parity import (
            compute_hrp_weights,
            hrp_with_turnover_control,
        )

        if current_weights and max_turnover is not None:
            hrp_weights = hrp_with_turnover_control(
                returns=returns,
                current_weights=current_weights,
                max_turnover=max_turnover,
                max_weight=max_weight,
            )
        else:
            hrp_weights = compute_hrp_weights(returns, max_weight=max_weight)

        if not hrp_weights:
            # Fallback: equal weight across all symbols in returns
            syms = list(returns.columns)
            hrp_weights = {s: 1.0 / len(syms) for s in syms} if syms else {}

        _log.debug(
            "[HRP] weights computed: n=%d, max=%.4f, min=%.4f",
            len(hrp_weights),
            max(hrp_weights.values()) if hrp_weights else 0.0,
            min(hrp_weights.values()) if hrp_weights else 0.0,
        )
    except Exception as exc:
        _log.debug("[HRP] computation failed, using equal weight: %s", exc)
        syms = list(returns.columns) if hasattr(returns, "columns") else []
        hrp_weights = {s: 1.0 / len(syms) for s in syms} if syms else {}

    return allocate_by_regime(
        regime=regime,
        equity_weights=hrp_weights,
        sector_weights=sector_weights,
        custom_allocations=custom_allocations,
    )


__all__ = [
    "RegimeAllocation",
    "RegimeDetectorConfig",
    "RegimeDetector",
    "REGIME_ALLOCATIONS",
    "allocate_by_regime",
    "allocate_by_regime_with_strategy_weights",
    "allocate_with_hrp",
]
