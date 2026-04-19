"""Tail Hedging — Portfolio Insurance and Tail Risk Management (M26).

Implements portfolio protection strategies for extreme downside events:
  1. Protective put overlay sizing (how much to spend on tail protection)
  2. VIX-based hedging triggers (when to activate protection)
  3. Tail risk budget allocation (constant proportion portfolio insurance)
  4. Expected shortfall-aware hedge ratios

The philosophy: spend a small, steady premium (the tail risk budget) to
protect against rare but devastating drawdowns. The cost of protection
should be viewed as insurance, not as performance drag.

Reference:
    Taleb, N.N. (2012). "Antifragile."
    Bhansali, V. (2014). "Tail Risk Hedging."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TailHedgeConfig:
    """Configuration for tail hedging.

    Attributes:
        tail_risk_budget_pct: Annual budget for tail protection as % of portfolio (default: 1.0%).
        vix_hedge_trigger: VIX level that activates hedging (default: 25).
        vix_full_hedge_level: VIX level for maximum hedge (default: 35).
        max_hedge_ratio: Maximum fraction of portfolio to hedge (default: 0.30).
        min_hedge_ratio: Minimum hedge when active (default: 0.05).
        put_otm_pct: How far out-of-the-money for puts (default: 0.05 = 5%).
        rebalance_frequency_days: How often to rebalance the hedge (default: 21).
        use_dynamic_sizing: Whether to size based on current vol regime.
    """

    tail_risk_budget_pct: float = 1.0
    vix_hedge_trigger: float = 25.0
    vix_full_hedge_level: float = 35.0
    max_hedge_ratio: float = 0.30
    min_hedge_ratio: float = 0.05
    put_otm_pct: float = 0.05
    rebalance_frequency_days: int = 21
    use_dynamic_sizing: bool = True


@dataclass
class HedgeRecommendation:
    """Recommended hedge position.

    Attributes:
        hedge_ratio: Fraction of portfolio to hedge (0 to max_hedge_ratio).
        trigger_reason: Why the hedge is recommended.
        estimated_annual_cost_pct: Estimated annual cost of protection.
        notional_to_hedge: Dollar amount to hedge.
        put_strike_pct: Recommended put strike as % of current price.
        urgency: 0-1 urgency of implementing the hedge.
    """

    hedge_ratio: float
    trigger_reason: str
    estimated_annual_cost_pct: float
    notional_to_hedge: float
    put_strike_pct: float
    urgency: float


def compute_hedge_ratio(
    current_vix: float,
    portfolio_vol: float,
    config: TailHedgeConfig | None = None,
) -> float:
    """Compute hedge ratio based on VIX level and portfolio volatility.

    Linearly scales between min and max hedge ratio as VIX moves
    from trigger to full hedge level.

    Args:
        current_vix: Current VIX level.
        portfolio_vol: Annualized portfolio volatility (decimal).
        config: Hedge configuration.

    Returns:
        Hedge ratio (0 to max_hedge_ratio).
    """
    cfg = config or TailHedgeConfig()

    if current_vix < cfg.vix_hedge_trigger:
        return 0.0

    # Linear interpolation between trigger and full hedge
    vix_range = cfg.vix_full_hedge_level - cfg.vix_hedge_trigger
    if vix_range <= 0:
        return cfg.max_hedge_ratio

    progress = (current_vix - cfg.vix_hedge_trigger) / vix_range
    progress = min(max(progress, 0.0), 1.0)

    base_ratio = cfg.min_hedge_ratio + progress * (cfg.max_hedge_ratio - cfg.min_hedge_ratio)

    # Dynamic adjustment: increase hedge if portfolio vol is elevated
    if cfg.use_dynamic_sizing and portfolio_vol > 0.15:
        vol_multiplier = min(portfolio_vol / 0.15, 1.5)
        base_ratio = min(base_ratio * vol_multiplier, cfg.max_hedge_ratio)

    return round(base_ratio, 4)


def compute_put_cost_estimate(
    portfolio_value: float,
    hedge_ratio: float,
    current_vol: float,
    otm_pct: float = 0.05,
    time_to_expiry_years: float = 1.0 / 12.0,
) -> float:
    """Estimate the cost of protective puts using simplified Black-Scholes.

    Uses a normal approximation for OTM put pricing:
        Put ~ S * N(-d2) * K/S - S * N(-d1)
    Simplified to: Put ~ vol * sqrt(T) * f(otm_pct/vol/sqrt(T))

    Args:
        portfolio_value: Total portfolio value.
        hedge_ratio: Fraction of portfolio to hedge.
        current_vol: Annualized implied volatility.
        otm_pct: How far out-of-the-money (0.05 = 5% OTM).
        time_to_expiry_years: Time to expiration.

    Returns:
        Estimated put premium cost in dollars.
    """
    notional = portfolio_value * hedge_ratio
    if notional <= 0 or current_vol <= 0:
        return 0.0

    # Standard-normal CDF via erf — avoids scipy dependency so this function
    # works in the base (non-ml) install profile.
    import math
    def _ncdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    sqrt_t = np.sqrt(time_to_expiry_years)
    d1 = (np.log(1.0 / (1.0 - otm_pct)) + (current_vol**2 / 2) * time_to_expiry_years) / (
        current_vol * sqrt_t
    )
    d2 = d1 - current_vol * sqrt_t

    put_pct = (1.0 - otm_pct) * _ncdf(-d2) - _ncdf(-d1)
    cost = notional * max(put_pct, 0.001)  # minimum 10 bps

    return round(float(cost), 2)


def recommend_hedge(
    portfolio_value: float,
    current_vix: float,
    portfolio_vol: float,
    recent_max_drawdown: float = 0.0,
    config: TailHedgeConfig | None = None,
) -> HedgeRecommendation:
    """Generate a tail hedge recommendation.

    Args:
        portfolio_value: Current portfolio value.
        current_vix: Current VIX level.
        portfolio_vol: Annualized portfolio volatility.
        recent_max_drawdown: Recent maximum drawdown (negative number).
        config: Hedge configuration.

    Returns:
        HedgeRecommendation with sizing and reasoning.
    """
    cfg = config or TailHedgeConfig()
    hedge_ratio = compute_hedge_ratio(current_vix, portfolio_vol, cfg)

    # Determine trigger reason
    reasons = []
    urgency = 0.0

    if current_vix >= cfg.vix_full_hedge_level:
        reasons.append(f"VIX={current_vix:.0f} >= full hedge level {cfg.vix_full_hedge_level}")
        urgency = 1.0
    elif current_vix >= cfg.vix_hedge_trigger:
        reasons.append(f"VIX={current_vix:.0f} >= trigger {cfg.vix_hedge_trigger}")
        urgency = 0.5 + 0.5 * (current_vix - cfg.vix_hedge_trigger) / (
            cfg.vix_full_hedge_level - cfg.vix_hedge_trigger
        )

    if recent_max_drawdown < -0.10:
        reasons.append(f"Recent drawdown {recent_max_drawdown:.1%} exceeds -10%")
        urgency = max(urgency, 0.7)
        hedge_ratio = max(hedge_ratio, cfg.min_hedge_ratio)

    if portfolio_vol > 0.25:
        reasons.append(f"Portfolio vol {portfolio_vol:.1%} elevated")
        urgency = max(urgency, 0.6)

    if not reasons:
        reasons.append("No hedge trigger active")

    notional = portfolio_value * hedge_ratio
    annual_cost = cfg.tail_risk_budget_pct / 100.0

    return HedgeRecommendation(
        hedge_ratio=hedge_ratio,
        trigger_reason="; ".join(reasons),
        estimated_annual_cost_pct=round(annual_cost, 3),
        notional_to_hedge=round(notional, 2),
        put_strike_pct=round(1.0 - cfg.put_otm_pct, 3),
        urgency=round(urgency, 2),
    )


def compute_tail_risk_metrics(
    returns: np.ndarray,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Compute tail risk metrics for hedge sizing decisions.

    Args:
        returns: Array of portfolio returns.
        confidence_level: VaR/ES confidence level.

    Returns:
        Dict with VaR, Expected Shortfall, tail ratio, max_drawdown.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]

    if len(r) < 10:
        return {
            "var_pct": 0.0,
            "expected_shortfall_pct": 0.0,
            "tail_ratio": 1.0,
            "max_drawdown": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
        }

    alpha = 1 - confidence_level
    var = float(np.percentile(r, alpha * 100))
    tail_losses = r[r <= var]
    es = float(tail_losses.mean()) if len(tail_losses) > 0 else var

    # Tail ratio: right tail / left tail
    p95 = np.percentile(r, 95)
    p05 = np.percentile(r, 5)
    tail_ratio = abs(p95 / p05) if abs(p05) > 1e-10 else 1.0

    # Max drawdown
    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(dd.min())

    # Higher moments
    mean = r.mean()
    std = r.std()
    if std > 1e-10:
        skew = float(((r - mean) ** 3).mean() / std**3)
        kurt = float(((r - mean) ** 4).mean() / std**4 - 3)
    else:
        skew = 0.0
        kurt = 0.0

    return {
        "var_pct": round(var * 100, 3),
        "expected_shortfall_pct": round(es * 100, 3),
        "tail_ratio": round(tail_ratio, 3),
        "max_drawdown": round(max_dd * 100, 3),
        "skewness": round(skew, 3),
        "kurtosis": round(kurt, 3),
    }


__all__ = [
    "TailHedgeConfig",
    "HedgeRecommendation",
    "compute_hedge_ratio",
    "compute_put_cost_estimate",
    "recommend_hedge",
    "compute_tail_risk_metrics",
]
