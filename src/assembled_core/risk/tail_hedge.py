"""Tail Risk Hedging — Collar Strategy and Put Spread Overlay (M38 Task 38.1).

Implements systematic tail-risk hedging:
1. Collar strategy: buy protective puts, sell covered calls
2. Put spread overlay for cost-efficient protection
3. Dynamic hedge ratio based on regime and VIX level
4. Cost-benefit analysis of hedge vs drawdown protection

Reference:
    Bhansali (2014) "Tail Risk Hedging"
    Israelov & Nielsen (2015) "Still Not Cheap: Portfolio Protection in Calm Markets"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CollarConfig:
    """Collar strategy configuration."""
    put_delta: float = -0.25         # OTM put delta target
    call_delta: float = 0.25         # OTM call delta target
    put_otm_pct: float = 0.05       # 5% OTM for puts
    call_otm_pct: float = 0.05      # 5% OTM for calls
    hedge_ratio: float = 1.0        # Fraction of portfolio hedged
    rebalance_days: int = 30        # Rebalance frequency
    max_hedge_cost_bps: float = 50  # Max acceptable net cost per year


@dataclass
class TailHedgeResult:
    """Result of tail hedge computation."""
    hedge_type: str                 # "collar", "put_spread", "dynamic"
    put_strike_pct: float           # Put strike as % of spot
    call_strike_pct: float          # Call strike as % of spot (collar)
    net_premium_bps: float          # Net premium cost in bps
    max_loss_pct: float             # Max portfolio loss with hedge
    upside_cap_pct: float           # Max upside with collar
    hedge_ratio: float              # Fraction of portfolio hedged
    breakeven_dd_pct: float         # Drawdown at which hedge pays for itself


def estimate_option_premium(
    spot: float,
    strike: float,
    vol: float,
    days_to_expiry: int = 30,
    risk_free_rate: float = 0.05,
    option_type: str = "put",
) -> float:
    """Estimate option premium using Black-Scholes approximation.

    Args:
        spot: Current price.
        strike: Strike price.
        vol: Annualized volatility.
        days_to_expiry: Days until expiration.
        risk_free_rate: Risk-free rate.
        option_type: "put" or "call".

    Returns:
        Estimated premium as fraction of spot.
    """
    T = days_to_expiry / 252
    if T <= 0 or vol <= 0:
        return 0.0

    try:
        from scipy.stats import norm
        d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)

        if option_type == "call":
            price = spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * T) * norm.cdf(d2)
        else:
            price = strike * np.exp(-risk_free_rate * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)

        return max(0, price / spot)
    except ImportError:
        # Simple approximation: premium ~ vol * sqrt(T) * moneyness factor
        moneyness = abs(spot - strike) / spot
        intrinsic = max(0, (strike - spot) / spot if option_type == "put" else (spot - strike) / spot)
        time_value = 0.4 * vol * np.sqrt(T) * np.exp(-moneyness / (vol * np.sqrt(T) + 1e-8))
        return max(intrinsic, intrinsic + time_value)


def compute_collar(
    portfolio_value: float,
    portfolio_vol: float,
    vix_level: float = 20.0,
    config: CollarConfig | None = None,
) -> TailHedgeResult:
    """Compute collar strategy parameters.

    Buy OTM puts for downside protection, sell OTM calls to finance.

    Args:
        portfolio_value: Current portfolio value.
        portfolio_vol: Portfolio annualized volatility.
        vix_level: Current VIX level for implied vol.
        config: Collar configuration.

    Returns:
        TailHedgeResult with collar parameters.
    """
    cfg = config or CollarConfig()
    implied_vol = vix_level / 100  # VIX is annualized % vol

    # Strike levels
    put_strike_pct = 1.0 - cfg.put_otm_pct
    call_strike_pct = 1.0 + cfg.call_otm_pct

    # Estimate premiums
    put_premium = estimate_option_premium(
        1.0, put_strike_pct, implied_vol, cfg.rebalance_days, option_type="put",
    )
    call_premium = estimate_option_premium(
        1.0, call_strike_pct, implied_vol, cfg.rebalance_days, option_type="call",
    )

    # Net premium (negative = net credit from collar)
    net_premium = (put_premium - call_premium) * cfg.hedge_ratio
    net_premium_annual_bps = net_premium * (252 / cfg.rebalance_days) * 10000

    # Max loss: down to put strike
    max_loss = cfg.put_otm_pct * cfg.hedge_ratio + (1 - cfg.hedge_ratio)  # Unhedged portion unlimited
    max_loss_pct = cfg.put_otm_pct  # With 100% hedge ratio

    # Upside cap: call strike
    upside_cap = cfg.call_otm_pct

    # Breakeven: drawdown where hedge cost is recovered
    if net_premium > 0:
        breakeven = net_premium / cfg.hedge_ratio if cfg.hedge_ratio > 0 else float("inf")
    else:
        breakeven = 0.0  # Net credit collar always beneficial

    logger.info(
        "[TailHedge] Collar: put@%.1f%% OTM, call@%.1f%% OTM, net=%.1f bps/yr, "
        "max_loss=%.1f%%, upside_cap=%.1f%%",
        cfg.put_otm_pct * 100, cfg.call_otm_pct * 100,
        net_premium_annual_bps, max_loss_pct * 100, upside_cap * 100,
    )

    return TailHedgeResult(
        hedge_type="collar",
        put_strike_pct=round(put_strike_pct, 4),
        call_strike_pct=round(call_strike_pct, 4),
        net_premium_bps=round(net_premium_annual_bps, 2),
        max_loss_pct=round(max_loss_pct, 4),
        upside_cap_pct=round(upside_cap, 4),
        hedge_ratio=cfg.hedge_ratio,
        breakeven_dd_pct=round(breakeven, 4),
    )


def compute_put_spread(
    portfolio_value: float,
    portfolio_vol: float,
    vix_level: float = 20.0,
    near_otm_pct: float = 0.05,
    far_otm_pct: float = 0.15,
    rebalance_days: int = 30,
) -> TailHedgeResult:
    """Compute put spread overlay for cost-efficient tail protection.

    Buy near-OTM put, sell far-OTM put to reduce cost.
    Protects between near and far strike levels.

    Args:
        portfolio_value: Current portfolio value.
        portfolio_vol: Portfolio annualized volatility.
        vix_level: Current VIX for implied vol.
        near_otm_pct: Near put OTM percentage.
        far_otm_pct: Far put OTM percentage.
        rebalance_days: Rebalance frequency.

    Returns:
        TailHedgeResult.
    """
    implied_vol = vix_level / 100
    near_strike = 1.0 - near_otm_pct
    far_strike = 1.0 - far_otm_pct

    near_premium = estimate_option_premium(1.0, near_strike, implied_vol, rebalance_days, option_type="put")
    far_premium = estimate_option_premium(1.0, far_strike, implied_vol, rebalance_days, option_type="put")

    net_premium = near_premium - far_premium
    net_premium_annual_bps = net_premium * (252 / rebalance_days) * 10000

    # Max protection: between near and far strikes
    max_protection = far_otm_pct - near_otm_pct

    logger.info(
        "[TailHedge] Put spread: buy %.0f%% / sell %.0f%% OTM, cost=%.1f bps/yr, "
        "protection=%.0f%%–%.0f%%",
        near_otm_pct * 100, far_otm_pct * 100, net_premium_annual_bps,
        near_otm_pct * 100, far_otm_pct * 100,
    )

    return TailHedgeResult(
        hedge_type="put_spread",
        put_strike_pct=round(near_strike, 4),
        call_strike_pct=round(far_strike, 4),  # Used as far put strike
        net_premium_bps=round(net_premium_annual_bps, 2),
        max_loss_pct=round(near_otm_pct, 4),
        upside_cap_pct=1.0,  # No upside cap
        hedge_ratio=1.0,
        breakeven_dd_pct=round(near_otm_pct + net_premium, 4),
    )


def dynamic_hedge_ratio(
    vix_level: float,
    regime: str = "normal",
    portfolio_dd: float = 0.0,
    base_ratio: float = 0.5,
) -> float:
    """Compute dynamic hedge ratio based on market conditions.

    Higher hedge in high-vol / crisis regimes, lower in calm markets.

    Args:
        vix_level: Current VIX level.
        regime: Market regime ("crisis", "stressed", "normal", "calm").
        portfolio_dd: Current portfolio drawdown (negative).
        base_ratio: Base hedge ratio.

    Returns:
        Adjusted hedge ratio [0, 1].
    """
    regime_multipliers = {
        "crisis": 1.5,
        "stressed": 1.2,
        "normal": 1.0,
        "calm": 0.7,
    }

    # VIX adjustment: hedge more when VIX elevated
    vix_mult = np.clip(vix_level / 20.0, 0.5, 2.0)

    # Drawdown adjustment: hedge more when already in drawdown
    dd_mult = 1.0 + max(0, abs(portfolio_dd) - 0.05) * 5  # Scale up after 5% DD

    regime_mult = regime_multipliers.get(regime, 1.0)
    ratio = base_ratio * vix_mult * regime_mult * dd_mult
    ratio = float(np.clip(ratio, 0.0, 1.0))

    return round(ratio, 4)


__all__ = [
    "CollarConfig",
    "TailHedgeResult",
    "estimate_option_premium",
    "compute_collar",
    "compute_put_spread",
    "dynamic_hedge_ratio",
]
