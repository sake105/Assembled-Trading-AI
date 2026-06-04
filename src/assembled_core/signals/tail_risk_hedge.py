"""Tail Risk Hedge configuration and sizing rules.

From 13_FREE_MODULE.md §13.16.
Systematic OTM put buying sleeve (Universa-style).
2-5% OTM, 30-45 DTE, 2% of portfolio.

Note: Full execution requires Alpaca Options live access + IV-Rank feature.
This module provides the rules engine and sizing logic — not order execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TailHedgeConfig:
    allocation_pct: float = 0.02
    strike_otm_pct: float = 0.05
    dte_target: int = 35
    roll_dte_trigger: int = 15
    roll_delta_trigger: float = -0.05
    max_iv_rank_for_buy: float = 70.0


@dataclass
class TailHedgeOrder:
    action: str
    strike_pct_otm: float
    target_dte: int
    portfolio_allocation_pct: float
    reason: str


def tail_hedge_rules(config: TailHedgeConfig | None = None) -> dict:
    """Return tail hedge configuration as a plain dict.

    This is the authoritative rule spec — pass to execution layer when live.
    """
    cfg = config or TailHedgeConfig()
    return {
        "allocation_pct": cfg.allocation_pct,
        "strike_otm_pct": cfg.strike_otm_pct,
        "dte_target": cfg.dte_target,
        "roll_trigger": {
            "dte_remaining": cfg.roll_dte_trigger,
            "delta_threshold": cfg.roll_delta_trigger,
        },
        "max_iv_rank_for_buy": cfg.max_iv_rank_for_buy,
    }


def should_buy_hedge(
    iv_rank: float,
    portfolio_has_hedge: bool,
    config: TailHedgeConfig | None = None,
) -> TailHedgeOrder | None:
    """Decide whether to initiate a new tail hedge position.

    Args:
        iv_rank: Current IV rank [0-100]. Buy only below max_iv_rank_for_buy.
        portfolio_has_hedge: True if hedge already in place.
        config: Hedge configuration.

    Returns:
        TailHedgeOrder if a new hedge should be initiated, None otherwise.
    """
    cfg = config or TailHedgeConfig()

    if portfolio_has_hedge:
        return None

    if iv_rank > cfg.max_iv_rank_for_buy:
        logger.debug(
            "IV rank %.1f above threshold %.1f — skipping tail hedge buy",
            iv_rank,
            cfg.max_iv_rank_for_buy,
        )
        return None

    return TailHedgeOrder(
        action="buy_put",
        strike_pct_otm=cfg.strike_otm_pct,
        target_dte=cfg.dte_target,
        portfolio_allocation_pct=cfg.allocation_pct,
        reason=f"tail_hedge_initiate iv_rank={iv_rank:.1f}",
    )


def should_roll_hedge(
    current_dte: int,
    current_delta: float,
    config: TailHedgeConfig | None = None,
) -> TailHedgeOrder | None:
    """Decide whether to roll an existing tail hedge position.

    Args:
        current_dte: Days to expiration of current hedge.
        current_delta: Current delta of the put (negative, e.g. -0.03).
        config: Hedge configuration.

    Returns:
        TailHedgeOrder for roll if triggered, None otherwise.
    """
    cfg = config or TailHedgeConfig()

    dte_trigger = current_dte <= cfg.roll_dte_trigger
    delta_trigger = current_delta > cfg.roll_delta_trigger  # delta became less negative

    if dte_trigger:
        return TailHedgeOrder(
            action="roll_put",
            strike_pct_otm=cfg.strike_otm_pct,
            target_dte=cfg.dte_target,
            portfolio_allocation_pct=cfg.allocation_pct,
            reason=f"roll_dte current_dte={current_dte}",
        )

    if delta_trigger:
        return TailHedgeOrder(
            action="roll_put",
            strike_pct_otm=cfg.strike_otm_pct,
            target_dte=cfg.dte_target,
            portfolio_allocation_pct=cfg.allocation_pct,
            reason=f"roll_delta current_delta={current_delta:.3f}",
        )

    return None


def hedge_cost_estimate(
    portfolio_value: float,
    iv: float,
    dte: int,
    strike_otm_pct: float = 0.05,
    allocation_pct: float = 0.02,
) -> float:
    """Rough cost estimate for tail hedge (Black-Scholes ATM proxy).

    Args:
        portfolio_value: Total portfolio NAV in USD.
        iv: Implied volatility (annualized, e.g. 0.20 = 20%).
        dte: Days to expiration.
        strike_otm_pct: How far OTM (e.g. 0.05 = 5%).
        allocation_pct: Fraction of portfolio allocated to hedge.

    Returns:
        Estimated annual cost in USD as fraction of portfolio.
        Rough proxy only — not a real options pricer.
    """
    # Simplified: OTM put premium ~ IV * sqrt(t) * 0.4 (ATM proxy) * OTM_discount
    t = dte / 252
    atm_premium_pct = iv * (t**0.5) * 0.4
    otm_discount = max(0.1, 1.0 - strike_otm_pct * 5)
    put_cost_pct = atm_premium_pct * otm_discount

    # Annual rolls: ~252/dte per year
    annual_rolls = 252 / max(dte, 1)
    annual_cost_pct = put_cost_pct * annual_rolls * allocation_pct

    return float(portfolio_value * annual_cost_pct)


__all__ = [
    "TailHedgeConfig",
    "TailHedgeOrder",
    "tail_hedge_rules",
    "should_buy_hedge",
    "should_roll_hedge",
    "hedge_cost_estimate",
]
