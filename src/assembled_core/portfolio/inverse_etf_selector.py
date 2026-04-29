"""Inverse ETF selector: picks the optimal short instrument per sector.

Decision logic:
  - Short holding < 5 days → inverse ETF (no compounding decay risk)
  - Holding 5-30 days     → 1x inverse ETF preferred, direct short for large caps
  - Holding > 30 days     → direct short strongly preferred (avoid vol decay)
  - Liquidity: avg volume > 500k shares/day required
  - Never recommend 2x/3x unless explicitly permitted
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instrument database
# ---------------------------------------------------------------------------

@dataclass
class InverseETFProfile:
    """Profile of an inverse ETF instrument."""

    symbol: str
    name: str
    sector: str
    leverage: float              # -1.0 = 1x inverse, -2.0 = 2x inverse
    benchmark: str               # Index it tracks inversely
    avg_daily_volume_k: float    # Average daily volume in thousands
    expense_ratio_bps: float     # Annual expense ratio in basis points
    bid_ask_spread_bps: float    # Typical bid-ask spread
    daily_vol_decay_factor: float  # Daily vol decay (annualized, for leveraged only)
    liquidity_tier: str          # "high", "medium", "low"


INVERSE_ETF_PROFILES: dict[str, InverseETFProfile] = {
    # Broad market
    "SH": InverseETFProfile(
        symbol="SH", name="ProShares Short S&P500", sector="BROAD",
        leverage=-1.0, benchmark="S&P500", avg_daily_volume_k=5000,
        expense_ratio_bps=88, bid_ask_spread_bps=2, daily_vol_decay_factor=0.0,
        liquidity_tier="high",
    ),
    "PSQ": InverseETFProfile(
        symbol="PSQ", name="ProShares Short QQQ", sector="TECH",
        leverage=-1.0, benchmark="NASDAQ100", avg_daily_volume_k=3000,
        expense_ratio_bps=95, bid_ask_spread_bps=3, daily_vol_decay_factor=0.0,
        liquidity_tier="high",
    ),
    "RWM": InverseETFProfile(
        symbol="RWM", name="ProShares Short Russell2000", sector="SMALL_CAP",
        leverage=-1.0, benchmark="Russell2000", avg_daily_volume_k=800,
        expense_ratio_bps=95, bid_ask_spread_bps=5, daily_vol_decay_factor=0.0,
        liquidity_tier="medium",
    ),
    # Sector shorts (1x)
    "SEF": InverseETFProfile(
        symbol="SEF", name="ProShares Short Financials", sector="FINANCE",
        leverage=-1.0, benchmark="DJ US Financials", avg_daily_volume_k=200,
        expense_ratio_bps=95, bid_ask_spread_bps=8, daily_vol_decay_factor=0.0,
        liquidity_tier="medium",
    ),
    "SSG": InverseETFProfile(
        symbol="SSG", name="ProShares UltraShort Semiconductors", sector="SEMIS",
        leverage=-2.0, benchmark="DJ US Semiconductors", avg_daily_volume_k=150,
        expense_ratio_bps=95, bid_ask_spread_bps=15, daily_vol_decay_factor=0.35,
        liquidity_tier="low",
    ),
    "MYY": InverseETFProfile(
        symbol="MYY", name="ProShares Short MidCap400", sector="MID_CAP",
        leverage=-1.0, benchmark="S&P MidCap400", avg_daily_volume_k=100,
        expense_ratio_bps=95, bid_ask_spread_bps=10, daily_vol_decay_factor=0.0,
        liquidity_tier="low",
    ),
    "REK": InverseETFProfile(
        symbol="REK", name="ProShares Short Real Estate", sector="REAL_ESTATE",
        leverage=-1.0, benchmark="DJ US Real Estate", avg_daily_volume_k=80,
        expense_ratio_bps=95, bid_ask_spread_bps=12, daily_vol_decay_factor=0.0,
        liquidity_tier="low",
    ),
    "DDG": InverseETFProfile(
        symbol="DDG", name="ProShares Short Oil & Gas", sector="ENERGY",
        leverage=-1.0, benchmark="DJ US Oil & Gas", avg_daily_volume_k=60,
        expense_ratio_bps=95, bid_ask_spread_bps=15, daily_vol_decay_factor=0.0,
        liquidity_tier="low",
    ),
    # Volatility products (special — longs for hedging)
    "VIXY": InverseETFProfile(
        symbol="VIXY", name="ProShares VIX Short-Term Futures ETF", sector="VOL",
        leverage=1.0, benchmark="VIX Futures", avg_daily_volume_k=5000,
        expense_ratio_bps=85, bid_ask_spread_bps=5, daily_vol_decay_factor=0.65,
        liquidity_tier="high",
    ),
}

# Sector → candidate instruments (ordered by preference)
SECTOR_INSTRUMENT_CANDIDATES: dict[str, list[str]] = {
    "BROAD": ["SH"],
    "TECH": ["PSQ", "SH"],
    "SEMIS": ["PSQ", "SH"],       # SSG is 2x, prefer PSQ for semis
    "SMALL_CAP": ["RWM"],
    "MID_CAP": ["MYY", "SH"],
    "FINANCE": ["SEF", "SH"],
    "ENERGY": ["DDG", "SH"],
    "REAL_ESTATE": ["REK", "SH"],
    "CONSUMER": ["SH"],
    "AUTO": ["SH"],
    "MINING": ["MYY", "SH"],
    "SHIPPING": ["SH"],
    "RUSSELL": ["RWM"],
    "VOL": ["VIXY"],
}


class InverseETFSelector:
    """Selects the optimal inverse ETF for a given sector and holding period."""

    def __init__(self, allow_2x: bool = False, allow_3x: bool = False):
        self.allow_2x = allow_2x
        self.allow_3x = allow_3x

    def select_best_short_instrument(
        self,
        sector: str,
        severity: float,
        holding_period_days: int = 14,
        liquidity_min_k: float = 200.0,
    ) -> str | None:
        """Select the best short instrument for a sector and holding period.

        Args:
            sector: Target sector (TECH, SEMIS, FINANCE, etc.)
            severity: Expected crash severity 0-1
            holding_period_days: Expected holding period in days
            liquidity_min_k: Minimum daily volume in thousands

        Returns:
            Symbol string, or None if no suitable instrument found.
        """
        candidates = SECTOR_INSTRUMENT_CANDIDATES.get(sector, SECTOR_INSTRUMENT_CANDIDATES["BROAD"])

        for symbol in candidates:
            profile = INVERSE_ETF_PROFILES.get(symbol)
            if profile is None:
                continue

            # Skip if liquidity too low
            if profile.avg_daily_volume_k < liquidity_min_k:
                logger.debug("[InverseETFSelector] %s skipped: low volume %dk", symbol, profile.avg_daily_volume_k)
                continue

            # Skip 2x/3x unless allowed
            if abs(profile.leverage) >= 2.0 and not self.allow_2x:
                continue
            if abs(profile.leverage) >= 3.0 and not self.allow_3x:
                continue

            # For long holding periods, warn about vol decay
            if holding_period_days > 5 and profile.daily_vol_decay_factor > 0.1:
                logger.info(
                    "[InverseETFSelector] %s has vol decay %.0f%% for %dd hold — "
                    "consider direct short",
                    symbol, profile.daily_vol_decay_factor * 100, holding_period_days,
                )
                # Still allow, but prefer non-decaying alternative
                continue

            return symbol

        # Fallback to broad market short
        if sector != "BROAD" and "SH" not in candidates:
            return "SH"

        return candidates[0] if candidates else None

    def compute_decay_adjusted_return(
        self,
        symbol: str,
        annualized_return: float,
        annual_volatility: float,
        holding_days: int,
    ) -> float:
        """Compute expected return after vol decay for leveraged ETFs.

        For non-leveraged (leverage = -1.0): no decay, return = simple inverse.
        For leveraged: apply daily rebalancing decay formula.

        Formula (approximation):
            Leveraged return ≈ L * R - (L² - L) / 2 * σ² * T
        where L = |leverage|, R = return, σ = daily vol, T = trading days.
        """
        profile = INVERSE_ETF_PROFILES.get(symbol)
        if profile is None:
            return annualized_return

        leverage = abs(profile.leverage)
        if leverage <= 1.0:
            return annualized_return  # No decay

        # Daily variance
        daily_vol = annual_volatility / (252 ** 0.5)
        daily_var = daily_vol ** 2

        # Compounding loss over holding period
        decay = (leverage ** 2 - leverage) / 2 * daily_var * holding_days

        # Gross leveraged return
        daily_underlying = annualized_return / 252
        leveraged_daily = leverage * daily_underlying
        net_return = leveraged_daily * holding_days - decay

        logger.debug(
            "[InverseETFSelector] %s decay=%.3f gross_return=%.3f net=%.3f (hold=%dd)",
            symbol, decay, leveraged_daily * holding_days, net_return, holding_days,
        )
        return net_return

    def rank_instruments_by_efficiency(
        self,
        sector: str,
        holding_days: int = 14,
        crash_magnitude: float = 0.10,
    ) -> list[dict]:
        """Rank all available instruments for a sector by efficiency score.

        Efficiency = expected return / (expense_ratio + bid_ask_spread + decay_cost)
        """
        candidates = SECTOR_INSTRUMENT_CANDIDATES.get(sector, ["SH"])
        results = []

        for symbol in candidates:
            profile = INVERSE_ETF_PROFILES.get(symbol)
            if profile is None:
                continue
            if abs(profile.leverage) >= 2.0 and not self.allow_2x:
                continue

            # Expected return (simple approximation)
            expected_return = abs(profile.leverage) * crash_magnitude

            # Costs
            annual_cost = (profile.expense_ratio_bps + profile.bid_ask_spread_bps) / 10000
            period_cost = annual_cost * holding_days / 252

            # Vol decay cost (for leveraged)
            decay_cost = profile.daily_vol_decay_factor * holding_days / 252

            total_cost = period_cost + decay_cost
            efficiency = expected_return / max(total_cost, 0.001)

            results.append({
                "symbol": symbol,
                "expected_return": round(expected_return, 4),
                "total_cost": round(total_cost, 5),
                "efficiency_score": round(efficiency, 2),
                "leverage": profile.leverage,
                "liquidity_tier": profile.liquidity_tier,
            })

        return sorted(results, key=lambda x: x["efficiency_score"], reverse=True)
