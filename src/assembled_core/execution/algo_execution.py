"""Algorithmic Execution: TWAP and VWAP Order Schedulers.

Provides institutional-grade order slicing algorithms to minimize market impact
for large orders. Rather than executing as a single market order, these
schedulers split orders into smaller slices spread over time or volume.

Classes:
    SlicedOrder      — dataclass representing a single execution slice
    TWAPScheduler    — Time-Weighted Average Price scheduler
    VWAPScheduler    — Volume-Weighted Average Price scheduler
    ImplementationShortfallModel — estimates execution cost via Kyle lambda proxy

Usage:
    from src.assembled_core.execution.algo_execution import TWAPScheduler, VWAPScheduler

    twap = TWAPScheduler(n_slices=10)
    slices = twap.schedule(symbol="AAPL", total_qty=1000, side="BUY",
                            start_time=open_time, end_time=close_time)

    vwap = VWAPScheduler(n_slices=10)
    slices = vwap.schedule(symbol="AAPL", total_qty=1000, side="BUY",
                            volume_profile=hourly_volume_df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SlicedOrder:
    """A single execution slice of a larger parent order.

    Attributes:
        symbol: Instrument symbol.
        side: "BUY" or "SELL".
        quantity: Quantity for this slice.
        scheduled_time: Target execution time for this slice.
        slice_idx: Index of this slice in the schedule (0-based).
        total_slices: Total number of slices in the parent order.
        algo: Algorithm name ("TWAP" or "VWAP").
        parent_order_id: Optional parent order identifier.
    """

    symbol: str
    side: str
    quantity: float
    scheduled_time: datetime
    slice_idx: int
    total_slices: int
    algo: str = "TWAP"
    parent_order_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "scheduled_time": self.scheduled_time.isoformat(),
            "slice_idx": self.slice_idx,
            "total_slices": self.total_slices,
            "algo": self.algo,
            "parent_order_id": self.parent_order_id,
        }


class TWAPScheduler:
    """Time-Weighted Average Price execution scheduler.

    Divides the total order quantity into equal slices distributed uniformly
    across the specified time window. Execution at each interval aims to
    achieve the TWAP benchmark.

    Attributes:
        n_slices: Number of execution slices (default: 10).
        randomize: Add ±10% random variation to each slice qty to avoid
                   scheduled order detection by other market participants (default: True).
    """

    def __init__(self, n_slices: int = 10, randomize: bool = True) -> None:
        self.n_slices = n_slices
        self.randomize = randomize

    def schedule(
        self,
        symbol: str,
        total_qty: float,
        side: str,
        start_time: datetime,
        end_time: datetime,
        parent_order_id: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> list[SlicedOrder]:
        """Generate TWAP execution schedule.

        Args:
            symbol: Instrument symbol.
            total_qty: Total order quantity (positive float).
            side: "BUY" or "SELL".
            start_time: Scheduled start of execution window.
            end_time: Scheduled end of execution window.
            parent_order_id: Optional identifier for tracking.
            random_seed: Random seed for randomization (default: None).

        Returns:
            List of SlicedOrder objects, one per slice.
        """
        if total_qty <= 0:
            raise ValueError("total_qty must be positive")
        if end_time <= start_time:
            raise ValueError("end_time must be after start_time")

        n = min(self.n_slices, int(total_qty))  # can't have more slices than units
        n = max(n, 1)

        # Time interval between slices
        window = (end_time - start_time).total_seconds()
        interval = window / n

        # Base quantity per slice
        base_qty = total_qty / n

        if self.randomize:
            rng = np.random.default_rng(random_seed)
            # Random multipliers around 1.0 (±10%), normalize to preserve total
            mults = 1.0 + rng.uniform(-0.1, 0.1, n)
            mults /= mults.sum()
            quantities = mults * total_qty
        else:
            quantities = np.full(n, base_qty)

        # Ensure quantities sum exactly to total_qty
        quantities[-1] = total_qty - quantities[:-1].sum()

        slices = []
        for i in range(n):
            t = start_time + timedelta(seconds=interval * (i + 0.5))
            slices.append(
                SlicedOrder(
                    symbol=symbol,
                    side=side,
                    quantity=float(quantities[i]),
                    scheduled_time=t,
                    slice_idx=i,
                    total_slices=n,
                    algo="TWAP",
                    parent_order_id=parent_order_id,
                )
            )

        logger.debug(
            "[TWAP] %s %s %.0f in %d slices (%s → %s)",
            side,
            symbol,
            total_qty,
            n,
            start_time,
            end_time,
        )
        return slices


class VWAPScheduler:
    """Volume-Weighted Average Price execution scheduler.

    Distributes order slices proportionally to the historical volume profile,
    attempting to participate at each interval in proportion to its typical
    share of daily volume.

    Attributes:
        n_slices: Number of execution slices (default: 10).
        participation_rate: Maximum fraction of interval volume to trade (default: 0.10 = 10%).
    """

    def __init__(self, n_slices: int = 10, participation_rate: float = 0.10) -> None:
        self.n_slices = n_slices
        self.participation_rate = participation_rate

    def schedule(
        self,
        symbol: str,
        total_qty: float,
        side: str,
        start_time: datetime,
        end_time: datetime,
        volume_profile: Optional[pd.DataFrame] = None,
        parent_order_id: Optional[str] = None,
    ) -> list[SlicedOrder]:
        """Generate VWAP execution schedule.

        Args:
            symbol: Instrument symbol.
            total_qty: Total order quantity (positive float).
            side: "BUY" or "SELL".
            start_time: Execution window start.
            end_time: Execution window end.
            volume_profile: Optional DataFrame with columns 'time' (datetime) and
                'volume' (typical volume at that time interval). If None, falls back
                to TWAP-style equal distribution.
            parent_order_id: Optional identifier for tracking.

        Returns:
            List of SlicedOrder objects with volume-proportional quantities.
        """
        if total_qty <= 0:
            raise ValueError("total_qty must be positive")

        n = max(1, min(self.n_slices, int(total_qty)))

        if volume_profile is None or volume_profile.empty:
            logger.warning(
                "[VWAP] No volume profile — falling back to equal distribution"
            )
            twap = TWAPScheduler(n_slices=n, randomize=False)
            slices = twap.schedule(
                symbol, total_qty, side, start_time, end_time, parent_order_id
            )
            for s in slices:
                s.algo = "VWAP"
            return slices

        # Filter volume profile to execution window
        vol = volume_profile.copy()
        time_col = "time" if "time" in vol.columns else vol.columns[0]
        vol_col = "volume" if "volume" in vol.columns else vol.columns[1]
        vol[time_col] = pd.to_datetime(vol[time_col], format="mixed")
        mask = (vol[time_col] >= start_time) & (vol[time_col] <= end_time)
        vol = vol[mask].sort_values(time_col)

        if vol.empty:
            twap = TWAPScheduler(n_slices=n, randomize=False)
            slices = twap.schedule(
                symbol, total_qty, side, start_time, end_time, parent_order_id
            )
            for s in slices:
                s.algo = "VWAP"
            return slices

        # Resample to n_slices buckets
        vol_values = vol[vol_col].values.astype(float)
        times = vol[time_col].values

        # Bucket the volume into n equal-time buckets
        bucket_size = max(1, len(vol_values) // n)
        buckets = []
        for i in range(n):
            start_i = i * bucket_size
            end_i = start_i + bucket_size if i < n - 1 else len(vol_values)
            buckets.append(
                (
                    pd.Timestamp(times[start_i]).to_pydatetime(),
                    float(vol_values[start_i:end_i].sum()),
                )
            )

        total_bucket_vol = sum(b[1] for b in buckets)
        if total_bucket_vol <= 0:
            total_bucket_vol = 1.0

        slices = []
        remaining = total_qty
        for i, (bucket_time, bucket_vol) in enumerate(buckets):
            prop = bucket_vol / total_bucket_vol
            if i < n - 1:
                qty = total_qty * prop
                remaining -= qty
            else:
                qty = remaining  # last slice absorbs rounding

            slices.append(
                SlicedOrder(
                    symbol=symbol,
                    side=side,
                    quantity=max(0.0, qty),
                    scheduled_time=bucket_time,
                    slice_idx=i,
                    total_slices=n,
                    algo="VWAP",
                    parent_order_id=parent_order_id,
                )
            )

        logger.debug(
            "[VWAP] %s %s %.0f in %d volume-proportional slices",
            side,
            symbol,
            total_qty,
            n,
        )
        return slices


@dataclass
class ImplementationShortfallModel:
    """Estimates execution cost using Kyle lambda market-impact proxy.

    Implementation Shortfall (IS) = Arrival Price - VWAP + opportunity cost.
    This simplified model uses:
        IS ≈ market_impact + timing_risk + opportunity_cost
        market_impact ≈ kyle_lambda * quantity / ADV

    Attributes:
        kyle_lambda: Market impact coefficient (default: 0.1).
            Estimated from kyle_lambda_proxy feature in ta_liquidity_vol_factors.
        timing_risk_pct: Daily volatility fraction as timing risk (default: 0.5).
        opportunity_cost_bps: Fixed opportunity cost in basis points (default: 5).
    """

    kyle_lambda: float = 0.1
    timing_risk_pct: float = 0.5
    opportunity_cost_bps: float = 5.0

    def estimate_cost(
        self,
        quantity: float,
        adv: float,
        daily_vol: float,
        price: float,
        execution_days: float = 1.0,
    ) -> dict:
        """Estimate total implementation shortfall cost.

        Args:
            quantity: Order size (shares or contracts).
            adv: Average daily volume (same units as quantity).
            daily_vol: Daily return volatility (decimal, e.g. 0.015 = 1.5%).
            price: Current market price (for notional calculation).
            execution_days: Expected execution duration in trading days.

        Returns:
            Dict with keys: market_impact_bps, timing_risk_bps,
            opportunity_cost_bps, total_cost_bps, total_cost_pct, total_cost_notional.
        """
        if adv <= 0 or price <= 0:
            return {
                "market_impact_bps": 0.0,
                "timing_risk_bps": 0.0,
                "opportunity_cost_bps": self.opportunity_cost_bps,
                "total_cost_bps": self.opportunity_cost_bps,
                "total_cost_pct": self.opportunity_cost_bps / 10000,
                "total_cost_notional": quantity
                * price
                * self.opportunity_cost_bps
                / 10000,
            }

        participation = quantity / (adv * execution_days)
        # Market impact: proportional to participation rate × kyle_lambda
        market_impact = self.kyle_lambda * participation  # as fraction of price
        market_impact_bps = market_impact * 10000

        # Timing risk: vol × sqrt(participation) × timing_risk_pct
        timing_risk = daily_vol * np.sqrt(participation) * self.timing_risk_pct
        timing_risk_bps = timing_risk * 10000

        total_bps = market_impact_bps + timing_risk_bps + self.opportunity_cost_bps
        total_pct = total_bps / 10000
        total_notional = quantity * price * total_pct

        return {
            "market_impact_bps": float(market_impact_bps),
            "timing_risk_bps": float(timing_risk_bps),
            "opportunity_cost_bps": float(self.opportunity_cost_bps),
            "total_cost_bps": float(total_bps),
            "total_cost_pct": float(total_pct),
            "total_cost_notional": float(total_notional),
        }

    def estimate_optimal_horizon(
        self,
        quantity: float,
        adv: float,
        daily_vol: float,
        price: float,
        max_days: int = 5,
    ) -> dict:
        """Find the execution horizon that minimizes total implementation shortfall.

        Trade-off: longer horizon → lower market impact but higher timing risk.

        Args:
            quantity: Order size.
            adv: Average daily volume.
            daily_vol: Daily return volatility.
            price: Current price.
            max_days: Maximum execution horizon to consider (default: 5).

        Returns:
            Dict with optimal_days, min_cost_bps, and cost_by_day.
        """
        horizons = np.linspace(0.1, max_days, 50)
        costs = [
            self.estimate_cost(quantity, adv, daily_vol, price, d)["total_cost_bps"]
            for d in horizons
        ]
        optimal_idx = int(np.argmin(costs))
        return {
            "optimal_days": float(horizons[optimal_idx]),
            "min_cost_bps": float(costs[optimal_idx]),
            "cost_by_day": dict(zip(horizons.tolist(), costs)),
        }


# ---------------------------------------------------------------------------
# Intraday Volume Profile (Plan 6.4)
# ---------------------------------------------------------------------------

INTRADAY_VOLUME_PROFILE: dict[str, float] = {
    "09:30-10:00": 0.15,
    "10:00-12:00": 0.25,
    "12:00-14:00": 0.15,
    "14:00-15:30": 0.25,
    "15:30-16:00": 0.20,
}


def get_volume_fraction(time_bucket: str) -> float:
    """Get expected volume fraction for a time bucket.

    Args:
        time_bucket: Time range string (e.g., "09:30-10:00").

    Returns:
        Expected fraction of daily volume.
    """
    return INTRADAY_VOLUME_PROFILE.get(time_bucket, 0.10)


# ---------------------------------------------------------------------------
# Participation Rate Limit (Plan 6.6)
# ---------------------------------------------------------------------------


def compute_multi_day_execution_plan(
    order_qty: float,
    adv: float,
    max_participation_pct: float = 0.05,
) -> list[dict]:
    """Create multi-day execution plan for large orders.

    Args:
        order_qty: Total shares to execute.
        adv: Average daily volume.
        max_participation_pct: Max daily participation (default 5%).

    Returns:
        List of daily execution slices.
    """
    if adv <= 0:
        return []

    max_daily = adv * max_participation_pct
    remaining = abs(order_qty)
    plan = []
    day = 1

    while remaining > 0:
        slice_qty = min(remaining, max_daily)
        plan.append(
            {
                "day": day,
                "quantity": round(slice_qty, 0),
                "pct_of_adv": round(slice_qty / adv * 100, 2),
                "remaining_after": round(remaining - slice_qty, 0),
            }
        )
        remaining -= slice_qty
        day += 1

    return plan


# ---------------------------------------------------------------------------
# Implementation Shortfall (Plan 6.7)
# ---------------------------------------------------------------------------


def compute_implementation_shortfall(
    decision_price: float,
    avg_fill_price: float,
    side: str = "BUY",
) -> float:
    """Compute implementation shortfall in basis points.

    IS = (avg_fill - decision_price) / decision_price × 10000 (for BUY)
    IS = (decision_price - avg_fill) / decision_price × 10000 (for SELL)

    Args:
        decision_price: Price at time of decision.
        avg_fill_price: Volume-weighted average fill price.
        side: BUY or SELL.

    Returns:
        Implementation shortfall in bps (positive = cost).
    """
    if decision_price <= 0:
        return 0.0

    if side.upper() == "BUY":
        is_bps = (avg_fill_price - decision_price) / decision_price * 10000
    else:
        is_bps = (decision_price - avg_fill_price) / decision_price * 10000

    return round(is_bps, 2)
