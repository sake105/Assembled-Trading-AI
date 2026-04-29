"""Execution routing — dispatches parent orders to direct/TWAP/Almgren-Chriss child slices."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal


Algorithm = Literal["direct", "twap", "almgren_chriss"]


@dataclass
class Order:
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    price: float
    order_id: str = ""


@dataclass
class ChildOrder:
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    price: float
    algo: Algorithm
    slice_idx: int = 0
    total_slices: int = 1
    scheduled_time: datetime | None = None
    parent_order_id: str = ""

    @classmethod
    def from_parent(cls, order: Order, algo: Algorithm = "direct") -> "ChildOrder":
        return cls(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=order.price,
            algo=algo,
            parent_order_id=order.order_id,
        )


@dataclass
class ExecutionConfig:
    """Thresholds (as fraction of ADV) that gate routing decisions."""
    direct_threshold: float = 0.01   # < 1% ADV → direct
    twap_threshold: float = 0.10     # 1-10% ADV → TWAP
    twap_slices: int = 10
    twap_duration_minutes: int = 60
    almgren_eta: float = 0.1         # temporary impact coefficient
    almgren_gamma: float = 0.1       # permanent impact coefficient
    almgren_lambda: float = 1e-6     # risk aversion


def twap_split(
    order: Order,
    n_slices: int,
    start_time: datetime | None = None,
    duration_minutes: int = 60,
) -> list[ChildOrder]:
    """Split order uniformly over n_slices time intervals."""
    if n_slices <= 0:
        return [ChildOrder.from_parent(order, "twap")]
    base_qty = order.quantity // n_slices
    remainder = order.quantity % n_slices
    start = start_time or datetime.now(timezone.utc)
    interval = timedelta(minutes=duration_minutes / n_slices)
    slices: list[ChildOrder] = []
    for i in range(n_slices):
        qty = base_qty + (1 if i < remainder else 0)
        if qty == 0:
            continue
        slices.append(
            ChildOrder(
                symbol=order.symbol,
                side=order.side,
                quantity=qty,
                price=order.price,
                algo="twap",
                slice_idx=i,
                total_slices=n_slices,
                scheduled_time=start + interval * i,
                parent_order_id=order.order_id,
            )
        )
    return slices


def ac_split(order: Order, config: ExecutionConfig) -> list[ChildOrder]:
    """Almgren-Chriss optimal execution trajectory (closed-form solution).

    Splits the parent order into slices sized by the AC closed-form schedule
    which minimises expected cost + variance of execution.
    """
    import math  # noqa: PLC0415

    n = config.twap_slices
    eta = config.almgren_eta
    gamma = config.almgren_gamma
    lam = config.almgren_lambda

    # Closed-form: sinh-shaped schedule
    # x(t) = Q * sinh(kappa * (T - t)) / sinh(kappa * T)
    # where kappa = sqrt(2 * lambda / (eta - 0.5 * gamma))
    denom = max(eta - 0.5 * gamma, 1e-9)
    kappa = math.sqrt(2 * lam / denom)
    T = float(n)

    if kappa * T < 1e-6:
        # Near-linear (low risk aversion) → TWAP fallback
        return twap_split(order, n_slices=n)

    slices: list[ChildOrder] = []
    prev_remaining = order.quantity
    for i in range(n):
        t = float(i)
        t_next = float(i + 1)
        holding_t = math.sinh(kappa * (T - t)) / math.sinh(kappa * T)
        holding_t_next = math.sinh(kappa * (T - t_next)) / math.sinh(kappa * T) if t_next <= T else 0.0
        qty = max(0, round(order.quantity * (holding_t - holding_t_next)))
        if i == n - 1:
            qty = prev_remaining
        qty = min(qty, prev_remaining)
        if qty <= 0:
            continue
        prev_remaining -= qty
        slices.append(
            ChildOrder(
                symbol=order.symbol,
                side=order.side,
                quantity=qty,
                price=order.price,
                algo="almgren_chriss",
                slice_idx=i,
                total_slices=n,
                parent_order_id=order.order_id,
            )
        )
    return slices


# ---------------------------------------------------------------------------
# Adaptive Almgren-Chriss (B10)
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveACState:
    """Mutable state for the adaptive AC executor.

    Maintains an EWMA estimate of the temporary market-impact coefficient η
    updated after each observed fill.  Start with the prior from
    :class:`ExecutionConfig`.

    Parameters
    ----------
    eta_prior:
        Initial η (temporary impact coefficient) — copied from config.
    ewma_alpha:
        Learning rate in (0, 1].  Higher = faster adaptation to new fills.
        Typical range: 0.05–0.20.
    min_obs_before_update:
        Minimum number of observed fills before the prior is overridden.
    """

    eta_hat: float
    ewma_alpha: float = 0.10
    min_obs_before_update: int = 3
    obs_count: int = 0
    _cum_implied_eta: float = 0.0  # weighted accumulator (internal)

    @classmethod
    def from_config(
        cls, config: ExecutionConfig, ewma_alpha: float = 0.10
    ) -> "AdaptiveACState":
        return cls(eta_hat=config.almgren_eta, ewma_alpha=ewma_alpha)

    def update(
        self,
        qty_filled: float,
        expected_price: float,
        actual_price: float,
        side: str,
        sigma_daily: float = 0.015,
    ) -> None:
        """Update η from one observed fill.

        Implied η = |slippage_bps| * price / (qty * sigma²).
        Slippage  = (actual - expected) for BUY, (expected - actual) for SELL.

        Args:
            qty_filled: Shares filled in this child order.
            expected_price: Reference price at order submission (e.g. arrival price).
            actual_price: Observed fill price.
            sigma_daily: Estimated daily vol used in the AC model (annualised / sqrt(252)).
        """
        if expected_price <= 0 or qty_filled <= 0:
            return
        sign = 1.0 if side.upper() == "BUY" else -1.0
        slippage = sign * (actual_price - expected_price) / expected_price
        slippage = max(slippage, 0.0)  # only cost (positive) fills update η

        # η relates slippage to trade rate: slippage ≈ η * (qty / sigma)
        var_daily = sigma_daily ** 2
        implied_eta = slippage * expected_price / max(qty_filled * var_daily, 1e-10)
        implied_eta = max(implied_eta, 1e-6)

        self.obs_count += 1
        if self.obs_count < self.min_obs_before_update:
            return

        # EWMA update
        self.eta_hat = (
            self.ewma_alpha * implied_eta
            + (1.0 - self.ewma_alpha) * self.eta_hat
        )
        self.eta_hat = max(self.eta_hat, 1e-6)


def adaptive_ac_split(
    order: Order,
    config: ExecutionConfig,
    state: AdaptiveACState,
    remaining_qty: int | None = None,
    remaining_slices: int | None = None,
) -> list[ChildOrder]:
    """Almgren-Chriss split using live-updated market-impact parameters.

    Accepts a partially executed order by passing *remaining_qty* and
    *remaining_slices* — the schedule is re-optimised for the remainder
    using the current ``state.eta_hat``.

    Args:
        order: The parent (or residual) order to execute.
        config: Execution configuration (gamma, lambda_risk, slices).
        state: Adaptive state carrying the current η estimate.
        remaining_qty: Shares still to execute; defaults to order.quantity.
        remaining_slices: Time intervals remaining; defaults to config.twap_slices.

    Returns:
        List of :class:`ChildOrder` slices.
    """
    import math

    qty = remaining_qty if remaining_qty is not None else order.quantity
    n = remaining_slices if remaining_slices is not None else config.twap_slices
    if n <= 0 or qty <= 0:
        return []

    eta = state.eta_hat
    gamma = config.almgren_gamma
    lam = config.almgren_lambda

    denom = max(eta - 0.5 * gamma, 1e-9)
    kappa = math.sqrt(2 * lam / denom)
    T = float(n)

    if kappa * T < 1e-6:
        # Near-linear (low risk aversion) → uniform split
        base = qty // n
        rem = qty % n
        slices = []
        for i in range(n):
            q = base + (1 if i < rem else 0)
            if q > 0:
                slices.append(ChildOrder(
                    symbol=order.symbol, side=order.side,
                    quantity=q, price=order.price,
                    algo="almgren_chriss",
                    slice_idx=i, total_slices=n,
                    parent_order_id=order.order_id,
                ))
        return slices

    slices = []
    prev_remaining = qty
    for i in range(n):
        t = float(i)
        t_next = float(i + 1)
        holding_t = math.sinh(kappa * (T - t)) / math.sinh(kappa * T)
        holding_tn = math.sinh(kappa * (T - t_next)) / math.sinh(kappa * T) if t_next <= T else 0.0
        child_qty = max(0, round(qty * (holding_t - holding_tn)))
        if i == n - 1:
            child_qty = prev_remaining
        child_qty = min(child_qty, prev_remaining)
        if child_qty <= 0:
            continue
        prev_remaining -= child_qty
        slices.append(ChildOrder(
            symbol=order.symbol, side=order.side,
            quantity=child_qty, price=order.price,
            algo="almgren_chriss",
            slice_idx=i, total_slices=n,
            parent_order_id=order.order_id,
        ))
    return slices


def route_order(
    order: Order,
    avg_daily_volume: float,
    config: ExecutionConfig | None = None,
) -> list[ChildOrder]:
    """Dispatch parent order to direct / TWAP / Almgren-Chriss child orders.

    Parameters
    ----------
    order: Parent order to route.
    avg_daily_volume: Symbol's average daily volume in shares.
    config: Routing configuration; uses defaults if None.

    Returns
    -------
    List of ChildOrder slices.
    """
    if config is None:
        config = ExecutionConfig()

    notional_shares = order.quantity
    pct_of_adv = notional_shares / max(avg_daily_volume, 1)

    if pct_of_adv < config.direct_threshold:
        return [ChildOrder.from_parent(order, "direct")]
    elif pct_of_adv < config.twap_threshold:
        return twap_split(order, n_slices=config.twap_slices,
                          duration_minutes=config.twap_duration_minutes)
    else:
        return ac_split(order, config)
