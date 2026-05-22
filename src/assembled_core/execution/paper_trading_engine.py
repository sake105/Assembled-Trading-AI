"""Paper trading engine for in-memory order execution simulation.

This module provides an in-memory paper trading engine that simulates order execution
without any file I/O or network calls. Orders are immediately filled and positions
are aggregated in memory.

Fill model (optional):
    When ``fill_model`` is provided to the constructor, the engine applies
    realistic execution costs: spread, market impact, and timing noise.
    Without it, the engine fills at the exact order price (legacy behaviour).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from src.assembled_core.logging_utils import get_logger

logger = get_logger("assembled_core.execution.paper_trading_engine")


@dataclass
class FillModel:
    """Configurable fill-cost model for realistic paper execution.

    Costs applied per order:
        effective_price = order_price
            + spread_cost          (half bid-ask spread)
            + market_impact_cost   (Almgren-Chriss square-root model)

    All costs are expressed as price offsets (positive = worse fill).
    For BUY orders, costs are *added* to price; for SELL, *subtracted*.

    Attributes:
        half_spread_bps: Half bid-ask spread in basis points (default: 5 bps).
        impact_coefficient: Market-impact multiplier (default: 0.10).
            impact = coefficient * sigma * sqrt(order_qty / adv)
        default_adv: Default average daily volume if not provided per symbol.
        default_sigma: Default daily volatility if not provided per symbol.
    """

    half_spread_bps: float = 5.0
    impact_coefficient: float = 0.10
    default_adv: float = 1_000_000.0
    default_sigma: float = 0.02

    @classmethod
    def from_cost_model(cls, cost_model: object | None = None) -> FillModel:
        """Create a FillModel aligned with the central CostModel defaults.

        This ensures paper trading uses the **same** cost assumptions as
        the backtest engine.  If *cost_model* is ``None``, the project-wide
        default from ``costs.get_default_cost_model()`` is loaded.

        Mapping:
        - commission_bps is applied separately in the backtest engine and
          is NOT part of the fill-price offset; paper trades should add
          commission externally when computing P&L.
        - spread_w scales the half-spread: ``half_spread_bps = 5.0 * spread_w``
        - impact_w scales the impact coefficient: ``impact_coefficient = 0.10 * impact_w``
        """
        if cost_model is None:
            from src.assembled_core.costs import get_default_cost_model

            cost_model = get_default_cost_model()

        spread_w = getattr(cost_model, "spread_w", 0.25)
        impact_w = getattr(cost_model, "impact_w", 0.5)

        return cls(
            half_spread_bps=5.0 * spread_w,
            impact_coefficient=0.10 * impact_w,
        )

    def compute_fill_price(
        self,
        order_price: float,
        side: str,
        quantity: float,
        adv: float | None = None,
        sigma: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Compute realistic fill price including execution costs.

        Returns:
            (fill_price, cost_breakdown) where cost_breakdown has keys:
            spread_cost, impact_cost, total_cost_bps.
        """
        adv = adv or self.default_adv
        sigma = sigma or self.default_sigma

        # Spread cost (half-spread)
        spread_cost = order_price * self.half_spread_bps / 10_000

        # Market impact: Almgren-Chriss square-root model
        participation = quantity / adv if adv > 0 else 0.01
        impact_cost = (
            self.impact_coefficient * sigma * order_price * math.sqrt(participation)
        )

        total_cost = spread_cost + impact_cost
        total_cost_bps = (total_cost / order_price * 10_000) if order_price > 0 else 0.0

        # BUY → pay more, SELL → receive less
        if side == "BUY":
            fill_price = order_price + total_cost
        else:
            fill_price = order_price - total_cost

        return fill_price, {
            "spread_cost": round(spread_cost, 6),
            "impact_cost": round(impact_cost, 6),
            "total_cost_bps": round(total_cost_bps, 2),
        }


@dataclass(slots=True)
class PaperOrder:
    """Paper trading order representation.

    Attributes:
        order_id: Unique order identifier
        symbol: Ticker symbol
        side: BUY or SELL
        quantity: Order quantity (always positive)
        price: Order price (optional, can be None for market orders)
        status: Order status (NEW, FILLED, REJECTED)
        reason: Optional reason for rejection
        client_order_id: Optional client-provided order ID
        route: Optional routing information (default: "PAPER" for paper trading)
        source: Optional source identifier (e.g., "CLI", "API", "BACKTEST", "DASHBOARD")
        created_at: Order creation timestamp
        filled_at: Order fill timestamp (if filled)
    """

    order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: float
    price: float | None
    status: Literal["NEW", "FILLED", "REJECTED"]
    reason: str | None = None
    client_order_id: str | None = None
    route: str | None = "PAPER"
    source: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    filled_at: datetime | None = None
    fill_price: float | None = None
    fill_cost_breakdown: dict | None = None


@dataclass(slots=True)
class PaperPosition:
    """Paper trading position representation.

    Attributes:
        symbol: Ticker symbol
        quantity: Position quantity (positive = long, negative = short)
    """

    symbol: str
    quantity: float


class PaperTradingEngine:
    """In-memory paper trading engine.

    This engine maintains orders and positions in memory. All orders are immediately
    filled when submitted (no partial fills, no rejection logic beyond basic validation).

    Attributes:
        _orders: List of all orders (newest first)
        _positions: Dictionary mapping symbol -> net quantity
    """

    def __init__(
        self, fill_model: FillModel | None = None, initial_cash: float = 100_000.0
    ) -> None:
        """Initialize paper trading engine with empty state.

        Args:
            fill_model: Optional FillModel for realistic execution costs.
                If None, orders are filled at exact order price (legacy behaviour).
            initial_cash: Starting cash balance for cash-balance guard (default: 100,000).
        """
        self._orders: list[PaperOrder] = []
        self._positions: dict[str, float] = {}
        self._fill_model = fill_model
        self._cash: float = initial_cash
        self._lock = threading.RLock()
        logger.debug(
            "Paper trading engine initialized (fill_model=%s, cash=%.2f)",
            "enabled" if fill_model else "off",
            initial_cash,
        )

    def submit_orders(self, orders: list[PaperOrder]) -> list[PaperOrder]:
        """Submit orders for execution.

        Orders are immediately filled and positions are updated.

        Args:
            orders: List of PaperOrder objects to submit

        Returns:
            List of PaperOrder objects with status updated to FILLED

        Raises:
            ValueError: If order validation fails
        """
        filled_orders = []

        with self._lock:
            for order in orders:
                # Validate order
                if order.quantity <= 0:
                    order.status = "REJECTED"
                    order.reason = f"Invalid quantity: {order.quantity} (must be > 0)"
                    logger.warning(f"Order {order.order_id} rejected: {order.reason}")
                    filled_orders.append(order)
                    continue

                if not order.symbol or not order.symbol.strip():
                    order.status = "REJECTED"
                    order.reason = "Invalid symbol: empty or whitespace"
                    logger.warning(f"Order {order.order_id} rejected: {order.reason}")
                    filled_orders.append(order)
                    continue

                # Normalize symbol
                symbol = order.symbol.strip().upper()

                # Cash-balance guard for BUY orders (skip if price is None or 0)
                if order.side == "BUY" and order.price:
                    cost = order.price * order.quantity
                    if cost > self._cash:
                        order.status = "REJECTED"
                        order.reason = "Insufficient cash"
                        logger.warning(
                            "Order %s rejected: Insufficient cash (need=%.2f, have=%.2f)",
                            order.order_id,
                            cost,
                            self._cash,
                        )
                        filled_orders.append(order)
                        continue

                # Fill order immediately
                order.status = "FILLED"
                order.filled_at = datetime.now(tz=timezone.utc)
                order.symbol = symbol  # Store normalized symbol

                # Apply fill model for realistic execution costs
                if self._fill_model is not None and order.price is not None:
                    fill_px, cost_info = self._fill_model.compute_fill_price(
                        order_price=order.price,
                        side=order.side,
                        quantity=order.quantity,
                    )
                    order.fill_price = fill_px
                    order.fill_cost_breakdown = cost_info
                else:
                    order.fill_price = order.price

                # Update position
                if symbol not in self._positions:
                    self._positions[symbol] = 0.0

                # BUY adds to position, SELL subtracts from position; update cash balance
                effective_price = order.fill_price or order.price
                if order.side == "BUY":
                    self._positions[symbol] += order.quantity
                    if effective_price:
                        self._cash -= effective_price * order.quantity
                else:  # SELL
                    self._positions[symbol] -= order.quantity
                    if effective_price:
                        self._cash += effective_price * order.quantity

                filled_orders.append(order)
                logger.debug(
                    f"Order {order.order_id} filled: {order.side} {order.quantity} {symbol} "
                    f"@ {order.price or 'MARKET'}"
                )

            # Add orders to history (newest first)
            self._orders = filled_orders + self._orders

        logger.info(f"Submitted {len(filled_orders)} orders, all filled")
        return filled_orders

    def list_orders(self, limit: int | None = None) -> list[PaperOrder]:
        """List recent orders.

        Args:
            limit: Maximum number of orders to return (None = all)

        Returns:
            List of PaperOrder objects (newest first)
        """
        with self._lock:
            orders = list(self._orders)
            if limit is not None and limit > 0:
                orders = orders[:limit]

        return orders

    def get_positions(self) -> list[PaperPosition]:
        """Get current positions.

        Returns:
            List of PaperPosition objects (only non-zero positions)
        """
        with self._lock:
            positions = [
                PaperPosition(symbol=symbol, quantity=qty)
                for symbol, qty in self._positions.items()
                if abs(qty) > 1e-6  # Filter out essentially zero positions
            ]

        # Sort by symbol for consistent ordering
        positions.sort(key=lambda p: p.symbol)
        return positions

    def submit_algo_order(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        price: float | None,
        algo: str = "TWAP",
        n_slices: int = 10,
        participation_rate: float = 0.10,
    ) -> list[PaperOrder]:
        """Submit an algorithmic order using TWAP or VWAP slicing.

        D8: Algorithmic execution support for paper trading.
        In paper mode, all slices are filled immediately at the given price.
        The slicing schedule is computed via algo_execution module to maintain
        realistic order decomposition for cost estimation.

        Args:
            symbol: Ticker symbol.
            side: "BUY" or "SELL".
            total_quantity: Total shares to execute.
            price: Reference price (None = market).
            algo: "TWAP" or "VWAP" (default: "TWAP").
            n_slices: Number of execution slices.
            participation_rate: Target participation rate for VWAP (default: 10%).

        Returns:
            List of filled PaperOrder objects (one per slice).
        """
        from src.assembled_core.execution.algo_execution import (
            TWAPScheduler,
            VWAPScheduler,
        )

        if algo.upper() == "VWAP":
            scheduler = VWAPScheduler(
                n_slices=n_slices, participation_rate=participation_rate
            )
        else:
            scheduler = TWAPScheduler(n_slices=n_slices)

        slices = scheduler.schedule(
            total_quantity=total_quantity, reference_price=price or 0.0
        )
        slice_orders: list[PaperOrder] = []

        from src.assembled_core.execution.idempotency import (
            build_client_order_id,
            compute_intent_hash,
        )

        # Deterministic base hash for this algo order (symbol+side+qty+type)
        _base_hash = compute_intent_hash(
            symbol=symbol,
            side=side.lower(),
            qty=total_quantity,
            order_type=algo.lower(),
            limit_price=price,
        )
        for i, sl in enumerate(slices):
            qty = (
                sl.quantity if hasattr(sl, "quantity") else (total_quantity / n_slices)
            )
            slice_price = sl.price if hasattr(sl, "price") else price
            # Use slice index as signal_id so each slice gets a unique but deterministic ID
            _client_id = build_client_order_id(
                signal_id=f"{algo.lower()}_slice_{i + 1}_of_{n_slices}",
                intent_hash=_base_hash,
                attempt=0,
            )

            order = PaperOrder(
                order_id=_client_id,
                symbol=symbol,
                side=side.upper(),  # type: ignore[arg-type]
                quantity=abs(qty),
                price=slice_price,
                status="NEW",
                client_order_id=_client_id,
                route="PAPER",
                source="ALGO",
            )
            slice_orders.append(order)

        filled = self.submit_orders(slice_orders)
        logger.info(
            "ALGO_ORDER: %s %s %s qty=%.0f via %s (%d slices filled)",
            side,
            symbol,
            algo,
            total_quantity,
            algo,
            len(filled),
        )
        return filled

    def get_cash_balance(self) -> float:
        """Return the current cash balance."""
        with self._lock:
            return self._cash

    def reset(self) -> None:
        """Reset engine state (clear all orders and positions).

        This is primarily for testing purposes.
        """
        with self._lock:
            order_count = len(self._orders)
            position_count = len(
                [qty for qty in self._positions.values() if abs(qty) > 1e-6]
            )

            self._orders = []
            self._positions = {}

        logger.info(
            f"Engine reset: cleared {order_count} orders, {position_count} positions"
        )
