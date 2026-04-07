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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from src.assembled_core.logging_utils import setup_logging

logger = setup_logging(level="INFO")


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
        impact_cost = self.impact_coefficient * sigma * order_price * math.sqrt(participation)

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


@dataclass
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


@dataclass
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

    def __init__(self, fill_model: FillModel | None = None) -> None:
        """Initialize paper trading engine with empty state.

        Args:
            fill_model: Optional FillModel for realistic execution costs.
                If None, orders are filled at exact order price (legacy behaviour).
        """
        self._orders: list[PaperOrder] = []
        self._positions: dict[str, float] = {}
        self._fill_model = fill_model
        logger.debug("Paper trading engine initialized (fill_model=%s)",
                      "enabled" if fill_model else "off")

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

            # BUY adds to position, SELL subtracts from position
            if order.side == "BUY":
                self._positions[symbol] += order.quantity
            else:  # SELL
                self._positions[symbol] -= order.quantity

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
        orders = self._orders
        if limit is not None and limit > 0:
            orders = orders[:limit]

        return orders

    def get_positions(self) -> list[PaperPosition]:
        """Get current positions.

        Returns:
            List of PaperPosition objects (only non-zero positions)
        """
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
        from src.assembled_core.execution.algo_execution import TWAPScheduler, VWAPScheduler

        if algo.upper() == "VWAP":
            scheduler = VWAPScheduler(n_slices=n_slices, participation_rate=participation_rate)
        else:
            scheduler = TWAPScheduler(n_slices=n_slices)

        slices = scheduler.schedule(total_quantity=total_quantity, reference_price=price or 0.0)
        slice_orders: list[PaperOrder] = []

        for i, sl in enumerate(slices):
            import uuid
            qty = sl.quantity if hasattr(sl, "quantity") else (total_quantity / n_slices)
            slice_price = sl.price if hasattr(sl, "price") else price

            order = PaperOrder(
                order_id=str(uuid.uuid4()),
                symbol=symbol,
                side=side.upper(),  # type: ignore[arg-type]
                quantity=abs(qty),
                price=slice_price,
                status="NEW",
                client_order_id=f"{algo.lower()}_slice_{i+1}_of_{n_slices}",
                route="PAPER",
                source="ALGO",
            )
            slice_orders.append(order)

        filled = self.submit_orders(slice_orders)
        logger.info(
            "ALGO_ORDER: %s %s %s qty=%.0f via %s (%d slices filled)",
            side, symbol, algo, total_quantity, algo, len(filled),
        )
        return filled

    def reset(self) -> None:
        """Reset engine state (clear all orders and positions).

        This is primarily for testing purposes.
        """
        order_count = len(self._orders)
        position_count = len(
            [qty for qty in self._positions.values() if abs(qty) > 1e-6]
        )

        self._orders = []
        self._positions = {}

        logger.info(
            f"Engine reset: cleared {order_count} orders, {position_count} positions"
        )
