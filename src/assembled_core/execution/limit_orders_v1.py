# src/assembled_core/execution/limit_orders_v1.py
"""Limit order module — IOC-limit with midpoint pricing and spread-adaptive aggression.

Pure logic/data structures only.  No live broker calls, no module-level side
effects.  Intended for paper trading and backtest simulation.

Behaviour summary
-----------------
* IOC-limit default (TIF = "ioc"), 5-minute (300 s) timeout.
* Aggression scalar derived from bid/ask spread width:
  - narrow spread  → midpoint (aggression ≈ 0.5)
  - wide spread    → more aggressive (aggression → 0.8) to maximise fill chance
* If the limit cannot be filled within the timeout *and* fallback_to_market is
  True, a synthetic market fill is returned instead.
* Slippage is reported in basis points relative to the midpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LimitOrderSpec:
    """Specification for a single limit order.

    Attributes
    ----------
    symbol:
        Ticker symbol.
    side:
        ``"buy"`` or ``"sell"``.
    qty:
        Requested quantity (shares / contracts).  Must be > 0.
    limit_price:
        Limit price placed on the order.
    tif:
        Time-in-force flag.  ``"ioc"`` (default) cancels immediately if not
        filled at the limit; ``"day"`` stays open for the session.
    max_wait_seconds:
        Simulation timeout in seconds.  After this period the order is
        considered expired and the fallback path is evaluated.
    fallback_to_market:
        If ``True`` and the limit order expires unfilled, a market order is
        simulated at the prevailing market price.
    """

    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    limit_price: float
    tif: str = "ioc"
    max_wait_seconds: int = 300  # 5-minute timeout
    fallback_to_market: bool = True

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise ValueError(
                f"LimitOrderSpec.side must be 'buy' or 'sell', got {self.side!r}"
            )
        if self.qty <= 0:
            raise ValueError(f"LimitOrderSpec.qty must be > 0, got {self.qty}")
        if self.limit_price <= 0:
            raise ValueError(
                f"LimitOrderSpec.limit_price must be > 0, got {self.limit_price}"
            )
        if self.max_wait_seconds < 0:
            raise ValueError(
                f"LimitOrderSpec.max_wait_seconds must be >= 0, got {self.max_wait_seconds}"
            )


@dataclass
class LimitOrderResult:
    """Result of a simulated (or live) limit-order attempt.

    Attributes
    ----------
    symbol:
        Ticker symbol.
    side:
        ``"buy"`` or ``"sell"``.
    requested_qty:
        Quantity originally requested.
    filled_qty:
        Quantity actually filled.  May be 0 if unfilled and no fallback.
    fill_price:
        Average fill price achieved.  0.0 when filled_qty == 0.
    order_type:
        ``"limit"`` if the limit order filled; ``"market"`` if the market
        fallback was used; ``"unfilled"`` if nothing filled.
    slippage_bps:
        ``(fill_price - mid) / mid * 10_000``.  Positive = paid more than mid
        (bad for buyer), negative = received less than mid (bad for seller).
        0.0 when unfilled.
    """

    symbol: str
    side: str
    requested_qty: float
    filled_qty: float
    fill_price: float
    order_type: str  # "limit" | "market" | "unfilled"
    slippage_bps: float  # (fill_price - mid) / mid * 10_000


# ---------------------------------------------------------------------------
# Core pricing helpers
# ---------------------------------------------------------------------------


def compute_limit_price(
    bid: float,
    ask: float,
    side: str,
    *,
    aggression: float = 0.5,
) -> float:
    """Compute a limit price from a bid/ask quote.

    Parameters
    ----------
    bid:
        Best bid price.  Must be > 0 and <= ask.
    ask:
        Best ask price.  Must be >= bid.
    side:
        ``"buy"`` or ``"sell"``.
    aggression:
        Float in [0.0, 1.0].

        * 0.0 → passive: bid for a buy, ask for a sell.
        * 0.5 → midpoint (default).
        * 1.0 → aggressive: ask for a buy, bid for a sell.

    Returns
    -------
    float
        Computed limit price.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if bid <= 0:
        raise ValueError(f"bid must be > 0, got {bid}")
    if ask < bid:
        raise ValueError(f"ask ({ask}) must be >= bid ({bid})")
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    aggression = max(0.0, min(1.0, aggression))

    if side == "buy":
        # passive=bid, aggressive=ask
        price = bid + aggression * (ask - bid)
    else:
        # sell: passive=ask, aggressive=bid
        price = ask - aggression * (ask - bid)

    logger.debug(
        "compute_limit_price: side=%s bid=%.4f ask=%.4f aggression=%.3f → %.4f",
        side,
        bid,
        ask,
        aggression,
        price,
    )
    return price


def spread_aggression(
    bid: float,
    ask: float,
    *,
    max_spread_bps: float = 20.0,
) -> float:
    """Return an aggression scalar derived from the bid/ask spread width.

    Narrow spread → midpoint-ish (aggression near 0.5).
    Wide spread   → more aggressive (aggression → 0.8) to improve fill chance.

    The mapping is linear from spread=0 (aggression=0.3) to
    spread>=max_spread_bps (aggression=0.8).

    Parameters
    ----------
    bid:
        Best bid price.  Must be > 0.
    ask:
        Best ask price.  Must be >= bid.
    max_spread_bps:
        Spread width (in bps) considered "very wide".  Default 20 bps.

    Returns
    -------
    float
        Aggression in [0.3, 0.8].

    Raises
    ------
    ValueError
        If bid or ask are invalid.
    """
    if bid <= 0:
        raise ValueError(f"bid must be > 0, got {bid}")
    if ask < bid:
        raise ValueError(f"ask ({ask}) must be >= bid ({bid})")
    if max_spread_bps <= 0:
        raise ValueError(f"max_spread_bps must be > 0, got {max_spread_bps}")

    mid = (bid + ask) / 2.0
    spread_bps = ((ask - bid) / mid) * 10_000.0

    # Linear interpolation: 0 bps → 0.3, max_spread_bps → 0.8
    _min_agg = 0.3
    _max_agg = 0.8
    ratio = min(spread_bps / max_spread_bps, 1.0)
    aggression = _min_agg + ratio * (_max_agg - _min_agg)

    logger.debug(
        "spread_aggression: bid=%.4f ask=%.4f spread_bps=%.2f → aggression=%.3f",
        bid,
        ask,
        spread_bps,
        aggression,
    )
    return aggression


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_limit_fill(
    spec: LimitOrderSpec,
    market_price: float,
    *,
    slippage_model: str = "half_spread",
    bid_ask_spread_pct: float = 0.001,
) -> LimitOrderResult:
    """Simulate a limit order fill for backtest / paper trading.

    Fill logic
    ----------
    The order is considered **fillable** at the limit when:

    * ``side == "buy"``  and ``spec.limit_price >= market_price``
    * ``side == "sell"`` and ``spec.limit_price <= market_price``

    When fillable the simulated fill price equals the limit price (the market
    came to us), and slippage is computed relative to the mid.

    When **not** fillable within the timeout:

    * If ``spec.fallback_to_market`` is ``True``, a market fill is simulated
      at ``market_price ± half_spread`` depending on side.
    * Otherwise an ``"unfilled"`` result is returned.

    Parameters
    ----------
    spec:
        The limit order specification.
    market_price:
        Current market reference price (e.g. last trade, mid).
    slippage_model:
        Only ``"half_spread"`` is supported.  The half-spread cost is
        ``market_price * bid_ask_spread_pct / 2``.
    bid_ask_spread_pct:
        Fractional bid/ask spread (e.g. 0.001 = 10 bps).

    Returns
    -------
    LimitOrderResult
    """
    if market_price <= 0:
        raise ValueError(f"market_price must be > 0, got {market_price}")
    if bid_ask_spread_pct < 0:
        raise ValueError(f"bid_ask_spread_pct must be >= 0, got {bid_ask_spread_pct}")
    if slippage_model != "half_spread":
        raise ValueError(
            f"Unsupported slippage_model {slippage_model!r}. Only 'half_spread' is supported."
        )

    half_spread = market_price * bid_ask_spread_pct / 2.0
    mid = market_price  # market_price treated as mid for slippage calc

    # ------------------------------------------------------------------
    # Determine whether the limit price crosses the market
    # ------------------------------------------------------------------
    limit_fillable: bool
    if spec.side == "buy":
        # Buyer: we pay up to limit_price; fillable if market (ask proxy) <=
        # limit_price.  For simulation, ask ≈ mid + half_spread.
        effective_ask = market_price + half_spread
        limit_fillable = spec.limit_price >= effective_ask
    else:
        # Seller: we accept down to limit_price; fillable if bid >=
        # limit_price.  For simulation, bid ≈ mid - half_spread.
        effective_bid = market_price - half_spread
        limit_fillable = spec.limit_price <= effective_bid

    # ------------------------------------------------------------------
    # Branch: limit fills
    # ------------------------------------------------------------------
    if limit_fillable:
        fill_price = spec.limit_price
        slippage_bps = (fill_price - mid) / mid * 10_000.0
        logger.info(
            "simulate_limit_fill: %s %s qty=%.2f LIMIT filled @ %.4f slippage=%.2f bps",
            spec.side.upper(),
            spec.symbol,
            spec.qty,
            fill_price,
            slippage_bps,
        )
        return LimitOrderResult(
            symbol=spec.symbol,
            side=spec.side,
            requested_qty=spec.qty,
            filled_qty=spec.qty,
            fill_price=fill_price,
            order_type="limit",
            slippage_bps=slippage_bps,
        )

    # ------------------------------------------------------------------
    # Branch: limit not fillable within timeout
    # ------------------------------------------------------------------
    logger.debug(
        "simulate_limit_fill: %s %s limit_price=%.4f not fillable vs market=%.4f",
        spec.side.upper(),
        spec.symbol,
        spec.limit_price,
        market_price,
    )

    if not spec.fallback_to_market:
        logger.info(
            "simulate_limit_fill: %s %s UNFILLED (no market fallback)",
            spec.side.upper(),
            spec.symbol,
        )
        return LimitOrderResult(
            symbol=spec.symbol,
            side=spec.side,
            requested_qty=spec.qty,
            filled_qty=0.0,
            fill_price=0.0,
            order_type="unfilled",
            slippage_bps=0.0,
        )

    # Market fallback
    if spec.side == "buy":
        fill_price = market_price + half_spread  # pay the ask
    else:
        fill_price = market_price - half_spread  # receive the bid

    slippage_bps = (fill_price - mid) / mid * 10_000.0
    logger.info(
        "simulate_limit_fill: %s %s qty=%.2f MARKET fallback @ %.4f slippage=%.2f bps",
        spec.side.upper(),
        spec.symbol,
        spec.qty,
        fill_price,
        slippage_bps,
    )
    return LimitOrderResult(
        symbol=spec.symbol,
        side=spec.side,
        requested_qty=spec.qty,
        filled_qty=spec.qty,
        fill_price=fill_price,
        order_type="market",
        slippage_bps=slippage_bps,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "LimitOrderSpec",
    "LimitOrderResult",
    "compute_limit_price",
    "spread_aggression",
    "simulate_limit_fill",
]
