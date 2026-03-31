"""Broker Adapter — M12: Abstract broker interface for paper and live trading.

This module provides:
- BrokerAdapter: Abstract base class defining the broker interface
- AlpacaAdapter: Alpaca paper trading implementation
- BrokerOrder / BrokerPosition: Data contracts

IMPORTANT: All live/paper order submission is gated. AlpacaAdapter requires
explicit configuration to instantiate. No API keys are hardcoded.
API keys must be provided via environment variables or explicit constructor args.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BrokerOrder:
    """Normalized order representation returned by broker adapters."""

    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    qty: float
    order_type: str  # "market" or "limit"
    status: str  # "pending", "filled", "cancelled", "rejected"
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    submitted_at: str = ""
    filled_at: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class BrokerPosition:
    """Normalized position representation returned by broker adapters."""

    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    raw: dict[str, Any] = field(default_factory=dict)


class BrokerAdapter(ABC):
    """Abstract base class for broker adapters.

    All broker integrations must implement this interface.
    This ensures consistent behavior across paper and live environments.
    """

    @property
    @abstractmethod
    def is_paper(self) -> bool:
        """Return True if this adapter is connected to a paper trading environment."""
        ...

    @abstractmethod
    def get_account(self) -> dict[str, Any]:
        """Return account summary dict (equity, buying_power, etc.)."""
        ...

    @abstractmethod
    def get_positions(self) -> list[BrokerPosition]:
        """Return list of current open positions."""
        ...

    @abstractmethod
    def get_open_orders(self) -> list[BrokerOrder]:
        """Return list of currently open (unfilled) orders."""
        ...

    @abstractmethod
    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        *,
        time_in_force: str = "day",
        comment: str = "",
    ) -> BrokerOrder:
        """Submit a market order.

        Args:
            symbol: Ticker symbol.
            qty: Number of shares (positive).
            side: "buy" or "sell".
            time_in_force: "day", "gtc", etc.
            comment: Optional order comment/tag.

        Returns:
            BrokerOrder with order_id and initial status.
        """
        ...

    @abstractmethod
    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns number of orders cancelled."""
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> BrokerOrder:
        """Get current status of a specific order by ID."""
        ...

    def health_check(self) -> dict[str, Any]:
        """Check broker connectivity. Returns dict with 'ok' bool and 'message'."""
        try:
            acct = self.get_account()
            return {
                "ok": True,
                "message": "connected",
                "account_equity": acct.get("equity"),
            }
        except Exception as e:
            return {"ok": False, "message": str(e)}


class AlpacaAdapter(BrokerAdapter):
    """Alpaca paper trading adapter.

    Uses Alpaca's paper trading API. Requires alpaca-trade-api or alpaca-py package.

    Configuration via environment variables:
        ALPACA_API_KEY:    Alpaca API key ID (paper trading key)
        ALPACA_API_SECRET: Alpaca API secret key (paper trading secret)
        ALPACA_BASE_URL:   Optional override (default: https://paper-api.alpaca.markets)

    IMPORTANT: Only paper trading is supported. Live trading requires explicit
    base_url override and is not recommended without additional safeguards.
    """

    PAPER_BASE_URL = "https://paper-api.alpaca.markets"

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str | None = None,
        *,
        force_paper: bool = True,
    ) -> None:
        """Initialize Alpaca adapter.

        Args:
            api_key: Alpaca API key (or set ALPACA_API_KEY env var).
            api_secret: Alpaca API secret (or set ALPACA_API_SECRET env var).
            base_url: API base URL. Defaults to paper trading URL.
            force_paper: If True (default), raises if base_url looks like live endpoint.
        """
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._api_secret = api_secret or os.environ.get("ALPACA_API_SECRET", "")
        self._base_url = base_url or os.environ.get(
            "ALPACA_BASE_URL", self.PAPER_BASE_URL
        )

        if force_paper and "paper" not in self._base_url.lower():
            raise ValueError(
                f"AlpacaAdapter: base_url does not look like a paper endpoint: {self._base_url!r}. "
                f"Set force_paper=False to override (not recommended)."
            )

        self._api: Any = None  # lazily initialized
        logger.info("[AlpacaAdapter] initialized (base_url=%s)", self._base_url)

    @property
    def is_paper(self) -> bool:
        return "paper" in self._base_url.lower()

    def _get_api(self) -> Any:
        """Lazily initialize the alpaca API client."""
        if self._api is not None:
            return self._api

        if not self._api_key or not self._api_secret:
            raise RuntimeError(
                "AlpacaAdapter: API key and secret required. "
                "Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables."
            )

        # Try alpaca-py (newer) first, then alpaca-trade-api
        try:
            from alpaca.trading.client import TradingClient  # type: ignore[import]

            self._api = TradingClient(
                self._api_key, self._api_secret, paper=self.is_paper
            )
            logger.info("[AlpacaAdapter] using alpaca-py TradingClient")
        except ImportError:
            try:
                import alpaca_trade_api as tradeapi  # type: ignore[import]

                self._api = tradeapi.REST(
                    self._api_key, self._api_secret, self._base_url
                )
                logger.info("[AlpacaAdapter] using alpaca-trade-api REST")
            except ImportError as exc:
                raise ImportError(
                    "AlpacaAdapter requires alpaca-py or alpaca-trade-api. "
                    "Install with: pip install alpaca-py"
                ) from exc

        return self._api

    def get_account(self) -> dict[str, Any]:
        """Return account summary."""
        api = self._get_api()
        try:
            acct = api.get_account()
            return {
                "equity": float(getattr(acct, "equity", 0)),
                "buying_power": float(getattr(acct, "buying_power", 0)),
                "cash": float(getattr(acct, "cash", 0)),
                "portfolio_value": float(getattr(acct, "portfolio_value", 0)),
                "status": str(getattr(acct, "status", "")),
            }
        except Exception as e:
            logger.error("[AlpacaAdapter] get_account error: %s", e)
            raise

    def get_positions(self) -> list[BrokerPosition]:
        """Return current open positions."""
        api = self._get_api()
        try:
            positions = api.get_all_positions()
        except AttributeError:
            positions = api.list_positions()

        result = []
        for pos in positions:
            result.append(
                BrokerPosition(
                    symbol=str(getattr(pos, "symbol", "")),
                    qty=float(getattr(pos, "qty", 0)),
                    avg_entry_price=float(getattr(pos, "avg_entry_price", 0)),
                    market_value=float(getattr(pos, "market_value", 0)),
                    unrealized_pnl=float(getattr(pos, "unrealized_pl", 0)),
                    unrealized_pnl_pct=float(getattr(pos, "unrealized_plpc", 0)),
                    raw={"_type": "alpaca_position"},
                )
            )
        return result

    def get_open_orders(self) -> list[BrokerOrder]:
        """Return open orders."""
        api = self._get_api()
        try:
            from alpaca.trading.enums import QueryOrderStatus  # type: ignore[import]
            from alpaca.trading.requests import GetOrdersRequest  # type: ignore[import]

            orders = api.get_orders(
                filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
        except ImportError:
            orders = api.list_orders(status="open")

        return [self._normalize_order(o) for o in orders]

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        *,
        time_in_force: str = "day",
        comment: str = "",
    ) -> BrokerOrder:
        """Submit a market order to Alpaca paper trading."""
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        side_lower = side.lower()
        if side_lower not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

        api = self._get_api()
        try:
            from alpaca.trading.enums import OrderSide, TimeInForce  # type: ignore[import]
            from alpaca.trading.requests import MarketOrderRequest  # type: ignore[import]

            order_side = OrderSide.BUY if side_lower == "buy" else OrderSide.SELL
            tif = TimeInForce.DAY if time_in_force == "day" else TimeInForce.GTC

            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
            )
            order = api.submit_order(order_data=request)
        except ImportError:
            order = api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side_lower,
                type="market",
                time_in_force=time_in_force,
            )

        normalized = self._normalize_order(order)
        logger.info(
            "[AlpacaAdapter] submitted %s %s qty=%.2f order_id=%s",
            side_lower.upper(),
            symbol,
            qty,
            normalized.order_id,
        )
        return normalized

    def cancel_all_orders(self) -> int:
        """Cancel all open orders."""
        api = self._get_api()
        try:
            result = api.cancel_orders()
            count = len(result) if hasattr(result, "__len__") else 0
        except Exception:
            orders = self.get_open_orders()
            for o in orders:
                try:
                    api.cancel_order_by_id(o.order_id)
                except Exception:
                    pass
            count = len(orders)
        logger.info("[AlpacaAdapter] cancelled %d orders", count)
        return count

    def get_order_status(self, order_id: str) -> BrokerOrder:
        """Get status of a specific order."""
        api = self._get_api()
        try:
            order = api.get_order_by_id(order_id)
        except AttributeError:
            order = api.get_order(order_id)
        return self._normalize_order(order)

    def _normalize_order(self, order: Any) -> BrokerOrder:
        """Normalize an Alpaca order object to BrokerOrder."""
        return BrokerOrder(
            order_id=str(getattr(order, "id", "")),
            symbol=str(getattr(order, "symbol", "")),
            side=str(getattr(order, "side", "")).replace("OrderSide.", "").lower(),
            qty=float(getattr(order, "qty", 0) or 0),
            order_type=str(
                getattr(order, "order_type", getattr(order, "type", "market"))
            ),
            status=str(getattr(order, "status", ""))
            .replace("OrderStatus.", "")
            .lower(),
            filled_qty=float(getattr(order, "filled_qty", 0) or 0),
            filled_avg_price=_safe_float(getattr(order, "filled_avg_price", None)),
            submitted_at=str(getattr(order, "submitted_at", "") or ""),
            filled_at=str(getattr(order, "filled_at", "") or ""),
            raw={"_type": "alpaca_order"},
        )


def _safe_float(value: Any) -> float | None:
    """Convert value to float or return None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def create_adapter_from_env(
    adapter_type: str = "alpaca",
    *,
    force_paper: bool = True,
) -> BrokerAdapter:
    """Factory: create a broker adapter from environment variables.

    Args:
        adapter_type: "alpaca" (only supported type in v1.x).
        force_paper: If True, raises if base_url is not a paper endpoint.

    Returns:
        Configured BrokerAdapter instance.
    """
    if adapter_type == "alpaca":
        return AlpacaAdapter(force_paper=force_paper)
    raise ValueError(f"Unknown adapter_type: {adapter_type!r}. Supported: 'alpaca'")


__all__ = [
    "BrokerAdapter",
    "AlpacaAdapter",
    "BrokerOrder",
    "BrokerPosition",
    "create_adapter_from_env",
]
