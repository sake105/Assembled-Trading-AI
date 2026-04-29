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

from src.assembled_core.errors import PriceLookupError

logger = logging.getLogger(__name__)


@dataclass
class BrokerOrder:
    """Normalized order representation returned by broker adapters."""

    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    qty: float
    order_type: str  # "market", "limit", "stop", "stop_limit", "moc", "loc"
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
    def submit_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        *,
        time_in_force: str = "day",
        comment: str = "",
    ) -> BrokerOrder:
        """Submit a limit order.

        Args:
            symbol: Ticker symbol.
            qty: Number of shares (positive).
            side: "buy" or "sell".
            limit_price: Limit price for the order.
            time_in_force: "day", "gtc", etc.
            comment: Optional order comment/tag.

        Returns:
            BrokerOrder with order_id and initial status.
        """
        ...

    @abstractmethod
    def submit_stop_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        stop_price: float,
        *,
        limit_price: float | None = None,
        time_in_force: str = "day",
        comment: str = "",
    ) -> BrokerOrder:
        """Submit a stop or stop-limit order.

        Args:
            symbol: Ticker symbol.
            qty: Number of shares (positive).
            side: "buy" or "sell".
            stop_price: Trigger price for the stop.
            limit_price: If provided, creates a stop-limit order.
            time_in_force: "day", "gtc", etc.
            comment: Optional order comment/tag.

        Returns:
            BrokerOrder with order_id and initial status.
        """
        ...

    def submit_moc_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        *,
        comment: str = "",
    ) -> BrokerOrder:
        """Submit a Market-On-Close (MOC) order.

        Default implementation submits a market order with time_in_force='cls'.
        Override for broker-specific MOC support.

        Args:
            symbol: Ticker symbol.
            qty: Number of shares (positive).
            side: "buy" or "sell".
            comment: Optional order comment/tag.

        Returns:
            BrokerOrder with order_id and initial status.
        """
        return self.submit_market_order(
            symbol, qty, side, time_in_force="cls", comment=comment or "MOC",
        )

    def submit_loc_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        *,
        comment: str = "",
    ) -> BrokerOrder:
        """Submit a Limit-On-Close (LOC) order.

        Default implementation submits a limit order with time_in_force='cls'.
        Override for broker-specific LOC support.

        Args:
            symbol: Ticker symbol.
            qty: Number of shares (positive).
            side: "buy" or "sell".
            limit_price: Limit price for the order.
            comment: Optional order comment/tag.

        Returns:
            BrokerOrder with order_id and initial status.
        """
        return self.submit_limit_order(
            symbol, qty, side, limit_price, time_in_force="cls", comment=comment or "LOC",
        )

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
        enforce_market_hours: bool = True,
        max_orders_per_cycle: int = 50,
        max_notional_per_cycle: float = 100_000.0,
    ) -> None:
        """Initialize Alpaca adapter.

        Args:
            api_key: Alpaca API key (or set ALPACA_API_KEY env var).
            api_secret: Alpaca API secret (or set ALPACA_API_SECRET env var).
            base_url: API base URL. Defaults to paper trading URL.
            force_paper: If True (default), raises if base_url looks like live endpoint.
            enforce_market_hours: If True, reject orders outside NYSE regular hours.
            max_orders_per_cycle: Maximum orders per cycle (safety limit).
            max_notional_per_cycle: Maximum total notional value per cycle.
        """
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._api_secret = api_secret or os.environ.get("ALPACA_API_SECRET", "")
        self._base_url = base_url or os.environ.get(
            "ALPACA_BASE_URL", self.PAPER_BASE_URL
        )

        is_paper_url = "paper" in self._base_url.lower()

        if force_paper and not is_paper_url:
            raise ValueError(
                f"AlpacaAdapter: base_url does not look like a paper endpoint: {self._base_url!r}. "
                f"Set force_paper=False to override (not recommended)."
            )

        # Secondary live-trading guard: even with force_paper=False, live trading
        # requires the explicit env var ALPACA_ALLOW_LIVE=true to prevent accidents.
        if not force_paper and not is_paper_url:
            allow_live = os.environ.get("ALPACA_ALLOW_LIVE", "").strip().lower()
            if allow_live != "true":
                raise ValueError(
                    f"AlpacaAdapter: connecting to a live endpoint ({self._base_url!r}) requires "
                    "ALPACA_ALLOW_LIVE=true to be set in the environment. "
                    "This is a two-step safety gate. Set force_paper=True to use paper trading."
                )
            logger.warning(
                "[AlpacaAdapter] LIVE TRADING MODE — base_url=%s (ALPACA_ALLOW_LIVE=true)",
                self._base_url,
            )

        self._api: Any = None  # lazily initialized
        self._enforce_market_hours = enforce_market_hours
        self._max_orders_per_cycle = max_orders_per_cycle
        self._max_notional_per_cycle = max_notional_per_cycle
        self._cycle_order_count = 0
        self._cycle_notional_total = 0.0
        logger.warning(
            "[AlpacaAdapter] initialized — base_url=%s is_paper=%s",
            self._base_url,
            is_paper_url,
        )

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

    def reset_cycle_counters(self) -> None:
        """Reset per-cycle order and notional counters. Call at start of each trading cycle."""
        self._cycle_order_count = 0
        self._cycle_notional_total = 0.0
        logger.info("[AlpacaAdapter] cycle counters reset")

    def _validate_market_hours(self) -> None:
        """Check if NYSE is currently in regular trading hours.

        Uses zoneinfo for correct US/Eastern timezone handling (EST/EDT).
        Raises MarketClosedError if market is closed and enforce_market_hours is True.
        """
        if not self._enforce_market_hours:
            return
        from src.assembled_core.execution.api_resilience import MarketClosedError

        try:
            import datetime

            from src.assembled_core.data.calendar import is_trading_day_safe

            # Use proper US/Eastern timezone (handles EST/EDT automatically)
            try:
                from zoneinfo import ZoneInfo
                et_tz = ZoneInfo("America/New_York")
            except ImportError:
                # Python < 3.9 fallback or missing tzdata
                et_tz = datetime.timezone(datetime.timedelta(hours=-5))
                logger.debug(
                    "[AlpacaAdapter] zoneinfo unavailable, using UTC-5 fallback"
                )

            now_et = datetime.datetime.now(et_tz)
            if not is_trading_day_safe(now_et):
                raise MarketClosedError(
                    f"Market is closed today ({now_et.strftime('%Y-%m-%d %A')}). "
                    "Set enforce_market_hours=False to override."
                )
            hour, minute = now_et.hour, now_et.minute
            market_open = (hour > 9) or (hour == 9 and minute >= 30)
            market_close = hour < 16
            if not (market_open and market_close):
                raise MarketClosedError(
                    f"Outside regular market hours ({now_et.strftime('%H:%M')} ET). "
                    "NYSE regular session: 09:30–16:00 ET. "
                    "Set enforce_market_hours=False to override."
                )
        except ImportError:
            logger.warning(
                "[AlpacaAdapter] calendar module not available, skipping market hours check"
            )

    def _check_cycle_limits(self, qty: float, estimated_price: float = 0.0) -> None:
        """Check per-cycle order count and notional limits.

        Raises ValueError if limits would be exceeded.
        """
        if self._cycle_order_count >= self._max_orders_per_cycle:
            raise ValueError(
                f"[AlpacaAdapter] per-cycle order limit reached "
                f"({self._cycle_order_count}/{self._max_orders_per_cycle}). "
                "Call reset_cycle_counters() at cycle start."
            )
        if estimated_price > 0:
            notional = qty * estimated_price
            if self._cycle_notional_total + notional > self._max_notional_per_cycle:
                raise ValueError(
                    f"[AlpacaAdapter] per-cycle notional limit would be exceeded: "
                    f"${self._cycle_notional_total + notional:,.0f} > "
                    f"${self._max_notional_per_cycle:,.0f}"
                )

    def _estimate_price(self, symbol: str) -> float:
        """Get price estimate for notional limit checks.

        Tries to get the last trade price from the broker.
        Raises PriceLookupError if price cannot be obtained — callers must
        not silently proceed with a zero/garbage price.
        """
        errors: list[str] = []
        try:
            api = self._get_api()
        except Exception as exc:
            raise PriceLookupError(
                symbol, f"Cannot get API client: {exc}"
            ) from exc

        # Try alpaca-py style
        try:
            from alpaca.data import StockHistoricalDataClient  # type: ignore[import]
            from alpaca.data.requests import (
                StockLatestTradeRequest,  # type: ignore[import]
            )

            data_client = StockHistoricalDataClient(
                api_key=api._api_key if hasattr(api, "_api_key") else None,
                secret_key=api._secret_key if hasattr(api, "_secret_key") else None,
            )
            trade = data_client.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=symbol)
            )
            if hasattr(trade, symbol):
                return float(getattr(trade, symbol).price)
            return float(trade[symbol].price)
        except ImportError:
            errors.append("alpaca-py SDK not installed")
        except Exception as exc:
            errors.append(f"alpaca-py call failed: {exc}")

        # Fallback: try legacy alpaca_trade_api
        try:
            trade = api.get_latest_trade(symbol)
            return float(trade.price)
        except Exception as exc:
            errors.append(f"legacy alpaca_trade_api failed: {exc}")

        # All methods failed — raise instead of returning garbage
        logger.error(
            "[AlpacaAdapter] Price lookup failed for %s. Errors: %s",
            symbol,
            "; ".join(errors),
        )
        raise PriceLookupError(symbol, "; ".join(errors))

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

        # Safety gates
        self._validate_market_hours()
        estimated_price = self._estimate_price(symbol)
        self._check_cycle_limits(qty, estimated_price)

        api = self._get_api()
        try:
            from alpaca.trading.enums import (  # type: ignore[import]
                OrderSide,
                TimeInForce,
            )
            from alpaca.trading.requests import (
                MarketOrderRequest,  # type: ignore[import]
            )

            order_side = OrderSide.BUY if side_lower == "buy" else OrderSide.SELL
            tif = TimeInForce.DAY if time_in_force == "day" else TimeInForce.GTC

            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
            )
            try:
                order = api.submit_order(order_data=request)
            except Exception as _broker_err:
                from src.assembled_core.execution.idempotency import is_duplicate_error
                if is_duplicate_error(str(_broker_err)):
                    logger.warning("[AlpacaAdapter] duplicate client_order_id detected — skipping retry: %s", _broker_err)
                    raise
                raise
        except ImportError:
            order = api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side_lower,
                type="market",
                time_in_force=time_in_force,
            )

        normalized = self._normalize_order(order)
        self._cycle_order_count += 1
        if estimated_price > 0:
            self._cycle_notional_total += qty * estimated_price
        logger.info(
            "[AlpacaAdapter] submitted %s %s qty=%.2f order_id=%s "
            "(cycle %d/%d, notional $%.0f/$%.0f)",
            side_lower.upper(),
            symbol,
            qty,
            normalized.order_id,
            self._cycle_order_count,
            self._max_orders_per_cycle,
            self._cycle_notional_total,
            self._max_notional_per_cycle,
        )
        return normalized

    def submit_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        *,
        time_in_force: str = "day",
        comment: str = "",
    ) -> BrokerOrder:
        """Submit a limit order to Alpaca."""
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        if limit_price <= 0:
            raise ValueError(f"limit_price must be positive, got {limit_price}")
        side_lower = side.lower()
        if side_lower not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

        self._validate_market_hours()
        self._check_cycle_limits(qty, limit_price)

        api = self._get_api()
        try:
            from alpaca.trading.enums import (  # type: ignore[import]
                OrderSide,
                TimeInForce,
            )
            from alpaca.trading.requests import (
                LimitOrderRequest,  # type: ignore[import]
            )

            order_side = OrderSide.BUY if side_lower == "buy" else OrderSide.SELL
            tif_map = {"day": TimeInForce.DAY, "gtc": TimeInForce.GTC, "cls": TimeInForce.CLS}
            tif = tif_map.get(time_in_force.lower(), TimeInForce.DAY)

            request = LimitOrderRequest(
                symbol=symbol, qty=qty, side=order_side,
                time_in_force=tif, limit_price=limit_price,
            )
            order = api.submit_order(order_data=request)
        except ImportError:
            order = api.submit_order(
                symbol=symbol, qty=qty, side=side_lower,
                type="limit", time_in_force=time_in_force,
                limit_price=str(limit_price),
            )

        normalized = self._normalize_order(order)
        self._cycle_order_count += 1
        self._cycle_notional_total += qty * limit_price
        logger.info(
            "[AlpacaAdapter] submitted LIMIT %s %s qty=%.2f limit=$%.2f order_id=%s",
            side_lower.upper(), symbol, qty, limit_price, normalized.order_id,
        )
        return normalized

    def submit_stop_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        stop_price: float,
        *,
        limit_price: float | None = None,
        time_in_force: str = "day",
        comment: str = "",
    ) -> BrokerOrder:
        """Submit a stop or stop-limit order to Alpaca."""
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        if stop_price <= 0:
            raise ValueError(f"stop_price must be positive, got {stop_price}")
        side_lower = side.lower()
        if side_lower not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

        self._validate_market_hours()
        self._check_cycle_limits(qty, stop_price)

        is_stop_limit = limit_price is not None
        api = self._get_api()

        try:
            from alpaca.trading.enums import (  # type: ignore[import]
                OrderSide,
                TimeInForce,
            )
            from alpaca.trading.requests import (  # type: ignore[import]
                StopLimitOrderRequest,
                StopOrderRequest,
            )

            order_side = OrderSide.BUY if side_lower == "buy" else OrderSide.SELL
            tif_map = {"day": TimeInForce.DAY, "gtc": TimeInForce.GTC, "cls": TimeInForce.CLS}
            tif = tif_map.get(time_in_force.lower(), TimeInForce.DAY)

            if is_stop_limit:
                request = StopLimitOrderRequest(
                    symbol=symbol, qty=qty, side=order_side,
                    time_in_force=tif, stop_price=stop_price,
                    limit_price=limit_price,
                )
            else:
                request = StopOrderRequest(
                    symbol=symbol, qty=qty, side=order_side,
                    time_in_force=tif, stop_price=stop_price,
                )
            order = api.submit_order(order_data=request)
        except ImportError:
            order_type = "stop_limit" if is_stop_limit else "stop"
            kwargs: dict[str, Any] = {
                "symbol": symbol, "qty": qty, "side": side_lower,
                "type": order_type, "time_in_force": time_in_force,
                "stop_price": str(stop_price),
            }
            if is_stop_limit:
                kwargs["limit_price"] = str(limit_price)
            order = api.submit_order(**kwargs)

        normalized = self._normalize_order(order)
        self._cycle_order_count += 1
        self._cycle_notional_total += qty * stop_price
        order_desc = "STOP_LIMIT" if is_stop_limit else "STOP"
        logger.info(
            "[AlpacaAdapter] submitted %s %s %s qty=%.2f stop=$%.2f%s order_id=%s",
            order_desc, side_lower.upper(), symbol, qty, stop_price,
            f" limit=${limit_price:.2f}" if is_stop_limit else "",
            normalized.order_id,
        )
        return normalized

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns the count actually cancelled."""
        api = self._get_api()
        try:
            result = api.cancel_orders()
            count = len(result) if hasattr(result, "__len__") else 0
        except Exception as bulk_exc:
            logger.warning(
                "[AlpacaAdapter] bulk cancel failed: %s — falling back to per-order",
                bulk_exc,
            )
            orders = self.get_open_orders()
            count = 0
            for o in orders:
                try:
                    api.cancel_order_by_id(o.order_id)
                    count += 1
                except Exception as per_exc:
                    # Surface per-order failures so a kill-switch flatten
                    # does not report a false-positive full-cancel count.
                    logger.warning(
                        "[AlpacaAdapter] cancel_order_by_id(%s) failed: %s",
                        o.order_id, per_exc,
                    )
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
