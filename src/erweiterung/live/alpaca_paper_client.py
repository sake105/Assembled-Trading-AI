"""Alpaca Paper-Trading-Client (alpaca-py).

Args
----
ALPACA_API_KEY und ALPACA_SECRET_KEY werden via env-Variablen gelesen.
**Niemals** im Code hardcoden.

Funktionen
----------
- ``get_account``  : Konto-Status (cash, equity, buying_power)
- ``get_positions``: aktuelle Positionen
- ``submit_order`` : Market/Limit-Order
- ``cancel_all``   : Alle offenen Orders schließen
- ``stream_trades``: Websocket-Trades-Streaming (callback-based)

Hinweis
-------
Diese Klasse ist ein **dünner Wrapper** um alpaca-py.  Für Production-Use bitte
zusätzliches Logging, Retry und Circuit-Breaker auf Anwender-Ebene
implementieren.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    symbol: str
    qty: float
    side: str  # 'buy' or 'sell'
    order_type: str = "market"  # 'market' | 'limit' | 'stop'
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"


class AlpacaPaperClient:
    """Wrapper um alpaca-py.TradingClient (paper=True)."""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not self.api_key or not self.secret_key:
            raise RuntimeError(
                "Alpaca API keys missing. Set ALPACA_API_KEY + ALPACA_SECRET_KEY env vars."
            )
        try:
            from alpaca.trading.client import TradingClient  # type: ignore
        except ImportError as e:
            raise RuntimeError("pip install alpaca-py") from e
        self._client = TradingClient(self.api_key, self.secret_key, paper=True)

    def get_account(self) -> dict:
        acct = self._client.get_account()
        return {
            "account_number": acct.account_number,
            "cash": float(acct.cash),
            "equity": float(acct.equity),
            "buying_power": float(acct.buying_power),
            "currency": acct.currency,
            "status": str(acct.status),
        }

    def get_positions(self) -> list[dict]:
        out = []
        for p in self._client.get_all_positions():
            out.append(
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": (
                        float(p.current_price) if p.current_price else None
                    ),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "side": p.side,
                }
            )
        return out

    def submit_order(self, request: OrderRequest) -> dict:
        from alpaca.trading.requests import (  # type: ignore
            LimitOrderRequest,
            MarketOrderRequest,
            StopOrderRequest,
        )
        from alpaca.trading.enums import OrderSide, TimeInForce  # type: ignore

        side = OrderSide.BUY if request.side.lower() == "buy" else OrderSide.SELL
        tif_map = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
        }
        tif = tif_map.get(request.time_in_force.lower(), TimeInForce.DAY)
        if request.order_type == "market":
            req = MarketOrderRequest(
                symbol=request.symbol, qty=request.qty, side=side, time_in_force=tif
            )
        elif request.order_type == "limit":
            if request.limit_price is None:
                raise ValueError("limit_price required for limit order")
            req = LimitOrderRequest(
                symbol=request.symbol,
                qty=request.qty,
                side=side,
                time_in_force=tif,
                limit_price=request.limit_price,
            )
        elif request.order_type == "stop":
            if request.stop_price is None:
                raise ValueError("stop_price required for stop order")
            req = StopOrderRequest(
                symbol=request.symbol,
                qty=request.qty,
                side=side,
                time_in_force=tif,
                stop_price=request.stop_price,
            )
        else:
            raise ValueError(f"unknown order type: {request.order_type}")
        order = self._client.submit_order(req)
        return {
            "id": order.id,
            "symbol": order.symbol,
            "qty": float(order.qty),
            "side": str(order.side),
            "status": str(order.status),
            "submitted_at": str(order.submitted_at),
        }

    def cancel_all(self) -> int:
        canceled = self._client.cancel_orders()
        return len(canceled)


def emergency_kill_switch(client: AlpacaPaperClient) -> dict:
    """Cancel all orders + close all positions. For absolute emergencies."""
    n_canceled = client.cancel_all()
    closed_summary = []
    for pos in client.get_positions():
        side = "sell" if pos["side"] == "long" else "buy"
        try:
            r = client.submit_order(
                OrderRequest(
                    symbol=pos["symbol"],
                    qty=abs(pos["qty"]),
                    side=side,
                    order_type="market",
                )
            )
            closed_summary.append(
                {"symbol": pos["symbol"], "ok": True, "order_id": r.get("id")}
            )
        except Exception as e:  # noqa: BLE001
            closed_summary.append(
                {"symbol": pos["symbol"], "ok": False, "error": str(e)}
            )
    return {"orders_canceled": n_canceled, "positions_closed": closed_summary}


__all__ = ["OrderRequest", "AlpacaPaperClient", "emergency_kill_switch"]
