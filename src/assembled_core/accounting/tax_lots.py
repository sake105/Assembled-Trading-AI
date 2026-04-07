"""Tax Lot Tracking — FIFO (Plan 8.4).

Tracks cost basis per lot for tax-loss harvesting and P&L attribution.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date


@dataclass
class TaxLot:
    """A single tax lot."""
    symbol: str
    buy_date: date
    quantity: float
    cost_basis_per_share: float

    @property
    def total_cost(self) -> float:
        return self.quantity * self.cost_basis_per_share


@dataclass
class TaxLotTracker:
    """FIFO tax lot tracker for a portfolio."""
    lots: dict[str, deque] = field(default_factory=dict)

    def buy(self, symbol: str, quantity: float, price: float, trade_date: date) -> None:
        """Record a buy, creating a new tax lot."""
        if symbol not in self.lots:
            self.lots[symbol] = deque()
        self.lots[symbol].append(TaxLot(
            symbol=symbol,
            buy_date=trade_date,
            quantity=quantity,
            cost_basis_per_share=price,
        ))

    def sell(self, symbol: str, quantity: float, price: float, trade_date: date) -> float:
        """Sell shares using FIFO. Returns realized P&L."""
        if symbol not in self.lots:
            return 0.0

        remaining = quantity
        realized_pnl = 0.0
        lots = self.lots[symbol]

        while remaining > 0 and lots:
            lot = lots[0]
            sold_from_lot = min(remaining, lot.quantity)
            realized_pnl += sold_from_lot * (price - lot.cost_basis_per_share)
            lot.quantity -= sold_from_lot
            remaining -= sold_from_lot
            if lot.quantity <= 1e-10:
                lots.popleft()

        return round(realized_pnl, 4)

    def unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """Compute unrealized P&L for a symbol."""
        if symbol not in self.lots:
            return 0.0
        return sum(
            lot.quantity * (current_price - lot.cost_basis_per_share)
            for lot in self.lots[symbol]
        )

    def tax_loss_harvesting_candidates(
        self, current_prices: dict[str, float], min_loss: float = -100.0,
    ) -> list[dict]:
        """Find lots with unrealized losses for tax-loss harvesting."""
        candidates = []
        for symbol, lots in self.lots.items():
            price = current_prices.get(symbol, 0.0)
            for lot in lots:
                loss = lot.quantity * (price - lot.cost_basis_per_share)
                if loss < min_loss:
                    candidates.append({
                        "symbol": symbol,
                        "buy_date": str(lot.buy_date),
                        "quantity": lot.quantity,
                        "cost_basis": lot.cost_basis_per_share,
                        "current_price": price,
                        "unrealized_loss": round(loss, 2),
                    })
        return sorted(candidates, key=lambda x: x["unrealized_loss"])


__all__ = ["TaxLot", "TaxLotTracker"]
