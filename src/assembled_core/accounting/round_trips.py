"""Round-Trip P&L Analysis (Plan 8.7).

Computes per-trade P&L: entry_date, exit_date, gross_pnl, net_pnl, holding_days.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass
class RoundTrip:
    """A complete buy→sell round trip."""
    symbol: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    commission: float = 0.0

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.commission

    @property
    def holding_days(self) -> int:
        return (self.exit_date - self.entry_date).days

    @property
    def return_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price * 100


def compute_round_trips(trades_df: pd.DataFrame) -> list[RoundTrip]:
    """Compute round trips from a trades DataFrame.

    Expects columns: symbol, date, side (BUY/SELL), price, quantity, commission.

    Returns:
        List of RoundTrip objects.
    """
    if trades_df.empty:
        return []

    trips = []
    # Simple FIFO matching per symbol
    open_positions: dict[str, list] = {}

    for _, row in trades_df.sort_values("date").iterrows():
        sym = str(row["symbol"])
        side = str(row.get("side", "BUY")).upper()
        price = float(row["price"])
        qty = float(row["quantity"])
        trade_date = pd.Timestamp(row["date"]).date()
        comm = float(row.get("commission", 0.0))

        if side == "BUY":
            if sym not in open_positions:
                open_positions[sym] = []
            open_positions[sym].append({
                "date": trade_date,
                "price": price,
                "quantity": qty,
                "commission": comm,
            })
        elif side == "SELL" and sym in open_positions:
            remaining = qty
            while remaining > 0 and open_positions[sym]:
                entry = open_positions[sym][0]
                matched = min(remaining, entry["quantity"])
                gross = matched * (price - entry["price"])
                trips.append(RoundTrip(
                    symbol=sym,
                    entry_date=entry["date"],
                    exit_date=trade_date,
                    entry_price=entry["price"],
                    exit_price=price,
                    quantity=matched,
                    gross_pnl=round(gross, 4),
                    commission=round(comm * matched / qty + entry["commission"] * matched / entry["quantity"], 4),
                ))
                entry["quantity"] -= matched
                remaining -= matched
                if entry["quantity"] <= 1e-10:
                    open_positions[sym].pop(0)

    return trips


def round_trip_summary(trips: list[RoundTrip]) -> dict:
    """Summarize round trip statistics."""
    if not trips:
        return {"n_trips": 0}

    pnls = [t.net_pnl for t in trips]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    return {
        "n_trips": len(trips),
        "total_pnl": round(sum(pnls), 2),
        "win_rate": round(len(winners) / len(pnls), 4) if pnls else 0,
        "avg_win": round(sum(winners) / len(winners), 2) if winners else 0,
        "avg_loss": round(sum(losers) / len(losers), 2) if losers else 0,
        "profit_factor": round(sum(winners) / abs(sum(losers)), 4) if losers and sum(losers) != 0 else float("inf"),
        "avg_holding_days": round(sum(t.holding_days for t in trips) / len(trips), 1),
    }


__all__ = ["RoundTrip", "compute_round_trips", "round_trip_summary"]
