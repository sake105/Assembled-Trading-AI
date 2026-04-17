"""Borrow cost computation for short positions.

Minimal, dependency-free helper for the paper engine. Flat-rate borrow by
default, with an optional per-symbol override table for hard-to-borrow (HTB)
names. Rates are quoted in **annualised basis points**.

Design
------
- Rates are annualised; the daily cost is ``rate_bps / 10_000 * notional / 365``.
- ``days_held`` defaults to 1 so the engine can call this once per trading day
  per short position.
- Only positions with ``qty < 0`` accrue borrow cost. Long positions are a noop.
- Symbols absent from ``rate_table`` fall back to ``default_rate_bps``.

The engine calls :func:`compute_borrow_cost` for each short position at the
end of the day and decrements ``cash`` by the total. This keeps shorting
realistic without pulling in an external borrow-rate feed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


EASY_TO_BORROW_BPS = 50.0
HARD_TO_BORROW_BPS = 500.0


@dataclass
class BorrowRateTable:
    """Per-symbol borrow rate overrides (annualised bps).

    Unknown symbols use ``default_rate_bps``. ``htb_symbols`` is a convenience
    shortcut that marks a set of symbols as hard-to-borrow without spelling
    out the exact basis-point rate.
    """

    default_rate_bps: float = EASY_TO_BORROW_BPS
    htb_rate_bps: float = HARD_TO_BORROW_BPS
    overrides: dict[str, float] = field(default_factory=dict)
    htb_symbols: set[str] = field(default_factory=set)

    def rate_bps(self, symbol: str) -> float:
        """Return annualised borrow rate in bps for ``symbol``."""
        if symbol in self.overrides:
            return float(self.overrides[symbol])
        if symbol in self.htb_symbols:
            return float(self.htb_rate_bps)
        return float(self.default_rate_bps)


def compute_borrow_cost(
    qty: float,
    price: float,
    rate_bps_annual: float,
    *,
    days_held: int = 1,
    days_in_year: int = 365,
) -> float:
    """Return the borrow cost (USD) for holding a short position.

    Long positions (``qty >= 0``) and zero-price rows return 0.0.

    Args:
        qty: Position quantity in shares. Negative for shorts.
        price: Mark price per share.
        rate_bps_annual: Annualised borrow rate in basis points.
        days_held: Number of calendar days the short was held (default 1).
        days_in_year: Calendar days used for annualisation (default 365).

    Returns:
        Cost in USD (always non-negative; engine subtracts it from cash).
    """
    if qty >= 0 or price <= 0 or rate_bps_annual <= 0 or days_held <= 0:
        return 0.0
    notional = abs(qty) * float(price)
    daily_rate = float(rate_bps_annual) / 10_000.0 / float(days_in_year)
    return notional * daily_rate * float(days_held)


def compute_borrow_cost_for_positions(
    positions: dict[str, float],
    prices: dict[str, float],
    rate_table: BorrowRateTable | None = None,
    *,
    days_held: int = 1,
) -> dict[str, float]:
    """Return a ``{symbol: borrow_cost_usd}`` dict for all short positions.

    Symbols with a non-negative quantity or missing price are skipped.
    """
    table = rate_table or BorrowRateTable()
    out: dict[str, float] = {}
    for sym, qty in positions.items():
        if qty >= 0:
            continue
        price = float(prices.get(sym, 0.0))
        rate = table.rate_bps(sym)
        cost = compute_borrow_cost(qty, price, rate, days_held=days_held)
        if cost > 0:
            out[sym] = cost
    return out
