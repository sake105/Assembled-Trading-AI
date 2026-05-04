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

import os
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


def load_rate_table_from_yaml(
    path: str | "os.PathLike[str]",
) -> BorrowRateTable:
    """Load a :class:`BorrowRateTable` from ``configs/htb_symbols.yaml``.

    The YAML schema is:

    .. code-block:: yaml

        default_rates_bps:
          easy: 50
          htb: 300
          special: 500
        symbols:
          GME:
            borrow_bps: 500
            tier: "special"

    Per-symbol ``borrow_bps`` wins over tier defaults. Unknown symbols keep
    the caller-supplied default rate.
    """
    import os  # noqa: F401  (typing support)
    from pathlib import Path

    import yaml  # type: ignore[import-untyped]

    text = Path(path).read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"[borrow_costs] Malformed YAML in {path}: {exc}") from exc

    default_rates = data.get("default_rates_bps") or {}
    easy = float(default_rates.get("easy", EASY_TO_BORROW_BPS))
    htb = float(default_rates.get("htb", HARD_TO_BORROW_BPS))

    overrides: dict[str, float] = {}
    htb_symbols: set[str] = set()
    symbols = data.get("symbols") or {}
    for sym, entry in symbols.items():
        entry = entry or {}
        bps = entry.get("borrow_bps")
        if bps is not None:
            overrides[str(sym)] = float(bps)
        tier = (entry.get("tier") or "").lower()
        if tier in {"htb", "special"}:
            htb_symbols.add(str(sym))

    return BorrowRateTable(
        default_rate_bps=easy,
        htb_rate_bps=htb,
        overrides=overrides,
        htb_symbols=htb_symbols,
    )


def compute_borrow_cost_for_positions(
    positions: dict[str, float],
    prices: dict[str, float],
    rate_table: BorrowRateTable | None = None,
    *,
    days_held: int = 1,
) -> dict[str, float]:
    """Return a ``{symbol: borrow_cost_usd}`` dict for all short positions.

    Symbols with a non-negative quantity or missing price are skipped. A
    missing-price short is a silent attribution bug — the cost is reported
    as 0 but financing drag is still economically real. We emit a WARN per
    occurrence so a downstream attribution diff can be traced back to a
    price-feed gap rather than look like a healthy zero-cost short.
    """
    import logging

    log = logging.getLogger(__name__)
    table = rate_table or BorrowRateTable()
    out: dict[str, float] = {}
    for sym, qty in positions.items():
        if qty >= 0:
            continue
        if sym not in prices or prices.get(sym) is None:
            log.warning(
                "[BORROW] missing price for short %s qty=%s — borrow cost "
                "reported as 0 but financing drag is real",
                sym,
                qty,
            )
            continue
        price = float(prices.get(sym, 0.0))
        rate = table.rate_bps(sym)
        cost = compute_borrow_cost(qty, price, rate, days_held=days_held)
        if cost > 0:
            out[sym] = cost
    return out
