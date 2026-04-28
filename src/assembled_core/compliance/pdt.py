"""Pattern Day Trader (PDT) rule compliance helpers.

From 50_COMPLIANCE_RECHT.md §50.5.

The SEC PDT rule: traders with < $25,000 equity who execute 4+ day-trades
within any rolling 5-business-day window are flagged as Pattern Day Traders
and their accounts are restricted for 90 days.

This module provides pure-Python helpers (no DB dependency) that can be used
by the execution layer to enforce the 3-day-trade limit when equity < $25k.

A "day trade" is defined as opening and closing the same security on the
same calendar day.
"""
from __future__ import annotations

from datetime import date, timedelta

PDT_EQUITY_THRESHOLD_USD: float = 25_000.0
PDT_MAX_DAY_TRADES_IN_5_DAYS: int = 3


def count_day_trades(
    trades: list[dict],
    reference_date: date | None = None,
    window_days: int = 5,
) -> int:
    """Count day-trades in the rolling *window_days* window ending on *reference_date*.

    Args:
        trades: List of dicts with at minimum keys:
            - ``date``: :class:`datetime.date` of the trade
            - ``symbol``: str ticker
            - ``side``: ``"buy"`` or ``"sell"``
        reference_date: End of the rolling window (defaults to today).
        window_days: Size of the rolling window in calendar days (default 5).

    Returns:
        Number of day-trade round-trips detected.

    A day-trade is counted whenever both a buy and a sell for the same symbol
    occur on the same calendar day within the window.
    """
    if reference_date is None:
        reference_date = date.today()

    cutoff = reference_date - timedelta(days=window_days - 1)
    window_trades = [t for t in trades if cutoff <= t["date"] <= reference_date]

    # Group by (date, symbol)
    pairs: dict[tuple[date, str], set[str]] = {}
    for t in window_trades:
        key = (t["date"], t["symbol"])
        pairs.setdefault(key, set()).add(t["side"].lower())

    # Day-trade = both sides present on the same day for the same symbol
    return sum(
        1 for sides in pairs.values() if "buy" in sides and "sell" in sides
    )


def can_day_trade(
    equity_usd: float,
    trades: list[dict],
    reference_date: date | None = None,
) -> tuple[bool, str]:
    """Return whether a new day-trade is permitted.

    Args:
        equity_usd: Current account equity in USD.
        trades: Recent trade history (see :func:`count_day_trades`).
        reference_date: Reference date for the rolling window.

    Returns:
        ``(allowed, reason)`` where *reason* is a short status string.
    """
    if equity_usd >= PDT_EQUITY_THRESHOLD_USD:
        return True, "pdt_threshold_met"

    count = count_day_trades(trades, reference_date=reference_date)
    if count >= PDT_MAX_DAY_TRADES_IN_5_DAYS:
        return False, f"pdt_limit_reached ({count}/{PDT_MAX_DAY_TRADES_IN_5_DAYS})"
    return True, f"pdt_ok ({count}/{PDT_MAX_DAY_TRADES_IN_5_DAYS})"
