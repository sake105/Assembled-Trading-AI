"""Multi-Currency Support (Plan 8.8).

FX rate management and position USD conversion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Default FX rates (vs USD) — fallback for offline usage
DEFAULT_FX_RATES: dict[str, float] = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.26,
    "JPY": 0.0067,
    "CHF": 1.12,
    "CAD": 0.74,
    "AUD": 0.65,
    "CNY": 0.14,
}


@dataclass
class FXConverter:
    """Convert positions to USD using FX rates."""

    rates: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FX_RATES))

    def to_usd(self, amount: float, currency: str) -> float:
        """Convert amount to USD.

        Args:
            amount: Amount in local currency.
            currency: ISO currency code.

        Returns:
            Amount in USD.
        """
        rate = self.rates.get(currency.upper(), None)
        if rate is None:
            logger.warning("[FX] Unknown currency %s, assuming 1:1 with USD", currency)
            return amount
        return amount * rate

    def convert_positions_to_usd(
        self,
        positions: dict[str, dict],
        currency_map: dict[str, str],
    ) -> dict[str, dict]:
        """Convert all positions to USD.

        Args:
            positions: Symbol → {notional, ...} dict.
            currency_map: Symbol → currency code.

        Returns:
            Positions with added usd_notional field.
        """
        for sym, pos in positions.items():
            ccy = currency_map.get(sym, "USD")
            notional = pos.get("notional", 0.0)
            pos["usd_notional"] = self.to_usd(notional, ccy)
            pos["currency"] = ccy
        return positions


__all__ = ["FXConverter", "DEFAULT_FX_RATES"]
