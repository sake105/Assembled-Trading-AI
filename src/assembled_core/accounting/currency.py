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

# B-acct-4: the hard-coded DEFAULT_FX_RATES carry NO as_of / freshness check —
# a stale rate silently mis-states cross-currency exposure. Emit a one-time
# WARNING the first time an FXConverter falls back to these defaults (i.e. it was
# constructed without operator-supplied rates), so a silent stale-FX conversion is
# at least observable. Module-level so it fires once per process, not per call.
_DEFAULT_FX_WARNED = False


def _warn_default_fx_once() -> None:
    global _DEFAULT_FX_WARNED
    if not _DEFAULT_FX_WARNED:
        _DEFAULT_FX_WARNED = True
        logger.warning(
            "[FX] Using hard-coded DEFAULT_FX_RATES fallback (no operator-supplied "
            "rates, no as_of/freshness check). Cross-currency USD conversion may be "
            "STALE — supply fresh rates to FXConverter(rates=...) for live use."
        )


@dataclass
class FXConverter:
    """Convert positions to USD using FX rates."""

    rates: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FX_RATES))

    def __post_init__(self) -> None:
        # True iff no operator-supplied rates were passed → we are running on the
        # stale hard-coded defaults. The one-time WARNING fires on first actual use
        # (to_usd), not at construction, so merely importing/constructing is silent.
        self._using_default_rates: bool = self.rates == DEFAULT_FX_RATES

    def to_usd(self, amount: float, currency: str) -> float:
        """Convert amount to USD.

        Args:
            amount: Amount in local currency.
            currency: ISO currency code.

        Returns:
            Amount in USD.
        """
        # B-acct-4: observable stale-FX fallback (one-time WARNING per process).
        if getattr(self, "_using_default_rates", False):
            _warn_default_fx_once()
        rate = self.rates.get(currency.upper(), None)
        if rate is None:
            # Silently assuming 1:1 on a typo (e.g. "GBp" pence vs "GBP", "EU"
            # vs "EUR") mis-states cross-currency exposure by 10-100×. The
            # operator must explicitly register the currency before it can
            # price through — fail closed.
            raise ValueError(
                f"[FX] Unknown currency {currency!r}; add an explicit rate "
                f"to FXConverter.rates before calling to_usd()"
            )
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
