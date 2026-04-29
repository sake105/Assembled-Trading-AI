"""Liquidity-aware position sizer — ADV, market-cap, and time-to-liquidate caps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SizeResult:
    final_qty: int
    signal_qty: int
    adv_cap: int
    mcap_cap: int
    liq_cap: int
    binding_constraint: str


class LiquidityAwareSizer:
    """Cap position size by ADV%, market-cap fraction, and time-to-liquidate.

    Implements Beraldi-Lehalle-Almgren participation-rate model.
    """

    def __init__(
        self,
        max_pct_adv: float = 0.05,
        max_pct_market_cap: float = 0.001,
        max_days_to_liquidate: float = 1.0,
        target_pov_pct: float = 0.10,
    ) -> None:
        self.max_pct_adv = max_pct_adv
        self.max_pct_market_cap = max_pct_market_cap
        self.max_days_to_liquidate = max_days_to_liquidate
        self.target_pov_pct = target_pov_pct

    def size_position(self, signal_target_qty: int, symbol_data: dict[str, Any]) -> SizeResult:
        """Return the liquidity-capped position size.

        Parameters
        ----------
        signal_target_qty:
            Desired quantity from signal/portfolio layer.
        symbol_data:
            Dict with keys: ``adv`` (avg daily volume in shares), ``price``,
            ``market_cap`` (total market cap in USD). All required.
        """
        adv = float(symbol_data.get("adv", 0))
        price = float(symbol_data.get("price", 1))
        market_cap = float(symbol_data.get("market_cap", 0))

        adv_cap = int(adv * self.max_pct_adv)
        mcap_cap = int(market_cap / max(price, 1e-9) * self.max_pct_market_cap)
        liq_cap = int(self.max_days_to_liquidate * self.target_pov_pct * adv)

        effective_cap = min(adv_cap, mcap_cap, liq_cap)
        final_qty = min(signal_target_qty, max(0, effective_cap))

        if effective_cap == adv_cap:
            binding = "adv"
        elif effective_cap == mcap_cap:
            binding = "mcap"
        else:
            binding = "liq"

        return SizeResult(
            final_qty=final_qty,
            signal_qty=signal_target_qty,
            adv_cap=adv_cap,
            mcap_cap=mcap_cap,
            liq_cap=liq_cap,
            binding_constraint=binding,
        )

    def is_liquid_enough(self, qty: int, symbol_data: dict[str, Any]) -> bool:
        """Return True if qty does not exceed any cap."""
        result = self.size_position(qty, symbol_data)
        return result.final_qty == qty
