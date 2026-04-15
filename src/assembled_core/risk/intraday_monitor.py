"""Intraday Risk Monitoring (M23 Task 23.5).

Real-time PnL calculation, intraday VaR breach detection, and
kill-switch triggers based on streaming prices.

Currently all risk checks run only EOD — this module closes that gap.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IntradayRiskConfig:
    """Configuration for intraday risk monitor."""
    max_intraday_drawdown_pct: float = 3.0   # Kill switch trigger
    warning_drawdown_pct: float = 1.5        # Warning alert
    var_confidence: float = 0.99             # VaR confidence level
    var_horizon_minutes: int = 60            # VaR horizon
    check_interval_seconds: float = 60.0     # How often to check
    max_single_loss_pct: float = 5.0         # Max single-position loss before alert


@dataclass
class PositionSnapshot:
    """A position with real-time pricing."""
    symbol: str
    shares: int
    entry_price: float
    current_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0

    def update_price(self, price: float) -> None:
        self.current_price = price
        self.pnl = (price - self.entry_price) * self.shares
        if self.entry_price > 0:
            self.pnl_pct = (price / self.entry_price - 1.0) * 100


@dataclass
class RiskAlert:
    """An intraday risk alert."""
    level: str         # "WARNING", "CRITICAL", "KILL_SWITCH"
    message: str
    metric: str        # Which metric triggered
    value: float       # Current value
    threshold: float   # Threshold that was breached
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class IntradayRiskMonitor:
    """Real-time intraday risk monitor.

    Tracks positions, computes real-time PnL, and triggers alerts
    when risk thresholds are breached.
    """

    def __init__(self, config: IntradayRiskConfig | None = None):
        self.config = config or IntradayRiskConfig()
        self._positions: dict[str, PositionSnapshot] = {}
        self._starting_equity: float = 0.0
        self._high_water_mark: float = 0.0
        self._alerts: list[RiskAlert] = []
        self._kill_switch_triggered = False

    def initialize(
        self,
        positions: dict[str, tuple[int, float]],
        starting_equity: float,
    ) -> None:
        """Initialize monitor with current positions.

        Args:
            positions: {symbol: (shares, entry_price)}.
            starting_equity: Portfolio equity at start of day.
        """
        self._positions.clear()
        for sym, (shares, entry) in positions.items():
            self._positions[sym] = PositionSnapshot(
                symbol=sym, shares=shares, entry_price=entry, current_price=entry,
            )
        self._starting_equity = starting_equity
        self._high_water_mark = starting_equity
        self._kill_switch_triggered = False
        self._alerts.clear()

    def update_price(self, symbol: str, price: float) -> list[RiskAlert]:
        """Update price for a symbol and check risk.

        Args:
            symbol: Ticker symbol.
            price: Current price.

        Returns:
            List of new risk alerts (empty if all OK).
        """
        if symbol in self._positions:
            self._positions[symbol].update_price(price)

        return self.check_risk()

    def update_prices(self, prices: dict[str, float]) -> list[RiskAlert]:
        """Batch update prices.

        Args:
            prices: {symbol: current_price}.

        Returns:
            List of new risk alerts.
        """
        for sym, price in prices.items():
            if sym in self._positions:
                self._positions[sym].update_price(price)

        return self.check_risk()

    def check_risk(self) -> list[RiskAlert]:
        """Run all intraday risk checks.

        Returns:
            List of new risk alerts.
        """
        if self._kill_switch_triggered:
            return []

        new_alerts = []
        cfg = self.config

        # Total PnL
        total_pnl = sum(p.pnl for p in self._positions.values())
        current_equity = self._starting_equity + total_pnl
        self._high_water_mark = max(self._high_water_mark, current_equity)

        # Intraday drawdown from high water mark
        if self._high_water_mark > 0:
            intraday_dd = (current_equity / self._high_water_mark - 1.0) * 100
        else:
            intraday_dd = 0.0

        # Check intraday drawdown
        if intraday_dd <= -cfg.max_intraday_drawdown_pct:
            alert = RiskAlert(
                level="KILL_SWITCH",
                message=f"Intraday drawdown {intraday_dd:.2f}% exceeds -{cfg.max_intraday_drawdown_pct}%",
                metric="intraday_drawdown",
                value=intraday_dd,
                threshold=-cfg.max_intraday_drawdown_pct,
            )
            new_alerts.append(alert)
            self._kill_switch_triggered = True
            logger.critical("[IntradayRisk] KILL SWITCH: %s", alert.message)

        elif intraday_dd <= -cfg.warning_drawdown_pct:
            alert = RiskAlert(
                level="WARNING",
                message=f"Intraday drawdown {intraday_dd:.2f}% approaching limit",
                metric="intraday_drawdown",
                value=intraday_dd,
                threshold=-cfg.warning_drawdown_pct,
            )
            new_alerts.append(alert)

        # Check single position losses
        for pos in self._positions.values():
            if pos.pnl_pct <= -cfg.max_single_loss_pct:
                alert = RiskAlert(
                    level="CRITICAL",
                    message=f"{pos.symbol} down {pos.pnl_pct:.2f}% (${pos.pnl:,.0f})",
                    metric="single_position_loss",
                    value=pos.pnl_pct,
                    threshold=-cfg.max_single_loss_pct,
                )
                new_alerts.append(alert)

        # Parametric intraday VaR estimate
        if self._starting_equity > 0 and len(self._positions) > 0:
            # Simple position-level VaR (assume 20% annualized vol)
            pos_values = [abs(p.current_price * p.shares) for p in self._positions.values()]
            total_exposure = sum(pos_values)
            # Scale to intraday: sqrt(horizon_minutes / (252 * 390))
            daily_vol = 0.02  # ~2% daily vol assumption
            intraday_scale = np.sqrt(cfg.var_horizon_minutes / 390)
            z = 2.326 if cfg.var_confidence == 0.99 else 1.645  # z-score
            var_estimate = total_exposure * daily_vol * intraday_scale * z
            var_pct = var_estimate / self._starting_equity * 100

            if var_pct > cfg.max_intraday_drawdown_pct * 0.8:
                alert = RiskAlert(
                    level="WARNING",
                    message=f"Intraday VaR {var_pct:.2f}% near drawdown limit",
                    metric="intraday_var",
                    value=var_pct,
                    threshold=cfg.max_intraday_drawdown_pct * 0.8,
                )
                new_alerts.append(alert)

        self._alerts.extend(new_alerts)
        return new_alerts

    def get_pnl_summary(self) -> dict:
        """Get current PnL summary."""
        total_pnl = sum(p.pnl for p in self._positions.values())
        current_equity = self._starting_equity + total_pnl

        return {
            "starting_equity": self._starting_equity,
            "current_equity": round(current_equity, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / max(self._starting_equity, 1) * 100, 4),
            "high_water_mark": round(self._high_water_mark, 2),
            "positions": len(self._positions),
            "kill_switch": self._kill_switch_triggered,
            "alerts": len(self._alerts),
        }

    def get_position_pnl(self) -> list[dict]:
        """Get per-position PnL."""
        return [
            {
                "symbol": p.symbol,
                "shares": p.shares,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "pnl": round(p.pnl, 2),
                "pnl_pct": round(p.pnl_pct, 4),
            }
            for p in sorted(self._positions.values(), key=lambda x: x.pnl)
        ]

    @property
    def kill_switch_triggered(self) -> bool:
        return self._kill_switch_triggered


__all__ = [
    "IntradayRiskConfig",
    "PositionSnapshot",
    "RiskAlert",
    "IntradayRiskMonitor",
]
