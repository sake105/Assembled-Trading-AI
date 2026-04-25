"""Mean-Reversion Signal Generator for Pairs Trading (M36.2).

Generates trading signals from cointegrated pairs:
- Spread computation: spread_t = price_A - hedge_ratio * price_B
- Z-score: z_t = (spread_t - mean) / std
- Entry: |z| > entry_threshold (default 2.0)
- Exit: |z| < exit_threshold (default 0.5)
- Stop-loss: |z| > stop_threshold (default 4.0)

References:
    Gatev, Goetzmann & Rouwenhorst (2006) "Pairs Trading"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PairPosition(str, Enum):
    """Pair trading position state."""
    FLAT = "flat"
    LONG_SPREAD = "long_spread"  # Long A, Short B
    SHORT_SPREAD = "short_spread"  # Short A, Long B
    STOPPED_OUT = "stopped_out"


@dataclass
class PairSignal:
    """Signal for a single pair at a point in time."""
    stock_a: str
    stock_b: str
    spread: float
    z_score: float
    position: PairPosition
    hedge_ratio: float
    signal_strength: float  # abs(z_score) capped


class PairSignalGenerator:
    """Generate mean-reversion signals for a cointegrated pair.

    Args:
        hedge_ratio: Beta from cointegration regression.
        lookback: Window for rolling mean/std of spread.
        entry_z: Z-score threshold to enter (default 2.0).
        exit_z: Z-score threshold to exit (default 0.5).
        stop_z: Z-score threshold for stop-loss (default 4.0).
    """

    def __init__(
        self,
        hedge_ratio: float,
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
    ) -> None:
        self.hedge_ratio = hedge_ratio
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self._position = PairPosition.FLAT

    def compute_spread(
        self, prices_a: pd.Series, prices_b: pd.Series,
    ) -> pd.Series:
        """Compute spread = A - hedge_ratio * B."""
        common = prices_a.index.intersection(prices_b.index)
        return prices_a.reindex(common) - self.hedge_ratio * prices_b.reindex(common)

    def compute_z_score(self, spread: pd.Series) -> pd.Series:
        """Rolling z-score of spread."""
        rolling_mean = spread.rolling(self.lookback, min_periods=20).mean()
        rolling_std = spread.rolling(self.lookback, min_periods=20).std()
        return (spread - rolling_mean) / (rolling_std + 1e-10)

    def generate_signals(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        stock_a: str = "A",
        stock_b: str = "B",
    ) -> list[PairSignal]:
        """Generate signals for all dates.

        Args:
            prices_a: Price series for stock A.
            prices_b: Price series for stock B.
            stock_a: Symbol A.
            stock_b: Symbol B.

        Returns:
            List of PairSignal for each date.
        """
        spread = self.compute_spread(prices_a, prices_b)
        z_scores = self.compute_z_score(spread)

        signals = []
        position = PairPosition.FLAT

        for i in range(len(z_scores)):
            z = z_scores.iloc[i]
            if np.isnan(z):
                signals.append(PairSignal(
                    stock_a=stock_a, stock_b=stock_b,
                    spread=float(spread.iloc[i]) if not np.isnan(spread.iloc[i]) else 0.0,
                    z_score=0.0, position=PairPosition.FLAT,
                    hedge_ratio=self.hedge_ratio, signal_strength=0.0,
                ))
                continue

            # State machine
            if position == PairPosition.FLAT:
                if z > self.entry_z:
                    position = PairPosition.SHORT_SPREAD
                elif z < -self.entry_z:
                    position = PairPosition.LONG_SPREAD

            elif position == PairPosition.LONG_SPREAD:
                if abs(z) < self.exit_z:
                    position = PairPosition.FLAT
                elif z > self.stop_z:
                    position = PairPosition.STOPPED_OUT

            elif position == PairPosition.SHORT_SPREAD:
                if abs(z) < self.exit_z:
                    position = PairPosition.FLAT
                elif z < -self.stop_z:
                    position = PairPosition.STOPPED_OUT

            elif position == PairPosition.STOPPED_OUT:
                if abs(z) < self.exit_z:
                    position = PairPosition.FLAT

            signals.append(PairSignal(
                stock_a=stock_a,
                stock_b=stock_b,
                spread=round(float(spread.iloc[i]), 6),
                z_score=round(float(z), 4),
                position=position,
                hedge_ratio=self.hedge_ratio,
                signal_strength=round(min(abs(float(z)), 5.0) / 5.0, 4),
            ))

        return signals

    def backtest_pair(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
    ) -> dict[str, float]:
        """Simple backtest of pair trading strategy.

        Returns:
            Dict with sharpe, total_return, max_dd, n_trades.
        """
        spread = self.compute_spread(prices_a, prices_b)
        z_scores = self.compute_z_score(spread)
        ret_a = prices_a.pct_change().reindex(spread.index)
        ret_b = prices_b.pct_change().reindex(spread.index)

        daily_pnl = []
        position = PairPosition.FLAT
        n_trades = 0

        for i in range(1, len(z_scores)):
            z = z_scores.iloc[i - 1]  # signal from yesterday
            if np.isnan(z):
                daily_pnl.append(0.0)
                continue

            r_a = ret_a.iloc[i] if not np.isnan(ret_a.iloc[i]) else 0.0
            r_b = ret_b.iloc[i] if not np.isnan(ret_b.iloc[i]) else 0.0

            old_pos = position

            if position == PairPosition.FLAT:
                if z > self.entry_z:
                    position = PairPosition.SHORT_SPREAD
                elif z < -self.entry_z:
                    position = PairPosition.LONG_SPREAD
            elif position in (PairPosition.LONG_SPREAD, PairPosition.SHORT_SPREAD):
                if abs(z) < self.exit_z or abs(z) > self.stop_z:
                    position = PairPosition.FLAT

            if old_pos != position:
                n_trades += 1

            # PnL
            if old_pos == PairPosition.LONG_SPREAD:
                daily_pnl.append(r_a - self.hedge_ratio * r_b)
            elif old_pos == PairPosition.SHORT_SPREAD:
                daily_pnl.append(-r_a + self.hedge_ratio * r_b)
            else:
                daily_pnl.append(0.0)

        if not daily_pnl:
            return {"sharpe": 0.0, "total_return": 0.0, "max_dd": 0.0, "n_trades": 0}

        pnl = np.array(daily_pnl)
        ann_ret = float(np.mean(pnl)) * 252
        ann_vol = float(np.std(pnl)) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 1e-8 else 0.0

        cumret = np.cumprod(1 + pnl)
        peak = np.maximum.accumulate(cumret)
        dd = (cumret - peak) / np.maximum(peak, 1e-10)

        return {
            "sharpe": round(sharpe, 4),
            "total_return": round(float(cumret[-1] - 1), 6),
            "max_dd": round(float(dd.min()), 6),
            "n_trades": n_trades,
        }


__all__ = [
    "PairPosition",
    "PairSignal",
    "PairSignalGenerator",
]
