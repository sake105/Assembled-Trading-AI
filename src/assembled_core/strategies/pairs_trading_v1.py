"""Pairs-trading strategy wrapper (Plan 11/10 §1.3).

Wraps signals/pairs_trading.py for potential ensemble integration.
NOT wired into the live pipeline yet — requires isolated backtest
validation first (A/B: Sharpe > 0.8, MDD < 15%, Trades > 50).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class PairsTradingStrategy:
    """Mean-reversion on cointegrated pairs.

    Requires isolated backtesting before ensemble integration:
        python scripts/run_backtest_strategy.py --strategy pairs_trading_v1
            --start 2020-01-01 --end 2024-12-31
    Activation threshold: Sharpe > 0.8, MDD < -15%, Trades > 50.
    """

    name = "pairs_trading_v1"

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.lookback_days: int = int(cfg.get("lookback_days", 252))
        self.min_cointegration_p: float = float(cfg.get("min_cointegration_p", 0.05))
        self.entry_zscore: float = float(cfg.get("entry_zscore", 2.0))
        self.exit_zscore: float = float(cfg.get("exit_zscore", 0.5))
        self.max_pairs: int = int(cfg.get("max_pairs", 20))

    # ------------------------------------------------------------------
    def discover_pairs(self, prices: pd.DataFrame) -> list[tuple[str, str]]:
        """Identify cointegrated pairs from price history.

        Runs quarterly (expensive). Pair list should be cached.
        """
        from src.assembled_core.signals.pairs_trading import cointegration_score

        symbols = prices["symbol"].unique().tolist()
        pairs: list[tuple[str, str, float]] = []

        for i, s1 in enumerate(symbols):
            for s2 in symbols[i + 1:]:
                c1 = prices[prices["symbol"] == s1]["close"]
                c2 = prices[prices["symbol"] == s2]["close"]
                if len(c1) < self.lookback_days or len(c2) < self.lookback_days:
                    continue
                try:
                    p_val = cointegration_score(c1.iloc[-self.lookback_days:],
                                                c2.iloc[-self.lookback_days:])
                    if p_val < self.min_cointegration_p:
                        pairs.append((s1, s2, p_val))
                except Exception as exc:
                    logger.debug("[pairs] %s/%s: %s", s1, s2, exc)

        pairs.sort(key=lambda t: t[2])
        result = [(s1, s2) for s1, s2, _ in pairs[: self.max_pairs]]
        logger.info("[pairs] discovered %d cointegrated pairs", len(result))
        return result

    def generate_signals(
        self,
        prices: pd.DataFrame,
        pairs: list[tuple[str, str]] | None = None,
    ) -> pd.DataFrame:
        """Generate long/short signals for discovered pairs."""
        from src.assembled_core.signals.pairs_trading import generate_pairs_signals

        if pairs is None:
            pairs = self.discover_pairs(prices)

        if not pairs:
            logger.warning("[pairs] no pairs available — returning empty signals")
            return pd.DataFrame(columns=["symbol", "direction", "score", "timestamp"])

        return generate_pairs_signals(
            prices,
            pairs,
            entry_z=self.entry_zscore,
            exit_z=self.exit_zscore,
        )
