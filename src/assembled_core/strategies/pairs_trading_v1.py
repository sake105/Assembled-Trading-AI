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

        date_col = next((c for c in prices.columns if c in ("date", "timestamp")), None)
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i + 1:]:
                sub1 = prices[prices["symbol"] == s1]
                sub2 = prices[prices["symbol"] == s2]
                if date_col:
                    c1 = sub1.set_index(date_col)["close"]
                    c2 = sub2.set_index(date_col)["close"]
                else:
                    c1 = sub1["close"]
                    c2 = sub2["close"]
                if len(c1) < max(30, self.lookback_days // 4) or len(c2) < max(30, self.lookback_days // 4):
                    continue
                try:
                    # Use full available history for cointegration (more data = more power)
                    p_val = cointegration_score(c1, c2)
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
        """Generate long/short signals for discovered pairs.

        prices: long-format DataFrame with columns [date/timestamp, symbol, close, ...]
        Returns: DataFrame with columns [symbol, direction, score] for backtest loop.
        """
        from src.assembled_core.signals.pairs_trading import generate_pairs_signals_from_panel

        if pairs is None:
            pairs = self.discover_pairs(prices)

        if not pairs:
            logger.warning("[pairs] no pairs available — returning empty signals")
            return pd.DataFrame(columns=["symbol", "direction", "score"])

        # Convert long-format → wide (dates × symbols) for generate_pairs_signals_from_panel
        date_col = next((c for c in prices.columns if c in ("date", "timestamp")), None)
        if date_col is None:
            logger.warning("[pairs] no date column found — returning empty signals")
            return pd.DataFrame(columns=["symbol", "direction", "score"])

        wide = prices.pivot_table(index=date_col, columns="symbol", values="close")
        wide.index = pd.to_datetime(wide.index).tz_localize(None)

        try:
            result = generate_pairs_signals_from_panel(
                wide,
                pairs=pairs,
                entry_z=self.entry_zscore,
                exit_z=self.exit_zscore,
                window=min(60, max(20, len(wide) // 5)),
            )
        except Exception as exc:
            logger.debug("[pairs] signal generation failed: %s", exc)
            return pd.DataFrame(columns=["symbol", "direction", "score"])

        if result.empty:
            return pd.DataFrame(columns=["symbol", "direction", "score"])

        # Translate direction to LONG/SHORT/EXIT for backtest loop
        last_row = result[result.index == result.index.max()] if isinstance(result.index, pd.DatetimeIndex) else result.tail(1)

        rows = []
        for _, row in last_row.iterrows():
            direction = str(row.get("direction", "HOLD")).upper()
            if direction in ("HOLD",):
                continue
            sym_a = row.get("symbol_a", "")
            sym_b = row.get("symbol_b", "")
            if direction == "LONG_A":
                rows.append({"symbol": sym_a, "direction": "LONG", "score": float(abs(row.get("z_score", 0)))})
                rows.append({"symbol": sym_b, "direction": "SHORT", "score": float(abs(row.get("z_score", 0)))})
            elif direction == "SHORT_A":
                rows.append({"symbol": sym_a, "direction": "SHORT", "score": float(abs(row.get("z_score", 0)))})
                rows.append({"symbol": sym_b, "direction": "LONG", "score": float(abs(row.get("z_score", 0)))})
            elif direction == "EXIT":
                rows.append({"symbol": sym_a, "direction": "EXIT", "score": 0.0})
                rows.append({"symbol": sym_b, "direction": "EXIT", "score": 0.0})

        return pd.DataFrame(rows)
