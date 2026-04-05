"""Long-short portfolio balance manager.

Manages the aggregate exposure balance between long and short books:
  - Net exposure = sum(weights) — can be negative (net short)
  - Gross exposure = sum(|weights|) — total risk deployed
  - Long exposure = sum(positive weights)
  - Short exposure = sum(absolute negative weights)

Enforces limits from policy and rebalances proportionally when breached.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExposureMetrics:
    """Current portfolio exposure breakdown."""

    long_exposure: float
    short_exposure: float
    net_exposure: float
    gross_exposure: float
    long_count: int
    short_count: int

    @property
    def is_net_long(self) -> bool:
        return self.net_exposure > 0

    @property
    def is_net_short(self) -> bool:
        return self.net_exposure < 0


class LongShortBalancer:
    """Manages and enforces long-short exposure balance."""

    def __init__(
        self,
        max_gross_exposure: float = 1.50,
        max_net_short: float = 0.20,
        max_total_short: float = 0.30,
        max_long_weight: float = 1.00,
    ):
        self.max_gross = max_gross_exposure
        self.max_net_short = max_net_short
        self.max_total_short = max_total_short
        self.max_long = max_long_weight

    @classmethod
    def from_policy(cls, policy: dict) -> "LongShortBalancer":
        """Construct from policy.yaml shorts section."""
        shorts = policy.get("shorts", {})
        return cls(
            max_gross_exposure=shorts.get("max_gross_exposure", 1.50),
            max_net_short=shorts.get("max_net_short", 0.20),
            max_total_short=shorts.get("max_total_short_exposure", 0.30),
            max_long_weight=policy.get("risk_limits", {}).get("max_position_weight", 1.0),
        )

    def compute_exposure(self, positions: pd.DataFrame) -> ExposureMetrics:
        """Compute exposure metrics from a positions DataFrame.

        Expects a 'weight' or 'target_weight' column with signed weights.
        Positive = long, negative = short.
        """
        if positions.empty:
            return ExposureMetrics(0.0, 0.0, 0.0, 0.0, 0, 0)

        weight_col = "target_weight" if "target_weight" in positions.columns else "weight"
        if weight_col not in positions.columns:
            return ExposureMetrics(0.0, 0.0, 0.0, 0.0, 0, 0)

        weights = positions[weight_col].fillna(0.0)
        longs = weights[weights > 0]
        shorts = weights[weights < 0]

        long_exp = float(longs.sum())
        short_exp = float(shorts.abs().sum())

        return ExposureMetrics(
            long_exposure=round(long_exp, 4),
            short_exposure=round(short_exp, 4),
            net_exposure=round(long_exp - short_exp, 4),
            gross_exposure=round(long_exp + short_exp, 4),
            long_count=len(longs),
            short_count=len(shorts),
        )

    def enforce_exposure_limits(
        self, positions: pd.DataFrame, regime: str = "sideways"
    ) -> pd.DataFrame:
        """Scale positions to satisfy all exposure limits.

        Scaling priority:
          1. Enforce max_gross_exposure (scale down both sides proportionally)
          2. Enforce max_total_short (scale down shorts only)
          3. Enforce max_net_short (scale down shorts further if needed)
        """
        if positions.empty:
            return positions

        weight_col = "target_weight" if "target_weight" in positions.columns else "weight"
        result = positions.copy()
        weights = result[weight_col].fillna(0.0)

        long_mask = weights > 0
        short_mask = weights < 0

        # Step 1: Max gross exposure
        gross = weights.abs().sum()
        if gross > self.max_gross:
            scale = self.max_gross / gross
            result[weight_col] *= scale
            weights = result[weight_col].fillna(0.0)
            logger.info(
                "[LongShortBalancer] Gross %.2f > %.2f: scaled by %.3f",
                gross, self.max_gross, scale,
            )

        # Step 2: Max total short
        short_total = weights[short_mask].abs().sum()
        if short_total > self.max_total_short:
            scale = self.max_total_short / short_total
            result.loc[short_mask, weight_col] *= scale
            weights = result[weight_col].fillna(0.0)
            logger.info(
                "[LongShortBalancer] Short exposure %.2f > %.2f: scaled by %.3f",
                short_total, self.max_total_short, scale,
            )

        # Step 3: Max net short (net_exposure = long - |short|)
        long_total = weights[long_mask].sum()
        short_total = weights[short_mask].abs().sum()
        net = long_total - short_total
        if net < -self.max_net_short:
            # Too net short — reduce short exposure
            excess = (-net) - self.max_net_short
            if short_total > 0:
                scale = (short_total - excess) / short_total
                result.loc[short_mask, weight_col] *= max(scale, 0.0)
                logger.info(
                    "[LongShortBalancer] Net short %.2f > %.2f: scaled shorts by %.3f",
                    -net, self.max_net_short, scale,
                )

        return result

    def compute_optimal_hedge_ratio(
        self,
        long_portfolio: pd.DataFrame,
        market_beta: float = 1.0,
        target_beta: float = 0.5,
    ) -> float:
        """Compute optimal hedge ratio to reduce portfolio beta.

        Args:
            long_portfolio: Long positions with weight column.
            market_beta: Current portfolio beta vs market.
            target_beta: Desired portfolio beta after hedging.

        Returns:
            Optimal short weight as fraction (positive number = fraction to short).
        """
        if market_beta <= 0:
            return 0.0
        if target_beta >= market_beta:
            return 0.0  # Already at or below target

        # Hedge ratio = (current_beta - target_beta) / market_beta
        hedge_ratio = (market_beta - target_beta) / market_beta
        return min(hedge_ratio, self.max_total_short)

    def rebalance_long_short(
        self,
        current_positions: pd.DataFrame,
        targets: pd.DataFrame,
        transaction_cost_bps: float = 10.0,
    ) -> pd.DataFrame:
        """Compute rebalancing trades considering transaction costs.

        Returns DataFrame with 'symbol', 'trade_weight', 'trade_direction'.
        Only generates trades where the benefit exceeds the cost.
        """
        if targets.empty:
            return pd.DataFrame()

        weight_col = "target_weight" if "target_weight" in targets.columns else "weight"

        if current_positions.empty:
            trades = targets[[weight_col]].copy()
            trades["trade_weight"] = trades[weight_col]
            trades["trade_direction"] = trades["trade_weight"].apply(
                lambda w: "BUY" if w > 0 else "SELL_SHORT"
            )
            return trades

        # Compute trade sizes
        current_wt = current_positions.set_index("symbol")[weight_col] if "symbol" in current_positions.columns else pd.Series(dtype=float)
        target_wt = targets.set_index("symbol")[weight_col] if "symbol" in targets.columns else pd.Series(dtype=float)

        all_symbols = set(current_wt.index) | set(target_wt.index)
        trades = []

        cost_threshold = transaction_cost_bps / 10000

        for sym in all_symbols:
            cur = float(current_wt.get(sym, 0.0))
            tgt = float(target_wt.get(sym, 0.0))
            delta = tgt - cur

            if abs(delta) < cost_threshold:
                continue  # Trade too small to justify cost

            direction = "BUY" if delta > 0 else ("SELL_SHORT" if tgt < 0 else "SELL")
            trades.append({"symbol": sym, "trade_weight": round(delta, 4), "trade_direction": direction})

        return pd.DataFrame(trades)

    def compute_net_exposure(self, positions: pd.DataFrame) -> float:
        """Net Exposure = Long - |Short|"""
        return self.compute_exposure(positions).net_exposure

    def compute_gross_exposure(self, positions: pd.DataFrame) -> float:
        """Gross Exposure = Long + |Short|"""
        return self.compute_exposure(positions).gross_exposure
