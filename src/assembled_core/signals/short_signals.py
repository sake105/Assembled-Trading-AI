"""Short signal generator: converts crash predictions into concrete short positions.

Four strategies:
  1. Sector shorts via inverse ETFs (crash_prob > 0.60)
  2. Broad market hedge via SH/PSQ/RWM (crash_prob > 0.75)
  3. Single stock shorts: weak stocks in weak sectors (severity > 0.60)
  4. Volatility longs via VIXY (crash_prob > 0.80)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .crash_prediction import CrashSignal, SECTOR_TO_INVERSE_ETF

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sector → inverse ETF map (extended for all strategies)
# ---------------------------------------------------------------------------

SECTOR_INVERSE_ETF_MAP: dict[str, str] = {
    **SECTOR_TO_INVERSE_ETF,
    "SMALL_CAP": "RWM",    # ProShares Short Russell2000
    "MID_CAP": "MYY",      # ProShares Short MidCap400
    "REAL_ESTATE": "REK",  # ProShares Short Real Estate
    "UTILITIES": "SDP",    # ProShares Short Utilities
}

# Volatility products (only for crash_prob > 0.80, short holds < 2 days)
VOL_INSTRUMENTS = ["VIXY"]   # iPath Series B S&P 500 VIX Short-Term Futures ETN

# ---------------------------------------------------------------------------
# Short target dataclass
# ---------------------------------------------------------------------------

@dataclass
class ShortTarget:
    """A single candidate short position."""

    symbol: str
    direction: str = "SHORT"
    target_weight: float = 0.0         # Negative weight (e.g., -0.05 = 5% short)
    confidence: float = 0.0
    strategy: str = ""                  # "sector_etf", "broad_hedge", "single_stock", "vol"
    reason: str = ""
    stop_loss_pct: float = 0.12        # Default 12% stop-loss
    max_hold_days: int = 30


# ---------------------------------------------------------------------------
# Short Signal Generator
# ---------------------------------------------------------------------------

class ShortSignalGenerator:
    """Generates concrete short positions from crash predictions."""

    def __init__(self, policy: dict[str, Any] | None = None):
        self.policy = policy or {}
        self.max_short_per_pos = self.policy.get("max_short_weight_per_position", 0.10)
        self.max_total_short = self.policy.get("max_total_short_exposure", 0.30)
        self.min_confidence = self.policy.get("min_short_signal_confidence", 0.70)
        self.allow_direct = self.policy.get("allowed_instruments", {}).get("direct_short", True)
        self.allow_1x = self.policy.get("allowed_instruments", {}).get("inverse_etf_1x", True)
        self.require_stop_loss = self.policy.get("require_stop_loss", True)

    def generate_short_targets(
        self,
        crash_signal: CrashSignal,
        universe: pd.DataFrame | None = None,
        prices: pd.DataFrame | None = None,
        regime: Any = None,
    ) -> pd.DataFrame:
        """Generate short targets based on crash signal.

        Returns DataFrame with columns:
            symbol, direction, target_weight, confidence, strategy, reason, stop_loss_pct
        """
        targets: list[ShortTarget] = []

        regime_cap = self._get_regime_short_cap(regime)
        if regime_cap == 0.0:
            logger.info("[ShortSignals] Regime cap = 0 — no shorts generated")
            return pd.DataFrame()

        # Strategy 1: Sector ETF shorts
        if crash_signal.crash_probability >= 0.60 and self.allow_1x:
            sector_shorts = self._generate_sector_shorts(crash_signal)
            targets.extend(sector_shorts)

        # Strategy 2: Broad market hedge
        if crash_signal.crash_probability >= 0.75 and self.allow_1x:
            broad_shorts = self._generate_broad_market_shorts(crash_signal)
            targets.extend(broad_shorts)

        # Strategy 3: Single stock shorts (weak stocks in weak sectors)
        if (
            crash_signal.expected_severity >= 0.60
            and self.allow_direct
            and universe is not None
            and prices is not None
        ):
            single_shorts = self._generate_single_stock_shorts(crash_signal, universe, prices)
            targets.extend(single_shorts)

        # Strategy 4: Volatility longs (VIX products, but only briefly)
        if crash_signal.crash_probability >= 0.80 and self.allow_1x:
            vol_longs = self._generate_vol_longs(crash_signal)
            targets.extend(vol_longs)

        if not targets:
            return pd.DataFrame()

        df = pd.DataFrame([
            {
                "symbol": t.symbol,
                "direction": t.direction,
                "target_weight": t.target_weight,
                "confidence": t.confidence,
                "strategy": t.strategy,
                "reason": t.reason,
                "stop_loss_pct": t.stop_loss_pct,
                "max_hold_days": t.max_hold_days,
            }
            for t in targets
        ])

        return self._apply_risk_limits(df, regime_cap)

    def _generate_sector_shorts(self, crash_signal: CrashSignal) -> list[ShortTarget]:
        """Generate 1x inverse ETF shorts for vulnerable sectors."""
        targets = []
        sectors = crash_signal.recommended_sectors_short

        for sector in sectors:
            etf = SECTOR_INVERSE_ETF_MAP.get(sector)
            if etf is None:
                continue

            # Weight based on severity and confidence
            base_weight = -min(
                crash_signal.expected_severity * 0.08,
                self.max_short_per_pos
            )
            confidence = min(
                crash_signal.confidence * crash_signal.crash_probability,
                1.0
            )

            if confidence < self.min_confidence:
                continue

            targets.append(ShortTarget(
                symbol=etf,
                target_weight=round(base_weight, 4),
                confidence=round(confidence, 4),
                strategy="sector_etf",
                reason=f"Inverse ETF for {sector} sector (crash_prob={crash_signal.crash_probability:.2f})",
                stop_loss_pct=0.12,
                max_hold_days=21,
            ))

        return targets

    def _generate_broad_market_shorts(self, crash_signal: CrashSignal) -> list[ShortTarget]:
        """Generate broad market hedge via SH when crash probability is high."""
        weight = -min(
            crash_signal.crash_probability * 0.10,
            self.max_short_per_pos
        )
        confidence = crash_signal.confidence

        if confidence < self.min_confidence:
            return []

        targets = [ShortTarget(
            symbol="SH",
            target_weight=round(weight, 4),
            confidence=round(confidence, 4),
            strategy="broad_hedge",
            reason=f"Broad market hedge: crash_prob={crash_signal.crash_probability:.2f}",
            stop_loss_pct=0.10,
            max_hold_days=14,
        )]

        # Add Russell 2000 short for high-severity crashes (small caps hit hardest)
        if crash_signal.expected_severity > 0.7:
            targets.append(ShortTarget(
                symbol="RWM",
                target_weight=round(weight * 0.6, 4),
                confidence=round(confidence * 0.9, 4),
                strategy="broad_hedge",
                reason=f"Small-cap hedge: severity={crash_signal.expected_severity:.2f}",
                stop_loss_pct=0.12,
                max_hold_days=14,
            ))

        return targets

    def _generate_single_stock_shorts(
        self,
        crash_signal: CrashSignal,
        universe: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> list[ShortTarget]:
        """Generate single stock shorts: weak stocks in weak sectors.

        Criteria:
        - In a sector identified as vulnerable
        - Negative momentum (below 50-day MA)
        - High debt/leverage if available
        """
        if universe.empty or prices.empty:
            return []

        targets = []
        vulnerable_sectors = set(crash_signal.recommended_sectors_short)

        # Filter for stocks in vulnerable sectors
        if "sector" not in universe.columns:
            return []

        candidates = universe[universe["sector"].isin(vulnerable_sectors)]
        if candidates.empty:
            return []

        for _, row in candidates.iterrows():
            symbol = row.get("symbol", row.name)
            if symbol not in prices.columns:
                continue

            price_series = prices[symbol].dropna()
            if len(price_series) < 50:
                continue

            # Negative momentum: price below 50-day MA
            ma50 = price_series.rolling(50).mean().iloc[-1]
            current = price_series.iloc[-1]
            if current >= ma50:
                continue  # Only short stocks below MA50

            # Momentum strength
            momentum = (current - ma50) / ma50
            if momentum > -0.03:
                continue  # Need at least 3% below MA50

            weight = -min(abs(momentum) * 0.5, self.max_short_per_pos * 0.5)
            confidence = min(
                crash_signal.confidence * (1 + abs(momentum) * 2),
                1.0
            )

            if confidence < self.min_confidence:
                continue

            targets.append(ShortTarget(
                symbol=str(symbol),
                target_weight=round(weight, 4),
                confidence=round(confidence, 4),
                strategy="single_stock",
                reason=f"Below MA50 by {abs(momentum)*100:.1f}%, sector={row.get('sector', 'unknown')}",
                stop_loss_pct=0.15,
                max_hold_days=30,
            ))

        # Limit to top 5 single stock shorts by confidence
        targets.sort(key=lambda x: x.confidence, reverse=True)
        return targets[:5]

    def _generate_vol_longs(self, crash_signal: CrashSignal) -> list[ShortTarget]:
        """Generate volatility longs (VIX products) for extreme crash probability."""
        weight = min(crash_signal.crash_probability * 0.05, 0.05)  # Max 5%, held briefly

        return [ShortTarget(
            symbol="VIXY",
            direction="LONG",          # This is a long position in VIX futures
            target_weight=round(weight, 4),
            confidence=round(crash_signal.confidence, 4),
            strategy="vol",
            reason=f"VIX long hedge: crash_prob={crash_signal.crash_probability:.2f}",
            stop_loss_pct=0.20,        # VIX products are volatile; wider stop
            max_hold_days=5,           # Very short hold — vol decay is severe
        )]

    def _get_regime_short_cap(self, regime: Any) -> float:
        """Return max short exposure cap based on regime."""
        regime_scaling = self.policy.get("regime_scaling", {})
        if regime is None:
            return regime_scaling.get("sideways", 0.10)

        label = str(getattr(regime, "state", getattr(regime, "label", regime))).lower()
        for key, val in regime_scaling.items():
            if key in label:
                return val

        return regime_scaling.get("sideways", 0.10)

    def _apply_risk_limits(self, df: pd.DataFrame, regime_cap: float) -> pd.DataFrame:
        """Enforce maximum short exposure limits."""
        if df.empty:
            return df

        # Cap each position by max_short_per_pos
        df["target_weight"] = df["target_weight"].clip(
            lower=-self.max_short_per_pos,
            upper=self.max_short_per_pos,
        )

        # Calculate total short exposure
        short_df = df[df["direction"] == "SHORT"]
        total_short = short_df["target_weight"].abs().sum()

        if total_short > self.max_total_short:
            # Scale down proportionally
            scale = self.max_total_short / total_short
            df.loc[df["direction"] == "SHORT", "target_weight"] *= scale

        # Apply regime cap
        total_short = df[df["direction"] == "SHORT"]["target_weight"].abs().sum()
        if total_short > regime_cap:
            scale = regime_cap / total_short
            df.loc[df["direction"] == "SHORT", "target_weight"] *= scale

        return df.round({"target_weight": 4, "confidence": 4})
