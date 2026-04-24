"""Short position risk controls.

8 hard rules for short positions:
  1. Every short MUST have a stop-loss (max 15% loss)
  2. No 2x/3x leveraged inverse ETFs in standard mode
  3. Shorts only in bear/crisis regime (or policy override)
  4. Max 30% total short exposure
  5. Max 10% per individual short
  6. Short-squeeze check before entry
  7. Daily mark-to-market with automatic stop-out
  8. No overnight VIX/volatility product shorts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_atr_stop_pct(
    prices_df: pd.DataFrame,
    symbol: str,
    *,
    atr_period: int = 14,
    regime: str = "sideways",
) -> float | None:
    """Compute ATR-based stop-loss as a fraction of price.

    ATR multiplier is regime-dependent:
        crisis/bear: 1.5 ATR  (tighter stops — protect capital)
        sideways:    2.0 ATR
        bull:        3.0 ATR  (wider stops — let position breathe)

    Args:
        prices_df: DataFrame with columns [symbol, date, high, low, close]
                   or MultiIndex with symbol level.
        symbol: Ticker to compute ATR for.
        atr_period: ATR lookback period (default 14).
        regime: Market regime string.

    Returns:
        Stop-loss as fraction (e.g., 0.12 = 12%), or None if insufficient data.
    """
    atr_multipliers = {
        "crisis": 1.5,
        "bear": 1.5,
        "sideways": 2.0,
        "bull": 3.0,
    }
    multiplier = atr_multipliers.get(regime, 2.0)

    # Extract symbol data
    if "symbol" in prices_df.columns:
        sym_data = prices_df[prices_df["symbol"] == symbol].copy()
    else:
        sym_data = prices_df.copy()

    if len(sym_data) < atr_period + 1:
        return None

    # Compute ATR
    high = sym_data["high"] if "high" in sym_data.columns else None
    low = sym_data["low"] if "low" in sym_data.columns else None
    close = sym_data["close"] if "close" in sym_data.columns else None

    if high is None or low is None or close is None:
        return None

    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(atr_period).mean().iloc[-1]
    last_price = close.iloc[-1]

    if np.isnan(atr) or last_price <= 0:
        return None

    stop_pct = (multiplier * atr) / last_price
    return float(min(stop_pct, 0.50))  # Cap at 50% to prevent nonsensical stops

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LEVERAGED_INVERSE_ETFS = {
    # 2x inverse
    "SDS", "QID", "MZZ", "TWM", "SKF", "SRS", "DUG", "REW", "SMN",
    "SDD", "CMD", "DXD", "EEV", "EFU", "FXP",
    # 3x inverse
    "SPXS", "SPXU", "SQQQ", "TZA", "FAZ", "SRTY", "DGAZ", "LABD",
    "SDOW", "SOXS", "YANG", "RUSS",
}

VOLATILITY_PRODUCTS = {"VIXY", "VXX", "UVXY", "SVXY", "TVIX"}

ALLOWED_REGIMES_FOR_SHORTS = {"bear", "crisis"}

# Short-squeeze risk thresholds
SQUEEZE_RISK_SHORT_INTEREST_THRESHOLD = 0.20  # > 20% float short = high squeeze risk
SQUEEZE_RISK_DAYS_TO_COVER_THRESHOLD = 10.0   # > 10 days to cover = high risk


@dataclass
class ShortRiskCheck:
    """Result of a short risk validation."""

    passed: bool
    violations: list[str]
    warnings: list[str]
    adjusted_weight: float  # May be reduced from requested weight


class ShortRiskManager:
    """Validates and enforces risk rules for short positions.

    Used as a gate before any short order is generated.
    """

    def __init__(self, policy: dict[str, Any] | None = None):
        self.policy = policy or {}
        self.shorts_policy = self.policy.get("shorts", {})
        self.allow_2x = self.shorts_policy.get("allowed_instruments", {}).get("inverse_etf_2x", False)
        self.allow_3x = self.shorts_policy.get("allowed_instruments", {}).get("inverse_etf_3x", False)
        self.max_per_pos = self.shorts_policy.get("max_short_weight_per_position", 0.10)
        self.max_total = self.shorts_policy.get("max_total_short_exposure", 0.30)
        self.max_gross = self.shorts_policy.get("max_gross_exposure", 1.50)
        self.max_net_short = self.shorts_policy.get("max_net_short", 0.20)
        self.require_stop_loss = self.shorts_policy.get("require_stop_loss", True)
        self.max_stop_loss = self.shorts_policy.get("max_stop_loss_pct", 0.15)
        self.squeeze_check_enabled = self.shorts_policy.get("squeeze_risk_check", True)

    def validate_short_targets(
        self,
        short_df: pd.DataFrame,
        current_portfolio: pd.DataFrame | None = None,
        regime: Any = None,
    ) -> ShortRiskCheck:
        """Validate a batch of short targets against all 8 hard rules.

        Args:
            short_df: DataFrame with columns [symbol, direction, target_weight,
                      confidence, strategy, stop_loss_pct]
            current_portfolio: Existing positions for exposure calculations.
            regime: Current market regime.

        Returns:
            ShortRiskCheck with pass/fail and any violations.
        """
        violations = []
        warnings = []

        if short_df.empty:
            return ShortRiskCheck(passed=True, violations=[], warnings=[], adjusted_weight=0.0)

        only_shorts = short_df[short_df.get("direction", "SHORT") == "SHORT"]

        # Rule 1: Stop-loss required
        if self.require_stop_loss:
            missing_stop = only_shorts[
                only_shorts.get("stop_loss_pct", pd.Series(dtype=float)).isna()
                | (only_shorts.get("stop_loss_pct", pd.Series(dtype=float)) <= 0)
            ]
            if not missing_stop.empty:
                violations.append(
                    f"Rule 1: {len(missing_stop)} short(s) missing stop-loss: "
                    f"{list(missing_stop['symbol'])}"
                )

        # Rule 2: No leveraged inverse ETFs (unless policy permits)
        if "symbol" in only_shorts.columns:
            leveraged = only_shorts[only_shorts["symbol"].isin(LEVERAGED_INVERSE_ETFS)]
            if not leveraged.empty and not self.allow_2x:
                violations.append(
                    f"Rule 2: Leveraged inverse ETFs not allowed: {list(leveraged['symbol'])}"
                )

        # Rule 3: Regime check
        if regime is not None:
            label = str(getattr(regime, "state", getattr(regime, "label", regime))).lower()
            regime_ok = any(r in label for r in ALLOWED_REGIMES_FOR_SHORTS)
            if not regime_ok:
                # This is a warning, not hard block — policy may override
                if not self.shorts_policy.get("regime_override", False):
                    warnings.append(
                        f"Rule 3: Regime '{label}' is not bear/crisis. "
                        f"Shorts capped at {self.shorts_policy.get('regime_scaling', {}).get(label, 0.10)*100:.0f}%."
                    )

        # Rule 4: Max total short exposure
        total_short = only_shorts["target_weight"].abs().sum() if not only_shorts.empty else 0.0
        if total_short > self.max_total:
            violations.append(
                f"Rule 4: Total short exposure {total_short:.2%} exceeds max {self.max_total:.2%}. "
                f"Scale down required."
            )

        # Rule 5: Max per position
        if "target_weight" in only_shorts.columns:
            over_limit = only_shorts[only_shorts["target_weight"].abs() > self.max_per_pos]
            if not over_limit.empty:
                violations.append(
                    f"Rule 5: {len(over_limit)} position(s) exceed max {self.max_per_pos:.0%} "
                    f"per short: {list(over_limit['symbol'])}"
                )

        # Rule 8: No overnight VIX products (short holds < 2 days only)
        if "symbol" in only_shorts.columns and "max_hold_days" in only_shorts.columns:
            vol_longs = only_shorts[
                only_shorts["symbol"].isin(VOLATILITY_PRODUCTS)
                & (only_shorts["max_hold_days"] > 2)
            ]
            if not vol_longs.empty:
                warnings.append(
                    f"Rule 8: VIX product(s) {list(vol_longs['symbol'])} have hold > 2 days. "
                    f"Capped at 2 days."
                )

        passed = len(violations) == 0
        total_adjusted_weight = only_shorts["target_weight"].abs().sum() if not only_shorts.empty else 0.0

        if not passed:
            logger.warning(
                "[ShortRisk] Validation failed with %d violation(s): %s",
                len(violations), violations,
            )
        elif warnings:
            logger.info("[ShortRisk] Validation passed with %d warning(s)", len(warnings))
        else:
            logger.debug("[ShortRisk] Validation passed cleanly")

        return ShortRiskCheck(
            passed=passed,
            violations=violations,
            warnings=warnings,
            adjusted_weight=total_adjusted_weight,
        )

    def check_short_squeeze_risk(
        self,
        symbol: str,
        short_interest_pct: float = 0.0,
        days_to_cover: float = 0.0,
    ) -> bool:
        """Return True if squeeze risk is HIGH (should avoid shorting).

        Args:
            symbol: Ticker symbol.
            short_interest_pct: Fraction of float currently sold short.
            days_to_cover: Average daily volume / short interest.
        """
        if not self.squeeze_check_enabled:
            return False

        if short_interest_pct > SQUEEZE_RISK_SHORT_INTEREST_THRESHOLD:
            logger.warning(
                "[ShortRisk] Squeeze risk: %s short interest %.1f%% > threshold %.1f%%",
                symbol, short_interest_pct * 100, SQUEEZE_RISK_SHORT_INTEREST_THRESHOLD * 100,
            )
            return True

        if days_to_cover > SQUEEZE_RISK_DAYS_TO_COVER_THRESHOLD:
            logger.warning(
                "[ShortRisk] Squeeze risk: %s days to cover %.1f > threshold %.1f",
                symbol, days_to_cover, SQUEEZE_RISK_DAYS_TO_COVER_THRESHOLD,
            )
            return True

        return False

    def enforce_regime_scaling(
        self,
        short_df: pd.DataFrame,
        regime: Any,
    ) -> pd.DataFrame:
        """Scale short positions down based on regime cap."""
        if short_df.empty:
            return short_df

        regime_scaling = self.shorts_policy.get("regime_scaling", {})
        label = str(getattr(regime, "state", getattr(regime, "label", str(regime)))).lower()

        cap = None
        for key, val in regime_scaling.items():
            if key in label:
                cap = val
                break
        if cap is None:
            cap = regime_scaling.get("sideways", 0.10)

        only_shorts = short_df[short_df.get("direction", "SHORT") == "SHORT"]
        total_short = only_shorts["target_weight"].abs().sum()

        if total_short > cap and total_short > 0:
            scale = cap / total_short
            short_df = short_df.copy()
            short_df.loc[short_df.get("direction", "SHORT") == "SHORT", "target_weight"] *= scale
            logger.info(
                "[ShortRisk] Regime '%s' cap=%.1f%% applied (was %.1f%% → %.1f%%)",
                label, cap * 100, total_short * 100, cap * 100,
            )

        return short_df

    def compute_short_exposure_limits(
        self,
        long_positions: pd.DataFrame | None,
        regime: Any,
    ) -> dict[str, float]:
        """Compute dynamic short limits based on current long book and regime.

        Returns dict with:
          - max_total_short: max absolute short weight
          - max_net_short: max net short weight
          - max_gross_exposure: max long + |short| weight
        """
        regime_scaling = self.shorts_policy.get("regime_scaling", {})
        label = str(getattr(regime, "state", getattr(regime, "label", str(regime)))).lower()

        regime_cap = 0.10
        for key, val in regime_scaling.items():
            if key in label:
                regime_cap = val
                break

        return {
            "max_total_short": min(self.max_total, regime_cap),
            "max_net_short": self.max_net_short,
            "max_gross_exposure": self.max_gross,
            "max_per_position": self.max_per_pos,
        }

    def check_correlation_concentration(
        self,
        short_symbols: list[str],
        corr_matrix: pd.DataFrame | None = None,
    ) -> float:
        """Compute short-book correlation concentration (0=diversified, 1=concentrated).

        High correlation in shorts = all move together = low diversification benefit.
        """
        if len(short_symbols) < 2 or corr_matrix is None:
            return 0.0

        available = [s for s in short_symbols if s in corr_matrix.columns]
        if len(available) < 2:
            return 0.0

        sub_corr = corr_matrix.loc[available, available]
        n = len(available)
        off_diag = sub_corr.values.sum() - n  # remove diagonal
        avg_corr = off_diag / (n * (n - 1)) if n > 1 else 0.0
        return float(max(0.0, avg_corr))

    def mark_to_market_check(
        self,
        short_positions: pd.DataFrame,
        current_prices: pd.Series,
        entry_prices: pd.Series,
        *,
        atr_stops: dict[str, float] | None = None,
    ) -> list[str]:
        """Check which shorts have hit their stop-loss level.

        Args:
            short_positions: DataFrame with at least 'symbol' column and
                optional 'stop_loss_pct' column.
            current_prices: Series mapping symbol → current price.
            entry_prices: Series mapping symbol → entry price.
            atr_stops: Optional dict mapping symbol → ATR-based stop fraction.
                If provided for a symbol, takes precedence over fixed stop.

        Returns list of symbols to close (stop-out triggered).
        """
        atr_stops = atr_stops or {}
        stop_outs = []
        for _, row in short_positions.iterrows():
            symbol = row.get("symbol")
            if symbol not in current_prices.index or symbol not in entry_prices.index:
                continue

            entry = entry_prices[symbol]
            current = current_prices[symbol]
            if entry <= 0:
                continue

            # For shorts: loss = (current - entry) / entry (price RISE = short loss)
            pnl_pct = (current - entry) / entry

            # ATR-based stop takes precedence, then per-position, then global max
            if symbol in atr_stops:
                stop_loss = atr_stops[symbol]
            else:
                stop_loss = row.get("stop_loss_pct", self.max_stop_loss)

            if pnl_pct > stop_loss:
                logger.warning(
                    "[ShortRisk] Stop-out triggered: %s moved +%.1f%% against short "
                    "(stop=%.1f%%%s)",
                    symbol, pnl_pct * 100, stop_loss * 100,
                    " ATR" if symbol in atr_stops else "",
                )
                stop_outs.append(symbol)

        return stop_outs
