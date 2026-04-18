"""Multi-signal crash prediction engine.

Aggregates 16 signals across 4 categories to estimate crash probability,
severity, and time horizon. Used as input to the short-profit engine.

Signal categories:
  - Technical (30%): death cross, breadth, VIX, put/call, new lows, A/D
  - Regime (25%):    bear probability, HMM crisis probability, vol regime shift
  - Geopolitical (25%): geo crisis score, shock cascade, sanctions, military escalation
  - Macro (20%):     yield curve, credit spreads, leading indicators, monetary tightening
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CrashSignal:
    """Output of the CrashPredictionEngine."""

    crash_probability: float            # 0-1: probability of significant decline (>10%)
    expected_severity: float            # 0-1: 0=mild correction, 1=systemic crisis
    time_horizon_days: int              # Expected days to crash onset
    confidence: float                   # 0-1: confidence in the prediction
    contributing_signals: dict[str, float] = field(default_factory=dict)
    recommended_sectors_short: list[str] = field(default_factory=list)
    recommended_instruments: list[str] = field(default_factory=list)
    active: bool = False               # True when crash_probability >= threshold


# ---------------------------------------------------------------------------
# Signal weights by category
# ---------------------------------------------------------------------------

CATEGORY_WEIGHTS = {
    "technical": 0.30,
    "regime": 0.25,
    "geopolitical": 0.25,
    "macro": 0.20,
}

# Sector vulnerability map: which sectors get hurt first in a crash
SECTOR_SHORT_PRIORITY = {
    "TECH": 0.9,
    "CONSUMER": 0.8,
    "AUTO": 0.8,
    "FINANCE": 0.75,
    "SEMIS": 0.85,
    "SHIPPING": 0.7,
    "MINING": 0.65,
    "ENERGY": 0.5,   # Energy can be hedge
    "DEFENSE": 0.3,  # Defense rises in crises
    "GOLD": 0.0,     # Never short gold in crisis
}

# Inverse ETF recommendations per sector
SECTOR_TO_INVERSE_ETF = {
    "TECH": "PSQ",          # ProShares Short QQQ
    "SEMIS": "SSG",         # ProShares UltraShort Semiconductors
    "FINANCE": "SEF",       # ProShares Short Financials
    "ENERGY": "DDG",        # ProShares Short Oil & Gas
    "AUTO": "SH",           # ProShares Short S&P500 (broad)
    "CONSUMER": "SH",
    "SHIPPING": "SH",
    "MINING": "MYY",        # ProShares Short MidCap400
    "BROAD": "SH",          # Default: short S&P500
    "RUSSELL": "RWM",       # ProShares Short Russell2000
}


class CrashPredictionEngine:
    """Aggregates all available crash indicators into a composite estimate.

    Each signal returns a float 0-1. The engine combines them via
    category-weighted average, then applies a calibration sigmoid.
    """

    def predict(
        self,
        market_data: pd.DataFrame | None = None,
        regime: Any = None,
        intel_state: Any = None,
        macro_data: dict[str, float] | None = None,
    ) -> CrashSignal:
        """Compute composite crash probability.

        Args:
            market_data: OHLCV DataFrame indexed by date (most recent last).
            regime:      Regime state object with .state or .label attribute.
            intel_state: CrisisIntelState or dict with geo_score, shock_types, etc.
            macro_data:  Dict of macro indicators (yield_spread, vix, etc.).

        Returns:
            CrashSignal with composite probability and recommended instruments.
        """
        signals: dict[str, float] = {}

        # ------------------------------------------------------------------ #
        # 1. Technical signals (30%)
        # ------------------------------------------------------------------ #
        signals["death_cross"] = self._detect_death_cross(market_data)
        signals["breadth_collapse"] = self._detect_breadth_collapse(market_data)
        signals["vix_regime"] = self._check_vix_regime(macro_data)
        signals["put_call_extreme"] = self._check_put_call_extreme(macro_data)
        signals["new_lows_expansion"] = self._detect_new_lows(market_data)
        signals["ad_divergence"] = self._check_ad_divergence(market_data)

        # ------------------------------------------------------------------ #
        # 2. Regime signals (25%)
        # ------------------------------------------------------------------ #
        signals["regime_bear_prob"] = self._regime_signal(regime)
        signals["hmm_crisis_prob"] = self._hmm_signal(market_data, regime)
        signals["vol_regime_shift"] = self._vol_regime_shift(market_data, macro_data)

        # ------------------------------------------------------------------ #
        # 3. Geopolitical signals (25%)
        # ------------------------------------------------------------------ #
        signals["geo_crisis_score"] = self._geo_signal(intel_state)
        signals["shock_cascade_risk"] = self._shock_cascade_signal(intel_state)
        signals["sanctions_escalation"] = self._sanctions_signal(intel_state)
        signals["military_escalation"] = self._military_signal(intel_state)

        # ------------------------------------------------------------------ #
        # 4. Macro signals (20%)
        # ------------------------------------------------------------------ #
        signals["yield_curve_inversion"] = self._yield_curve_signal(macro_data)
        signals["credit_spread_widening"] = self._credit_spread_signal(macro_data)
        signals["monetary_tightening"] = self._monetary_signal(macro_data, regime)

        # ------------------------------------------------------------------ #
        # Weighted aggregation
        # ------------------------------------------------------------------ #
        crash_prob = self._weighted_aggregate(signals)
        severity = self._estimate_severity(signals, intel_state)
        horizon = self._estimate_horizon(signals, macro_data)
        sectors = self._identify_vulnerable_sectors(signals, intel_state)
        instruments = self._select_short_instruments(sectors, severity)

        result = CrashSignal(
            crash_probability=round(crash_prob, 4),
            expected_severity=round(severity, 4),
            time_horizon_days=horizon,
            confidence=round(self._compute_confidence(signals), 4),
            contributing_signals={k: round(v, 4) for k, v in signals.items()},
            recommended_sectors_short=sectors,
            recommended_instruments=instruments,
            active=crash_prob >= 0.60,
        )

        logger.info(
            "[CrashPrediction] prob=%.3f severity=%.3f horizon=%dd active=%s",
            crash_prob, severity, horizon, result.active,
        )
        return result

    # ------------------------------------------------------------------ #
    # Technical signal helpers
    # ------------------------------------------------------------------ #

    def _detect_death_cross(self, market_data: pd.DataFrame | None) -> float:
        """SMA50 < SMA200 = death cross signal."""
        if market_data is None or "close" not in market_data.columns:
            return 0.0
        close = market_data["close"]
        if len(close) < 200:
            return 0.0
        sma50 = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1]
        if pd.isna(sma50) or pd.isna(sma200):
            return 0.0
        if sma50 < sma200:
            # Strength proportional to divergence
            divergence = (sma200 - sma50) / sma200
            return min(divergence * 10, 1.0)
        return 0.0

    def _detect_breadth_collapse(self, market_data: pd.DataFrame | None) -> float:
        """Detect broad market weakness (fraction above SMA50 < 30%)."""
        if market_data is None:
            return 0.0
        if "breadth_pct_above_ma50" in market_data.columns:
            val = market_data["breadth_pct_above_ma50"].iloc[-1]
            if pd.isna(val):
                return 0.0
            if val < 0.30:
                return 1.0 - (val / 0.30)
        return 0.0

    def _check_vix_regime(self, macro_data: dict | None) -> float:
        """VIX above 25 = moderate, above 35 = severe."""
        if not macro_data:
            return 0.0
        vix = macro_data.get("vix", 0)
        if vix <= 20:
            return 0.0
        elif vix <= 25:
            return 0.20
        elif vix <= 30:
            return 0.45
        elif vix <= 40:
            return 0.70
        else:
            return 1.0

    def _check_put_call_extreme(self, macro_data: dict | None) -> float:
        """Put/call ratio > 1.2 = bearish extreme."""
        if not macro_data:
            return 0.0
        pcr = macro_data.get("put_call_ratio", 0)
        if pcr <= 0.8:
            return 0.0
        elif pcr <= 1.0:
            return 0.20
        elif pcr <= 1.2:
            return 0.50
        elif pcr <= 1.5:
            return 0.80
        else:
            return 1.0

    def _detect_new_lows(self, market_data: pd.DataFrame | None) -> float:
        """52-week new lows expanding = breadth deterioration."""
        if market_data is None:
            return 0.0
        if "new_lows_pct" in market_data.columns:
            val = market_data["new_lows_pct"].iloc[-1]
            if pd.isna(val):
                return 0.0
            return min(val * 5, 1.0)
        return 0.0

    def _check_ad_divergence(self, market_data: pd.DataFrame | None) -> float:
        """A/D line diverging from price = distribution top signal."""
        if market_data is None:
            return 0.0
        if "ad_divergence" in market_data.columns:
            val = market_data["ad_divergence"].iloc[-1]
            if pd.isna(val):
                return 0.0
            return max(0.0, min(-val / 0.05, 1.0)) if val < 0 else 0.0
        return 0.0

    # ------------------------------------------------------------------ #
    # Regime signal helpers
    # ------------------------------------------------------------------ #

    def _regime_signal(self, regime: Any) -> float:
        """Extract bear/crisis probability from regime state."""
        if regime is None:
            return 0.0
        # Support both string labels and objects with .state attribute
        label = str(getattr(regime, "state", getattr(regime, "label", regime))).lower()
        mapping = {
            "crisis": 0.90,
            "bear": 0.70,
            "risk_off": 0.65,
            "sideways": 0.30,
            "reflation": 0.20,
            "bull": 0.05,
        }
        for key, val in mapping.items():
            if key in label:
                return val
        # Unknown/typo'd regime label used to silently contribute 0.20
        # (a non-neutral "mild crash" reading) to the aggregate crash
        # probability. Log and return 0.0 so a misconfigured regime does
        # not artificially lift crash_probability.
        logger.warning("[CrashPrediction] unknown regime label: %r", label)
        return 0.0

    def _hmm_signal(self, market_data: pd.DataFrame | None, regime: Any) -> float:
        """HMM-derived crisis state probability."""
        if regime is None:
            return 0.0
        # Try to get HMM crisis probability if available
        prob = getattr(regime, "crisis_probability", None)
        if prob is not None:
            return float(prob)
        # Fall back to regime-based estimate
        return self._regime_signal(regime) * 0.8

    def _vol_regime_shift(
        self, market_data: pd.DataFrame | None, macro_data: dict | None
    ) -> float:
        """Detect vol regime shift: recent vol >> historical vol."""
        if market_data is None or "close" not in market_data.columns:
            return 0.0
        close = market_data["close"]
        if len(close) < 60:
            return 0.0
        recent_vol = close.pct_change().rolling(10).std().iloc[-1]
        hist_vol = close.pct_change().rolling(60).std().iloc[-1]
        if pd.isna(recent_vol) or pd.isna(hist_vol) or hist_vol == 0:
            return 0.0
        vol_ratio = recent_vol / hist_vol
        if vol_ratio > 2.5:
            return 1.0
        elif vol_ratio > 2.0:
            return 0.75
        elif vol_ratio > 1.5:
            return 0.40
        return 0.0

    # ------------------------------------------------------------------ #
    # Geopolitical signal helpers
    # ------------------------------------------------------------------ #

    def _geo_signal(self, intel_state: Any) -> float:
        """Extract geo crisis score from intel state."""
        if intel_state is None:
            return 0.0
        # Support dict or object
        geo_score = (
            intel_state.get("geo_score", 0) if isinstance(intel_state, dict)
            else getattr(intel_state, "geo_score", 0)
        )
        return min(float(geo_score) / 3.0, 1.0)

    def _shock_cascade_signal(self, intel_state: Any) -> float:
        """Active shock cascade risk from intel pipeline."""
        if intel_state is None:
            return 0.0
        cascade = (
            intel_state.get("shock_cascade_risk", 0) if isinstance(intel_state, dict)
            else getattr(intel_state, "shock_cascade_risk", 0)
        )
        return min(float(cascade), 1.0)

    def _sanctions_signal(self, intel_state: Any) -> float:
        """New major sanctions package detected."""
        if intel_state is None:
            return 0.0
        shocks = (
            intel_state.get("active_shocks", []) if isinstance(intel_state, dict)
            else getattr(intel_state, "active_shocks", [])
        )
        sanctions_shocks = {"sanctions_exposure", "banking_isolation", "secondary_sanctions_risk"}
        matches = sum(1 for s in shocks if str(s).lower() in sanctions_shocks)
        return min(matches * 0.35, 1.0)

    def _military_signal(self, intel_state: Any) -> float:
        """Military escalation signal from active conflicts."""
        if intel_state is None:
            return 0.0
        shocks = (
            intel_state.get("active_shocks", []) if isinstance(intel_state, dict)
            else getattr(intel_state, "active_shocks", [])
        )
        military_shocks = {"nuclear_escalation_risk", "military_loss_surge", "supply_line_threat"}
        matches = sum(1 for s in shocks if str(s).lower() in military_shocks)
        # Nuclear escalation is especially severe
        if "nuclear_escalation_risk" in [str(s).lower() for s in shocks]:
            return min(0.70 + matches * 0.15, 1.0)
        return min(matches * 0.40, 1.0)

    # ------------------------------------------------------------------ #
    # Macro signal helpers
    # ------------------------------------------------------------------ #

    def _yield_curve_signal(self, macro_data: dict | None) -> float:
        """2s10s yield curve inversion depth."""
        if not macro_data:
            return 0.0
        spread = macro_data.get("yield_2s10s", 1.0)  # positive = normal, negative = inverted
        if spread >= 0.50:
            return 0.0
        elif spread >= 0:
            return 0.15
        elif spread >= -0.25:
            return 0.40
        elif spread >= -0.50:
            return 0.65
        else:
            return 0.90

    def _credit_spread_signal(self, macro_data: dict | None) -> float:
        """HY credit spread widening above 400bps = stress."""
        if not macro_data:
            return 0.0
        hy_spread = macro_data.get("hy_spread_bps", 300)
        if hy_spread < 300:
            return 0.0
        elif hy_spread < 400:
            return 0.20
        elif hy_spread < 600:
            return 0.50
        elif hy_spread < 900:
            return 0.75
        else:
            return 1.0

    def _monetary_signal(self, macro_data: dict | None, regime: Any) -> float:
        """Synchronized global tightening = liquidity drain."""
        if not macro_data:
            return 0.0
        # Check if we're in a tightening cycle
        rate_level = macro_data.get("fed_funds_rate", 0)
        rate_change_12m = macro_data.get("fed_rate_change_12m", 0)
        if rate_change_12m > 3.0:  # +300bps in 12 months = aggressive tightening
            return 0.85
        elif rate_change_12m > 2.0:
            return 0.60
        elif rate_change_12m > 1.0:
            return 0.35
        elif rate_level > 5.0:     # High absolute rate
            return 0.25
        return 0.0

    # ------------------------------------------------------------------ #
    # Aggregation and output helpers
    # ------------------------------------------------------------------ #

    def _weighted_aggregate(self, signals: dict[str, float]) -> float:
        """Category-weighted average of all signals."""
        tech_signals = ["death_cross", "breadth_collapse", "vix_regime",
                        "put_call_extreme", "new_lows_expansion", "ad_divergence"]
        regime_signals = ["regime_bear_prob", "hmm_crisis_prob", "vol_regime_shift"]
        geo_signals = ["geo_crisis_score", "shock_cascade_risk",
                       "sanctions_escalation", "military_escalation"]
        macro_signals = ["yield_curve_inversion", "credit_spread_widening", "monetary_tightening"]

        def cat_avg(keys: list[str]) -> float:
            vals = [signals[k] for k in keys if k in signals]
            return sum(vals) / len(vals) if vals else 0.0

        score = (
            CATEGORY_WEIGHTS["technical"] * cat_avg(tech_signals)
            + CATEGORY_WEIGHTS["regime"] * cat_avg(regime_signals)
            + CATEGORY_WEIGHTS["geopolitical"] * cat_avg(geo_signals)
            + CATEGORY_WEIGHTS["macro"] * cat_avg(macro_signals)
        )
        return min(score, 1.0)

    def _estimate_severity(self, signals: dict[str, float], intel_state: Any) -> float:
        """Estimate expected crash severity (0=mild correction, 1=systemic)."""
        # High nuclear/sanctions = severe; high VIX + death cross = moderate-severe
        nuclear = signals.get("military_escalation", 0)
        geo = signals.get("geo_crisis_score", 0)
        credit = signals.get("credit_spread_widening", 0)
        technical = (signals.get("death_cross", 0) + signals.get("breadth_collapse", 0)) / 2

        severity = (
            0.35 * nuclear
            + 0.25 * credit
            + 0.25 * geo
            + 0.15 * technical
        )
        return min(severity, 1.0)

    def _estimate_horizon(self, signals: dict[str, float], macro_data: dict | None) -> int:
        """Estimate days until crash onset."""
        # Yield curve inversion = 12-18 months lead time
        # Technical signals = days to weeks
        # Geopolitical = immediate to weeks
        if signals.get("yield_curve_inversion", 0) > 0.5 and signals.get("death_cross", 0) < 0.3:
            return 180  # 6 months lead
        elif signals.get("death_cross", 0) > 0.5:
            return 14   # 2 weeks
        elif signals.get("geo_crisis_score", 0) > 0.7:
            return 7    # 1 week
        elif signals.get("breadth_collapse", 0) > 0.5:
            return 30   # 1 month
        else:
            return 60   # 2 months default

    def _identify_vulnerable_sectors(
        self, signals: dict[str, float], intel_state: Any
    ) -> list[str]:
        """Identify sectors most vulnerable to predicted crash."""
        sectors = []

        # Tech/Semis vulnerable when geopolitical or tech decoupling active
        if signals.get("geo_crisis_score", 0) > 0.4 or signals.get("sanctions_escalation", 0) > 0.3:
            sectors.extend(["SEMIS", "TECH"])

        # Finance vulnerable in credit stress
        if signals.get("credit_spread_widening", 0) > 0.4:
            sectors.append("FINANCE")

        # Consumer/Auto vulnerable in rate shock or recession
        if signals.get("monetary_tightening", 0) > 0.4 or signals.get("yield_curve_inversion", 0) > 0.4:
            sectors.extend(["CONSUMER", "AUTO"])

        # Broad market short if severe technical breakdown
        if signals.get("death_cross", 0) > 0.5 and signals.get("breadth_collapse", 0) > 0.3:
            sectors.append("BROAD")

        # Energy vulnerable only in demand collapse (not supply shock)
        if signals.get("regime_bear_prob", 0) > 0.7 and signals.get("geo_crisis_score", 0) < 0.3:
            sectors.append("ENERGY")

        return list(dict.fromkeys(sectors))  # deduplicate while preserving order

    def _select_short_instruments(self, sectors: list[str], severity: float) -> list[str]:
        """Select specific short instruments based on sectors and severity."""
        instruments = []
        for sector in sectors:
            etf = SECTOR_TO_INVERSE_ETF.get(sector)
            if etf and etf not in instruments:
                instruments.append(etf)
        # For severe crashes, add small-cap short (smaller companies hit harder)
        if severity > 0.6 and "RWM" not in instruments:
            instruments.append("RWM")
        return instruments

    def _compute_confidence(self, signals: dict[str, float]) -> float:
        """Confidence in the prediction = agreement among signals."""
        vals = list(signals.values())
        if not vals:
            return 0.0
        n_active = sum(1 for v in vals if v > 0.3)
        n_total = len(vals)
        # High confidence when many signals agree
        agreement = n_active / n_total
        avg_strength = sum(vals) / n_total
        return min((agreement * 0.6 + avg_strength * 0.4), 1.0)


# ---------------------------------------------------------------------------
# Dynamic Threshold Computation (Plan 1.7)
# ---------------------------------------------------------------------------


def compute_rolling_percentile_thresholds(
    series: pd.Series,
    window: int = 252,
    percentiles: tuple[float, ...] = (0.75, 0.90, 0.99),
    min_periods: int = 60,
) -> pd.DataFrame:
    """Compute rolling percentile-based thresholds for crash indicators.

    Replaces hardcoded VIX thresholds (25/35/50) with adaptive
    rolling percentiles that adjust to the current volatility regime.

    Args:
        series: Time series (e.g., VIX, put/call ratio, yield curve).
        window: Rolling window in trading days (default: 252 = 1 year).
        percentiles: Percentile levels to compute.
        min_periods: Minimum observations for valid percentile.

    Returns:
        DataFrame with columns like ``p75``, ``p90``, ``p99``.
    """
    result = pd.DataFrame(index=series.index)
    for p in percentiles:
        col_name = f"p{int(p * 100)}"
        result[col_name] = series.rolling(window, min_periods=min_periods).quantile(p)
    return result
