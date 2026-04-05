"""Intel signal adapter: converts DependencySignal objects to trading signals.

Bridge between the geopolitical intel pipeline and the trading signal layer.
DependencySignal has beneficiaries/losers → mapping to concrete symbols.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector → ETF symbol map (used to convert sector nodes to tradeable symbols)
# ---------------------------------------------------------------------------

SECTOR_TO_ETF: dict[str, str] = {
    "ENERGY": "XLE",
    "DEFENSE": "ITA",
    "TECH": "QQQ",
    "SEMIS": "SOXX",
    "AUTO": "CARZ",
    "PHARMA": "XPH",
    "SHIPPING": "BOAT",
    "MINING": "XME",
    "AGRICULTURE": "DBA",
    "FINANCE": "XLF",
    "CYBER": "CIBR",
    "CONSUMER": "XLY",
    "RENEWABLE_ENERGY": "ICLN",
    "AEROSPACE": "ITA",
    "TELECOM": "IYZ",
    "GOLD": "GLD",
    "OIL": "USO",
    "MATERIALS": "XLB",
    "UTILITIES": "XLU",
    "REAL_ESTATE": "VNQ",
}

# Inverse ETF map for short signals (when sector is a loser)
SECTOR_TO_INVERSE_ETF: dict[str, str] = {
    "ENERGY": "DDG",
    "TECH": "PSQ",
    "SEMIS": "PSQ",
    "FINANCE": "SEF",
    "CONSUMER": "SH",
    "AUTO": "SH",
    "BROAD": "SH",
    "SMALL_CAP": "RWM",
}

# Known beneficiary nodes for various shock types
SHOCK_BENEFICIARY_MAP: dict[str, list[str]] = {
    "oil_supply_risk": ["XLE", "GLD"],
    "energy_price_spike": ["XLE", "GLD"],
    "defense_demand_surge": ["ITA", "LMT", "RTX", "NOC"],
    "global_risk_off": ["GLD", "TLT", "SHY"],
    "shipping_cost_risk": ["ZIM", "MATX"],
    "insurance_cost_risk": ["GLD"],
    "cyber_risk": ["CIBR", "PANW", "CRWD"],
    "rare_earth_supply_risk": ["MP", "REMX"],
    "semiconductor_supply_risk": ["SOXX", "TSM"],
    "inflation_spike": ["GLD", "PDBC", "XLE"],
    "currency_crisis": ["GLD", "DXY"],
    "rate_shock": ["TBT", "SH"],  # Short-term bearish for both equities and bonds
    "food_supply_risk": ["DBA", "MOO"],
    "climate_disruption": ["ICLN"],
    "nuclear_escalation_risk": ["GLD", "ITA"],
}


@dataclass
class IntelTradingSignal:
    """A trading signal derived from geopolitical intel."""

    symbol: str
    direction: str          # "LONG", "SHORT", "FLAT"
    score: float            # 0-1 (absolute), positive = long, negative = short
    confidence: float       # 0-1
    source_trigger: str     # originating trigger ID
    horizon: str            # "intraday", "short", "medium"
    reason: str = ""
    severity: int = 1       # 0-3


class IntelSignalAdapter:
    """Converts DependencySignal objects into trading signals.

    Handles:
    - Sector node → ETF translation
    - Beneficiary nodes → LONG signals
    - Loser nodes → SHORT signals (via inverse ETFs if configured)
    - Pair trade identification (long beneficiary / short loser)
    """

    def __init__(
        self,
        allow_short_signals: bool = True,
        min_confidence: float = 0.40,
        min_severity: int = 1,
    ):
        self.allow_short = allow_short_signals
        self.min_confidence = min_confidence
        self.min_severity = min_severity

    def convert_to_trading_signals(
        self,
        dep_signals: list[Any],
        symbol_sector_map: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Convert DependencySignal objects to a trading signals DataFrame.

        Args:
            dep_signals: List of DependencySignal objects from intel pipeline.
            symbol_sector_map: Optional map from sector node ID to ETF symbol.

        Returns:
            DataFrame with columns: symbol, direction, score, confidence,
                                    source_trigger, horizon, reason, severity
        """
        if not dep_signals:
            return pd.DataFrame()

        sector_map = symbol_sector_map or SECTOR_TO_ETF
        signals: list[IntelTradingSignal] = []

        for dep_sig in dep_signals:
            # Filter by confidence and severity
            sig_conf = float(getattr(dep_sig, "confidence", 0))
            sig_sev = int(getattr(dep_sig, "severity", 0))

            if sig_conf < self.min_confidence:
                continue
            if sig_sev < self.min_severity:
                continue

            trigger_id = str(getattr(dep_sig, "trigger_id", "unknown"))
            horizon = str(getattr(dep_sig, "time_horizon", "medium"))

            # Long signals from beneficiaries
            beneficiaries = getattr(dep_sig, "beneficiaries", []) or []
            for node_id in beneficiaries:
                etf = sector_map.get(node_id) or SECTOR_TO_ETF.get(node_id)
                if etf is None:
                    # Try direct symbol if it looks like a ticker
                    if len(node_id) <= 5 and node_id.isupper():
                        etf = node_id
                    else:
                        continue

                signals.append(IntelTradingSignal(
                    symbol=etf,
                    direction="LONG",
                    score=sig_conf,
                    confidence=sig_conf,
                    source_trigger=trigger_id,
                    horizon=horizon,
                    reason=f"Intel beneficiary: {node_id} (trigger={trigger_id})",
                    severity=sig_sev,
                ))

            # Short signals from losers (if allowed)
            if self.allow_short:
                losers = getattr(dep_sig, "losers", []) or []
                for node_id in losers:
                    # Try inverse ETF first
                    inv_etf = SECTOR_TO_INVERSE_ETF.get(node_id)
                    if inv_etf is None:
                        # Fall back to sector ETF (for potential direct short)
                        etf = sector_map.get(node_id) or SECTOR_TO_ETF.get(node_id)
                        inv_etf = SECTOR_TO_INVERSE_ETF.get(etf, "SH") if etf else None

                    if inv_etf is None:
                        continue

                    signals.append(IntelTradingSignal(
                        symbol=inv_etf,
                        direction="SHORT",
                        score=-sig_conf,
                        confidence=sig_conf,
                        source_trigger=trigger_id,
                        horizon=horizon,
                        reason=f"Intel loser: {node_id} (trigger={trigger_id})",
                        severity=sig_sev,
                    ))

        if not signals:
            return pd.DataFrame()

        df = pd.DataFrame([
            {
                "symbol": s.symbol,
                "direction": s.direction,
                "score": round(s.score, 4),
                "confidence": round(s.confidence, 4),
                "source_trigger": s.source_trigger,
                "horizon": s.horizon,
                "reason": s.reason,
                "severity": s.severity,
            }
            for s in signals
        ])

        # Aggregate: if same symbol appears multiple times, take max confidence
        df = (
            df.sort_values("confidence", ascending=False)
            .drop_duplicates(subset=["symbol", "direction"], keep="first")
            .reset_index(drop=True)
        )

        logger.info(
            "[IntelSignalAdapter] %d dep_signals -> %d trading signals (%d long, %d short)",
            len(dep_signals),
            len(df),
            len(df[df["direction"] == "LONG"]),
            len(df[df["direction"] == "SHORT"]),
        )
        return df

    def compute_sector_impact_scores(
        self,
        dep_signals: list[Any],
    ) -> dict[str, float]:
        """Aggregate all signals per sector node to a net impact score.

        Positive = net beneficiary, Negative = net loser.
        """
        sector_scores: dict[str, float] = {}

        for dep_sig in dep_signals:
            conf = float(getattr(dep_sig, "confidence", 0))
            sev = int(getattr(dep_sig, "severity", 1))
            weight = conf * sev

            for node_id in (getattr(dep_sig, "beneficiaries", []) or []):
                sector_scores[node_id] = sector_scores.get(node_id, 0) + weight

            for node_id in (getattr(dep_sig, "losers", []) or []):
                sector_scores[node_id] = sector_scores.get(node_id, 0) - weight

        return sector_scores

    def identify_pair_trades(
        self,
        dep_signals: list[Any],
    ) -> list[tuple[str, str, float]]:
        """Identify long-short pairs from beneficiary/loser relationships.

        Returns list of (long_etf, short_etf, confidence) tuples.
        """
        pairs = []

        for dep_sig in dep_signals:
            conf = float(getattr(dep_sig, "confidence", 0))
            if conf < self.min_confidence:
                continue

            beneficiaries = getattr(dep_sig, "beneficiaries", []) or []
            losers = getattr(dep_sig, "losers", []) or []

            for ben in beneficiaries:
                long_etf = SECTOR_TO_ETF.get(ben)
                if long_etf is None:
                    continue
                for loser in losers:
                    short_inv = SECTOR_TO_INVERSE_ETF.get(loser)
                    if short_inv is None:
                        continue
                    pairs.append((long_etf, short_inv, round(conf, 4)))

        return pairs

    def enrich_signals_with_shock_beneficiaries(
        self,
        active_shocks: list[str],
        base_confidence: float = 0.60,
    ) -> pd.DataFrame:
        """Generate signals directly from active shock types using built-in map.

        Useful when intel pipeline produces shocks but no full dependency traversal.
        """
        signals = []
        for shock in active_shocks:
            shock_lower = shock.lower()
            beneficiaries = SHOCK_BENEFICIARY_MAP.get(shock_lower, [])
            for sym in beneficiaries:
                signals.append({
                    "symbol": sym,
                    "direction": "LONG",
                    "score": round(base_confidence, 4),
                    "confidence": round(base_confidence, 4),
                    "source_trigger": shock_lower,
                    "horizon": "short",
                    "reason": f"Direct shock beneficiary: {shock}",
                    "severity": 1,
                })

        if not signals:
            return pd.DataFrame()

        df = pd.DataFrame(signals)
        return df.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)
