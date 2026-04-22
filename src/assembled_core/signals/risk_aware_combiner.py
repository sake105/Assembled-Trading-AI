"""Risk-Aware Signal Combiner.

Standard-Signal-Kombination: gewichtete Summe mit fixen Weights.
Problem: Weights sind regime-agnostisch; ein Signal das in RISK_ON funktioniert
kann in CRISIS scheitern.

Lösung: Regime-bedingte Signal-Gewichtung basierend auf historischer
Signal-Performance pro Regime.

    final_signal = Σ (weight[signal, regime] × raw_signal)

Wobei weight[s, r] aus historischer Sharpe des Signals im Regime r abgeleitet wird.

Ergänzt ensemble.py (simple averaging) um regime-bewusste Aggregation.

PIT-Invariante: weights werden auf Trainingsdaten berechnet; zur Inferenz
werden sie als read-only Tabelle genutzt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REGIMES = ["RISK_ON", "NEUTRAL", "RISK_OFF", "CRISIS"]


@dataclass
class SignalPerformance:
    """Performance eines Signals in einem Regime."""

    signal_name: str
    regime: str
    n_obs: int
    mean_return: float
    sharpe: float
    hit_rate: float


@dataclass
class CombinerState:
    """Persistenter State des Combiners."""

    performance_table: dict[tuple[str, str], SignalPerformance] = field(default_factory=dict)
    """{(signal, regime): Performance}"""

    weights: dict[tuple[str, str], float] = field(default_factory=dict)
    """{(signal, regime): weight}. Sum über signals pro regime = 1."""

    default_weight: float = 0.0
    """Weight für (signal, regime) nicht in Tabelle."""

    min_sharpe_for_inclusion: float = 0.0
    """Signals mit Regime-Sharpe < threshold → Gewicht = 0."""


class RiskAwareSignalCombiner:
    """Kombiniert mehrere Signale regime-bedingt gewichtet.

    Usage:
        combiner = RiskAwareSignalCombiner()
        combiner.fit(signal_df, returns, regime_series)
        combined = combiner.combine(current_signal_df, current_regime="NEUTRAL")
    """

    def __init__(
        self,
        min_obs_per_bucket: int = 20,
        min_sharpe_for_inclusion: float = 0.0,
        weight_floor: float = 0.0,
        softmax_temperature: float = 1.0,
    ) -> None:
        """Args:
            min_obs_per_bucket: Benötigte Samples pro (signal, regime)-Kombi.
            min_sharpe_for_inclusion: Signals unter diesem Sharpe kriegen weight=0.
            weight_floor: Minimum-Gewicht (vermeidet 0-Weights für alle in einem Regime).
            softmax_temperature: Höher → flacher, niedriger → Winner-takes-all.
        """
        self.min_obs_per_bucket = min_obs_per_bucket
        self.min_sharpe_for_inclusion = min_sharpe_for_inclusion
        self.weight_floor = weight_floor
        self.softmax_temperature = softmax_temperature
        self.state = CombinerState(
            min_sharpe_for_inclusion=min_sharpe_for_inclusion,
        )

    def fit(
        self,
        signal_df: pd.DataFrame,
        returns: pd.Series,
        regime_series: pd.Series,
    ) -> "RiskAwareSignalCombiner":
        """Berechnet pro (Signal, Regime)-Kombi die Performance.

        Args:
            signal_df: DataFrame mit Signalen (Spalten = Signal-Namen)
            returns: Zeit-aligned Returns (gleicher Index)
            regime_series: Zeit-aligned Regime-Labels (gleicher Index)

        Returns:
            Self mit populiertem state.
        """
        aligned = pd.concat([signal_df, returns.rename("_ret"), regime_series.rename("_regime")], axis=1).dropna()

        for regime in _REGIMES:
            mask = aligned["_regime"] == regime
            sub = aligned[mask]
            if len(sub) < self.min_obs_per_bucket:
                logger.debug(
                    "[Combiner] Regime=%s: nur %d Samples (<%d) — skip",
                    regime, len(sub), self.min_obs_per_bucket,
                )
                continue

            rets = sub["_ret"].values
            for sig_name in signal_df.columns:
                sig_vals = sub[sig_name].values
                # Strategy: sign(signal) * next_return
                strat_rets = np.sign(sig_vals) * rets

                if strat_rets.std() > 1e-9:
                    sharpe = float(strat_rets.mean() / strat_rets.std() * np.sqrt(252))
                else:
                    sharpe = 0.0

                hit_rate = float((np.sign(sig_vals) == np.sign(rets)).mean())

                self.state.performance_table[(sig_name, regime)] = SignalPerformance(
                    signal_name=sig_name,
                    regime=regime,
                    n_obs=len(sub),
                    mean_return=float(strat_rets.mean()),
                    sharpe=sharpe,
                    hit_rate=hit_rate,
                )

        self._compute_weights(list(signal_df.columns))
        logger.info(
            "[Combiner] fitted: %d (signal,regime)-Buckets, %d weights",
            len(self.state.performance_table),
            len(self.state.weights),
        )
        return self

    def _compute_weights(self, signal_names: list[str]) -> None:
        """Softmax-Weights pro Regime über Signal-Sharpes."""
        for regime in _REGIMES:
            # Sammle Sharpes dieses Regimes
            sharpes_by_signal: dict[str, float] = {}
            for sig in signal_names:
                perf = self.state.performance_table.get((sig, regime))
                if perf is None or perf.sharpe < self.min_sharpe_for_inclusion:
                    continue
                sharpes_by_signal[sig] = perf.sharpe

            if not sharpes_by_signal:
                # Equal-weight fallback
                for sig in signal_names:
                    self.state.weights[(sig, regime)] = 1.0 / len(signal_names)
                continue

            # Softmax über Sharpes
            arr = np.array(list(sharpes_by_signal.values()))
            scaled = arr / max(self.softmax_temperature, 1e-6)
            scaled = scaled - scaled.max()
            exp_u = np.exp(scaled)
            w = exp_u / exp_u.sum()

            # Assign mit floor
            for sig, weight in zip(sharpes_by_signal.keys(), w):
                self.state.weights[(sig, regime)] = max(float(weight), self.weight_floor)

            # Signals die nicht in sharpes_by_signal sind → 0 oder floor
            for sig in signal_names:
                if (sig, regime) not in self.state.weights:
                    self.state.weights[(sig, regime)] = self.weight_floor

            # Renormalize (falls floor > 0)
            total = sum(self.state.weights[(s, regime)] for s in signal_names)
            if total > 1e-9:
                for sig in signal_names:
                    self.state.weights[(sig, regime)] /= total

    def combine(
        self,
        signal_df: pd.DataFrame,
        current_regime: str,
    ) -> pd.Series:
        """Kombiniert Signals regime-bedingt.

        Args:
            signal_df: DataFrame mit Signalen (Spalten = Signal-Namen)
            current_regime: Aktuelles Regime-Label

        Returns:
            pd.Series mit kombiniertem Signal.
        """
        if current_regime not in _REGIMES:
            logger.warning("[Combiner] Unbekanntes Regime %s — NEUTRAL Fallback", current_regime)
            current_regime = "NEUTRAL"

        combined = np.zeros(len(signal_df))
        for sig in signal_df.columns:
            w = self.state.weights.get((sig, current_regime), self.state.default_weight)
            combined += w * signal_df[sig].fillna(0.0).values

        return pd.Series(combined, index=signal_df.index, name="risk_aware_combined")

    def combine_auto_regime(
        self,
        signal_df: pd.DataFrame,
        regime_series: pd.Series,
    ) -> pd.Series:
        """Kombiniert pro Zeile mit Regime aus regime_series."""
        combined = np.zeros(len(signal_df))
        for i, (idx, row) in enumerate(signal_df.iterrows()):
            regime = regime_series.get(idx, "NEUTRAL")
            if regime not in _REGIMES:
                regime = "NEUTRAL"
            val = 0.0
            for sig in signal_df.columns:
                w = self.state.weights.get((sig, regime), self.state.default_weight)
                val += w * (0.0 if pd.isna(row[sig]) else float(row[sig]))
            combined[i] = val
        return pd.Series(combined, index=signal_df.index, name="risk_aware_combined")

    def get_weights(self, regime: str) -> dict[str, float]:
        """Gibt aktuelle Weights für ein Regime zurück."""
        return {
            sig: w for (sig, r), w in self.state.weights.items() if r == regime
        }

    def summary(self) -> dict:
        """Kompakte Zusammenfassung."""
        regime_summary = {}
        for regime in _REGIMES:
            perf = [
                p for (s, r), p in self.state.performance_table.items()
                if r == regime
            ]
            weights = self.get_weights(regime)
            regime_summary[regime] = {
                "n_signals_with_perf": len(perf),
                "mean_sharpe": round(float(np.mean([p.sharpe for p in perf])), 3) if perf else 0.0,
                "weights": {k: round(v, 4) for k, v in weights.items()},
            }
        return regime_summary


__all__ = [
    "SignalPerformance",
    "CombinerState",
    "RiskAwareSignalCombiner",
]
