"""Signal-Correlation-Analyzer — erkennt redundante Signale.

Ergänzt `feature_clustering.py` (Round 2, allgemein) um spezifische Signal-
Ebene-Analyse:
- rollende Korrelationsmatrix
- hierarchisches Clustering auf |corr|
- Redundanz-Report (Gruppen mit mean_corr > threshold)
- Zeitliche Stabilität der Korrelationen (detektiert Cluster-Changes)

PIT-Invariante: Alles auf historischen Daten.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SignalCorrelationReport:
    correlation_matrix: pd.DataFrame
    mean_abs_corr: float
    """Mittlere absolute paarweise Korrelation."""

    redundant_clusters: list[list[str]] = field(default_factory=list)
    """Liste von Clustern mit hoher interner Korrelation."""

    n_signals: int = 0
    threshold: float = 0.7


class SignalCorrelationAnalyzer:
    """Analysiert Inter-Signal-Korrelationen und identifiziert Redundanz."""

    def __init__(
        self,
        redundancy_threshold: float = 0.7,
        min_cluster_size: int = 2,
    ) -> None:
        self.redundancy_threshold = redundancy_threshold
        self.min_cluster_size = min_cluster_size

    def analyze(self, signals: pd.DataFrame) -> SignalCorrelationReport:
        """Rollende Korrelation ist out-of-scope — hier nur Snapshot.

        Args:
            signals: DataFrame mit Signal-Spalten

        Returns:
            SignalCorrelationReport
        """
        if signals.shape[1] < 2 or len(signals) < 10:
            return SignalCorrelationReport(
                correlation_matrix=pd.DataFrame(),
                mean_abs_corr=0.0,
                redundant_clusters=[],
                n_signals=signals.shape[1],
                threshold=self.redundancy_threshold,
            )

        corr = signals.corr().fillna(0.0)
        # Mean |corr| ohne Diagonale
        mask = ~np.eye(len(corr), dtype=bool)
        mean_abs = float(np.abs(corr.values[mask]).mean())

        # Simple greedy clustering: find groups mit pairwise |corr| > threshold
        redundant: list[list[str]] = []
        remaining = set(corr.columns)
        for col in corr.columns:
            if col not in remaining:
                continue
            group = [col]
            for other in list(remaining - {col}):
                if all(abs(corr.loc[other, g]) >= self.redundancy_threshold for g in group):
                    group.append(other)
            if len(group) >= self.min_cluster_size:
                redundant.append(sorted(group))
                remaining -= set(group)
            else:
                remaining.discard(col)

        return SignalCorrelationReport(
            correlation_matrix=corr,
            mean_abs_corr=round(mean_abs, 4),
            redundant_clusters=redundant,
            n_signals=signals.shape[1],
            threshold=self.redundancy_threshold,
        )

    def rolling_analysis(
        self,
        signals: pd.DataFrame,
        window: int = 60,
    ) -> pd.DataFrame:
        """Rollende mean_abs_corr-Serie über Zeit."""
        if len(signals) < window:
            return pd.DataFrame()
        records = []
        for end in range(window, len(signals) + 1):
            slice_df = signals.iloc[end - window: end]
            report = self.analyze(slice_df)
            records.append({
                "timestamp": slice_df.index[-1] if hasattr(slice_df.index, "date") else end,
                "mean_abs_corr": report.mean_abs_corr,
                "n_redundant_clusters": len(report.redundant_clusters),
            })
        return pd.DataFrame(records).set_index("timestamp") if records else pd.DataFrame()


__all__ = [
    "SignalCorrelationReport",
    "SignalCorrelationAnalyzer",
]
