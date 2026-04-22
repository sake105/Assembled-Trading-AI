"""Rolling Feature-Importance Tracking + Auto-Pruning.

Standard-Problem: mit 200+ Features hat man viele, die keinen echten Signalwert
mehr tragen (decayed features) oder per Chance selektiert wurden.

Lösung — Rolling Permutation Importance + MDA (Mean Decrease Accuracy):
1. Monatlich / wöchentlich: Berechne Feature-Importance via Permutation
2. Tracke Zeitreihe der Importance-Werte
3. Features mit konstant niedrigem Wert + abnehmender Tendenz → prune

Komplementär zu SHAP:
- SHAP misst Attribution einzelner Predictions
- Permutation misst globale Vorhersage-Wichtigkeit
- Beide zusammen → robuste Feature-Auswahl

PIT-Invariante: Importance wird auf historischem Validation-Set gemessen,
nicht auf Future-Daten.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ImportanceSnapshot:
    """Eine Feature-Importance-Messung zu einem Zeitpunkt."""

    as_of: str
    """ISO-Datum"""
    importances: dict[str, float]
    """Feature → mean Permutation-Importance"""
    baseline_score: float
    n_features: int
    n_samples: int


@dataclass
class PruningDecision:
    feature: str
    action: str
    """'prune' | 'keep' | 'review'"""
    reason: str
    last_importance: float
    trend: str
    """'falling' | 'stable' | 'rising'"""


class FeatureImportanceTracker:
    """Persistenter Tracker für Rolling Feature-Importance.

    State-Datei: output/ml/feature_importance_history.json
    Jeder Snapshot wird angehängt; History behält letzte N Snapshots.
    """

    def __init__(
        self,
        state_path: Path | None = None,
        history_window: int = 12,
    ) -> None:
        """Args:
            state_path: Pfad für Persistenz (default: output/ml/feature_importance_history.json)
            history_window: Anzahl Snapshots für Trend-Analyse
        """
        self.state_path = state_path or Path("output/ml/feature_importance_history.json")
        self.history_window = history_window
        self._snapshots: list[ImportanceSnapshot] = self._load()

    def _load(self) -> list[ImportanceSnapshot]:
        if not self.state_path.exists():
            return []
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return [ImportanceSnapshot(**s) for s in data.get("snapshots", [])]
        except Exception as exc:
            logger.warning("[FI-Tracker] Konnte history nicht laden: %s", exc)
            return []

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep only last history_window snapshots
        trimmed = self._snapshots[-self.history_window:]
        data = {
            "snapshots": [
                {
                    "as_of": s.as_of,
                    "importances": s.importances,
                    "baseline_score": s.baseline_score,
                    "n_features": s.n_features,
                    "n_samples": s.n_samples,
                }
                for s in trimmed
            ]
        }
        self.state_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def compute_permutation_importance(
        self,
        model: object,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: list[str],
        n_repeats: int = 5,
        scoring: callable | None = None,
    ) -> tuple[dict[str, float], float]:
        """Permutation-Importance via In-Sample-Resampling.

        Args:
            model: gefittetes sklearn-kompatibles Modell
            X: Feature-DataFrame
            y: Target
            feature_cols: Zu prüfende Features
            n_repeats: Permutations pro Feature
            scoring: callable(model, X, y) → float. Default: IC.

        Returns:
            ({feature: mean_importance_drop}, baseline_score)
        """
        if scoring is None:
            def scoring(mdl, X_arr, y_arr):
                preds = mdl.predict(X_arr)
                if np.std(preds) < 1e-9:
                    return 0.0
                ic = np.corrcoef(preds, y_arr)[0, 1]
                return float(ic) if not np.isnan(ic) else 0.0

        X_vals = X[feature_cols].fillna(0.0).values
        y_vals = y.values
        baseline = scoring(model, X_vals, y_vals)

        rng = np.random.default_rng(42)
        importances: dict[str, float] = {}

        for fi, feat in enumerate(feature_cols):
            drops = []
            for _ in range(n_repeats):
                X_perm = X_vals.copy()
                rng.shuffle(X_perm[:, fi])
                score_perm = scoring(model, X_perm, y_vals)
                drops.append(baseline - score_perm)
            importances[feat] = float(np.mean(drops))

        return importances, float(baseline)

    def record_snapshot(
        self,
        model: object,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: list[str],
        as_of: str | pd.Timestamp | None = None,
        n_repeats: int = 5,
    ) -> ImportanceSnapshot:
        """Berechnet Importance und speichert als Snapshot in History."""
        if as_of is None:
            as_of = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
        elif isinstance(as_of, pd.Timestamp):
            as_of = as_of.strftime("%Y-%m-%d")

        imp, baseline = self.compute_permutation_importance(
            model=model, X=X, y=y, feature_cols=feature_cols, n_repeats=n_repeats,
        )
        snap = ImportanceSnapshot(
            as_of=as_of,
            importances=imp,
            baseline_score=baseline,
            n_features=len(feature_cols),
            n_samples=len(X),
        )
        self._snapshots.append(snap)
        self._save()
        logger.info(
            "[FI-Tracker] Snapshot %s: n_feat=%d baseline=%.4f",
            as_of, len(feature_cols), baseline,
        )
        return snap

    def get_trend(
        self,
        feature: str,
        min_snapshots: int = 3,
    ) -> str:
        """Trend-Analyse: 'falling' / 'stable' / 'rising' / 'unknown'."""
        recent = [s.importances.get(feature, 0.0) for s in self._snapshots[-min_snapshots:]]
        if len(recent) < min_snapshots:
            return "unknown"
        # Einfacher linearer Trend via Differenzen
        diffs = np.diff(recent)
        mean_diff = float(np.mean(diffs))
        threshold = 0.005
        if mean_diff < -threshold:
            return "falling"
        if mean_diff > threshold:
            return "rising"
        return "stable"

    def prune_recommendations(
        self,
        importance_threshold: float = 0.001,
        min_snapshots: int = 3,
    ) -> list[PruningDecision]:
        """Generiert Prune-Empfehlungen basierend auf History.

        Prune wenn:
        - letzte N Snapshots → mean importance < threshold
        - Trend ist 'falling'

        Args:
            importance_threshold: Minimum-Importance zum Behalten
            min_snapshots: Benötigte History-Tiefe

        Returns:
            Liste von PruningDecision pro Feature.
        """
        if len(self._snapshots) < min_snapshots:
            return []

        all_features = set()
        for s in self._snapshots:
            all_features.update(s.importances.keys())

        decisions: list[PruningDecision] = []
        recent_snaps = self._snapshots[-min_snapshots:]

        for feat in sorted(all_features):
            recent_imps = [s.importances.get(feat, 0.0) for s in recent_snaps]
            mean_imp = float(np.mean(recent_imps))
            last_imp = float(recent_imps[-1])
            trend = self.get_trend(feat, min_snapshots=min_snapshots)

            if mean_imp < importance_threshold and trend != "rising":
                action = "prune"
                reason = f"mean_imp={mean_imp:.4f} < threshold, trend={trend}"
            elif mean_imp < importance_threshold and trend == "rising":
                action = "review"
                reason = "low importance but rising trend"
            else:
                action = "keep"
                reason = f"mean_imp={mean_imp:.4f}, trend={trend}"

            decisions.append(PruningDecision(
                feature=feat,
                action=action,
                reason=reason,
                last_importance=last_imp,
                trend=trend,
            ))

        n_prune = sum(1 for d in decisions if d.action == "prune")
        logger.info(
            "[FI-Tracker] Prune-Recommendations: %d prune / %d review / %d keep",
            n_prune,
            sum(1 for d in decisions if d.action == "review"),
            sum(1 for d in decisions if d.action == "keep"),
        )
        return decisions

    def history_length(self) -> int:
        return len(self._snapshots)


__all__ = [
    "ImportanceSnapshot",
    "PruningDecision",
    "FeatureImportanceTracker",
]
