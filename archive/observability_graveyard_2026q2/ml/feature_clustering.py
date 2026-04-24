"""Feature-Clustering für Multikollinearitäts-Reduktion + ClusteredMDA.

Standard-Problem: 200+ Features haben oft starke Korrelationen. Lineare
Modelle sind instabil, Tree-Modelle verteilen Importance zufällig zwischen
kollinearen Features.

Lösung (Lopez de Prado AIFML Ch.6):
1. Hierarchical Clustering auf 1 - |corr| als Distanz
2. Pro Cluster: repräsentatives Feature wählen (höchste individuelle IC)
3. ClusteredMDA: Mean-Decrease-Accuracy aggregiert pro Cluster

Vorteile:
- Reduziert Feature-Redundanz
- Stabilere Importance-Schätzungen
- Schnelleres Training

PIT-Invariante: Clustering auf historischen Korrelationen, keine Future-Leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureClusterResult:
    clusters: dict[int, list[str]]
    """Cluster-ID → Features."""

    representatives: dict[int, str]
    """Cluster-ID → Repräsentatives Feature."""

    cluster_ic: dict[int, float] = field(default_factory=dict)
    """Cluster-ID → mean IC des Repräsentativ-Features."""

    linkage_matrix: np.ndarray | None = None
    n_original_features: int = 0
    n_clusters: int = 0

    def get_selected_features(self) -> list[str]:
        """Gibt eine repräsentative Feature pro Cluster zurück (reduzierter Satz)."""
        return [self.representatives[cid] for cid in sorted(self.representatives)]


def cluster_features_by_correlation(
    X: pd.DataFrame,
    feature_cols: list[str] | None = None,
    n_clusters: int | None = None,
    distance_threshold: float | None = 0.3,
    linkage_method: str = "average",
) -> FeatureClusterResult:
    """Hierarchisches Clustering auf Korrelations-Distanz.

    Distanz = 1 - |correlation|. Feature-Paare mit |corr| > 0.7 landen in
    gleichem Cluster (bei distance_threshold=0.3).

    Args:
        X: Feature-DataFrame
        feature_cols: Zu clusterende Features (None = alle numerischen)
        n_clusters: Zielanzahl Cluster (None = durch distance_threshold bestimmt)
        distance_threshold: Abbruch-Distanz (0.3 = |corr| > 0.7 zusammen)
        linkage_method: 'single' / 'complete' / 'average' / 'ward'

    Returns:
        FeatureClusterResult
    """
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except ImportError:
        logger.warning("[FeatureCluster] scipy fehlt — Fallback auf 1-Feature-1-Cluster")
        cols = feature_cols or list(X.select_dtypes(include="number").columns)
        clusters = {i: [c] for i, c in enumerate(cols)}
        reps = {i: c for i, c in enumerate(cols)}
        return FeatureClusterResult(
            clusters=clusters,
            representatives=reps,
            n_original_features=len(cols),
            n_clusters=len(cols),
        )

    cols = feature_cols or list(X.select_dtypes(include="number").columns)
    if len(cols) < 2:
        return FeatureClusterResult(
            clusters={0: cols},
            representatives={0: cols[0]} if cols else {},
            n_original_features=len(cols),
            n_clusters=1 if cols else 0,
        )

    # Korrelationsmatrix → Distanzmatrix
    corr = X[cols].corr().fillna(0.0).values
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    # Symmetrisieren (Floating-Point-Drift)
    dist = (dist + dist.T) / 2.0
    condensed = squareform(dist, checks=False)

    Z = linkage(condensed, method=linkage_method)

    if n_clusters is not None:
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    elif distance_threshold is not None:
        labels = fcluster(Z, t=distance_threshold, criterion="distance")
    else:
        labels = fcluster(Z, t=4, criterion="maxclust")

    clusters: dict[int, list[str]] = {}
    for feat, label in zip(cols, labels):
        clusters.setdefault(int(label), []).append(feat)

    # Repräsentatives Feature pro Cluster: Mittlere Absolut-Korrelation innerhalb Cluster
    reps: dict[int, str] = {}
    for cid, feats in clusters.items():
        if len(feats) == 1:
            reps[cid] = feats[0]
            continue
        # Feature mit höchster mean |corr| zu anderen Cluster-Mitgliedern = Zentrum
        sub = X[feats].corr().abs().fillna(0.0)
        np.fill_diagonal(sub.values, 0.0)
        mean_corr = sub.mean(axis=1)
        reps[cid] = str(mean_corr.idxmax())

    logger.info(
        "[FeatureCluster] %d Features → %d Cluster (linkage=%s, threshold=%s)",
        len(cols), len(clusters), linkage_method, distance_threshold,
    )
    return FeatureClusterResult(
        clusters=clusters,
        representatives=reps,
        linkage_matrix=Z,
        n_original_features=len(cols),
        n_clusters=len(clusters),
    )


def select_features_by_cluster_ic(
    X: pd.DataFrame,
    y: pd.Series,
    cluster_result: FeatureClusterResult,
) -> FeatureClusterResult:
    """Innerhalb jedes Clusters das Feature mit höchstem |IC| zu y auswählen.

    Ergebnis ersetzt `representatives` durch IC-basierte Auswahl.
    """
    new_reps: dict[int, str] = {}
    new_ic: dict[int, float] = {}

    for cid, feats in cluster_result.clusters.items():
        best_feat = feats[0]
        best_ic = 0.0
        for f in feats:
            if X[f].std() < 1e-9:
                continue
            try:
                corr = X[f].corr(y)
                ic = float(corr) if not pd.isna(corr) else 0.0
            except Exception:
                ic = 0.0
            if abs(ic) > abs(best_ic):
                best_ic = ic
                best_feat = f
        new_reps[cid] = best_feat
        new_ic[cid] = best_ic

    cluster_result.representatives = new_reps
    cluster_result.cluster_ic = new_ic
    logger.info(
        "[FeatureCluster] IC-based selection: %d Cluster, mean_ic=%.4f",
        len(new_reps),
        float(np.mean(list(new_ic.values()))) if new_ic else 0.0,
    )
    return cluster_result


def clustered_mda(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    cluster_result: FeatureClusterResult,
    n_repeats: int = 3,
) -> dict[int, float]:
    """Clustered Mean Decrease Accuracy: Permutiert ganzen Cluster auf einmal.

    Unterschied zu Standard-PermutationImportance: Permutation aller Features
    eines Clusters zusammen → misst die tatsächliche Prädiktor-Kraft des
    Clusters, nicht redundant-aufgeteilte Anteile.

    Returns:
        {cluster_id: mean_accuracy_drop}
    """
    from sklearn.metrics import roc_auc_score

    all_feats = [f for feats in cluster_result.clusters.values() for f in feats]
    X_vals = X[all_feats].fillna(0.0).values
    y_vals = y.values

    feat_to_idx = {f: i for i, f in enumerate(all_feats)}

    # Baseline
    try:
        baseline_preds = model.predict(X_vals)  # type: ignore[attr-defined]
        if hasattr(model, "predict_proba"):
            baseline_preds = model.predict_proba(X_vals)[:, -1]  # type: ignore[attr-defined]
    except Exception as exc:
        logger.warning("[ClusteredMDA] baseline predict failed: %s", exc)
        return {}

    if len(np.unique(y_vals)) == 2:
        def score_fn(preds, actual):
            try:
                return float(roc_auc_score(actual, preds))
            except Exception:
                return 0.0
    else:
        def score_fn(preds, actual):
            if np.std(preds) < 1e-9:
                return 0.0
            corr = np.corrcoef(preds, actual)[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0

    baseline_score = score_fn(baseline_preds, y_vals)
    rng = np.random.default_rng(42)

    cluster_mda: dict[int, float] = {}
    for cid, feats in cluster_result.clusters.items():
        drops = []
        cluster_col_idx = [feat_to_idx[f] for f in feats]
        for _ in range(n_repeats):
            X_perm = X_vals.copy()
            for ci in cluster_col_idx:
                rng.shuffle(X_perm[:, ci])
            try:
                if hasattr(model, "predict_proba"):
                    preds_perm = model.predict_proba(X_perm)[:, -1]  # type: ignore[attr-defined]
                else:
                    preds_perm = model.predict(X_perm)  # type: ignore[attr-defined]
                drops.append(baseline_score - score_fn(preds_perm, y_vals))
            except Exception:
                drops.append(0.0)
        cluster_mda[cid] = float(np.mean(drops))

    return cluster_mda


__all__ = [
    "FeatureClusterResult",
    "cluster_features_by_correlation",
    "select_features_by_cluster_ic",
    "clustered_mda",
]
