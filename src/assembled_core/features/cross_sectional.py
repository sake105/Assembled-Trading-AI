"""Cross-Sectional Feature Normalisierung.

Standard quantitativer Practice: Features werden täglich cross-sectional
(über alle Symbole) rank- oder z-score-normalisiert.

Warum:
- Absolute Feature-Werte sind regime-abhängig (z.B. P/E während Bubbles)
- Ranks sind monoton-invariant und stabiler
- ML-Modelle lernen dadurch relative Signale, nicht absolute Niveaus

PIT-Invariante: Ranking nur über Symbole AM GLEICHEN TAG
(kein Time-Leakage, da tagesweise groupby).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def rank_cross_sectional(
    panel_df: pd.DataFrame,
    feature_cols: list[str],
    timestamp_col: str = "timestamp",
    normalize_to: str = "percentile",
    suffix: str = "_xrank",
) -> pd.DataFrame:
    """Cross-sectional Rank pro Timestamp.

    Args:
        panel_df: Panel mit mind. timestamp_col und feature_cols
        feature_cols: Feature-Spalten für Ranking
        normalize_to: "percentile" (0..1), "symmetric" (-1..+1), "integer" (1..N)
        suffix: Suffix für neue Spalten

    Returns:
        panel_df mit zusätzlichen Spalten `{col}{suffix}`.
    """
    result = panel_df.copy()
    for col in feature_cols:
        if col not in result.columns:
            logger.debug("[XRank] Spalte %s fehlt — übersprungen", col)
            continue
        new_col = f"{col}{suffix}"
        if normalize_to == "percentile":
            result[new_col] = result.groupby(timestamp_col)[col].rank(pct=True)
        elif normalize_to == "symmetric":
            pct = result.groupby(timestamp_col)[col].rank(pct=True)
            result[new_col] = 2.0 * pct - 1.0
        elif normalize_to == "integer":
            result[new_col] = result.groupby(timestamp_col)[col].rank(method="min")
        else:
            raise ValueError(f"Unbekanntes normalize_to: {normalize_to}")
    return result


def zscore_cross_sectional(
    panel_df: pd.DataFrame,
    feature_cols: list[str],
    timestamp_col: str = "timestamp",
    winsorize_std: float | None = 3.0,
    suffix: str = "_xz",
) -> pd.DataFrame:
    """Cross-sectional Z-Score pro Timestamp mit optional Winsorization.

    Args:
        panel_df: Panel
        feature_cols: Feature-Spalten
        winsorize_std: Ausreißer beschneiden auf ±N·σ (None = keine Winsorization)
        suffix: Suffix für neue Spalten

    Returns:
        panel_df mit `{col}{suffix}`-Spalten.
    """
    result = panel_df.copy()
    for col in feature_cols:
        if col not in result.columns:
            continue
        new_col = f"{col}{suffix}"

        def _zscore(s: pd.Series) -> pd.Series:
            mu = s.mean()
            sigma = s.std()
            if pd.isna(sigma) or sigma < 1e-9:
                return pd.Series(np.zeros(len(s)), index=s.index)
            z = (s - mu) / sigma
            if winsorize_std is not None:
                z = z.clip(lower=-winsorize_std, upper=winsorize_std)
            return z

        result[new_col] = result.groupby(timestamp_col, group_keys=False)[
            col
        ].transform(_zscore)
    return result


def neutralize_cross_sectional(
    panel_df: pd.DataFrame,
    target_col: str,
    neutralize_by: list[str],
    timestamp_col: str = "timestamp",
    suffix: str = "_neu",
) -> pd.DataFrame:
    """Regressiert target_col cross-sectional gegen neutralize_by und gibt Residuen zurück.

    Typischer Use-Case: Faktor gegen Sektoren / Marketcap / Beta neutralisieren.
    Residuen = pure Alpha ohne Sektor/Size-Tilt.

    Args:
        panel_df: Panel
        target_col: Faktor zu neutralisieren
        neutralize_by: Kontrollvariablen (z.B. ["sector_id", "log_marketcap"])
        suffix: Neue Spalte = f"{target_col}{suffix}"

    Returns:
        panel_df mit `{target_col}{suffix}`-Spalte (Residuen).
    """
    from sklearn.linear_model import LinearRegression

    result = panel_df.copy()
    new_col = f"{target_col}{suffix}"

    if target_col not in result.columns:
        return result
    missing = [c for c in neutralize_by if c not in result.columns]
    if missing:
        logger.warning("[XNeutralize] Fehlende Spalten: %s — übersprungen", missing)
        return result

    residuals = np.full(len(result), np.nan)

    for ts, group in result.groupby(timestamp_col):
        mask = group[[target_col, *neutralize_by]].notna().all(axis=1)
        if mask.sum() < len(neutralize_by) + 2:
            continue
        X = group.loc[mask, neutralize_by].values
        y = group.loc[mask, target_col].values

        try:
            reg = LinearRegression()
            reg.fit(X, y)
            pred = reg.predict(X)
            res = y - pred
            idx_positions = np.where(result.index.isin(group.loc[mask].index))[0]
            residuals[idx_positions] = res
        except Exception as exc:
            logger.debug("[XNeutralize] %s failed at %s: %s", target_col, ts, exc)

    result[new_col] = residuals
    return result


__all__ = [
    "rank_cross_sectional",
    "zscore_cross_sectional",
    "neutralize_cross_sectional",
]
