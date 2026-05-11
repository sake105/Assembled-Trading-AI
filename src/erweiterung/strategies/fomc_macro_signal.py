"""FOMC-Macro-Signal — verbindet FOMC-Tone-Score mit Equity-Allokation.

Konzept
-------
Sieben Tage nach FOMC-Meeting bis zum nächsten Meeting:
- Wenn Δhd_score positiv (= mehr hawkish als letztes Meeting) → reduziere Equity-Exposure
- Wenn Δhd_score negativ (= dovisher) → erhöhe oder behalte Exposure

Empirisch (Lucca/Trebbi 2009, Hansen et al. 2018): Hawkish-Surprises
führen zu kurzfristig negativen Equity-Returns (post-FOMC drift).

API
---
- ``build_fomc_signal_series``: aus statements + dates → daily Allokations-Override
- ``apply_fomc_allocation_override``: wende auf bestehende Portfolio-Returns an

Daten-Limitierung
-----------------
Funktioniert nur mit echten FOMC-Statement-Texten. Hier wird die Pipeline
implementiert; Beschaffung der historischen Texte (z. B. aus FED-Archives
oder federalreserve.gov) ist Caller-Sache.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from erweiterung.transcripts.fomc_tone import score_fomc_statements


@dataclass
class FOMCMacroConfig:
    hawkish_delta_threshold: float = 0.10
    dovish_delta_threshold: float = -0.10
    exposure_hawkish: float = 0.50
    exposure_dovish: float = 1.20  # leicht overweight bei dove-surprise
    exposure_neutral: float = 1.00
    decay_days: int = 14
    """Effekt dauert decay_days nach Statement-Datum; danach Rückkehr zu 1.0."""


def build_fomc_signal_series(
    statements: list[str],
    dates: list[pd.Timestamp],
    target_index: pd.DatetimeIndex,
    config: FOMCMacroConfig | None = None,
) -> pd.Series:
    """Erzeuge tägliche FOMC-Allokations-Override-Series.

    Args:
        statements: Liste FOMC-Statement-Texte (chronologisch).
        dates: Korrespondierende FOMC-Meeting-Daten.
        target_index: DatetimeIndex der Backtest-Periode.
        config: FOMCMacroConfig.

    Returns:
        pd.Series mit Werten in {exposure_hawkish, exposure_neutral, exposure_dovish}
        je Tag, aligned auf target_index.
    """
    cfg = config or FOMCMacroConfig()
    if not statements or not dates or len(statements) != len(dates):
        return pd.Series(cfg.exposure_neutral, index=target_index)

    scored = score_fomc_statements(statements, dates=dates)
    scored = scored.sort_values("date").reset_index(drop=True)
    scored["delta_hd"] = scored["hd_score"].diff()

    # Initialisiere mit neutral
    out = pd.Series(cfg.exposure_neutral, index=target_index)

    for i, row in scored.iterrows():
        raw_date = row["date"]
        ts = pd.Timestamp(raw_date)
        meeting_date = (
            ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        )
        delta = row["delta_hd"]
        if pd.isna(delta):
            continue

        if delta > cfg.hawkish_delta_threshold:
            target = cfg.exposure_hawkish
        elif delta < cfg.dovish_delta_threshold:
            target = cfg.exposure_dovish
        else:
            target = cfg.exposure_neutral

        # Decay-Window: meeting_date+1 bis meeting_date + decay_days
        # WICHTIG: > meeting_date (nicht >=) verhindert Lookahead.
        # FOMC-Statements werden ~14:00 ET veröffentlicht, US-Close 16:00 ET.
        # Daher kann der Statement-Inhalt erst ab dem NÄCHSTEN Trading-Day
        # für Allokations-Entscheidungen genutzt werden (close-based returns).
        end_date = meeting_date + pd.Timedelta(days=cfg.decay_days)
        mask = (out.index > meeting_date) & (out.index <= end_date)
        out[mask] = target

    return out


def apply_fomc_allocation_override(
    portfolio_returns: pd.Series,
    fomc_signal: pd.Series,
) -> pd.Series:
    """Wende FOMC-Allokations-Override auf Portfolio-Returns an.

    Args:
        portfolio_returns: Daily returns.
        fomc_signal: Allokations-Override (Series mit gleichem Index).

    Returns:
        Modifizierte Returns.
    """
    aligned = pd.concat({"r": portfolio_returns, "e": fomc_signal}, axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    return aligned["r"] * aligned["e"]


__all__ = [
    "FOMCMacroConfig",
    "build_fomc_signal_series",
    "apply_fomc_allocation_override",
]
