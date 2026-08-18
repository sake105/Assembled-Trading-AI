# -*- coding: utf-8 -*-
"""Earnings-Integration: Pre-Earnings-Suppression + PEAD-Drift (Phase 9).

IMPLEMENTIERT 2026-08-17 (Audit-Plan 4.5, nach Deny-Lift): der Call-Site
``pipeline/_tc_signals.py`` (Step 3.3) importierte dieses Modul seit
2026-04-22 — es existierte nie. Der Guard wurde bei ``enabled=true`` vom
umgebenden except still geskippt, waehrend der
``earnings-calendar-refresh``-CI seinen einzigen Konsumenten fuetterte, den
es nicht gab (Phantom-Gewicht 0.15, Audit §3; Flag seit Plan 1.2 ehrlich
``false``). Der Vertrag hier ist EXAKT der bestehende Call-Site-Vertrag —
keine Signatur-Erfindung.

Semantik (aus dem Policy-Block ``signal_generation.earnings_guard``):

1. **Pre-Earnings-Suppression**: Symbole mit anstehendem Earnings-Termin in
   ``(as_of, as_of + suppress_window Tage]`` bekommen ``score = 0.0`` —
   Mean-Reversion-Signale sollen nicht in eine binaere Event-Lotterie
   hineinpositionieren. Das Signal bleibt als Zeile sichtbar (Diagnose),
   traegt aber kein Gewicht.
2. **PEAD-Drift**: Symbole mit BERICHTETEM Earnings innerhalb der letzten
   ``pead_window_days`` und bekannter Surprise bekommen
   ``score += pead_weight * sign(surprise)`` (Post-Earnings-Announcement-
   Drift: positive Ueberraschungen driften nach, negative nach unten).
   Ergebnis wird auf [-1, 1] geclippt.

PIT-Disziplin (CLAUDE.md):
- Suppression nutzt NUR Termine ``earnings_date > as_of`` (Kalendertermine
  sind vorab oeffentlich angekuendigt — PIT-sauber).
- PEAD nutzt NUR Events, deren ``disclosure_date <= as_of`` (bzw.
  ``earnings_date <= as_of`` beim Kalender mit berichtetem ``eps_actual``)
  — niemals zukuenftige oder noch nicht offengelegte Surprises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EarningsIntegrationResult:
    """Nachvollziehbarkeit fuer Log/QA: was wurde unterdrueckt/gedriftet."""

    suppressed_symbols: list[str] = field(default_factory=list)
    pead_symbols: dict[str, float] = field(default_factory=dict)
    n_signals_in: int = 0
    n_signals_out: int = 0


def _to_utc(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    return ts


def apply_earnings_integration(
    signals: pd.DataFrame,
    *,
    earnings_calendar: pd.DataFrame | None = None,
    earnings_events: pd.DataFrame | None = None,
    as_of: pd.Timestamp,
    suppress_window: int = 3,
    pead_window_days: int = 60,
    pead_weight: float = 0.15,
) -> tuple[pd.DataFrame, EarningsIntegrationResult]:
    """Apply pre-earnings suppression and PEAD drift to signal scores.

    Args:
        signals: DataFrame mit mindestens ``symbol`` und ``score``.
        earnings_calendar: Optional; Spalten ``symbol``, ``earnings_date``,
            optional ``eps_actual``/``surprise_pct``
            (scripts/fetch_earnings_calendar.py-Cache).
        earnings_events: Optional; Spalten ``symbol``, ``eps_surprise_pct``
            und ``disclosure_date`` (bevorzugt) oder ``event_date``
            (output/events_earnings.parquet-Schema).
        as_of: PIT-Zeitpunkt der Signalgenerierung.
        suppress_window: Kalendertage VOR dem Termin, in denen unterdrueckt wird.
        pead_window_days: Kalendertage NACH dem Report, in denen gedriftet wird.
        pead_weight: Score-Addition pro Symbol (Vorzeichen = Surprise-Richtung).

    Returns:
        (adjusted_signals, EarningsIntegrationResult) — Kopie, Input unveraendert.
    """
    result = EarningsIntegrationResult(n_signals_in=len(signals))
    if signals.empty or "symbol" not in signals.columns:
        result.n_signals_out = len(signals)
        return signals, result

    out = signals.copy()
    if "score" not in out.columns:
        result.n_signals_out = len(out)
        return out, result

    as_of = pd.Timestamp(as_of)
    as_of = (
        as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    )
    sym_u = out["symbol"].astype(str).str.upper()

    # --- 1. Pre-Earnings-Suppression (nur ZUKUENFTIGE Termine, PIT) -------
    if (
        earnings_calendar is not None
        and not earnings_calendar.empty
        and "symbol" in earnings_calendar.columns
        and "earnings_date" in earnings_calendar.columns
    ):
        cal = earnings_calendar[["symbol", "earnings_date"]].copy()
        cal["earnings_date"] = _to_utc(cal["earnings_date"])
        cal = cal.dropna(subset=["earnings_date"])
        # F-senior-13: der REPORTTAG SELBST muss unterdrueckt werden. Mit
        # "> as_of" positionierte der Guard bei einem After-Close-Report am
        # Tag as_of genau in die Event-Lotterie, die er verhindern soll.
        day0 = as_of.normalize()
        horizon = day0 + pd.Timedelta(days=int(suppress_window))
        upcoming = cal[
            (cal["earnings_date"] >= day0) & (cal["earnings_date"] <= horizon)
        ]
        suppress_set = set(upcoming["symbol"].astype(str).str.upper())
        if suppress_set:
            mask = sym_u.isin(suppress_set)
            if bool(mask.any()):
                out.loc[mask, "score"] = 0.0
                result.suppressed_symbols = sorted(set(sym_u[mask]))

    # --- 2. PEAD-Drift (nur BERICHTETE, offengelegte Surprises, PIT) ------
    pead_src: pd.DataFrame | None = None
    if (
        earnings_events is not None
        and not earnings_events.empty
        and "symbol" in earnings_events.columns
        and "eps_surprise_pct" in earnings_events.columns
    ):
        date_col = (
            "disclosure_date"
            if "disclosure_date" in earnings_events.columns
            else ("event_date" if "event_date" in earnings_events.columns else None)
        )
        if date_col is not None:
            pead_src = earnings_events[["symbol", date_col, "eps_surprise_pct"]].rename(
                columns={date_col: "reported_at", "eps_surprise_pct": "surprise"}
            )
    if (
        pead_src is None
        and earnings_calendar is not None
        and not earnings_calendar.empty
        and "surprise_pct" in earnings_calendar.columns
        and "earnings_date" in earnings_calendar.columns
    ):
        _cal = earnings_calendar
        if "eps_actual" in _cal.columns:
            _cal = _cal[_cal["eps_actual"].notna()]  # nur berichtete
        pead_src = _cal[["symbol", "earnings_date", "surprise_pct"]].rename(
            columns={"earnings_date": "reported_at", "surprise_pct": "surprise"}
        )

    if pead_src is not None and not pead_src.empty:
        pead_src = pead_src.copy()
        pead_src["reported_at"] = _to_utc(pead_src["reported_at"])
        pead_src["surprise"] = pd.to_numeric(pead_src["surprise"], errors="coerce")
        pead_src = pead_src.dropna(subset=["reported_at", "surprise"])
        window_start = as_of - pd.Timedelta(days=int(pead_window_days))
        recent = pead_src[
            (pead_src["reported_at"] <= as_of)  # PIT: nur offengelegte
            & (pead_src["reported_at"] >= window_start)
            & (pead_src["surprise"] != 0.0)
        ]
        if not recent.empty:
            # Juengster Report je Symbol gewinnt.
            recent = (
                recent.sort_values("reported_at")
                .groupby(recent["symbol"].astype(str).str.upper())
                .last()
            )
            drift = {
                sym: float(pead_weight) * float(np.sign(row["surprise"]))
                for sym, row in recent.iterrows()
                # Suppression hat Vorrang: kein Drift auf unterdrueckte Symbole.
                if sym not in set(result.suppressed_symbols)
            }
            if drift:
                add = sym_u.map(drift).fillna(0.0)
                out["score"] = (out["score"] + add).clip(-1.0, 1.0)
                result.pead_symbols = drift

    result.n_signals_out = len(out)
    return out, result
