# -*- coding: utf-8 -*-
"""Tests fuer signals/earnings_integration (Audit-Plan 4.5, 2026-08-17).

Das Modul war 4 Monate lang ein Phantom-Import: _tc_signals Step 3.3
importierte es bei enabled=true, der ImportError wurde still geskippt.
Diese Tests pinnen den Call-Site-Vertrag + die PIT-Disziplin.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.signals.earnings_integration import (
    apply_earnings_integration,
)

pytestmark = pytest.mark.fast

AS_OF = pd.Timestamp("2026-08-17", tz="UTC")


def _signals(**scores):
    syms = list(scores)
    return pd.DataFrame(
        {
            "timestamp": [AS_OF] * len(syms),
            "symbol": syms,
            "direction": ["LONG"] * len(syms),
            "score": [scores[s] for s in syms],
        }
    )


def test_pre_earnings_suppression_zeroes_score_pit():
    """Termin IM Fenster -> score 0; Termin AUSSERHALB + VERGANGEN -> unveraendert."""
    cal = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "JNJ"],
            "earnings_date": [
                AS_OF + pd.Timedelta(days=2),  # im 3-Tage-Fenster
                AS_OF + pd.Timedelta(days=10),  # ausserhalb
                AS_OF - pd.Timedelta(days=1),  # VERGANGEN: keine Suppression
            ],
        }
    )
    out, res = apply_earnings_integration(
        _signals(AAPL=0.8, MSFT=0.6, JNJ=0.4),
        earnings_calendar=cal,
        as_of=AS_OF,
        suppress_window=3,
    )
    row = out.set_index("symbol")
    assert row.loc["AAPL", "score"] == 0.0
    assert row.loc["MSFT", "score"] == pytest.approx(0.6)
    assert row.loc["JNJ", "score"] == pytest.approx(0.4)
    assert res.suppressed_symbols == ["AAPL"]


def test_pead_drift_direction_and_pit():
    """Nur OFFENGELEGTE Surprises im Fenster driften; Richtung = sign(surprise);
    ein Zukunfts-Event darf NIE wirken (PIT)."""
    events = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "XOM"],
            "disclosure_date": [
                AS_OF - pd.Timedelta(days=5),  # frisch berichtet
                AS_OF - pd.Timedelta(days=5),
                AS_OF + pd.Timedelta(days=2),  # ZUKUNFT -> tabu
            ],
            "eps_surprise_pct": [12.0, -8.0, 50.0],
        }
    )
    out, res = apply_earnings_integration(
        _signals(AAPL=0.2, MSFT=0.2, XOM=0.2),
        earnings_events=events,
        as_of=AS_OF,
        pead_window_days=60,
        pead_weight=0.15,
    )
    row = out.set_index("symbol")
    assert row.loc["AAPL", "score"] == pytest.approx(0.35)  # +0.15
    assert row.loc["MSFT", "score"] == pytest.approx(0.05)  # -0.15
    assert row.loc["XOM", "score"] == pytest.approx(0.2)  # Zukunft ignoriert
    assert res.pead_symbols == {"AAPL": 0.15, "MSFT": -0.15}


def test_pead_window_expiry_and_clipping():
    """Report aelter als pead_window_days -> kein Drift; Score wird geclippt."""
    events = pd.DataFrame(
        {
            "symbol": ["OLD", "HOT"],
            "disclosure_date": [
                AS_OF - pd.Timedelta(days=90),  # ausserhalb 60-Tage-Fenster
                AS_OF - pd.Timedelta(days=1),
            ],
            "eps_surprise_pct": [30.0, 10.0],
        }
    )
    out, _res = apply_earnings_integration(
        _signals(OLD=0.5, HOT=0.95),
        earnings_events=events,
        as_of=AS_OF,
        pead_window_days=60,
        pead_weight=0.15,
    )
    row = out.set_index("symbol")
    assert row.loc["OLD", "score"] == pytest.approx(0.5)
    assert row.loc["HOT", "score"] == pytest.approx(1.0)  # 0.95+0.15 -> clip 1.0


def test_suppression_beats_pead():
    """Symbol mit anstehendem Termin UND altem Report: Suppression gewinnt."""
    cal = pd.DataFrame(
        {"symbol": ["AAPL"], "earnings_date": [AS_OF + pd.Timedelta(days=1)]}
    )
    events = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "disclosure_date": [AS_OF - pd.Timedelta(days=30)],
            "eps_surprise_pct": [20.0],
        }
    )
    out, res = apply_earnings_integration(
        _signals(AAPL=0.7),
        earnings_calendar=cal,
        earnings_events=events,
        as_of=AS_OF,
    )
    assert out.set_index("symbol").loc["AAPL", "score"] == 0.0
    assert res.pead_symbols == {}


def test_empty_and_missing_inputs_are_noops():
    sig = _signals(AAPL=0.3)
    out, res = apply_earnings_integration(sig, as_of=AS_OF)
    pd.testing.assert_frame_equal(out, sig)
    assert res.n_signals_in == res.n_signals_out == 1
    out2, _res2 = apply_earnings_integration(
        pd.DataFrame(), as_of=AS_OF, earnings_calendar=pd.DataFrame()
    )
    assert out2.empty


def test_calendar_fallback_for_pead_uses_only_reported():
    """Kalender-Fallback: nur Zeilen mit eps_actual (berichtete) driften."""
    cal = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "earnings_date": [
                AS_OF - pd.Timedelta(days=10),
                AS_OF - pd.Timedelta(days=10),
            ],
            "eps_actual": [1.5, None],  # MSFT: Termin vorbei, aber nichts berichtet
            "surprise_pct": [9.0, 5.0],
        }
    )
    out, res = apply_earnings_integration(
        _signals(AAPL=0.1, MSFT=0.1),
        earnings_calendar=cal,
        as_of=AS_OF,
        pead_weight=0.15,
    )
    row = out.set_index("symbol")
    assert row.loc["AAPL", "score"] == pytest.approx(0.25)
    assert row.loc["MSFT", "score"] == pytest.approx(0.1)
    assert res.pead_symbols == {"AAPL": 0.15}
