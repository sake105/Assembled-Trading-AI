"""Tests für P12d — die Survivorship-Schranke.

Das Modul hat zwei Schutzmechanismen — aber nur einer trägt, und welcher das
ist, war mir zunächst falsch klar:

Der **Glitch-Filter** ist der Mechanismus, der wirklich trägt: Vendor-Fehler
wie MEL 7,73 → 141.630 liegen zwischen zwei gültigen Kursen, überleben also
jede Lücken-Behandlung, und ließen das PIT-Universum im ersten Lauf auf 10^81
eskalieren. Mutationsgeprüft: Schwelle aushebeln → zwei Tests fallen.

Die **Lückenmaske** ist dagegen redundant — ``pct_change(fill_method=None)``
liefert über NaN-Strecken hinweg ohnehin NaN. Ein früherer Docstring schrieb
ihr die Rettung zu; der Mutationstest widerlegte das. Der Test unten prüft
deshalb das VERHALTEN (keine Scheinrendite nach einer Lücke) und behauptet
nicht, welche Codezeile es bewirkt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.mandat2.p12d_survivorship_schranke import (
    GLITCH_SCHWELLE,
    buy_and_hold,
    glitch_verdaechtig,
    kennzahlen,
)

pytestmark = pytest.mark.fast


def _tage(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2010-01-04", periods=n, tz="UTC")


class TestGlitchErkennung:
    def test_erkennt_unmoeglichen_sprung(self):
        idx = _tage(20)
        s = pd.Series(50.0, index=idx)
        s.iloc[10:] = 5000.0  # +9.900 %
        g = glitch_verdaechtig(pd.DataFrame({"X": s}), {"X"})
        assert "X" in g
        assert g["X"]["von"] == pytest.approx(50.0)
        assert g["X"]["auf"] == pytest.approx(5000.0)

    def test_ignoriert_starke_aber_moegliche_bewegung(self):
        """+80 % an einem Tag ist selten, aber real — kein Glitch."""
        idx = _tage(20)
        s = pd.Series(50.0, index=idx)
        s.iloc[10:] = 90.0
        assert glitch_verdaechtig(pd.DataFrame({"X": s}), {"X"}) == {}

    def test_mikropreis_sprung_bleibt_der_alten_regel_ueberlassen(self):
        """Vortagskurs < 1 USD: das faengt bereits campaign_data (E-074-Klasse).

        Wuerde diese Regel hier ebenfalls greifen, wuerde sie Delisting-Verlaeufe
        wegwerfen, die ECHT sind — genau die Namen, um die es in P12d geht.
        """
        idx = _tage(20)
        s = pd.Series(0.5, index=idx)
        s.iloc[10:] = 50.0  # +9.900 %, aber Vortagskurs unter 1 USD
        assert glitch_verdaechtig(pd.DataFrame({"X": s}), {"X"}) == {}

    def test_schwelle_ist_bei_200_prozent_verankert(self):
        """ABSOLUTE Werte, nicht aus GLITCH_SCHWELLE abgeleitet.

        Die erste Fassung baute ihre Fixtures aus der Konstanten selbst und war
        damit gegen jede Aenderung des Werts immun: die Schwelle auf 500
        zu setzen (Filter faktisch aus) liess sie unberuehrt (F-test-6).
        """
        idx = _tage(10)
        knapp_drunter = pd.Series(10.0, index=idx)
        knapp_drunter.iloc[5:] = 25.0  # +150 %
        assert glitch_verdaechtig(pd.DataFrame({"X": knapp_drunter}), {"X"}) == {}
        drueber = pd.Series(10.0, index=idx)
        drueber.iloc[5:] = 35.0  # +250 %
        assert "X" in glitch_verdaechtig(pd.DataFrame({"X": drueber}), {"X"})
        assert GLITCH_SCHWELLE == pytest.approx(2.0), (
            "Schwelle geaendert — die Grenzfaelle oben (+150 %/+250 %) und die "
            "Befundzahlen muessen mitgezogen werden."
        )


class TestBuyAndHold:
    def test_ohne_delisting_sind_beide_varianten_identisch(self):
        """DER Regressionsanker fuer F-test-1 (BLOCKER).

        Unterscheiden sich ``halten`` und ``umschichten`` ohne ein einziges
        Delisting, ist ``umschichten`` kein Buy-and-Hold mehr, sondern ein
        rebalanciertes Portfolio. Genau das war der Fall: die alte
        Implementierung wich hier um 9 % ab und erzeugte damit eine
        Entwarnung, die die Daten nicht hergaben.

        Zwei GEGENLAEUFIGE Titel sind Pflicht — mit flachen oder gleichlaufenden
        Kursen faellt der Unterschied nicht auf, und exakt daran war die alte
        Testabdeckung blind.
        """
        idx = _tage(30)
        px = pd.DataFrame(
            {
                "A": 100.0 * np.cumprod(np.full(30, 1.03)),
                "B": 100.0 * np.cumprod(np.full(30, 0.97)),
            },
            index=idx,
        )
        k_h, _ = buy_and_hold(px, {"A", "B"}, umschichten=False)
        k_u, _ = buy_and_hold(px, {"A", "B"}, umschichten=True)
        assert k_h.iloc[-1] == pytest.approx(k_u.iloc[-1], rel=1e-12)
        # ... und beide muessen das analytische Buy-and-Hold treffen.
        erwartet = 0.5 * (1.03**29) + 0.5 * (0.97**29)
        assert k_h.iloc[-1] == pytest.approx(erwartet, rel=1e-12)

    def test_erloes_wird_pro_rata_verteilt(self):
        """Gegen die Handrechnung, nicht gegen ein Erwartungsband."""
        idx = _tage(4)
        # B stirbt nach Tag 2 (letzter Kurs 100). A laeuft weiter.
        a = pd.Series([100.0, 100.0, 200.0, 400.0], index=idx)
        b = pd.Series([100.0, 100.0, np.nan, np.nan], index=idx)
        px = pd.DataFrame({"A": a, "B": b})
        k_u, d = buy_and_hold(px, {"A", "B"}, umschichten=True)
        k_h, _ = buy_and_hold(px, {"A", "B"}, umschichten=False)
        assert d["n_delistings_im_fenster"] == 1
        # Tag 1 (Index 1): A=0.5, B=0.5 -> B stirbt, 0.5 wandert auf A.
        # Danach verdoppelt A zweimal: 1.0 -> 2.0 -> 4.0
        assert k_u.iloc[-1] == pytest.approx(4.0, rel=1e-12)
        # Totes Geld: A 0.5 -> 2.0, plus 0.5 liegengeblieben = 2.5
        assert k_h.iloc[-1] == pytest.approx(2.5, rel=1e-12)
        assert k_u.iloc[-1] > k_h.iloc[-1]

    def test_diagnose_meldet_wer_nicht_mitspielt(self):
        """Stille Filterung waere hier besonders teuer (F-test-3)."""
        idx = _tage(5)
        px = pd.DataFrame(
            {"A": [10.0] * 5, "SPAET": [np.nan, np.nan, 1.0, 1.0, 1.0]}, index=idx
        )
        _, d = buy_and_hold(px, {"A", "SPAET", "GIBTESNICHT"}, umschichten=False)
        assert d["n_dabei"] == 1
        assert d["n_ohne_preisspalte"] == 1
        assert d["n_ohne_startkurs"] == 1
        assert "SPAET" in d["ohne_startkurs"]

    def test_halten_gegen_analytisch_bekannte_kurve(self):
        idx = _tage(10)
        px = pd.DataFrame(
            {"A": np.linspace(10.0, 20.0, 10), "B": np.linspace(5.0, 5.0, 10)},
            index=idx,
        )
        k, _ = buy_and_hold(px, {"A", "B"}, umschichten=False)
        assert k.iloc[-1] == pytest.approx((2.0 + 1.0) / 2)

    def test_delisteter_name_friert_ein_statt_zu_verschwinden(self):
        """Der Kern der Variante ‚Erlös gehalten': kein stiller Bonus."""
        idx = _tage(10)
        a = pd.Series(np.linspace(10.0, 20.0, 10), index=idx)
        b = pd.Series(10.0, index=idx)
        b.iloc[5:] = np.nan  # ab hier delistet, letzter Kurs 10.0
        k, _ = buy_and_hold(
            pd.DataFrame({"A": a, "B": b}), {"A", "B"}, umschichten=False
        )
        # B bleibt bei 1.0 stehen, A verdoppelt -> Mittel 1.5
        assert k.iloc[-1] == pytest.approx(1.5)

    def test_umschichten_rechnet_nicht_ueber_datenluecken(self):
        """Verhalten, nicht Mechanismus: nach einer Luecke keine Scheinrendite.

        Bewirkt wird das von ``pct_change(fill_method=None)``, nicht von der
        expliziten Maske — deshalb faengt dieser Test eine Abschwaechung der
        Maske auch NICHT. Er sichert die Eigenschaft, nicht ihre Herkunft.
        """
        idx = _tage(12)
        a = pd.Series(100.0, index=idx)
        b = pd.Series(100.0, index=idx)
        b.iloc[3:8] = np.nan
        b.iloc[8:] = 100_000.0  # Sprung NACH der Luecke: darf nicht zaehlen
        k, _ = buy_and_hold(
            pd.DataFrame({"A": a, "B": b}), {"A", "B"}, umschichten=True
        )
        assert k.iloc[-1] == pytest.approx(1.0, abs=1e-9)

    def test_umschichten_zaehlt_echte_renditen(self):
        idx = _tage(3)
        px = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=idx)
        k, _ = buy_and_hold(px, {"A"}, umschichten=True)
        assert k.iloc[-1] == pytest.approx(1.21)

    def test_leeres_universum_kracht(self):
        px = pd.DataFrame({"A": [1.0, 2.0]}, index=_tage(2))
        with pytest.raises(ValueError):
            buy_and_hold(px, {"GIBTESNICHT"}, umschichten=False)


class TestKennzahlen:
    def test_cagr_und_maxdd(self):
        idx = _tage(5)
        k = pd.Series([1.0, 1.0, 0.5, 0.75, 2.0], index=idx)
        a = kennzahlen(k, jahre=2.0)
        assert a["endwert"] == pytest.approx(2.0)
        assert a["cagr"] == pytest.approx(2.0**0.5 - 1.0)
        assert a["maxdd"] == pytest.approx(-0.5)
