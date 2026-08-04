"""Tests für P12d — die Survivorship-Schranke.

Das Modul hat zwei Schutzmechanismen — aber nur einer trägt, und welcher das
ist, war mir zunächst falsch klar:

Der **Glitch-Filter** ist der Mechanismus, der wirklich trägt: Vendor-Fehler
wie MEL 7,73 → 141.630 liegen zwischen zwei gültigen Kursen, überleben also
jede Lücken-Behandlung, und ließen das PIT-Universum im ersten Lauf
eskalieren (Größenordnung über 10^70; die exakte Potenz ist nicht belegt und
wird bewusst nur als Schranke genannt). Mutationsgeprüft: Schwelle aushebeln → zwei Tests fallen.

Die **Lückenmaske** war in der ersten, renditebasierten Implementierung
redundant — dort erledigte ``pct_change(fill_method=None)`` die Arbeit. Seit
dem Umbau auf wertbasierte Simulation gibt es kein ``pct_change`` mehr in
``buy_and_hold``: die Maske ``lebt & ~isnan(p0) & ~isnan(p1)`` IST jetzt der
Mechanismus, und ``test_umschichten_rechnet_nicht_ueber_datenluecken`` fängt
ihre Entfernung (mutationsgeprüft: Maske auf ``lebt`` reduzieren → Test fällt).

Der Docstring hatte den Rewrite überlebt und behauptete anschließend das
Gegenteil des gemessenen Verhaltens — er redete die eigene Abdeckung klein
(E-098).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.mandat2.p12d_survivorship_schranke import (
    GLITCH_SCHWELLE,
    buy_and_hold,
    glitch_verdaechtig,
    korruptions_spannen,
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

    def test_erloes_wird_PRO_RATA_verteilt_nicht_gleich(self):
        """Muss Pro-rata von Gleichverteilung UNTERSCHEIDEN koennen.

        Die erste Fassung hatte nur EINEN Ueberlebenden — der bekommt unter
        jeder Verteilungsregel denselben Betrag, und eine Mutation zu
        Gleichverteilung ueberlebte die ganze Suite (Stage-2-Finding
        F-senior-3). Ein Test unterscheidet nur, was sein Fixture unterscheidet.

        Hier: ZWEI Ueberlebende mit ungleichen Positionswerten, und eine
        Bewegung DANACH — denn im Moment der Verteilung sind beide Regeln noch
        nicht auseinanderzuhalten.
        """
        idx = _tage(5)
        # Start je 1/3. A verdreifacht sich bis zum Delisting-Tag, B bleibt.
        # C stirbt an Tag 1 mit Wert 1/3.
        a = pd.Series([100.0, 300.0, 300.0, 600.0, 600.0], index=idx)
        b = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=idx)
        c = pd.Series([100.0, 100.0, np.nan, np.nan, np.nan], index=idx)
        px = pd.DataFrame({"A": a, "B": b, "C": c})
        k, d = buy_and_hold(px, {"A", "B", "C"}, umschichten=True)
        assert d["n_delistings_im_fenster"] == 1
        # An Tag 1: A=1.0, B=1/3, C=1/3. C stirbt, 1/3 pro rata auf A:B = 3:1
        #   -> A = 1.0 + 1/3*0.75 = 1.25 ; B = 1/3 + 1/3*0.25 = 0.41667
        # Danach verdoppelt A -> 2.5 ; B unveraendert -> Summe 2.91667
        assert k.iloc[-1] == pytest.approx(1.25 * 2 + 1 / 3 + 1 / 12, rel=1e-12)
        # Gegenprobe: bei GLEICHverteilung waere A = 1.0 + 1/6 = 1.16667,
        # verdoppelt 2.33333, plus B 0.5 -> 2.83333. Der Test trennt beide.
        assert k.iloc[-1] != pytest.approx(2.83333, rel=1e-4)

    def test_totes_geld_bleibt_liegen(self):
        """Die Gegenvariante: der Erloes verzinst sich nicht."""
        idx = _tage(4)
        a = pd.Series([100.0, 100.0, 200.0, 400.0], index=idx)
        b = pd.Series([100.0, 100.0, np.nan, np.nan], index=idx)
        px = pd.DataFrame({"A": a, "B": b})
        k_h, _ = buy_and_hold(px, {"A", "B"}, umschichten=False)
        k_u, _ = buy_and_hold(px, {"A", "B"}, umschichten=True)
        # A 0.5 -> 2.0, B als totes Geld konstant 0.5 -> 2.5
        assert k_h.iloc[-1] == pytest.approx(2.5, rel=1e-12)
        # Umgeschichtet: die vollen 1.0 laufen mit A -> 4.0
        assert k_u.iloc[-1] == pytest.approx(4.0, rel=1e-12)

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
        """Nach einer Luecke darf keine Scheinrendite entstehen.

        Seit dem Umbau auf wertbasierte Simulation traegt die NaN-Maske in
        ``buy_and_hold`` diesen Schutz — mutationsgeprueft: Maske auf ``lebt``
        reduzieren laesst genau diesen Test fallen (nan statt 1.0).
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


class TestKorruptionsSpannen:
    """Die Funktion, die den teuersten Fehler dieser Reihe erzeugt hat.

    ``korruptions_spannen`` bestimmt, WELCHE Kurse als falsch gelten — und
    damit, was die Bereinigung anfasst. Eine frühere Fassung schloss die Spanne
    bei der ersten Gegenbewegung, ohne deren Betrag zu prüfen; bei YRCW war das
    ein echter Kurssturz von −77,2 %, und die Division überkorrigierte den Rand
    zu einer Rendite von **+6.802 %**. Die Bereinigung war dort schlimmer als
    der Fehler (E-107). Getestet wird deshalb jede Verzweigung einzeln.
    """

    def _px(self, werte: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"X": werte}, index=_tage(len(werte)))

    def test_offene_spanne_reicht_HINTER_den_letzten_tag(self):
        """E-106: mit ``index[-1]`` als Ende fiele der letzte Tag heraus.

        Die Masken arbeiten mit ``>= a & < b``. Endet die Spanne auf dem
        letzten Handelstag, bleibt genau dieser Tag unbereinigt und behält
        seinen Sprung — still, weil alle anderen Tage korrekt aussehen.
        """
        px = self._px([10.0] * 4 + [100.0] * 4)
        aus = korruptions_spannen(px, {"X"})
        ((a, b),) = [tuple(s) for s in aus["X"]["spannen"]]
        assert pd.Timestamp(b) > px.index[-1].tz_localize(None)
        # Alle vier Tage ab dem Sprung zaehlen als falsch, inklusive letztem.
        assert aus["X"]["n_tage_falsch"] == 4
        assert aus["X"]["unaufloesbar"] is False
        assert a == f"{px.index[4]:%Y-%m-%d}"

    def test_passende_rueckkehr_schliesst_die_spanne(self):
        """Ein Sprung um f und ein Fall um 1/f − 1 gehoeren zusammen."""
        px = self._px([10.0] * 3 + [100.0] * 3 + [10.0] * 3)  # x10, dann -90 %
        aus = korruptions_spannen(px, {"X"})
        ((a, b),) = [tuple(s) for s in aus["X"]["spannen"]]
        assert a == f"{px.index[3]:%Y-%m-%d}"
        assert b == f"{px.index[6]:%Y-%m-%d}"
        assert aus["X"]["n_tage_falsch"] == 3

    def test_echter_kurssturz_schliesst_die_spanne_NICHT(self):
        """Der YRCW-Fall — der eigentliche BLOCKER (F-test-1).

        Der Sprung ist x300; ein Fall um −77 % passt dazu nicht
        ((1−0,77)·300 = 69, nicht 1). Er ist ein echter Kurssturz auf der
        falschen Skala. Wird er als Rückkehr gewertet, überkorrigiert die
        Bereinigung den Rand um Faktor 69.
        """
        px = self._px([1.5] * 3 + [450.0] * 3 + [102.6] * 3)  # x300, dann -77,2 %
        aus = korruptions_spannen(px, {"X"})
        ((a, b),) = [tuple(s) for s in aus["X"]["spannen"]]
        assert a == f"{px.index[3]:%Y-%m-%d}"
        # Spanne laeuft weiter bis hinter das Panelende, statt am Sturz zu enden.
        assert pd.Timestamp(b) > px.index[-1].tz_localize(None)
        assert aus["X"]["n_tage_falsch"] == 6

    def test_zweiter_sprung_macht_den_namen_unaufloesbar(self):
        """Verschraenkte Skalen: melden, nicht raten (F-test-6).

        Bei zwei offenen Spruengen ohne Rueckkehr ist nicht bestimmbar, welcher
        Kurs auf welcher Skala liegt. Der Name wird gemeldet und NICHT
        bereinigt — 13 der 25 auffaelligen Namen der Kampagne fallen hierunter
        (12 verschraenkt, einer ueber den Vendor-Sentinel).
        """
        px = self._px([10.0] * 3 + [100.0] * 3 + [1000.0] * 3)
        aus = korruptions_spannen(px, {"X"})
        assert aus["X"]["unaufloesbar"] is True
        assert aus["X"]["unaufloesbar_grund"] == "verschraenkt"
        # MESSUNG BLEIBT (Stage-2-Finding F-senior-2): „nicht reparierbar" ist
        # nicht „nicht kaputt". Eine fruehere Fassung leerte hier die Messfelder
        # — und der Konsument, der die Kontamination BEZIFFERT, bekam fuer die
        # kaputtesten Namen null gemeldet, also eine Entwarnung.
        assert aus["X"]["n_tage_falsch"] > 0
        assert len(aus["X"]["spannen"]) == 1

    def test_ruhiger_name_taucht_gar_nicht_auf(self):
        px = self._px(list(np.linspace(10.0, 20.0, 12)))
        assert korruptions_spannen(px, {"X"}) == {}

    def test_mikropreis_sprung_wird_nicht_als_glitch_gewertet(self):
        """Unter 1 USD sind Verzehnfachungen real — dieselbe Schwelle wie im
        Glitch-Filter, hier eigenstaendig geprueft."""
        px = self._px([0.4] * 3 + [40.0] * 3)
        assert korruptions_spannen(px, {"X"}) == {}

    def test_toleranz_ist_die_stellschraube(self):
        """Mutationsprobe: ohne die Paarungsbedingung faellt dieser Test.

        Ein Fall knapp ausserhalb der Toleranz darf NICHT schliessen. Bei
        f = 10 waere die exakte Rueckkehr −90 %; hier sind es −85 %
        ((1−0,85)·10 = 1,5, Abweichung 0,5 > 0,15).
        """
        px = self._px([10.0] * 3 + [100.0] * 3 + [15.0] * 3)
        aus = korruptions_spannen(px, {"X"})
        ((_, b),) = [tuple(s) for s in aus["X"]["spannen"]]
        assert pd.Timestamp(b) > px.index[-1].tz_localize(None)

    def test_saettigungs_sentinel_macht_den_namen_unaufloesbar(self):
        """F7: 999.999,9999 ist kein Kurs, sondern ein Deckelwert.

        Vier Namen (COMS, MCIC, WFT, YRCW) stehen an 3.799 Tagen exakt auf
        diesem Wert. Eine frühere Fassung bereinigte WFT darauf über 1.375 Tage
        — Konstante geteilt durch Konstante — und erzeugte dabei einen
        Ausstiegstag von +14,87 %. Der höchste echte Kurs im Panel liegt
        darunter, die Schwelle trennt also sauber.
        """
        px = self._px([10.0] * 3 + [999_999.9999] * 3 + [10.0] * 3)
        aus = korruptions_spannen(px, {"X"})
        assert aus["X"]["unaufloesbar"] is True
        assert aus["X"]["unaufloesbar_grund"] == "sentinel"
        # Auch hier: gemeldet wird der volle Umfang, repariert wird nichts.
        assert aus["X"]["n_tage_falsch"] > 0

    def test_hoher_aber_echter_kurs_bleibt_bereinigbar(self):
        """Die Gegenprobe: knapp unter dem Sentinel wird normal behandelt.

        Sonst wäre die Schwelle ein stiller Ausschluss großer Kurse statt eines
        gezielten Sentinel-Filters.
        """
        px = self._px([1_000.0] * 3 + [900_000.0] * 3 + [1_000.0] * 3)
        aus = korruptions_spannen(px, {"X"})
        assert aus["X"]["unaufloesbar"] is False
        assert aus["X"]["unaufloesbar_grund"] == ""
        assert len(aus["X"]["spannen"]) == 1
