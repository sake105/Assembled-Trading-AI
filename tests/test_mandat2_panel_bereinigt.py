"""Tests für die Panel-Bereinigung (P12f).

Diese Funktion verändert die Kurse, auf denen die Kampagne rechnet. Ein Fehler
hier wäre der teuerste der ganzen Reihe: er würde ein Verdikt drehen und dabei
aussehen wie ein Befund. Getestet wird deshalb gegen analytisch bekannte
Reihen, und in beide Richtungen — was verschwinden **muss** und was
unangetastet **bleiben muss**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.mandat2.panel_bereinigt import auffaellig, bereinige, gegenprobe

pytestmark = pytest.mark.fast


def _tage(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2010-01-04", periods=n, tz="UTC")


class TestSpleissen:
    def test_uebergangsrendite_wird_null(self):
        """Der künstliche Sprung verschwindet — das ist der ganze Zweck."""
        idx = _tage(10)
        s = pd.Series([100.0] * 5 + [350.0] * 5, index=idx)  # +250 % bei idx[5]
        px = pd.DataFrame({"X": s})
        ende = idx[-1] + pd.Timedelta(days=1)
        sp = {"X": {"spannen": [[f"{idx[5]:%Y-%m-%d}", f"{ende:%Y-%m-%d}"]]}}
        neu, _, prot = bereinige(px, sp)
        r = neu["X"].pct_change(fill_method=None)
        assert r.iloc[5] == pytest.approx(0.0, abs=1e-12)
        assert prot["X"][0]["faktor"] == pytest.approx(3.5)

    def test_renditen_innerhalb_der_spanne_bleiben_exakt(self):
        """Ein konstanter Faktor kürzt sich im Quotienten heraus.

        Das ist der Grund, warum nur ZWEI Renditen je Spanne ersetzt werden und
        nicht Hunderte — und der Grund, warum ich einem Review-Einwand
        widersprochen habe, der das Gegenteil behauptete (E-105).
        """
        idx = _tage(12)
        basis = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        s = pd.Series(np.concatenate([basis, basis * 3.5]), index=idx)
        px = pd.DataFrame({"X": s})
        ende = idx[-1] + pd.Timedelta(days=1)
        sp = {"X": {"spannen": [[f"{idx[6]:%Y-%m-%d}", f"{ende:%Y-%m-%d}"]]}}
        neu, _, _ = bereinige(px, sp)
        r_alt = px["X"].pct_change(fill_method=None)
        r_neu = neu["X"].pct_change(fill_method=None)
        # Alles ausser dem Uebergang bei idx[6] bleibt Bit fuer Bit gleich.
        for i in list(range(1, 6)) + list(range(7, 12)):
            assert r_neu.iloc[i] == pytest.approx(r_alt.iloc[i], rel=1e-15)

    def test_ruecksprung_wird_ebenfalls_geglaettet(self):
        idx = _tage(12)
        s = pd.Series([100.0] * 4 + [350.0] * 4 + [100.0] * 4, index=idx)
        px = pd.DataFrame({"X": s})
        sp = {"X": {"spannen": [[f"{idx[4]:%Y-%m-%d}", f"{idx[8]:%Y-%m-%d}"]]}}
        neu, _, _ = bereinige(px, sp)
        r = neu["X"].pct_change(fill_method=None)
        assert r.iloc[4] == pytest.approx(0.0, abs=1e-12)
        assert r.iloc[8] == pytest.approx(0.0, abs=1e-12)

    def test_unbeteiligte_spalten_bleiben_unangetastet(self):
        idx = _tage(10)
        px = pd.DataFrame(
            {"X": [100.0] * 5 + [350.0] * 5, "Y": np.linspace(50.0, 60.0, 10)},
            index=idx,
        )
        ende = idx[-1] + pd.Timedelta(days=1)
        sp = {"X": {"spannen": [[f"{idx[5]:%Y-%m-%d}", f"{ende:%Y-%m-%d}"]]}}
        neu, _, _ = bereinige(px, sp)
        assert neu["Y"].equals(px["Y"])

    def test_ohne_vortag_wird_nicht_gespleisst(self):
        """Am Panelanfang gibt es keinen Faktor — dann lieber nichts tun."""
        idx = _tage(6)
        px = pd.DataFrame({"X": [350.0] * 6}, index=idx)
        sp = {"X": {"spannen": [[f"{idx[0]:%Y-%m-%d}", f"{idx[-1]:%Y-%m-%d}"]]}}
        neu, _, prot = bereinige(px, sp)
        assert neu["X"].equals(px["X"])
        assert prot == {}

    def test_unbekanntes_symbol_wird_uebersprungen(self):
        idx = _tage(5)
        px = pd.DataFrame({"X": [1.0] * 5}, index=idx)
        neu, _, prot = bereinige(
            px, {"GIBTESNICHT": {"spannen": [["2010-01-05", "2010-01-08"]]}}
        )
        assert neu.equals(px)
        assert prot == {}

    def test_protokoll_nennt_faktor_und_umfang(self):
        """Stille Bereinigung waere hier besonders teuer."""
        idx = _tage(10)
        px = pd.DataFrame({"X": [10.0] * 5 + [50.0] * 5}, index=idx)
        ende = idx[-1] + pd.Timedelta(days=1)
        sp = {"X": {"spannen": [[f"{idx[5]:%Y-%m-%d}", f"{ende:%Y-%m-%d}"]]}}
        _, _, prot = bereinige(px, sp)
        e = prot["X"][0]
        assert e["faktor"] == pytest.approx(5.0)
        assert e["n_tage"] == 5
        assert e["von"] == f"{idx[5]:%Y-%m-%d}"

    def test_mehrere_spannen_werden_einzeln_behandelt(self):
        idx = _tage(16)
        s = pd.Series([100.0] * 4 + [300.0] * 4 + [100.0] * 4 + [500.0] * 4, index=idx)
        px = pd.DataFrame({"X": s})
        sp = {
            "X": {
                "spannen": [
                    [f"{idx[4]:%Y-%m-%d}", f"{idx[8]:%Y-%m-%d}"],
                    [
                        f"{idx[12]:%Y-%m-%d}",
                        f"{idx[-1] + pd.Timedelta(days=1):%Y-%m-%d}",
                    ],
                ]
            }
        }
        neu, _, prot = bereinige(px, sp)
        assert len(prot["X"]) == 2
        r = neu["X"].pct_change(fill_method=None)
        for i in (4, 8, 12):
            assert r.iloc[i] == pytest.approx(0.0, abs=1e-12)


class TestSchutzmechanismen:
    """Die Guards — jeder aus einem echten Fehlschlag entstanden."""

    def test_unaufloesbare_namen_werden_nicht_angetastet(self):
        """Verschraenkte Skalen: melden, nicht raten (F-test-6).

        Bei YRCW schloss eine frueher Fassung die Spanne an einem echten
        Kurssturz und ueberkorrigierte den Rand um Faktor 69 — sie ERZEUGTE
        eine Rendite von +6.802 %.
        """
        idx = _tage(10)
        px = pd.DataFrame({"X": [100.0] * 5 + [350.0] * 5}, index=idx)
        sp = {
            "X": {
                "spannen": [[f"{idx[5]:%Y-%m-%d}", f"{idx[-1]:%Y-%m-%d}"]],
                "unaufloesbar": True,
            }
        }
        neu, _, prot = bereinige(px, sp)
        assert neu["X"].equals(px["X"])
        assert prot == {}

    def test_neue_spruenge_lassen_die_bereinigung_krachen(self):
        """Fail-closed: die Reparatur darf nicht schlimmer sein als der Fehler.

        Hier wird eine falsche Spanne uebergeben (Faktor passt nicht zum
        Rueckweg) — genau der Fall, der bei YRCW/CIN/CFC durchrutschte.
        """
        idx = _tage(12)
        # Sprung x50 hoch, aber nur -50 % zurueck: die Division durch 50
        # erzeugt am Rand einen riesigen kuenstlichen Sprung.
        s = pd.Series([10.0] * 4 + [500.0] * 4 + [250.0] * 4, index=idx)
        px = pd.DataFrame({"X": s})
        sp = {"X": {"spannen": [[f"{idx[4]:%Y-%m-%d}", f"{idx[8]:%Y-%m-%d}"]]}}
        with pytest.raises(SystemExit, match="NEUE Ausreisser"):
            bereinige(px, sp)

    def test_dividenden_werden_mitskaliert(self):
        """div_panel ist ein ABGELEITETES Feld (F-test-3).

        Ohne Mitskalierung stiege die implizite Dividendenrendite in der Spanne
        um genau f — bei WIN gemessen von 26 % auf 274 %. Der Vergleich zweier
        Panels waere dann unfair auf genau der Achse, um die es geht.
        """
        idx = _tage(10)
        px = pd.DataFrame({"X": [100.0] * 5 + [350.0] * 5}, index=idx)
        div = pd.DataFrame({"X": [0.0] * 5 + [3.5] * 5}, index=idx)
        ende = idx[-1] + pd.Timedelta(days=1)
        sp = {"X": {"spannen": [[f"{idx[5]:%Y-%m-%d}", f"{ende:%Y-%m-%d}"]]}}
        _, div_neu, _ = bereinige(px, sp, div)
        # Faktor 3.5 -> die Dividende in der Spanne muss auf 1.0 fallen.
        assert div_neu["X"].iloc[7] == pytest.approx(1.0)
        # Ausserhalb der Spanne unveraendert.
        assert div_neu["X"].iloc[0] == pytest.approx(0.0)

    def test_dividendenrendite_bleibt_erhalten(self):
        """Die eigentliche Invariante: Dividende je Kurseinheit."""
        idx = _tage(10)
        px = pd.DataFrame({"X": [100.0] * 5 + [350.0] * 5}, index=idx)
        div = pd.DataFrame({"X": [0.0] * 5 + [3.5] * 5}, index=idx)
        ende = idx[-1] + pd.Timedelta(days=1)
        sp = {"X": {"spannen": [[f"{idx[5]:%Y-%m-%d}", f"{ende:%Y-%m-%d}"]]}}
        neu, div_neu, _ = bereinige(px, sp, div)
        alt = div["X"].iloc[7] / px["X"].iloc[7]
        jetzt = div_neu["X"].iloc[7] / neu["X"].iloc[7]
        assert jetzt == pytest.approx(alt, rel=1e-12)

    def test_nan_am_spannenanfang_wird_uebersprungen(self):
        idx = _tage(10)
        px = pd.DataFrame({"X": [100.0] * 4 + [np.nan] + [350.0] * 5}, index=idx)
        sp = {"X": {"spannen": [[f"{idx[5]:%Y-%m-%d}", f"{idx[-1]:%Y-%m-%d}"]]}}
        neu, _, prot = bereinige(px, sp)
        assert prot == {}
        assert neu["X"].equals(px["X"])

    def test_nicht_positiver_vortagskurs_wird_uebersprungen(self):
        idx = _tage(10)
        px = pd.DataFrame(
            {"X": [1.0] * 4 + [0.0] + [350.0] * 5, "Y": [1.0] * 10}, index=idx
        )
        sp = {"X": {"spannen": [[f"{idx[5]:%Y-%m-%d}", f"{idx[-1]:%Y-%m-%d}"]]}}
        neu, _, prot = bereinige(px, sp)
        assert prot == {}
        assert neu["X"].equals(px["X"])

    def test_neuer_absturz_laesst_die_bereinigung_ebenfalls_krachen(self):
        """Der Waechter muss ZWEISEITIG sein (F-test-4).

        Wird durch einen zu GROSSEN Faktor geteilt, entsteht kein Sprung nach
        oben, sondern ein Absturz. Ein kuenstlicher -80-%-Tag faellt in einem
        Momentum-Backtest niemandem auf — er verschiebt nur still die Rangliste.
        """
        idx = _tage(12)
        # Der Faktor wird aus idx[4]/idx[3] = 50 bestimmt. Die Spanne endet
        # aber schon bei idx[8], wo der Kurs auf 500 zurueckgefallen ist:
        # 500/50 = 10 gegen 500 am Vortag -> -98 %.
        s = pd.Series([10.0] * 4 + [500.0] * 4 + [500.0] * 4, index=idx)
        px = pd.DataFrame({"X": s})
        sp = {"X": {"spannen": [[f"{idx[4]:%Y-%m-%d}", f"{idx[8]:%Y-%m-%d}"]]}}
        with pytest.raises(SystemExit, match="NEUE Ausreisser"):
            bereinige(px, sp)

    def test_spleissrichtung_ist_fuer_renditen_unerheblich(self):
        """Begruendeter Widerspruch zu F-test-2.

        Der Einwand lautete, bei einer Spanne, die den GROSSTEIL der Historie
        ausmacht, sei es besser, das kurze Anfangsstueck hochzuskalieren statt
        die lange Spanne herunterzuteilen. Fuer die Backtests ist das
        gleichgueltig: gerechnet wird auf Renditen und auf Dividenden JE
        KURSEINHEIT, beides ist skaleninvariant. Der Test belegt das, statt es
        zu behaupten (E-105).

        Was NICHT invariant ist: absolute Kursniveaus. Solange kein Filter auf
        Kurshoehe wirkt, ist das folgenlos — der Glitch-Detektor selbst nutzt
        eine 1-USD-Schwelle, arbeitet aber auf dem ORIGINALPANEL.
        """
        idx = _tage(12)
        basis = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        s = pd.Series(np.concatenate([basis, basis * 3.5]), index=idx)
        px = pd.DataFrame({"X": s})
        ende = idx[-1] + pd.Timedelta(days=1)
        sp = {"X": {"spannen": [[f"{idx[6]:%Y-%m-%d}", f"{ende:%Y-%m-%d}"]]}}
        runter, _, _ = bereinige(px, sp)
        # Andere Richtung von Hand: Anfangsstueck mit f multiplizieren. f ist
        # der GEMESSENE Uebergangsfaktor 350/105, nicht die 3.5, mit der die
        # Reihe konstruiert wurde — die beiden fallen hier auseinander, weil
        # der Sprung von 105 auf 350 geht und nicht von 100 auf 350.
        f = 350.0 / 105.0
        hoch = px["X"].copy()
        hoch.iloc[:6] = hoch.iloc[:6] * f
        r_runter = runter["X"].pct_change(fill_method=None)
        r_hoch = hoch.pct_change(fill_method=None)
        for i in range(1, 12):
            assert r_runter.iloc[i] == pytest.approx(r_hoch.iloc[i], rel=1e-12)
        # Und die Niveaus unterscheiden sich um genau f — das ist der Preis.
        assert (hoch / runter["X"]).round(9).nunique() == 1


class TestWaechterBlindstellen:
    """Der Fail-Closed-Wächter darf sich nicht selbst blind machen.

    Alle Fälle hier stammen aus Stage-1-Findings — jeder war ein Pfad, auf dem
    die Bereinigung Schaden anrichten konnte, ohne dass ihre eigene Gegenprobe
    es sah.
    """

    def test_penny_schwelle_gilt_fuer_das_ORIGINAL_nicht_fuer_die_reparatur(self):
        """F1 (BLOCKER): die Reparatur schob Kurse unter die eigene Schwelle.

        Identisch zum Guard-Test, nur alle Kurse ÷ 20. Vorher lief er durch,
        weil der bereinigte Kurs unter 1 USD lag — die 1-USD-Schwelle beantwortet
        aber die Frage „war das ein Penny-Stock?", und die entscheidet das
        Originalpanel, nicht das Ergebnis der Bereinigung.
        """
        idx = _tage(12)
        s = pd.Series([0.5] * 4 + [25.0] * 4 + [12.5] * 4, index=idx)
        px = pd.DataFrame({"X": s})
        sp = {"X": {"spannen": [[f"{idx[4]:%Y-%m-%d}", f"{idx[8]:%Y-%m-%d}"]]}}
        with pytest.raises(SystemExit, match="NEUE Ausreisser"):
            bereinige(px, sp)

    def test_echter_pennystock_loest_keinen_fehlalarm_aus(self):
        """Die Gegenprobe: unter 1 USD im ORIGINAL bleibt der Wächter still.

        Sonst wäre der Fix ein Rückschritt — die Schwelle hat einen fachlichen
        Grund (unter einem Dollar sind Verzehnfachungen real).
        """
        idx = _tage(12)
        # Beide fuer die Maske relevanten Vortage (idx[3] und idx[7]) liegen im
        # ORIGINAL unter 1 USD — genau der Fall, den die Schwelle schuetzen soll.
        s = pd.Series([0.05] * 4 + [0.9] * 4 + [0.45] * 4, index=idx)
        px = pd.DataFrame({"X": s})
        sp = {"X": {"spannen": [[f"{idx[4]:%Y-%m-%d}", f"{idx[8]:%Y-%m-%d}"]]}}
        neu, _, prot = bereinige(px, sp)  # kein SystemExit
        assert prot["X"][0]["faktor"] == pytest.approx(18.0)
        # Der Eingriff erzeugt hier rechnerisch einen Sprung — der Waechter
        # schweigt bewusst, weil unter 1 USD keine Aussage moeglich ist.
        assert neu["X"].pct_change(fill_method=None).iloc[8] > 1.0

    def test_abwaertsarm_faengt_was_der_aufwaertsarm_nicht_sieht(self):
        """F2: der bisherige „Zweiseitigkeits"-Test feuerte in Wahrheit oben.

        Erst die Mutationsprobe zeigte es: Abwärtsarm entfernen → kein Test
        fällt. Der Grund ist strukturell. Am Spannenende gilt
        ``r_neu = f · (1 + r_alt) − 1``; für **f > 1** ist ``r_neu > r_alt``,
        die Bereinigung verschiebt die Rendite dort also immer nach **oben**.
        Ein neuer Absturz ist mit f > 1 gar nicht konstruierbar — jeder
        vermeintliche Abwärtstest feuerte über den Aufwärtsarm.

        Der Abwärtsarm greift bei **f < 1**, also bei einer Spanne, die nach
        UNTEN von der Basisskala abweicht. ``korruptions_spannen`` erzeugt
        solche Spannen heute nicht (es öffnet nur bei Sprüngen nach oben) —
        der Arm sichert also gegen künftige Aufrufer und gegen von Hand
        übergebene Spannen. Das ist der ehrliche Geltungsbereich, und er steht
        hier, statt „zweiseitig geprüft" pauschal zu behaupten.
        """
        idx = _tage(12)
        # f = 50/100 = 0,5: die Spanne wird VERDOPPELT. Am Spannenende steht
        # original 45 (nur -10 % gegenüber 50) — nach der Bereinigung aber 45
        # gegen 100, also -55 %. Ein Absturz, den es vorher nicht gab.
        s = pd.Series([100.0] * 4 + [50.0] * 4 + [45.0] * 4, index=idx)
        px = pd.DataFrame({"X": s})
        sp = {"X": {"spannen": [[f"{idx[4]:%Y-%m-%d}", f"{idx[8]:%Y-%m-%d}"]]}}
        with pytest.raises(SystemExit, match="NEUE Ausreisser"):
            bereinige(px, sp)

    def test_aufwaertssprung_am_spannenende_wird_gefangen(self):
        """Der Fall mit f > 1 — die Morphologie des echten YRCW-Fehlers."""
        idx = _tage(12)
        s = pd.Series([100.0] * 4 + [1000.0] * 4 + [1000.0] * 4, index=idx)
        px = pd.DataFrame({"X": s})
        # Faktor 10, Spanne endet mitten im Hochplateau -> danach +900 %.
        sp = {"X": {"spannen": [[f"{idx[4]:%Y-%m-%d}", f"{idx[8]:%Y-%m-%d}"]]}}
        with pytest.raises(SystemExit, match="NEUE Ausreisser"):
            bereinige(px, sp)

    def test_ruecksprungtag_ist_NICHT_exakt_null(self):
        """F3: der Docstring behauptete das lange — es stimmt nicht.

        Die Paarungsbedingung lässt bis zu 15 % Abweichung zu; genau diese
        Rendite setzt die Bereinigung am Rücksprungtag ein. Der Test hält die
        Größenordnung fest, damit sie nicht wieder aus der Doku verschwindet.
        """
        idx = _tage(12)
        # Sprung x10, Rueckweg nur -88 % statt -90 %: (1-0.88)*10 = 1.2.
        s = pd.Series([10.0] * 4 + [100.0] * 4 + [12.0] * 4, index=idx)
        px = pd.DataFrame({"X": s})
        sp = {"X": {"spannen": [[f"{idx[4]:%Y-%m-%d}", f"{idx[8]:%Y-%m-%d}"]]}}
        neu, _, _ = bereinige(px, sp)
        r = neu["X"].pct_change(fill_method=None)
        assert r.iloc[4] == pytest.approx(0.0, abs=1e-12)  # Eintritt: exakt 0
        assert r.iloc[8] == pytest.approx(0.2, rel=1e-9)  # Austritt: +20 %


class TestGegenprobeZaehlt:
    """F-senior-5: nur der Abbruchpfad war getestet, die Zahlen nicht.

    Diese vier Felder tragen im Befund den Satz „beseitigt werden 5 % der
    auffälligen Tage; 433 bleiben stehen — die Bereinigung ist eine
    Untergrenze". Das ist die einzige quantitative Selbstbeschränkung des
    Abschnitts; wäre `beseitigt` mit `neu_entstanden` vertauscht, läse sich das
    Dokument als Erfolgsmeldung.
    """

    def _panel(self):
        """Zwei Namen: bei X wird der Ausreißer beseitigt, bei Y bleibt er."""
        idx = _tage(10)
        return pd.DataFrame(
            {
                "X": [100.0] * 5 + [350.0] * 5,  # wird gespleisst
                "Y": [100.0] * 5 + [350.0] * 5,  # bleibt unangetastet
            },
            index=idx,
        )

    def test_zaehlt_beseitigt_und_verbleibend_getrennt(self):
        px = self._panel()
        idx = px.index
        ende = idx[-1] + pd.Timedelta(days=1)
        sp = {"X": {"spannen": [[f"{idx[5]:%Y-%m-%d}", f"{ende:%Y-%m-%d}"]]}}
        neu, _, _ = bereinige(px, sp)
        g = gegenprobe(px, neu)
        assert g["auffaellig_original"] == 2  # X und Y springen je einmal
        assert g["auffaellig_bereinigt"] == 1  # nur noch Y
        assert g["beseitigt"] == 1
        assert g["neu_entstanden"] == 0
        assert g["wo_neu"] == []

    def test_beseitigt_und_neu_entstanden_sind_nicht_vertauschbar(self):
        """Mutationsprobe in Testform: `(a & ~n)` gegen `(n & ~a)`."""
        px = self._panel()
        idx = px.index
        ende = idx[-1] + pd.Timedelta(days=1)
        sp = {"X": {"spannen": [[f"{idx[5]:%Y-%m-%d}", f"{ende:%Y-%m-%d}"]]}}
        neu, _, _ = bereinige(px, sp)
        g = gegenprobe(px, neu)
        # Asymmetrisch: hier verschwindet etwas, es entsteht nichts.
        assert g["beseitigt"] != g["neu_entstanden"]
        assert g["auffaellig_bereinigt"] < g["auffaellig_original"]

    def test_identische_panels_ergeben_lauter_nullen(self):
        px = self._panel()
        g = gegenprobe(px, px)
        assert g["beseitigt"] == 0 and g["neu_entstanden"] == 0
        assert g["auffaellig_original"] == g["auffaellig_bereinigt"] == 2

    def test_auffaellig_zaehlt_beide_richtungen(self):
        idx = _tage(6)
        px = pd.DataFrame(
            {
                "HOCH": [10.0, 10.0, 100.0, 100.0, 100.0, 100.0],  # +900 %
                "TIEF": [10.0, 10.0, 2.0, 2.0, 2.0, 2.0],  # -80 %
                "RUHIG": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
            },
            index=idx,
        )
        a = auffaellig(px)
        assert bool(a["HOCH"].any()) and bool(a["TIEF"].any())
        assert not bool(a["RUHIG"].any())
