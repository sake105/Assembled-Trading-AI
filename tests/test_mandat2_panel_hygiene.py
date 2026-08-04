"""Tests für P12e — Panel-Hygiene der Kampagne.

Die tragende Logik dieses Moduls ist die **Kanal-Unterscheidung**, und sie
entstand aus einem Fehler: die erste Fassung meldete „13 korrumpierte Namen
wurden gewählt → Verdicts kontaminiert". Das war die falsche Frage. Ein Name,
der zehn Jahre vor oder nach seinem Preisfehler gewählt wurde, ist über keinen
Kanal berührt. Erst die Unterscheidung

* **Halte-Kanal** — über den Glitch-Tag hinweg gehalten, der falsche Sprung
  geht direkt in die Portfoliorendite ein;
* **Auswahl-Kanal** — im Rückblickfenster nach dem Glitch gewählt, der Fehler
  hat das Momentum aufgebläht;

Und der Halte-Kanal muss aus dem **echten Bestand** kommen, nicht aus der
Auswahlmenge: die Engine verkauft erst bei rang > rank_out, hält also weit
über den letzten Top-20-Termin hinaus. Ein Proxy aus der Auswahl übersah, dass
GPS am Tag seines Vendor-Fehlers im Portfolio lag — mit +12,4 % Tagesrendite,
dem zweitgrößten Einzeltag in 21 Jahren (E-102).

Getestet wird deshalb die Zuordnung selbst, **importiert** statt nachgebaut.
Die erste Fassung dieser Datei baute sie im Test nach und konnte den Fehler im
Produktionscode deshalb nicht sehen.
"""

from __future__ import annotations

import pandas as pd
import pytest

from research.mandat2.p12e_panel_hygiene import (
    MOM_FENSTER,
    MOM_LAG,
    abdeckung_je_termin,
    austritts_anreicherung,
    kanal_auswahl,
    kanal_halten,
)

pytestmark = pytest.mark.fast


def _monate(n: int, start: str = "2000-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME", tz="UTC")


class TestAbdeckung:
    def test_zaehlt_nur_mitglieder_mit_preisspalte(self):
        idx = _monate(2)
        m = pd.Series([frozenset({"A", "B", "C"}), frozenset({"A", "B"})], index=idx)
        aus = abdeckung_je_termin(m, {"A", "B"})
        assert aus[0]["n_mitglieder"] == 3
        assert aus[0]["n_mit_spalte"] == 2
        assert aus[0]["abdeckung"] == pytest.approx(2 / 3)
        assert aus[1]["abdeckung"] == pytest.approx(1.0)

    def test_leere_mitgliederliste_wird_uebersprungen(self):
        idx = _monate(2)
        m = pd.Series([frozenset(), frozenset({"A"})], index=idx)
        assert len(abdeckung_je_termin(m, {"A"})) == 1

    def test_spalten_ohne_mitgliedschaft_zaehlen_nicht(self):
        """Sonst waere die Abdeckung > 100 % — die Kennzahl misst EINE Richtung."""
        idx = _monate(1)
        m = pd.Series([frozenset({"A"})], index=idx)
        aus = abdeckung_je_termin(m, {"A", "B", "C"})
        assert aus[0]["abdeckung"] == pytest.approx(1.0)


class TestAustrittsAnreicherung:
    def test_erkennt_ueberrepraesentierte_austritte(self):
        """Der Kern: sind die Namen OHNE Preisspalte oefter ausgeschieden?"""
        idx = _monate(2)
        start = frozenset({"MIT1", "MIT2", "OHNE1", "OHNE2"})
        ende = frozenset({"MIT1", "MIT2"})  # beide OHNE-Namen sind weg
        m = pd.Series([start, ende], index=idx)
        a = austritts_anreicherung(m, {"MIT1", "MIT2"})
        assert a["ueberlebensquote_mit_spalte"] == pytest.approx(1.0)
        assert a["ueberlebensquote_ohne_spalte"] == pytest.approx(0.0)

    def test_neutrale_luecke_ergibt_gleiche_quoten(self):
        """Gegenprobe — sonst faende der Test seine Anreicherung immer."""
        idx = _monate(2)
        start = frozenset({"MIT1", "MIT2", "OHNE1", "OHNE2"})
        ende = frozenset({"MIT1", "OHNE1"})  # je einer ueberlebt
        m = pd.Series([start, ende], index=idx)
        a = austritts_anreicherung(m, {"MIT1", "MIT2"})
        assert a["ueberlebensquote_mit_spalte"] == pytest.approx(0.5)
        assert a["ueberlebensquote_ohne_spalte"] == pytest.approx(0.5)

    def test_ohne_luecke_kein_nenner(self):
        idx = _monate(2)
        m = pd.Series([frozenset({"A"}), frozenset({"A"})], index=idx)
        a = austritts_anreicherung(m, {"A"})
        assert a["n_ohne_spalte"] == 0
        assert a["ueberlebensquote_ohne_spalte"] is None


class TestKanalHalten:
    """Kanal A — verzerrt ist die TAGESRENDITE, also nur am Uebergangstag.

    Wichtig und gegen einen Review-Einwand verteidigt: ein dauerhafter
    Niveaubruch verzerrt die Rendite NICHT dauerhaft. Liegen Vortag und Tag auf
    derselben (falschen) Skala, kuerzt sich der Faktor im Quotienten heraus.
    Verzerrt ist ausschliesslich der Uebergang. Deshalb liest diese Funktion
    ``uebergaenge`` und nicht die ganze Korruptionsspanne.

    Importiert die Produktionsfunktion, statt sie nachzubauen (F-test-7).
    """

    @staticmethod
    def _r(werte: dict) -> pd.Series:
        return pd.Series(
            list(werte.values()),
            index=pd.DatetimeIndex(list(werte), tz="UTC"),
        )

    def test_haelt_an_uebergangstag_wird_erkannt(self):
        g = {"X": {"uebergaenge": ["2010-01-05", "2010-01-06"]}}
        bestand = {"2010-01-05": {"X", "Y"}, "2010-01-06": {"Y"}}
        aus = kanal_halten(g, bestand, self._r({"2010-01-05": 0.12}))
        assert aus["X"]["n_tage"] == 1
        assert aus["X"]["tage"] == ["2010-01-05"]
        assert aus["X"]["groesste_wirkung"] == pytest.approx(0.12)

    def test_nicht_im_bestand_ist_nicht_beruehrt(self):
        g = {"X": {"uebergaenge": ["2010-01-05"]}}
        assert kanal_halten(g, {"2010-01-05": {"Y", "Z"}}, self._r({})) == {}

    def test_vorzeichen_bleibt_erhalten(self):
        """Der Extremwert kann ein VERLUSTtag sein.

        Eine fruehere Fassung speicherte ``max(abs(...))``; die Ausgabe
        formatierte den Betrag dann mit ``+`` und nannte ihn einen Gewinn
        (F-senior-4). Fuer CFC war der Extremtag tatsaechlich -6,7 %.
        """
        g = {"X": {"uebergaenge": ["2010-01-05", "2010-01-06"]}}
        bestand = {"2010-01-05": {"X"}, "2010-01-06": {"X"}}
        aus = kanal_halten(
            g, bestand, self._r({"2010-01-05": 0.03, "2010-01-06": -0.09})
        )
        assert aus["X"]["groesste_wirkung"] == pytest.approx(-0.09)
        assert aus["X"]["groesste_wirkung_betrag"] == pytest.approx(0.09)
        assert aus["X"]["groesste_wirkung_tag"] == "2010-01-06"

    def test_fehlende_rendite_KRACHT_statt_zu_schweigen(self):
        """Fail-loud (F-senior-1).

        Ein fehlender Renditewert zu einem gehaltenen korrupten Tag wuerde als
        0,0 % gerendert und als Entwarnung gelesen. Die Messung darf nicht in
        die beruhigende Richtung ausfallen — die vorige Testfassung hielt genau
        dieses Schweigen als Sollverhalten fest.
        """
        g = {"X": {"uebergaenge": ["2010-01-05"]}}
        with pytest.raises(SystemExit):
            kanal_halten(g, {"2010-01-05": {"X"}}, self._r({}))

    def test_leeres_bestandsprotokoll_kracht(self):
        with pytest.raises(SystemExit):
            kanal_halten({"X": {"uebergaenge": ["2010-01-05"]}}, {}, self._r({}))

    def test_rang_wird_gemessen_nicht_behauptet(self):
        """F-senior-9: 'zweitgroesster Einzeltag' stand im Log, ohne gerechnet
        zu sein. Jetzt kommt der Rang aus den Daten."""
        g = {"X": {"uebergaenge": ["2010-01-06"]}}
        r = self._r({"2010-01-05": 0.20, "2010-01-06": 0.12, "2010-01-07": 0.01})
        aus = kanal_halten(g, {"2010-01-06": {"X"}}, r)
        assert aus["X"]["rang_unter_allen_tagen"] == 2
        assert aus["X"]["n_handelstage"] == 3


class TestKanalAuswahl:
    """Kanal B — kontaminiert ist der Score, wenn die BEIDEN BEINE auf
    verschiedenen Preisskalen liegen.

    ``momentum_score`` ist ein Quotient aus genau zwei Stuetzstellen. Daraus
    folgt beides: ein Fehlertag ZWISCHEN den Beinen beruehrt den Score nicht
    (meine erste Fassung markierte ~230 Termine zu viel), und zwei Beine auf
    DERSELBEN falschen Skala ergeben trotzdem den richtigen Quotienten (die
    naheliegende Korrektur „ein Bein liegt falsch" waere also auch falsch).
    """

    @staticmethod
    def _idx(n: int = 400) -> pd.DatetimeIndex:
        return pd.bdate_range("2010-01-04", periods=n, tz="UTC")

    def test_beine_auf_verschiedenen_skalen_ist_kontaminiert(self):
        idx = self._idx(700)
        bruch = idx[300]
        sp = {"X": {"spannen": [[f"{bruch:%Y-%m-%d}", f"{idx[-1]:%Y-%m-%d}"]]}}
        # Termin so, dass das kurze Bein NACH, das lange VOR dem Bruch liegt.
        # Zugleich >= MOM_FENSTER, sonst existiert gar kein volles Fenster.
        t = idx[300 + MOM_LAG]
        assert kanal_auswahl(sp, {f"{t:%Y-%m-%d}": ["X"]}, idx) == {
            "X": [f"{t:%Y-%m-%d}"]
        }

    def test_beide_beine_auf_derselben_falschen_skala_ist_sauber(self):
        """Der Faktor kuerzt sich heraus — der Score stimmt trotz falscher Kurse."""
        idx = self._idx()
        bruch = idx[10]
        sp = {"X": {"spannen": [[f"{bruch:%Y-%m-%d}", f"{idx[-1]:%Y-%m-%d}"]]}}
        t = idx[300]  # beide Beine (t-21, t-252) liegen nach dem Bruch
        assert kanal_auswahl(sp, {f"{t:%Y-%m-%d}": ["X"]}, idx) == {}

    def test_beide_beine_vor_dem_bruch_ist_sauber(self):
        idx = self._idx()
        bruch = idx[350]
        sp = {"X": {"spannen": [[f"{bruch:%Y-%m-%d}", f"{idx[-1]:%Y-%m-%d}"]]}}
        t = idx[300]
        assert kanal_auswahl(sp, {f"{t:%Y-%m-%d}": ["X"]}, idx) == {}

    def test_einzelspike_trifft_nur_die_beiden_beine(self):
        """Ein Ein-Tages-Spike kontaminiert genau ZWEI Termine, nicht ~230.

        Das ist der Punkt, an dem meine erste Fassung am weitesten danebenlag:
        sie markierte alles im Intervall nach dem Spike.
        """
        idx = self._idx(900)
        spike = idx[300]
        sp = {"X": {"spannen": [[f"{spike:%Y-%m-%d}", f"{idx[301]:%Y-%m-%d}"]]}}
        gewaehlt = {f"{x:%Y-%m-%d}": ["X"] for x in idx[MOM_FENSTER:800]}
        treffer = kanal_auswahl(sp, gewaehlt, idx)["X"]
        assert len(treffer) == 2
        assert f"{idx[300 + MOM_LAG]:%Y-%m-%d}" in treffer
        assert f"{idx[300 + MOM_FENSTER]:%Y-%m-%d}" in treffer

    def test_anderer_name_zaehlt_nicht(self):
        idx = self._idx(700)
        bruch = idx[300]
        sp = {"X": {"spannen": [[f"{bruch:%Y-%m-%d}", f"{idx[-1]:%Y-%m-%d}"]]}}
        t = idx[300 + MOM_LAG]
        assert kanal_auswahl(sp, {f"{t:%Y-%m-%d}": ["Y"]}, idx) == {}

    def test_termin_vor_dem_ersten_vollen_fenster_wird_uebersprungen(self):
        idx = self._idx()
        sp = {"X": {"spannen": [[f"{idx[0]:%Y-%m-%d}", f"{idx[5]:%Y-%m-%d}"]]}}
        t = idx[MOM_FENSTER - 5]
        assert kanal_auswahl(sp, {f"{t:%Y-%m-%d}": ["X"]}, idx) == {}
