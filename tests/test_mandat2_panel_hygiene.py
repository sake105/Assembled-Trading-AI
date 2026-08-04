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
    """Kanal A — der Kanal, dessen Fehleinschaetzung die falsche Entwarnung trug.

    Importiert die Produktionsfunktion, statt sie nachzubauen. Die erste Fassung
    dieser Tests baute die Logik im Test nach und konnte deshalb nicht merken,
    dass der Produktionscode die falsche Groesse mass (F-test-7).
    """

    def test_haelt_an_korruptem_tag_wird_erkannt(self):
        g = {"X": {"tage": ["2010-01-05", "2010-01-06"]}}
        bestand = {"2010-01-05": {"X", "Y"}, "2010-01-06": {"Y"}}
        r = pd.Series([0.12], index=pd.DatetimeIndex(["2010-01-05"], tz="UTC"))
        aus = kanal_halten(g, bestand, r)
        assert aus["X"]["n_tage"] == 1
        assert aus["X"]["tage"] == ["2010-01-05"]
        assert aus["X"]["groesste_wirkung"] == pytest.approx(0.12)

    def test_nicht_im_bestand_ist_nicht_beruehrt(self):
        g = {"X": {"tage": ["2010-01-05"]}}
        bestand = {"2010-01-05": {"Y", "Z"}}
        assert kanal_halten(g, bestand, pd.Series(dtype=float)) == {}

    def test_groesste_wirkung_nimmt_den_betrag(self):
        """Ein Einbruch ist genauso kontaminierend wie ein Sprung."""
        g = {"X": {"tage": ["2010-01-05", "2010-01-06"]}}
        bestand = {"2010-01-05": {"X"}, "2010-01-06": {"X"}}
        r = pd.Series(
            [0.03, -0.09],
            index=pd.DatetimeIndex(["2010-01-05", "2010-01-06"], tz="UTC"),
        )
        assert kanal_halten(g, bestand, r)["X"]["groesste_wirkung"] == pytest.approx(
            0.09
        )

    def test_fehlende_rendite_kippt_nicht(self):
        g = {"X": {"tage": ["2010-01-05"]}}
        aus = kanal_halten(g, {"2010-01-05": {"X"}}, pd.Series(dtype=float))
        assert aus["X"]["n_tage"] == 1
        assert aus["X"]["groesste_wirkung"] == 0.0


class TestKanalAuswahl:
    """Kanal B — das Fenster wird in HANDELSTAGEN gerechnet, nicht in Kalendertagen."""

    @staticmethod
    def _idx(n: int = 400) -> pd.DatetimeIndex:
        return pd.bdate_range("2010-01-04", periods=n, tz="UTC")

    def test_termin_im_fenster_wird_erkannt(self):
        idx = self._idx()
        g = {"X": {"tage": [f"{idx[0]:%Y-%m-%d}"]}}
        drin = idx[MOM_LAG + 5]
        gewaehlt = {f"{drin:%Y-%m-%d}": ["X"]}
        assert kanal_auswahl(g, gewaehlt, idx) == {"X": [f"{drin:%Y-%m-%d}"]}

    def test_zu_frueh_zaehlt_nicht(self):
        """Die ersten MOM_LAG Handelstage sind NICHT kontaminiert — beide Beine
        des Momentums liegen dort noch vor dem Fehlertag."""
        idx = self._idx()
        g = {"X": {"tage": [f"{idx[0]:%Y-%m-%d}"]}}
        zu_frueh = idx[MOM_LAG - 3]
        assert kanal_auswahl(g, {f"{zu_frueh:%Y-%m-%d}": ["X"]}, idx) == {}

    def test_zu_spaet_zaehlt_nicht(self):
        idx = self._idx()
        g = {"X": {"tage": [f"{idx[0]:%Y-%m-%d}"]}}
        zu_spaet = idx[MOM_FENSTER + 5]
        assert kanal_auswahl(g, {f"{zu_spaet:%Y-%m-%d}": ["X"]}, idx) == {}

    def test_anderer_name_zaehlt_nicht(self):
        idx = self._idx()
        g = {"X": {"tage": [f"{idx[0]:%Y-%m-%d}"]}}
        drin = idx[MOM_LAG + 5]
        assert kanal_auswahl(g, {f"{drin:%Y-%m-%d}": ["Y"]}, idx) == {}

    def test_fenster_zaehlt_indexpositionen_nicht_kalendertage(self):
        """Der eigentliche Punkt — und er ist mit einem Kalender-Loch beweisbar.

        Der Index bekommt hier eine lange Luecke (Boersenschliessung). Ein
        Termin, der kalendarisch weit jenseits von 365 Tagen liegt, ist
        positionell trotzdem noch im Fenster. Eine Kalendertage-Grenze haette
        ihn verworfen — genau der Fehler, den F-test-4 an der alten
        365-Tage-Schwelle bemaengelte.

        (Ein reiner ``bdate_range`` taugt dafuer nicht: er kennt keine
        Feiertage, dort sind 252 Werktage nur 352 Kalendertage. Diese
        Testannahme hatte ich zuerst falsch — der Test fand sie.)
        """
        vorn = pd.bdate_range("2010-01-04", periods=MOM_FENSTER, tz="UTC")
        hinten = pd.bdate_range("2012-06-01", periods=50, tz="UTC")  # Luecke!
        idx = vorn.append(hinten)
        g = {"X": {"tage": [f"{idx[0]:%Y-%m-%d}"]}}
        rand = idx[MOM_FENSTER]  # positionell im Fenster
        assert (rand - idx[0]).days > 365, "Fixture muss die Luecke wirklich haben"
        assert kanal_auswahl(g, {f"{rand:%Y-%m-%d}": ["X"]}, idx) == {
            "X": [f"{rand:%Y-%m-%d}"]
        }
