"""Tests für P12h/P12i — recycelte Ticker im Tagespanel.

Der Fund: der Datenlieferant gibt unter einem Symbol die **heutige** Firma
zurück. 29 Panel-Spalten führen dadurch zwei oder mehr Unternehmen, und die
Delisting-Hygiene fällt bei ihnen aus — CGP lief 3.264 Handelstage im Bestand
weiter, obwohl die Firma seit 2001 nicht existierte.

Die Tests sichern vor allem die **Unterscheidung**, an der die erste Fassung
gescheitert ist: eine unterbrochene Serie ist nicht automatisch ein
Firmenwechsel. Coca-Cola Enterprises hat sechs jährliche Vendor-Lücken und
existierte durchgehend; bei einer Schwelle von 120 Handelstagen hätte die
Trennung diese eine Firma in sieben Stücke zerlegt (E-107-Klasse).
"""

from __future__ import annotations

import pandas as pd
import pytest

from research.mandat2.p12h_ticker_recycling import (
    tote_haltedauer,
    unterbrechungen,
    wirkung,
)
from research.mandat2.panel_getrennt import MIN_LUECKE, segmente, trenne

pytestmark = pytest.mark.fast


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2000-01-03", periods=n, tz="UTC")


def _serie(idx, stuecke: list[tuple[int, int, float]]) -> pd.Series:
    """Baut eine Serie aus (start, laenge, kurs)-Stuecken; Rest ist NaN."""
    s = pd.Series(float("nan"), index=idx)
    for start, laenge, kurs in stuecke:
        s.iloc[start : start + laenge] = kurs
    return s


class TestUnterbrechungen:
    def test_findet_die_luecke_und_die_kurse_an_ihren_raendern(self):
        idx = _idx(800)
        px = pd.DataFrame({"X": _serie(idx, [(0, 100, 50.0), (700, 100, 5.0)])})
        (t,) = unterbrechungen(px, min_luecke=500)
        assert t["symbol"] == "X"
        assert t["luecke_handelstage"] == 601
        assert t["kurs_vor"] == 50.0 and t["kurs_nach"] == 5.0
        assert t["faktor"] == pytest.approx(0.1)

    def test_luecke_unter_der_schwelle_wird_ignoriert(self):
        idx = _idx(400)
        px = pd.DataFrame({"X": _serie(idx, [(0, 100, 50.0), (200, 100, 51.0)])})
        assert unterbrechungen(px, min_luecke=500) == []

    def test_durchgehende_serie_ist_kein_treffer(self):
        idx = _idx(600)
        px = pd.DataFrame({"X": _serie(idx, [(0, 600, 10.0)])})
        assert unterbrechungen(px, min_luecke=500) == []

    def test_zu_kurze_serien_werden_uebersprungen(self):
        idx = _idx(800)
        px = pd.DataFrame({"X": _serie(idx, [(0, 5, 50.0), (700, 5, 5.0)])})
        assert unterbrechungen(px, min_luecke=500) == []

    def test_misst_in_handelstagen_nicht_in_kalendertagen(self):
        """Über 20 Jahre sind Wochenenden ein Drittel der Kalenderzeit.

        Die Serie hat hier eine Lücke von 300 Handelstagen ≈ 420 Kalendertagen.
        Wer in Kalendertagen misst, überschätzt jede Lücke um rund 40 %.
        """
        idx = _idx(800)
        px = pd.DataFrame({"X": _serie(idx, [(0, 100, 50.0), (400, 100, 5.0)])})
        (t,) = unterbrechungen(px, min_luecke=250)
        assert t["luecke_handelstage"] == 301
        kalender = (
            pd.Timestamp(t["naechster_kurs_am"]) - pd.Timestamp(t["letzter_kurs_am"])
        ).days
        assert kalender > 400  # deutlich mehr — genau der Unterschied


class TestSchwelleTrenntDieUrsachen:
    """Die Lehre aus dem Fehlversuch: unterbrochen ≠ recycelt.

    CCE (Coca-Cola Enterprises) hat jährliche Vendor-Lücken und existierte
    durchgehend. Bei 120 Handelstagen hätte die Trennung sie in sieben Stücke
    zerlegt.
    """

    def _cce_artig(self):
        idx = _idx(1500)
        # Vier Stuecke mit je ~250 Tagen Luecke — Kurs setzt jeweils fort.
        return pd.DataFrame(
            {
                "CCE": _serie(
                    idx,
                    [
                        (0, 100, 9.0),
                        (350, 100, 11.0),
                        (700, 100, 10.0),
                        (1050, 100, 12.0),
                    ],
                )
            }
        )

    def test_jaehrliche_datenloecher_werden_bei_500_nicht_getrennt(self):
        px = self._cce_artig()
        assert segmente(px, "CCE", MIN_LUECKE) == []

    def test_dieselbe_serie_waere_bei_120_zerlegt_worden(self):
        """Der Beleg, dass die Schwelle die Unterscheidung trägt."""
        px = self._cce_artig()
        assert len(segmente(px, "CCE", 120)) == 3

    def test_schwelle_steht_auf_500(self):
        assert MIN_LUECKE == 500


class TestTrennung:
    def _panel(self):
        idx = _idx(1400)
        close = pd.DataFrame(
            {
                "REC": _serie(idx, [(0, 200, 80.0), (1200, 200, 0.1)]),
                "OK": _serie(idx, [(0, 1400, 20.0)]),
            }
        )
        m = pd.Series(
            [frozenset({"REC", "OK"}), frozenset({"REC", "OK"})],
            index=[idx[100], idx[1300]],
        )
        return close, m

    def test_zerlegt_in_zwei_spalten(self):
        close, m = self._panel()
        (t,) = unterbrechungen(close, min_luecke=MIN_LUECKE)
        neu, m_neu, _, prot = trenne(close, m, [t])
        assert "REC#2" in neu.columns
        assert prot["REC"]["n_segmente"] == 2
        # Firma A endet vor dem Schnitt, Firma B beginnt dort.
        assert neu["REC"].dropna().iloc[-1] == 80.0
        assert neu["REC#2"].dropna().iloc[0] == 0.1
        assert neu["REC"].dropna().index[-1] < neu["REC#2"].dropna().index[0]

    def test_last_valid_liegt_danach_am_echten_ende(self):
        """Das ist der ganze Zweck: die Delisting-Regel muss greifen können.

        Vorher lag `last_valid` am Ende der Serie von Firma B — der
        Zwangsverkauf prüft `last_valid < t` und war nie erfüllt.
        """
        close, m = self._panel()
        (t,) = unterbrechungen(close, min_luecke=MIN_LUECKE)
        neu, *_ = trenne(close, m, [t])
        assert close["REC"].last_valid_index() > neu["REC"].last_valid_index()

    def test_unbeteiligte_spalten_bleiben_gleich(self):
        close, m = self._panel()
        (t,) = unterbrechungen(close, min_luecke=MIN_LUECKE)
        neu, *_ = trenne(close, m, [t])
        assert neu["OK"].equals(close["OK"])

    def test_mitgliedschaft_wandert_mit(self):
        """Sonst wählt die Engine `REC` zu Terminen, an denen dort NaN steht."""
        close, m = self._panel()
        (t,) = unterbrechungen(close, min_luecke=MIN_LUECKE)
        _, m_neu, _, _ = trenne(close, m, [t])
        frueh, spaet = m_neu.iloc[0], m_neu.iloc[-1]
        assert "REC" in frueh and "REC#2" not in frueh
        assert "REC#2" in spaet and "REC" not in spaet

    def test_dividenden_werden_mitgetrennt(self):
        close, m = self._panel()
        div = pd.DataFrame(
            {"REC": pd.Series(1.0, index=close.index), "OK": 0.0}, index=close.index
        )
        (t,) = unterbrechungen(close, min_luecke=MIN_LUECKE)
        _, _, div_neu, _ = trenne(close, m, [t], div)
        assert "REC#2" in div_neu.columns
        # Keine Dividende darf doppelt stehen ...
        beide = div_neu[["REC", "REC#2"]].notna().sum(axis=1)
        assert int(beide.max()) == 1
        # ... und keine darf VERSCHWINDEN. Die erste Fassung prüfte nur die
        # Doppelbuchung; „Firma B bekommt gar keine Dividenden" überlebte den
        # Test, weil REC in der Frühphase die 1 lieferte (F-senior-6). Ein
        # stiller Dividendenverlust hätte den p12i-Vergleich einseitig
        # verschoben — zulasten des getrennten Panels.
        assert div_neu["REC#2"].notna().any()
        assert div_neu[["REC", "REC#2"]].sum().sum() == pytest.approx(div["REC"].sum())

    def test_mehrfach_vergebenes_symbol_wird_vollstaendig_zerlegt(self):
        """Der Fall, an dem die erste Fassung scheiterte (RYC, drei Segmente)."""
        idx = _idx(2200)
        close = pd.DataFrame(
            {"M": _serie(idx, [(0, 100, 50.0), (900, 100, 5.0), (1900, 100, 200.0)])}
        )
        # Mitgliedschaft in ALLEN DREI Segmenten — sonst bleibt die Zuordnung
        # ungetestet, und `passend[0]` statt `passend[-1]` überlebt (F-senior-7).
        m = pd.Series([frozenset({"M"})] * 3, index=[idx[50], idx[950], idx[1950]])
        (t,) = unterbrechungen(close, min_luecke=MIN_LUECKE)
        neu, m_neu, _, prot = trenne(close, m, [t])
        assert prot["M"]["n_segmente"] == 3
        assert {"M", "M#2", "M#3"} <= set(neu.columns)
        # Jeder Termin gehört zu SEINEM Segment, nicht pauschal zum ersten.
        assert m_neu.iloc[0] == frozenset({"M"})
        assert m_neu.iloc[1] == frozenset({"M#2"})
        assert m_neu.iloc[2] == frozenset({"M#3"})

    def test_unvollstaendige_zerlegung_kracht(self, monkeypatch):
        """Fail-closed: bleibt eine Lücke stehen, war der Schnitt falsch.

        Genau dieser Wächter hat den RYC-Fall gefunden — damals schnitt
        ``trenne`` nur an der längsten Lücke. Seit ``segmente`` **alle** Lücken
        liefert, ist der Guard bei korrekter Verwendung unerreichbar; er sichert
        gegen eine Regression dort. Deshalb wird hier ``segmente`` verkürzt,
        nicht die Schwelle verstellt — sonst prüfte der Test nichts.
        """
        import research.mandat2.panel_getrennt as pg

        idx = _idx(2200)
        close = pd.DataFrame(
            {"M": _serie(idx, [(0, 100, 50.0), (900, 100, 5.0), (1900, 100, 200.0)])}
        )
        m = pd.Series([frozenset({"M"})], index=[idx[50]])
        echte = pg.segmente
        monkeypatch.setattr(pg, "segmente", lambda c, s, ml: echte(c, s, ml)[:1])
        with pytest.raises(SystemExit, match="nicht sauber zerlegt"):
            trenne(close, m, [{"symbol": "M"}], min_luecke=MIN_LUECKE)


class TestWirkungsmessung:
    def _bestand(self, tage: list[str], sym: str) -> dict[str, set[str]]:
        return {t: {sym} for t in tage}

    def test_tote_haltedauer_zaehlt_nur_tage_ohne_kurs(self):
        treffer = [
            {
                "symbol": "X",
                "letzter_kurs_am": "2005-01-03",
                "naechster_kurs_am": "2010-01-04",
            }
        ]
        bestand = self._bestand(
            ["2004-12-01", "2006-06-01", "2008-06-01", "2010-01-04", "2011-01-04"], "X"
        )
        (v,) = tote_haltedauer(treffer, bestand).values()
        # Nur 2006 und 2008 liegen strikt zwischen den Kursen.
        assert v["tote_haltetage"] == 2
        assert v["letzter_echter_kurs"] == "2005-01-03"

    def test_nicht_gehaltener_name_taucht_nicht_auf(self):
        treffer = [
            {
                "symbol": "X",
                "letzter_kurs_am": "2005-01-03",
                "naechster_kurs_am": "2010-01-04",
            }
        ]
        assert tote_haltedauer(treffer, {"2006-06-01": {"Y"}}) == {}

    def test_wirkung_meldet_fehlende_rendite_statt_zu_schweigen(self):
        """E-103: der Ausfallmodus darf nicht die beruhigende Antwort sein."""
        idx = pd.to_datetime(["2010-01-04", "2010-01-05"]).tz_localize("UTC")
        equity = pd.Series([100.0, 110.0], index=idx)
        treffer = [{"symbol": "X", "naechster_kurs_am": "2099-01-01", "faktor": 2.0}]
        bestand = {"2098-12-31": {"X"}}
        (v,) = wirkung(treffer, bestand, equity).values()
        assert v["gehalten_am_vortag"] is True
        assert v["rendite"] is None
        assert "keine Portfolio-Rendite" in v["hinweis"]

    def test_wirkung_beziffert_den_wiedereinstiegstag(self):
        idx = pd.bdate_range("2010-01-04", periods=5, tz="UTC")
        equity = pd.Series([100.0, 100.0, 90.0, 90.0, 90.0], index=idx)
        tag = f"{idx[2]:%Y-%m-%d}"
        treffer = [{"symbol": "X", "naechster_kurs_am": tag, "faktor": 0.1}]
        bestand = {f"{idx[1]:%Y-%m-%d}": {"X"}}
        (v,) = wirkung(treffer, bestand, equity).values()
        assert v["portfolio_tagesrendite"] == pytest.approx(-0.10)
        assert v["kurssprung"] == 0.1


class TestInstrumentierungFaellt_Laut:
    """MAJOR-1: der Fail-loud-Wächter war selbst ungetestet.

    Beide Mutationen überlebten: Wächter entfernen und Patch gar nicht
    installieren. Schlimmer noch war sein Zuschnitt — er zählte nur die
    *Anzahl* protokollierter Tage. Feuert der Patch pro Tag, liefert aber
    durchweg leere Mengen, war die Bedingung erfüllt und er schwieg. Genau der
    Zustand macht jede Wirkungsmessung wertlos (E-103).
    """

    class _Daten:
        def __init__(self, n: int):
            self.close = pd.DataFrame({"X": [1.0] * n}, index=_idx(n))

    def test_unvollstaendiges_protokoll_kracht(self):
        from research.mandat2.p12e_panel_hygiene import pruefe_protokoll

        d = self._Daten(10)
        teil = {f"{t:%Y-%m-%d}": {"X"} for t in d.close.index[:5]}
        with pytest.raises(SystemExit, match="unvollstaendig"):
            pruefe_protokoll(teil, d)

    def test_leeres_protokoll_kracht(self):
        from research.mandat2.p12e_panel_hygiene import pruefe_protokoll

        with pytest.raises(SystemExit, match="unvollstaendig"):
            pruefe_protokoll({}, self._Daten(10))

    def test_vollstaendig_aber_durchweg_LEER_kracht_ebenfalls(self):
        """Der Fall, den die erste Fassung des Wächters durchließ."""
        from research.mandat2.p12e_panel_hygiene import pruefe_protokoll

        d = self._Daten(10)
        leer = {f"{t:%Y-%m-%d}": set() for t in d.close.index}
        with pytest.raises(SystemExit, match="an JEDEM Tag leer"):
            pruefe_protokoll(leer, d)

    def test_gueltiges_protokoll_geht_durch(self):
        from research.mandat2.p12e_panel_hygiene import pruefe_protokoll

        d = self._Daten(10)
        gut = {
            f"{t:%Y-%m-%d}": ({"X"} if i % 2 else set())
            for i, t in enumerate(d.close.index)
        }
        pruefe_protokoll(gut, d)  # kein Krach

    def test_gehaltene_namen_gibt_es_nur_einmal(self):
        """MAJOR-2: zwei Fassungen derselben Messung sind eine zweite Wahrheit."""
        from research.mandat2 import p12e_panel_hygiene as e
        from research.mandat2 import p12h_ticker_recycling as h

        assert h.gehaltene_namen is e.gehaltene_namen


class TestWirkungBestandspruefung:
    """MAJOR-3: `vortage[-1]` -> `vortage[0]` überlebte.

    Die alten Tests hatten genau EINEN Tag im Bestand — dort sind erstes und
    letztes Element gleich. Im echten Lauf mit 5.548 Tagen hätte die Mutation
    gefragt „lag der Name am allerersten Kampagnentag im Bestand", also
    praktisch nie: CGP, NGH und NVLS wären herausgefallen und `wirkung` hätte
    {} geliefert. Falsche Entwarnung.
    """

    def _equity(self):
        idx = pd.bdate_range("2010-01-04", periods=6, tz="UTC")
        return pd.Series([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], index=idx), idx

    def test_es_zaehlt_der_VORTAG_nicht_irgendein_frueherer_tag(self):
        equity, idx = self._equity()
        tag = f"{idx[3]:%Y-%m-%d}"
        treffer = [{"symbol": "X", "naechster_kurs_am": tag, "faktor": 0.1}]
        # Am ersten Tag im Bestand, am Vortag des Wiedereinstiegs NICHT.
        bestand = {
            f"{idx[0]:%Y-%m-%d}": {"X"},
            f"{idx[1]:%Y-%m-%d}": set(),
            f"{idx[2]:%Y-%m-%d}": set(),
        }
        assert wirkung(treffer, bestand, equity) == {}

    def test_am_vortag_gehalten_wird_erfasst(self):
        equity, idx = self._equity()
        tag = f"{idx[3]:%Y-%m-%d}"
        treffer = [{"symbol": "X", "naechster_kurs_am": tag, "faktor": 0.1}]
        bestand = {
            f"{idx[0]:%Y-%m-%d}": set(),
            f"{idx[1]:%Y-%m-%d}": set(),
            f"{idx[2]:%Y-%m-%d}": {"X"},
        }
        (v,) = wirkung(treffer, bestand, equity).values()
        assert v["portfolio_tagesrendite"] == pytest.approx(-0.10)

    def test_nicht_gehaltener_name_faellt_heraus(self):
        """Für `tote_haltedauer` gab es diesen Test, für `wirkung` nicht."""
        equity, idx = self._equity()
        treffer = [
            {"symbol": "X", "naechster_kurs_am": f"{idx[3]:%Y-%m-%d}", "faktor": 0.1}
        ]
        bestand = {f"{t:%Y-%m-%d}": {"Y"} for t in idx[:3]}
        assert wirkung(treffer, bestand, equity) == {}


class TestFehltrefferKennzeichnung:
    """MAJOR-4: die 8-von-30-Zahl ruhte auf ungetestetem Code.

    `faktor_nahe_eins` immer auf False zu setzen ließ p12i „0 wahrscheinliche
    Fehltreffer" drucken — die E-117-Offenlegung wäre lautlos verschwunden.
    """

    def _mit_faktor(self, kurs_nach: float):
        idx = _idx(1400)
        close = pd.DataFrame(
            {"R": _serie(idx, [(0, 200, 80.0), (1200, 200, kurs_nach)])}
        )
        m = pd.Series([frozenset({"R"})], index=[idx[100]])
        (t,) = unterbrechungen(close, min_luecke=MIN_LUECKE)
        _, _, _, prot = trenne(close, m, [t])
        return prot["R"]

    def test_kurs_setzt_fort_wird_als_fehltreffer_markiert(self):
        info = self._mit_faktor(72.0)  # Faktor 0,9 — Firma laeuft weiter
        assert info["faktoren"][0] == pytest.approx(0.9)
        assert info["faktor_nahe_eins"] == [True]

    def test_echter_bruch_wird_nicht_markiert(self):
        info = self._mit_faktor(0.1)  # Faktor 0,00125
        assert info["faktor_nahe_eins"] == [False]

    def test_bandgrenzen(self):
        """Genau auf 0,5 und 2,0 gilt noch als „setzt fort"."""
        assert self._mit_faktor(40.0)["faktor_nahe_eins"] == [True]  # 0,5
        assert self._mit_faktor(160.0)["faktor_nahe_eins"] == [True]  # 2,0
        assert self._mit_faktor(39.0)["faktor_nahe_eins"] == [False]
        assert self._mit_faktor(161.0)["faktor_nahe_eins"] == [False]


class TestSchwellenraender:
    """MINOR-1: die alten Tests hatten fünffachen Abstand zur Schwelle.

    Eine Lücke von 101 gegen Schwelle 500 pinnt den Rand nicht — `>=` zu `>`
    zu mutieren blieb folgenlos.
    """

    def _mit_luecke(self, luecke: int):
        n = 200 + luecke + 200
        idx = _idx(n)
        return pd.DataFrame(
            {"X": _serie(idx, [(0, 200, 50.0), (200 + luecke, 200, 5.0)])}
        )

    def test_genau_auf_der_schwelle_zaehlt(self):
        px = self._mit_luecke(MIN_LUECKE - 1)  # diff == MIN_LUECKE
        (t,) = unterbrechungen(px, min_luecke=MIN_LUECKE)
        assert t["luecke_handelstage"] == MIN_LUECKE
        assert len(segmente(px, "X", MIN_LUECKE)) == 1

    def test_einen_tag_darunter_zaehlt_nicht(self):
        px = self._mit_luecke(MIN_LUECKE - 2)  # diff == MIN_LUECKE - 1
        assert unterbrechungen(px, min_luecke=MIN_LUECKE) == []
        assert segmente(px, "X", MIN_LUECKE) == []


class TestUngetesteteHelfer:
    """MINOR-2/3/4 aus der Stage-1-Prüfung — Coverage-Löcher schließen."""

    def test_betroffene_auswahl_zaehlt_mitgliedschaftstermine(self):
        from research.mandat2.p12h_ticker_recycling import betroffene_auswahl

        idx = _idx(3)
        m = pd.Series(
            [frozenset({"A", "B"}), frozenset({"A"}), frozenset({"C"})], index=idx
        )
        aus = betroffene_auswahl(m, {"A", "C", "NIE"})
        assert aus["A"]["n_termine_mitglied"] == 2
        assert aus["C"]["n_termine_mitglied"] == 1
        assert "NIE" not in aus  # nie Mitglied -> harmlos, taucht nicht auf

    def test_rohdaten_lage_trennt_fehlt_im_panel_von_fehlt_ueberall(self, tmp_path):
        from research.mandat2.p12h_ticker_recycling import rohdaten_lage

        idx = _idx(5)
        # Panel kennt nur A; die Rohdatei zusätzlich B; C fehlt überall.
        close = pd.DataFrame({"A": [1.0] * 5}, index=idx)
        m = pd.Series([frozenset({"A", "B", "C"})], index=[idx[0]])
        roh = pd.DataFrame(
            {
                "timestamp": list(idx) * 2,
                "symbol": ["A"] * 5 + ["B"] * 5,
                "close": [1.0] * 10,
            }
        )
        p = tmp_path / "roh.parquet"
        roh.to_parquet(p)
        lage = rohdaten_lage(m, close, p)
        assert lage["n_pit_mitglieder"] == 3
        assert lage["n_im_panel"] == 1
        assert lage["n_fehlt_im_panel"] == 2
        assert lage["n_fehlt_aber_in_rohdatei"] == 1  # B
        assert lage["n_fehlt_ueberall"] == 1  # C
        assert lage["beispiele"]["B"]["im_suchfenster"] == 5

    def test_p12h_schwelle_ist_gepinnt(self):
        """Sie landet als `min_luecke_handelstage` im Artefakt."""
        from research.mandat2.p12h_ticker_recycling import MIN_LUECKE as DETEKTOR

        assert DETEKTOR == 120

    def test_fehlender_bestandstag_wird_gemeldet_statt_verschwiegen(self):
        """MINOR-4: ein stiller Skip ist von „nicht betroffen" nicht zu trennen."""
        idx = pd.bdate_range("2010-01-04", periods=3, tz="UTC")
        equity = pd.Series([100.0, 100.0, 100.0], index=idx)
        treffer = [
            {"symbol": "X", "naechster_kurs_am": f"{idx[0]:%Y-%m-%d}", "faktor": 0.5}
        ]
        (v,) = wirkung(treffer, {f"{idx[2]:%Y-%m-%d}": {"X"}}, equity).values()
        assert v["gehalten_am_vortag"] is None
        assert "kein Bestandstag" in v["hinweis"]
