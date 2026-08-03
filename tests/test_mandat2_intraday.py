"""Tests fuer den Intraday-Strang (Mandat II, P12).

Die erste Fassung dieser Datei prueft die Bereinigung nur NACHGEBAUT — sie rief
``load_intraday`` nie auf. Ein Mutationstest der Stage-1-Review zeigte, was das
kostet: ein um eine Bar in die Zukunft verschobener Score (echtes Look-ahead)
und das Entfernen des Holdout-Schnitts ueberlebten die Suite unbemerkt
(Finding F-test-7). Deshalb laeuft der Produktionspfad hier jetzt gegen ein
Mini-Fixture aus echten Parquet-Dateien.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import research.mandat2.intraday_data as idata
from research.mandat2.intraday_data import (
    BOERSE,
    MIN_ABDECKUNG,
    RTH_BIS,
    RTH_VON,
    STUFEN_SCHWELLE,
    _n_spruenge,
    _split_diagnose,
    _stufig_machen,
    load_intraday,
)
from research.mandat2.p12_intraday_haltedauer import (
    buy_and_hold,
    kennzahlen,
    momentum,
    rebalanciert,
    simuliere,
    zufall,
)


def _stunden(n: int, start: str = "2010-01-04 09:00") -> pd.DatetimeIndex:
    """n Handelsstunden innerhalb der regulaeren Sitzung, als UTC-Index.

    Aufgebaut wird in BOERSENzeit — genau die Umstellung, die der erste
    Entwurf verpasst hatte (F-test-4).
    """
    idx: list[pd.Timestamp] = []
    t = pd.Timestamp(start, tz=BOERSE)
    while len(idx) < n:
        if RTH_VON <= t.hour <= RTH_BIS and t.dayofweek < 5:
            idx.append(t)
        t += pd.Timedelta(hours=1)
    return pd.DatetimeIndex(idx).tz_convert("UTC")


class TestSprungDetektor:
    def test_erkennt_konstruierten_split(self):
        s = pd.Series(100.0, index=_stunden(40))
        s.iloc[20:] = 50.0
        assert _n_spruenge(pd.DataFrame({"X": s})) == 1

    def test_ignoriert_normale_bewegung(self):
        rng = np.random.default_rng(0)
        s = pd.Series(
            100.0 * np.cumprod(1 + rng.normal(0, 0.004, 40)), index=_stunden(40)
        )
        assert _n_spruenge(pd.DataFrame({"X": s})) == 0

    def test_schwelle_wirkt(self):
        s = pd.Series(100.0, index=_stunden(10))
        s.iloc[5:] = 60.0
        df = pd.DataFrame({"X": s})
        assert _n_spruenge(df, schwelle=0.35) == 1
        assert _n_spruenge(df, schwelle=0.45) == 0

    def test_split_diagnose_nennt_symbol_und_zeitpunkt(self):
        s = pd.Series(100.0, index=_stunden(20))
        s.iloc[10:] = 50.0
        d = _split_diagnose(pd.DataFrame({"X": s}))
        assert len(d) == 1
        assert d[0]["symbol"] == "X"
        assert d[0]["roher_sprung"] == pytest.approx(-0.5)


class TestStufenFaktor:
    """Die Gegenprobe, auf der eine tragende Aussage des Befunds ruht.

    Der Satz „die Bruttokante ist KEIN Verfahrensartefakt" haengt allein an
    ``_stufig_machen``. Waere die Funktion ein No-op — Schwelle zu klein, ``lauf``
    nie aktualisiert —, waere die Gegenprobe eine Kopie des Originals, und das
    Ergebnis saehe GENAUSO aus (kleine Delta). Ohne diese Tests kann der Befund
    „Artefakt widerlegt" nicht von „Gegenprobe kaputt" unterscheiden
    (Stage-3-Finding F-auditor-3).
    """

    def test_kleiner_drift_wird_unterdrueckt(self):
        idx = _stunden(50)
        f = pd.Series(np.linspace(1.0, 1.0 + STUFEN_SCHWELLE * 0.5, 50), index=idx)
        aus = _stufig_machen(f)
        assert aus.nunique() == 1, "Drift unter der Schwelle darf keine Stufe erzeugen"
        assert aus.iloc[0] == pytest.approx(f.iloc[0])

    def test_echter_sprung_erzeugt_genau_eine_stufe(self):
        idx = _stunden(40)
        f = pd.Series(1.0, index=idx)
        f.iloc[20:] = 0.5
        aus = _stufig_machen(f)
        assert aus.nunique() == 2
        assert aus.iloc[19] == pytest.approx(1.0)
        assert aus.iloc[20] == pytest.approx(0.5)

    def test_ist_kein_noop(self):
        """Der entscheidende Test: die Funktion muss etwas VERAENDERN."""
        idx = _stunden(60)
        rng = np.random.default_rng(1)
        f = pd.Series(
            np.cumprod(1 + rng.normal(0, STUFEN_SCHWELLE * 0.2, 60)), index=idx
        )
        aus = _stufig_machen(f)
        assert not np.allclose(aus.to_numpy(), f.to_numpy())
        assert aus.nunique() < f.nunique()

    def test_stueckweise_konstant(self):
        idx = _stunden(80)
        rng = np.random.default_rng(2)
        f = pd.Series(
            np.cumprod(1 + rng.normal(0, STUFEN_SCHWELLE * 0.3, 80)), index=idx
        )
        aus = _stufig_machen(f)
        wechsel = (aus.diff().abs() > 1e-15).sum()
        assert wechsel < len(f) / 2, "Treppe muss deutlich weniger Wechsel haben"


class TestLoadIntradayEchterPfad:
    """Der Produktionspfad, gegen ein Mini-Fixture aus echten Parquet-Dateien.

    Geprueft wird, was der Mutationstest als ungedeckt entlarvt hat: der
    Holdout-Schnitt, die Wirksamkeit der Split-Bereinigung und der
    Sitzungsfilter.
    """

    @pytest.fixture
    def fixture_repo(self, tmp_path, monkeypatch):
        """Ein Symbol mit konstruiertem Split UND Bars jenseits des Cutoffs."""
        idx = _stunden(2450, start="2015-01-05 09:00")  # 350 Handelstage > MIN 250
        roh = pd.Series(np.linspace(200.0, 300.0, len(idx)), index=idx)
        # 2:1-Split in der Mitte -> alle spaeteren Rohkurse halbiert
        split_ab = len(idx) // 2
        roh.iloc[split_ab:] = roh.iloc[split_ab:] / 2.0
        df = pd.DataFrame({"close": roh.to_numpy()}, index=idx)
        df.index.name = "ts"
        (tmp_path / "TEST.parquet").parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(tmp_path / "TEST.parquet")
        monkeypatch.setattr(idata, "ROH", tmp_path)

        # Tagesanker: der SPLITBEREINIGTE Kurs, wie ihn das Tagespanel fuehrt.
        adj = roh.copy()
        adj.iloc[:split_ab] = adj.iloc[:split_ab] / 2.0
        tage = pd.Index(adj.index.tz_convert("UTC").date)
        anker = adj.groupby(tage).last()
        cutoff = pd.Timestamp("2016-12-31").date()
        anker = anker[anker.index <= cutoff]
        tages = pd.DataFrame(
            {"TEST": anker.to_numpy()},
            index=pd.DatetimeIndex(pd.to_datetime(list(anker.index)), tz="UTC"),
        )

        class _Camp:
            close = tages

        monkeypatch.setattr(idata, "load_campaign", lambda: _Camp())
        return tmp_path

    def test_holdout_wird_abgeschnitten(self, fixture_repo):
        d = load_intraday(["TEST"])
        assert d.close.index.max() <= pd.Timestamp("2016-12-31", tz="UTC")

    def test_split_verschwindet_durch_bereinigung(self, fixture_repo):
        d = load_intraday(["TEST"])
        assert d.roh_spruenge >= 1, "Fixture muss einen Rohsprung enthalten"
        assert d.rest_spruenge == 0, "Bereinigung hat den Split nicht entfernt"

    def test_nur_regulaere_sitzung(self, fixture_repo):
        d = load_intraday(["TEST"])
        et = d.close.index.tz_convert(BOERSE)
        assert et.hour.min() >= RTH_VON
        assert et.hour.max() <= RTH_BIS

    def test_fehlendes_symbol_wird_protokolliert_nicht_verschluckt(self, fixture_repo):
        d = load_intraday(["TEST", "GIBTESNICHT"])
        assert "GIBTESNICHT" in d.verworfen
        assert list(d.close.columns) == ["TEST"]

    def test_stufig_flag_wird_durchgereicht(self, fixture_repo):
        assert load_intraday(["TEST"], stufig=True).stufig is True
        assert load_intraday(["TEST"]).stufig is False

    def test_duenn_abgedecktes_symbol_faellt_aus_BEIDEN_seiten(
        self, fixture_repo, tmp_path, monkeypatch
    ):
        """Benchmark-Symmetrie: ein Name, den Buy-and-Hold nicht kaufen kann,
        darf auch fuer den Kandidaten nicht waehlbar sein (E-079, F-senior-5)."""
        voll = pd.read_parquet(tmp_path / "TEST.parquet")
        # Genug gemeinsame Tage, um den Ueberlappungsfilter zu passieren, aber
        # unter MIN_ABDECKUNG — sonst prueft der Test den falschen Guard.
        anteil = MIN_ABDECKUNG - 0.04
        duenn = voll.iloc[: int(len(voll) * anteil)].copy()
        assert len(duenn) / 7 > 250, "Fixture muss den Ueberlappungsfilter passieren"
        duenn.to_parquet(tmp_path / "DUENN.parquet")

        anker = idata.load_campaign().close
        anker2 = anker.copy()
        anker2["DUENN"] = anker2["TEST"]

        class _Camp:
            close = anker2

        monkeypatch.setattr(idata, "load_campaign", lambda: _Camp())
        d = load_intraday(["TEST", "DUENN"])
        assert "DUENN" in d.verworfen
        assert "Abdeckung" in d.verworfen["DUENN"]
        assert "DUENN" not in d.close.columns


class TestSimulation:
    @pytest.fixture
    def panel(self) -> pd.DataFrame:
        idx = _stunden(400)
        rng = np.random.default_rng(7)
        return pd.DataFrame(
            {
                f"S{i}": 100.0 * np.cumprod(1 + rng.normal(0.0001, 0.005, len(idx)))
                for i in range(8)
            },
            index=idx,
        )

    def test_score_nutzt_keine_zukunft(self, panel):
        """PIT per Zukunfts-RANDOMISIERUNG, nicht per Randabschneidung.

        Die erste Fassung verglich einen vollen mit einem abgeschnittenen Lauf.
        Das faengt Panelrand-Effekte, aber kein Look-ahead: eine Mutation
        ``score.iloc[t] -> score.iloc[t+1]`` ueberlebte sie nachweislich, weil
        t+1 auch im kurzen Panel existiert (Stage-2-Finding F-senior-6).

        Hier wird der Score NACH Bar k verfaelscht. Nutzt die Entscheidung an
        einem Termin <= k auch nur eine Bar Zukunft, aendert sich die Equity bis
        k — und der Test faellt.
        """
        sc = momentum(panel, 20)
        k = 250
        verfaelscht = sc.copy()
        rng = np.random.default_rng(99)
        verfaelscht.iloc[k + 1 :] = rng.standard_normal(verfaelscht.iloc[k + 1 :].shape)
        a, _, _, _ = simuliere(panel, halte_bars=10, score=sc, top_k=3, start=20)
        b, _, _, _ = simuliere(
            panel, halte_bars=10, score=verfaelscht, top_k=3, start=20
        )
        bis = a.index <= panel.index[k]
        assert a[bis].to_numpy() == pytest.approx(b[bis].to_numpy())
        assert a.iloc[-1] != pytest.approx(b.iloc[-1]), (
            "Verfaelschung muss NACH k wirken, sonst prueft der Test nichts"
        )

    def test_kuerzere_haltedauer_erzeugt_mehr_umschichtungen(self, panel):
        """Der Parameter muss WIRKEN — die Lehre aus E-072."""
        sc = momentum(panel, 20)
        _, _, kurz, _ = simuliere(panel, halte_bars=2, score=sc, top_k=3, start=20)
        _, _, lang, _ = simuliere(panel, halte_bars=40, score=sc, top_k=3, start=20)
        assert kurz > lang > 0

    def test_positionen_driften_zwischen_terminen(self, panel):
        """Zwischen zwei Terminen darf NICHT rebalanciert werden (F-test-5).

        Bei nur einem Termin ueber das ganze Fenster muss das Ergebnis exakt
        einem Kauf-und-Halten der gewaehlten Titel entsprechen.
        """
        sc = pd.DataFrame(
            [[1.0, 0.9, 0.8] + [0.0] * 5] * len(panel),
            index=panel.index,
            columns=panel.columns,
        )
        n, _, _, _ = simuliere(
            panel, halte_bars=len(panel), score=sc, top_k=3, kosten_bps=0.0, start=0
        )
        erwartet = (panel.iloc[:, :3] / panel.iloc[0, :3] / 3.0).sum(axis=1)
        assert n.iloc[-1] == pytest.approx(float(erwartet.iloc[-1]), rel=1e-12)

    def test_kosten_senken_netto_unter_brutto(self, panel):
        n, b, _, _ = simuliere(
            panel,
            halte_bars=2,
            score=momentum(panel, 20),
            top_k=3,
            kosten_bps=10.0,
            start=20,
        )
        assert n.iloc[-1] < b.iloc[-1]

    def test_ohne_kosten_faellt_netto_mit_brutto_zusammen(self, panel):
        n, b, _, _ = simuliere(
            panel,
            halte_bars=4,
            score=momentum(panel, 20),
            top_k=3,
            kosten_bps=0.0,
            start=20,
        )
        assert n.iloc[-1] == pytest.approx(b.iloc[-1], rel=1e-12)

    def test_kostenlast_steigt_mit_handelsfrequenz(self, panel):
        sc = momentum(panel, 20)
        lasten = []
        for bars in (1, 4, 40):
            n, b, _, _ = simuliere(panel, halte_bars=bars, score=sc, top_k=3, start=20)
            lasten.append(b.iloc[-1] / n.iloc[-1] - 1.0)
        assert lasten[0] > lasten[1] > lasten[2]

    def test_totalverlust_wird_gezeigt_nicht_maskiert(self):
        """Ein wertloser Titel muss als 0 erscheinen (F-test-8).

        Der frueher hier stehende Test pruefte ``(n > 0).all()`` — genau die
        Eigenschaft, die die damalige ``replace(0, nan).ffill()``-Zeile
        faelschte. Er konnte deshalb nie fehlschlagen.
        """
        idx = _stunden(20)
        px = pd.DataFrame({"A": np.linspace(100.0, 0.0, 20)}, index=idx)
        sc = pd.DataFrame(1.0, index=idx, columns=["A"])
        n, _, _, _ = simuliere(px, halte_bars=5, score=sc, top_k=1, kosten_bps=0.0)
        assert n.iloc[-1] == pytest.approx(0.0, abs=1e-12)

    def test_zufallsscore_ist_deterministisch(self, panel):
        assert zufall(panel, 3).equals(zufall(panel, 3))
        assert not zufall(panel, 3).equals(zufall(panel, 4))

    def test_zufallsscore_erbt_nan_maske(self, panel):
        """Ohne diese Maske vergleicht P12b Startzeitpunkte (F-test-3)."""
        sig = momentum(panel, 20)
        z = zufall(panel, 0, wie=sig)
        assert z.isna().equals(sig.isna())
        assert z.notna().any().any()

    def test_zu_wenige_kandidaten_bleiben_in_cash(self):
        """Fail-closed: lieber nicht investiert als willkuerlich investiert."""
        idx = _stunden(50)
        panel = pd.DataFrame({"A": np.linspace(100, 110, 50)}, index=idx)
        n, _, _, _ = simuliere(panel, halte_bars=4, score=momentum(panel, 5), top_k=5)
        assert n.iloc[-1] == pytest.approx(1.0)


class TestKennzahlen:
    """`kennzahlen` erzeugt JEDE berichtete Zahl und war ungetestet (F-senior-9)."""

    def test_cagr_und_endwert_gegen_bekannte_kurve(self):
        idx = _stunden(100)
        e = pd.Series(np.linspace(1.0, 2.0, 100), index=idx)
        k = kennzahlen(e, jahre=2.0)
        assert k["endwert"] == pytest.approx(2.0)
        assert k["cagr"] == pytest.approx(2.0**0.5 - 1.0)

    def test_maxdd_gegen_bekannten_einbruch(self):
        idx = _stunden(5)
        e = pd.Series([1.0, 1.0, 0.6, 0.8, 1.2], index=idx)
        assert kennzahlen(e, jahre=1.0)["maxdd"] == pytest.approx(-0.4)

    def test_normiert_auf_den_startwert(self):
        """Die Equity beginnt bei 1.0 — aber verlassen darf sich niemand darauf."""
        idx = _stunden(3)
        assert kennzahlen(pd.Series([2.0, 3.0, 4.0], index=idx), jahre=1.0)[
            "endwert"
        ] == pytest.approx(2.0)


class TestBenchmark:
    @pytest.fixture
    def panel(self) -> pd.DataFrame:
        idx = _stunden(100)
        return pd.DataFrame(
            {"A": np.linspace(100, 200, 100), "B": np.linspace(50, 75, 100)}, index=idx
        )

    def test_buy_and_hold_rebalanciert_nicht(self, panel):
        bh = buy_and_hold(panel, kosten_bps=0.0)
        erwartet = 0.5 * (200 / 100) + 0.5 * (75 / 50)
        assert bh.iloc[-1] / bh.iloc[0] == pytest.approx(erwartet, rel=1e-12)

    def test_rebalanciert_haelt_alle_namen(self, panel):
        """Konstanter Score -> ALLE Namen, nicht n-1 (F-senior-5)."""
        reb = rebalanciert(panel, alle_bars=10, kosten_bps=0.0)
        bh = buy_and_hold(panel, kosten_bps=0.0)
        # Beide starten gleichgewichtet; nach dem ersten Bar duerfen sie
        # auseinanderlaufen, aber keiner darf einen Namen ausgelassen haben.
        assert reb.iloc[0] == pytest.approx(bh.iloc[0], rel=1e-9)
        assert reb.iloc[-1] > 0

    def test_buy_and_hold_ignoriert_namen_ohne_startkurs(self, panel):
        panel = panel.copy()
        panel.loc[panel.index[0], "B"] = np.nan
        bh = buy_and_hold(panel, kosten_bps=0.0)
        assert bh.iloc[-1] / bh.iloc[0] == pytest.approx(200 / 100, rel=1e-12)


class TestBefundRenderer:
    """Der Renderer ist die einzige strukturelle Sperre gegen E-085 — und war
    selbst ungetestet (Stage-2-Finding F-senior-7).

    Geprueft wird nicht die Prosa, sondern dass die datenABHAENGIGEN Aussagen
    tatsaechlich von den Daten abhaengen: dreht man die Zahlen um, muss sich der
    Text mitdrehen. Ein Generator, der immer dasselbe schreibt, waere keine
    Sperre, sondern eine Attrappe.
    """

    @pytest.fixture
    def res(self, tmp_path, monkeypatch):
        import research.mandat2.render_befund_p12 as rb

        monkeypatch.setattr(rb, "RES", tmp_path)
        monkeypatch.setattr(rb, "ZIEL", tmp_path / "BEFUND.md")
        return tmp_path, rb

    @staticmethod
    def _zeile(bars, name, netto, brutto, zuf_n, zuf_b):
        return {
            "halte_bars": bars,
            "name": name,
            "rueckblick_bars": 100,
            "umschichtungen": 10,
            "netto_end": netto,
            "brutto_end": brutto,
            "cagr": 0.1,
            "maxdd": -0.4,
            "zufall_end_mittel": zuf_n,
            "zufall_end_alle": [zuf_n] * 5,
            "zufall_brutto_mittel": zuf_b,
            "anteil_cash": 0.0,
            "kostenlast": 0.2,
        }

    def _basis(self, netto=1.5, brutto=2.0, zuf_b=3.0):
        return {
            "universum": ["A", "B"],
            "verworfen": {"C": "Abdeckung nur 54.5%"},
            "fenster": "2006-01-01..2016-12-30",
            "jahre": 10.5,
            "warmup_bars": 100,
            "bars_pro_tag": 7,
            "kosten_bps": 10.0,
            "top_k": 5,
            "roh_spruenge": 16,
            "rest_spruenge": 8,
            "split_diagnose": [
                {"symbol": "A", "zeitpunkt": "2010-01-01 13:00", "roher_sprung": -0.5}
            ],
            "buy_and_hold": {"endwert": 3.138, "cagr": 0.115, "maxdd": -0.547},
            "ew_rebalanciert": {"endwert": 3.244, "cagr": 0.118, "maxdd": -0.616},
            "familien": {
                "A_fest": [self._zeile(1, "1 Stunde", netto, brutto, 0.0, zuf_b)]
            },
        }

    def _render(self, res, daten):
        tmp, rb = res
        (tmp / "p12_intraday_haltedauer.json").write_text(
            json.dumps(daten), encoding="utf-8"
        )
        rb.main()
        return (tmp / "BEFUND.md").read_text(encoding="utf-8")

    def test_laeuft_ohne_optionale_artefakte(self, res):
        text = self._render(res, self._basis())
        assert "P12 — Das kurze Ende" in text
        # Abschnitts-Ueberschriften, nicht blosse Erwaehnungen: "P12b" kommt
        # auch im Belastbarkeits-Absatz von Kernaussage 2 vor.
        assert "## P12b" not in text
        assert "## P12c" not in text
        assert "## Artefaktschranke" not in text

    def test_fehlendes_pflicht_artefakt_scheitert_laut(self, res):
        _, rb = res
        with pytest.raises(SystemExit):
            rb.main()

    def test_kernaussage_1_dreht_mit_den_daten(self, res):
        """Der eigentliche Test: schlaegt das kurze Ende, muss der Text kippen."""
        schlecht = self._render(res, self._basis(netto=1.5))
        assert "Das kurze Ende trägt nicht" in schlecht
        gut = self._render(res, self._basis(netto=9.9))
        assert "Das kurze Ende trägt nicht" not in gut
        assert "schlägt das Halten" in gut

    def test_kernaussage_2_dreht_mit_den_daten(self, res):
        plus = self._render(res, self._basis(brutto=2.0))
        assert "also im Plus" in plus
        minus = self._render(res, self._basis(brutto=0.5))
        assert "verliert schon brutto" in minus

    def test_zaehlt_zufallsvergleich_aus_den_daten(self, res):
        drunter = self._render(res, self._basis(brutto=2.0, zuf_b=3.0))
        assert "**1 von 1**" in drunter
        drueber = self._render(res, self._basis(brutto=4.0, zuf_b=3.0))
        assert "**0 von 1**" in drueber

    def test_fehlender_schluessel_kracht_statt_zu_luegen(self, res):
        """E-085-Kern: eine Datenluecke darf keine ueberzeugende Zahl ergeben."""
        daten = self._basis()
        del daten["familien"]["A_fest"][0]["zufall_brutto_mittel"]
        with pytest.raises(KeyError):
            self._render(res, daten)

    def test_verwurfsgrund_wird_eingedeutscht(self, res):
        text = self._render(res, self._basis())
        assert "54,5%" in text
        assert "54.5%" not in text

    def test_abdeckungsschwelle_nur_wenn_im_artefakt(self, res):
        ohne = self._render(res, self._basis())
        assert "noch nicht" in ohne
        daten = self._basis()
        daten["min_abdeckung"] = 0.9
        mit = self._render(res, daten)
        assert "bis zu 10,0 %" in mit

    @staticmethod
    def _p12c_zeile(bars, name, brutto, be, bench=3.138, bei_be=None):
        return {
            "halte_bars": bars,
            "name": name,
            "umschichtungen": 100,
            "brutto_end": brutto,
            "brutto_cagr": 0.18,
            # Der Break-even-Punkt MUSS in der Kurve liegen — der Renderer greift
            # direkt zu und kracht sonst (bewusst, siehe E-086).
            # Der Break-even-Punkt MUSS in der Kurve liegen — der Renderer
            # greift direkt zu und kracht sonst (bewusst, siehe E-086).
            "kurve": {"0.0": brutto, "10.0": 0.01}
            | {f"{be}": brutto * 0.5 if bei_be is None else bei_be},
            "breakeven_kapitalerhalt_bps": 2.0,
            "breakeven_schlaegt_benchmark_bps": be,
            "bei_10bps": 0.01,
            "bench_end": bench,
        }

    def _mit_p12c(self, res, zeilen, stufig_faktoren):
        """Legt p12c + p12c_stufig an; stufig_faktoren skaliert brutto je Zeile."""
        tmp, _ = res
        (tmp / "p12c_reversal_kostenschwelle.json").write_text(
            json.dumps({"kosten_raster": [0.0, 1.0, 10.0], "zeilen": zeilen}),
            encoding="utf-8",
        )
        st = [
            dict(z, brutto_end=z["brutto_end"] * f)
            for z, f in zip(zeilen, stufig_faktoren)
        ]
        (tmp / "p12c_reversal_stufig.json").write_text(
            json.dumps({"kosten_raster": [0.0, 1.0, 10.0], "zeilen": st}),
            encoding="utf-8",
        )
        return self._render(res, self._basis())

    def test_vorzeichenaussage_dreht_mit_den_daten(self, res):
        """E-087: `abs()` hatte genau diese Information weggeworfen."""
        zeilen = [self._p12c_zeile(1, "1 Stunde", 6.0, 1.0)]
        wechselnd = self._mit_p12c(res, zeilen * 2, [0.97, 1.05])
        assert "nicht systematisch gerichtet" in wechselnd
        gleichgerichtet = self._mit_p12c(res, zeilen * 2, [1.05, 1.08])
        assert "systematisch gerichtet" in gleichgerichtet
        assert "nicht systematisch gerichtet" not in gleichgerichtet
        assert "nicht** ausgeschlossen" in gleichgerichtet

    def test_tragende_zeile_ist_die_mit_hoechstem_brutto(self, res):
        """E-088: Schranke und Break-even müssen aus DERSELBEN Zeile stammen."""
        zeilen = [
            self._p12c_zeile(1, "1 Stunde", 6.0, 1.0),
            self._p12c_zeile(7, "1 Tag", 2.7, 1.0),
        ]
        text = self._mit_p12c(res, zeilen, [0.97, 1.12])
        assert "(**1 Stunde**" in text
        assert "(**1 Tag**" not in text
        # Die Schranke muss die der 1-Stunden-Zeile sein (3,0 %), nicht die
        # groesste ueber alle Zeilen (12,0 %).
        assert "3,0 %" in text
        assert "beträgt die Artefaktschranke 12,0 %" not in text

    def test_fehlender_kurvenpunkt_kracht(self, res):
        """E-086: lieber KeyError als eine überzeugend aussehende Falschzahl."""
        z = self._p12c_zeile(1, "1 Stunde", 6.0, 1.0)
        del z["kurve"]["1.0"]
        with pytest.raises(KeyError):
            self._mit_p12c(res, [z], [0.97])

    def test_basispunkt_wertung_dreht_mit_den_daten(self, res):
        """E-089: keine feste Wertung neben einer datengetriebenen Zahl."""
        # Teuer: Break-even bei 1 bps UND dort ist fast nichts mehr uebrig.
        teuer = self._mit_p12c(
            res, [self._p12c_zeile(1, "1 Stunde", 6.0, 1.0, bei_be=3.18)], [0.97]
        )
        assert "einzelner Basispunkt" in teuer
        # Robust: Kante traegt bis 20 bps.
        robust = self._mit_p12c(
            res, [self._p12c_zeile(1, "1 Stunde", 6.0, 20.0, bei_be=5.5)], [0.97]
        )
        assert "einzelner Basispunkt" not in robust
        assert "überlebt bis 20 bps" in robust
