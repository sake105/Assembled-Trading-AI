"""Tests für P12f — den Neulauf auf dem bereinigten Panel.

Das Skript hatte lange keinen einzigen Test, obwohl es die Frage beantwortet,
ob ein Kampagnen-Verdikt an Vendor-Preisfehlern hängt. Die Stage-1-Prüfung
zeigte drei Mutationen, die niemand fing: Zielfunktion zurück auf
Endwertvergleich, Auswahl des Besten nach Endwert, halbiertes Parametergitter.
Genau diese drei sind hier abgedeckt.

Kein Kampagnendatensatz nötig: die Engine wird gestellt, geprüft wird die
Verdrahtung — welche Größe gemessen und welche verglichen wird.
"""

from __future__ import annotations

import pandas as pd
import pytest

from research.mandat2 import p12f_neulauf_bereinigt as P
from research.mandat2 import p2_sweep

pytestmark = pytest.mark.fast


class _Lauf:
    def __init__(self, endwert: float):
        self.equity_netto = pd.Series(
            [1.0, endwert], index=pd.to_datetime(["2010-01-04", "2020-01-06"])
        )


class _Auswertung:
    """Minimale Auswertung — die Zahlen sind frei gewählt und wiedererkennbar."""

    def __init__(self, median: float, dd: float):
        self.median_kandidat = median
        self.median_benchmark = 1.95
        self.schlimmster_maxdd = dd
        self.deckel_eingehalten = dd >= -0.35
        self.schlaegt_benchmark = median > 1.95
        self.bestanden = self.schlaegt_benchmark and self.deckel_eingehalten


@pytest.fixture
def engine(monkeypatch):
    """Stellt Engine, Steuerregime und Auswertung; sammelt die Aufrufe ein."""
    aufrufe: list[dict] = []

    def fake_momentum(data, regime, **kw):
        aufrufe.append(kw)
        # Endwert und Median laufen bewusst GEGENLAEUFIG: wer nach Endwert
        # optimiert, waehlt garantiert den falschen.
        return _Lauf(endwert=1000.0 - kw["rank_out"])

    def fake_auswerten(kand, bench, *, label):
        end = float(kand.iloc[-1])
        # GEGENLAEUFIG zum Endwert: hoher Endwert -> niedriger Median. Sonst
        # waeren beide Auswahlkriterien aequivalent und der Test wertlos.
        return _Auswertung(median=3.0 - end / 1000.0, dd=-0.30 if end < 950 else -0.80)

    monkeypatch.setattr(P, "run_momentum", fake_momentum)
    monkeypatch.setattr(P, "run_buy_and_hold", lambda data, regime: _Lauf(500.0))
    monkeypatch.setattr(P, "make_regime", lambda name, **kw: ("regime", name, kw))
    monkeypatch.setattr(P, "auswerten", fake_auswerten)
    return aufrufe


class TestGitter:
    def test_misst_die_zielfunktion_nicht_den_endwert(self, engine):
        """Jede Zeile muss die Kampagnen-Kennzahlen tragen.

        Mutation „zurück auf Endwertvergleich" (M10 der Stage-1-Prüfung) fiel
        vorher durch jedes Netz.
        """
        zeilen, b_end = P.gitter(object(), "ZERO", "ZERO", {})
        pflicht = {
            "median_kandidat",
            "median_benchmark",
            "schlimmster_maxdd",
            "deckel_eingehalten",
            "bestanden",
        }
        assert pflicht <= set(zeilen[0])
        # `bestanden` kommt aus der Auswertung, nicht aus einem Endwertvergleich.
        for z in zeilen:
            erwartet = z["median_kandidat"] > z["median_benchmark"] and (
                z["schlimmster_maxdd"] >= -0.35
            )
            assert z["bestanden"] is erwartet

    def test_bestanden_ist_nicht_dasselbe_wie_schlaegt_benchmark(self, engine):
        """Genau dieses Auseinanderfallen macht die Unterscheidung nötig.

        Im echten Lauf schlagen 5–7 von 24 den Benchmark und **null** bestehen.
        Fällt die Deckel-Bedingung weg, sind beide Zahlen identisch.
        """
        zeilen, _ = P.gitter(object(), "ZERO", "ZERO", {})
        n_schlagen = sum(1 for z in zeilen if z["schlaegt_bench"])
        n_bestanden = sum(1 for z in zeilen if z["bestanden"])
        assert n_schlagen > n_bestanden > 0

    def test_gitter_hat_die_volle_groesse(self, engine):
        """Mutation „Gitter halbieren" (M12) blieb vorher unbemerkt."""
        zeilen, _ = P.gitter(object(), "ZERO", "ZERO", {})
        assert len(zeilen) == len(P.HALTETAGE) * len(P.RANK_OUT) * len(P.HEBEL)
        assert len(zeilen) == 24

    def test_top_in_ist_20_wie_in_P2(self, engine):
        P.gitter(object(), "ZERO", "ZERO", {})
        assert {kw["top_in"] for kw in engine} == {20}


class TestIdentitaetMitP2:
    """Ist es wirklich eine WIEDERHOLUNG — oder heimlich eine neue Suche?

    Weicht das Gitter ab, ist der Trial-Zähler zu Unrecht eingefroren (E-090)
    und der Vergleich mit P2 nicht mehr zulässig.
    """

    def test_parametergitter_identisch(self):
        assert P.HALTETAGE == p2_sweep.HALTETAGE
        assert P.RANK_OUT == p2_sweep.RANK_OUT
        assert P.HEBEL == p2_sweep.HEBEL

    def test_steuerwelten_identisch(self):
        assert P.WELTEN == p2_sweep.WELTEN


class TestBesterKandidat:
    def test_waehlt_nach_median_nicht_nach_endwert(self):
        """Mutation M11: `max(..., key=endwert)` überlebte die ganze Suite."""
        zeilen = [
            {"median_kandidat": 2.0, "endwert": 100.0, "name": "richtig"},
            {"median_kandidat": 1.0, "endwert": 999.0, "name": "falsch"},
        ]
        assert P.bester_kandidat(zeilen)["name"] == "richtig"

    def test_auf_echten_gitterzeilen(self, engine):
        zeilen, _ = P.gitter(object(), "ZERO", "ZERO", {})
        b = P.bester_kandidat(zeilen)
        assert b["median_kandidat"] == max(z["median_kandidat"] for z in zeilen)
        # Und er ist NICHT der mit dem hoechsten Endwert — die Fixture ist so
        # gebaut, dass beide auseinanderfallen.
        assert b["endwert"] != max(z["endwert"] for z in zeilen)


class TestMainVerdrahtung:
    """F-senior-4: `main()` war ungetestet — drei Mutationen überlebten.

    Die folgenschwerste: `d_neu = replace(d, close=close_neu, ...)` durch
    `d_neu = d` ersetzen. Dann läuft der „bereinigt"-Arm auf dem Originalpanel,
    das Artefakt meldet trotzdem „Verdikt dreht nicht" — und von außen ist das
    nicht unterscheidbar. Genau die Zeile trägt die ganze Aussage des Skripts.
    """

    @pytest.fixture
    def gestellter_lauf(self, monkeypatch, tmp_path):
        """Minimaler End-to-End-Lauf: echte Verdrahtung, gestellte Engine."""
        from dataclasses import dataclass

        idx = pd.bdate_range("2000-01-03", periods=40, tz="UTC")
        close = pd.DataFrame({"X": range(100, 140), "Y": range(200, 240)}, index=idx)
        close = close.astype(float)

        @dataclass
        class _Daten:
            close: pd.DataFrame
            div_panel: pd.DataFrame
            membership: pd.Series
            fenster: str
            von: pd.Timestamp
            bis: pd.Timestamp

        daten = _Daten(
            close=close,
            div_panel=close * 0.0,
            membership=pd.Series([frozenset({"X", "Y"})], index=idx[-1:]),
            fenster="TEST",
            von=idx[0],
            bis=idx[-1],
        )
        gesehen: list[int] = []

        def fake_momentum(data, regime, **kw):
            gesehen.append(id(data.close))
            return _Lauf(endwert=1000.0 - kw["rank_out"])

        monkeypatch.setattr(P, "load_campaign", lambda: daten)
        monkeypatch.setattr(P, "run_momentum", fake_momentum)
        monkeypatch.setattr(P, "run_buy_and_hold", lambda data, regime: _Lauf(500.0))
        monkeypatch.setattr(P, "make_regime", lambda name, **kw: ("regime", name, kw))
        monkeypatch.setattr(
            P,
            "auswerten",
            lambda k, b, *, label: _Auswertung(
                median=3.0 - float(k.iloc[-1]) / 1000.0,
                dd=-0.30 if float(k.iloc[-1]) < 950 else -0.80,
            ),
        )
        # Eine kuenstliche Korruption, damit die Bereinigung etwas zu tun hat.
        monkeypatch.setattr(
            P,
            "korruptions_spannen",
            lambda px, namen: {
                "X": {
                    "spannen": [
                        [
                            f"{idx[20]:%Y-%m-%d}",
                            f"{idx[-1] + pd.Timedelta(days=1):%Y-%m-%d}",
                        ]
                    ],
                    "unaufloesbar": False,
                    "unaufloesbar_grund": "",
                }
            },
        )
        monkeypatch.setattr(P, "OUT", tmp_path)
        return gesehen, tmp_path

    def test_bereinigter_arm_bekommt_ein_anderes_panel(self, gestellter_lauf):
        gesehen, _ = gestellter_lauf
        assert P.main() == 0
        # 3 Welten x 2 Panels x 24 Kombinationen
        assert len(gesehen) == 3 * 2 * 24
        # Genau ZWEI verschiedene close-Objekte — nicht eines.
        assert len(set(gesehen)) == 2, "beide Arme liefen auf demselben Panel"

    def test_artefakt_traegt_die_verdikt_felder(self, gestellter_lauf):
        import json

        _, out = gestellter_lauf
        assert P.main() == 0
        d = json.loads((out / "p12f_neulauf_bereinigt.json").read_text("utf-8"))
        assert set(d["welten"]) == {"ZERO", "PRIVAT_DE", "GMBH+FK"}
        for v in d["welten"].values():
            assert {"verdikt_dreht", "optimum_wandert", "schlaegt_dreht"} <= set(v)
            assert v["original"]["n_bestanden"] + v["original"]["n_schlagen_bench"] >= 0
        assert d["gegenprobe"]["neu_entstanden"] == 0
        assert "unaufloesbar_grund" in d

    def test_verdikt_dreht_folgt_den_bestandenen(self, gestellter_lauf):
        """Mutation `verdikt_dreht = False` überlebte vorher jede Prüfung."""
        import json

        _, out = gestellter_lauf
        assert P.main() == 0
        d = json.loads((out / "p12f_neulauf_bereinigt.json").read_text("utf-8"))
        for v in d["welten"].values():
            erwartet = (v["original"]["n_bestanden"] == 0) != (
                v["bereinigt"]["n_bestanden"] == 0
            )
            assert v["verdikt_dreht"] is erwartet

    def test_optimum_wandert_folgt_den_parametern(self, gestellter_lauf):
        import json

        _, out = gestellter_lauf
        assert P.main() == 0
        d = json.loads((out / "p12f_neulauf_bereinigt.json").read_text("utf-8"))
        for v in d["welten"].values():
            o, b = v["original"]["bester"], v["bereinigt"]["bester"]
            erwartet = any(o[k] != b[k] for k in ("haltetage", "rank_out", "hebel"))
            assert v["optimum_wandert"] is erwartet


class TestMainErkenntEinenDreher:
    """Die Gegenprobe zu TestMainVerdrahtung.

    Dort sind `verdikt_dreht` und `optimum_wandert` in allen Welten False —
    eine Mutation auf `= False` ist dann nicht unterscheidbar und überlebt
    (nachgewiesen). Ein Test, der ein Flag nur im Ruhezustand prüft, prüft es
    nicht. Hier reagiert die gestellte Engine auf das Panel, sodass beide Flags
    True werden müssen.
    """

    @pytest.fixture
    def lauf_mit_dreher(self, monkeypatch, tmp_path):
        from dataclasses import dataclass

        idx = pd.bdate_range("2000-01-03", periods=40, tz="UTC")
        close = pd.DataFrame(
            {"X": range(100, 140), "Y": range(200, 240)}, index=idx
        ).astype(float)

        @dataclass
        class _Daten:
            close: pd.DataFrame
            div_panel: pd.DataFrame
            membership: pd.Series
            fenster: str
            von: pd.Timestamp
            bis: pd.Timestamp

        daten = _Daten(
            close=close,
            div_panel=close * 0.0,
            membership=pd.Series([frozenset({"X", "Y"})], index=idx[-1:]),
            fenster="TEST",
            von=idx[0],
            bis=idx[-1],
        )
        erstes: list[int] = []

        def fake_momentum(data, regime, **kw):
            kennung = id(data.close)
            if not erstes:
                erstes.append(kennung)
            ist_original = kennung == erstes[0]
            # Auf dem ORIGINAL gewinnt rank_out=200, auf dem BEREINIGTEN
            # rank_out=30 -> das Optimum wandert. Und ALLE Original-Endwerte
            # bleiben unter 1.100, reissen also den Deckel; im bereinigten Panel
            # liegt keiner darunter -> das Verdikt dreht von 0/24 auf 24/24.
            r = kw["rank_out"] / 10.0
            return _Lauf(endwert=1000.0 + r if ist_original else 1500.0 - r)

        def fake_auswerten(k, b, *, label):
            end = float(k.iloc[-1])
            return _Auswertung(median=end / 400.0, dd=-0.80 if end < 1100 else -0.30)

        monkeypatch.setattr(P, "load_campaign", lambda: daten)
        monkeypatch.setattr(P, "run_momentum", fake_momentum)
        monkeypatch.setattr(P, "run_buy_and_hold", lambda data, regime: _Lauf(500.0))
        monkeypatch.setattr(P, "make_regime", lambda name, **kw: ("regime", name, kw))
        monkeypatch.setattr(P, "auswerten", fake_auswerten)
        monkeypatch.setattr(
            P,
            "korruptions_spannen",
            lambda px, namen: {
                "X": {
                    "spannen": [
                        [
                            f"{idx[20]:%Y-%m-%d}",
                            f"{idx[-1] + pd.Timedelta(days=1):%Y-%m-%d}",
                        ]
                    ],
                    "unaufloesbar": False,
                    "unaufloesbar_grund": "",
                }
            },
        )
        monkeypatch.setattr(P, "OUT", tmp_path)
        return tmp_path

    def _welten(self, out):
        import json

        assert P.main() == 0
        return json.loads((out / "p12f_neulauf_bereinigt.json").read_text("utf-8"))[
            "welten"
        ]

    def test_verdikt_dreht_wird_als_True_gemeldet(self, lauf_mit_dreher):
        welten = self._welten(lauf_mit_dreher)
        assert all(v["original"]["n_bestanden"] == 0 for v in welten.values())
        assert all(v["bereinigt"]["n_bestanden"] > 0 for v in welten.values())
        assert all(v["verdikt_dreht"] is True for v in welten.values())

    def test_optimum_wandert_wird_als_True_gemeldet(self, lauf_mit_dreher):
        welten = self._welten(lauf_mit_dreher)
        assert all(v["optimum_wandert"] is True for v in welten.values())
        for v in welten.values():
            assert (
                v["original"]["bester"]["rank_out"]
                != (v["bereinigt"]["bester"]["rank_out"])
            )
