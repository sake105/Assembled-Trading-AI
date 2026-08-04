"""Tests für P12g — kann der Intraday-Endpunkt die Survivorship-Lücke schließen?

Die erste Fassung dieses Skripts beantwortete die Frage aus dem Dateisystem und
lag falsch: Namen der vermeintlichen Fehlgruppe lieferten 7.000–8.000 Bars, sie
waren nur nie angefragt worden (E-112). Getestet wird deshalb vor allem, dass
die Aussage aus einer **Abfrage** stammt und dass die Randfälle nicht in
Richtung Entwarnung kippen.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from research.mandat2 import p12g_pull_bilanz as P
from research.mandat2.p12g_pull_bilanz import bilanz, probiere, vorhandene_symbole

pytestmark = pytest.mark.fast


def _membership(erste: set[str], letzte: set[str]) -> pd.Series:
    return pd.Series(
        [frozenset(erste), frozenset(letzte)],
        index=pd.to_datetime(["2006-06-30", "2016-12-30"]),
    )


class TestBilanzIstKeineEndpunktAussage:
    """Die Kennzahl beschreibt die Anfrageliste — und muss das auch sagen."""

    def test_traegt_einen_hinweis_auf_ihre_reichweite(self):
        b = bilanz(_membership({"A", "B"}, {"A"}), {"A"})
        assert "NICHT das Verhalten des Endpunkts" in b["hinweis"]

    def test_faktor_wird_berechnet_wenn_beide_quoten_positiv(self):
        m = _membership({"A", "B", "C", "D"}, {"A", "B", "C"})
        b = bilanz(m, {"A", "B"})
        # mit Datei: A, B -> beide ueberleben = 100 %
        # ohne Datei: C, D -> nur C ueberlebt = 50 %
        assert b["anreicherungsfaktor"] == pytest.approx(2.0)
        assert b["maximal_verzerrt"] is False

    def test_neutrales_fehlen_ergibt_faktor_eins(self):
        m = _membership({"A", "B", "C", "D"}, {"A", "C"})
        assert bilanz(m, {"A", "B"})["anreicherungsfaktor"] == pytest.approx(1.0)

    def test_staerkste_verzerrung_wird_als_solche_markiert(self):
        """F-senior-4: `faktor is None` hieß im Renderer „neutral".

        Der Fall q_ohne == 0 liefert keinen Quotienten — er ist aber das
        Gegenteil von neutral: kein einziger Name ohne Datei überlebt. Ohne das
        Flag fiel der Renderer hier auf Entwarnung.
        """
        m = _membership({"A", "B", "C", "D"}, {"A", "B"})
        b = bilanz(m, {"A", "B"})
        assert b["anreicherungsfaktor"] is None
        assert b["maximal_verzerrt"] is True

    def test_abdeckung_zaehlt_ueber_das_ganze_fenster(self):
        b = bilanz(_membership({"A", "B"}, {"C", "D"}), {"A", "C"})
        assert b["n_pit_mitglieder"] == 4
        assert b["n_mit_datei"] == 2
        assert b["abdeckung"] == pytest.approx(0.5)


class TestApiProbe:
    """Die eigentliche Messung — sie darf nur sagen, was abgefragt wurde."""

    def test_zaehlt_bars_und_meldet_fehler_getrennt(self, monkeypatch):
        gefragt = []

        def fake(sym, von, bis, tok):
            gefragt.append(sym)
            if sym == "KAPUTT":
                raise RuntimeError("boom")
            return [{}] * (0 if sym.endswith("Q") else 42)

        monkeypatch.setattr(
            "research.mandat2.intraday_pull.hole_fenster", fake, raising=True
        )
        aus = probiere(["AAA", "BBBQ", "KAPUTT", None], "tok", jahr=2008, monat=3)
        assert aus["AAA"] == 42
        assert aus["BBBQ"] == 0
        # Ein Fehler ist KEIN Negativbefund — er wird als solcher gekennzeichnet.
        assert aus["KAPUTT"] == "ERR:RuntimeError"
        assert None not in gefragt

    def test_probefenster_liegt_im_suchzeitraum(self, monkeypatch):
        """Holdout-Disziplin: keine Abfrage jenseits 2016-12-31."""
        fenster = []

        def fake(sym, von, bis, tok):
            fenster.append((von, bis))
            return []

        monkeypatch.setattr(
            "research.mandat2.intraday_pull.hole_fenster", fake, raising=True
        )
        for _, _, _, jahr, monat in P.AUSSCHEIDER:
            probiere(["X"], "tok", jahr=jahr, monat=monat)
        probiere(["X"], "tok", jahr=2006, monat=7, tage=30)
        assert all(b < pd.Timestamp("2017-01-01", tz="UTC") for _, b in fenster)

    def test_ausscheider_werden_unter_BEIDEN_symbolen_geprueft(self):
        """E-113: der Q-Ticker entsteht erst nach der Insolvenz.

        Ein Negativbefund allein auf ihm ist fast garantiert und beweist nichts
        über die Historie des Namens.
        """
        for sym, q, name, jahr, _ in P.AUSSCHEIDER:
            assert sym and name
            if q is not None:
                assert q != sym and q.endswith("Q")
            assert 2006 <= jahr <= 2016

    def test_kontrollgruppe_ist_nicht_leer(self):
        """Ohne sie wäre ein Negativbefund nicht von einem kaputten Aufruf
        zu unterscheiden."""
        assert len(P.KONTROLLE) >= 3
        assert not set(P.KONTROLLE) & {s for s, *_ in P.AUSSCHEIDER}


class TestMain:
    """F-senior-5/11: `main()` war ungetestet und stürzte am Extremfall ab."""

    @pytest.fixture
    def umgebung(self, tmp_path, monkeypatch):
        intraday = tmp_path / "intraday"
        intraday.mkdir()
        for s in ("A", "B"):
            (intraday / f"{s}.parquet").write_bytes(b"")
        out = tmp_path / "results"

        class _D:
            membership = _membership({"A", "B", "C", "D"}, {"A", "B"})

            def __repr__(self):
                return "CampaignData(TEST)"

        monkeypatch.setattr(P, "INTRADAY", intraday)
        monkeypatch.setattr(P, "OUT", out)
        monkeypatch.setattr(P, "load_campaign", lambda: _D())
        return out

    def test_laeuft_ohne_probe_und_sagt_dass_nichts_belegt_ist(self, umgebung, capsys):
        assert P.main() == 0
        text = capsys.readouterr().out
        assert "[SKIP] API-Probe" in text
        assert "BEFUND: keiner" in text
        d = json.loads((umgebung / "p12g_pull_bilanz.json").read_text("utf-8"))
        assert d["api_probe"] is None

    def test_stuerzt_am_extremfall_nicht_ab(self, umgebung, capsys):
        """q_ohne == 0 -> kein Faktor. Vorher: TypeError beim Formatieren."""
        assert P.main() == 0
        text = capsys.readouterr().out
        assert "Faktor n/a" in text
        assert "staerkste Verzerrung" in text

    def test_mit_probe_traegt_das_verdikt_die_abfrage(
        self, umgebung, monkeypatch, capsys
    ):
        monkeypatch.setattr("research.mandat2.intraday_pull.token", lambda: "x")
        monkeypatch.setattr(
            "research.mandat2.intraday_pull.hole_fenster",
            lambda sym, von, bis, tok: ([] if sym in _stumme() else [{}] * 7000),
            raising=True,
        )
        monkeypatch.setattr(P.sys, "argv", ["p12g", "--probe"])
        assert P.main() == 0
        text = capsys.readouterr().out
        assert "6 von 6 geprueften Ausscheidern NICHT" in text
        assert "Kontrollgruppe (Ueberlebende): 8/8" in text
        d = json.loads((umgebung / "p12g_pull_bilanz.json").read_text("utf-8"))
        assert d["api_probe"]["ausscheider"]["LEH"]["bars_mitgliedschaftssymbol"] == 0
        assert d["api_probe"]["kontrolle"]["AMZN"] == 7000

    def test_liefernde_ausscheider_kehren_das_verdikt_um(
        self, umgebung, monkeypatch, capsys
    ):
        """Die Gegenprobe — sonst wäre das Verdikt eine feste Behauptung."""
        monkeypatch.setattr("research.mandat2.intraday_pull.token", lambda: "x")
        monkeypatch.setattr(
            "research.mandat2.intraday_pull.hole_fenster",
            lambda sym, von, bis, tok: [{}] * 500,
            raising=True,
        )
        monkeypatch.setattr(P.sys, "argv", ["p12g", "--probe"])
        assert P.main() == 0
        text = capsys.readouterr().out
        assert "Alle geprueften Ausscheider liefern Bars" in text
        assert "geprueften Ausscheidern NICHT" not in text


def _stumme() -> set[str]:
    aus = set()
    for sym, q, *_ in P.AUSSCHEIDER:
        aus.add(sym)
        if q:
            aus.add(q)
    return aus


class TestSymbolerkennung:
    def test_liest_parquet_dateinamen_gross(self, tmp_path):
        (tmp_path / "aapl.parquet").write_bytes(b"")
        (tmp_path / "MSFT.parquet").write_bytes(b"")
        (tmp_path / "liesmich.txt").write_bytes(b"")
        assert vorhandene_symbole(tmp_path) == {"AAPL", "MSFT"}

    def test_leerer_ordner_ergibt_leere_menge(self, tmp_path):
        assert vorhandene_symbole(tmp_path) == set()


class TestVerdiktBrauchtVollstaendigeEvidenz:
    """F-auditor-1: die Weiche keyte auf „ein stummer Name" statt auf Evidenz.

    Zwei live reproduzierte Folgen: bei lauter fehlgeschlagenen Abfragen meldete
    das Skript „der Weg ist gangbar" — eine positive Behauptung aus null
    Messung. Und bei einem stummen von sechs stand im Dokument „die Ausscheider
    sind bei dieser Quelle nicht zu haben".
    """

    def _probe(self, bars_je_ausscheider: list, kontrolle: dict) -> dict:
        namen = [(s, q, n) for s, q, n, *_ in P.AUSSCHEIDER]
        aus = {}
        for (sym, q, name), bars in zip(namen, bars_je_ausscheider, strict=False):
            aus[sym] = {
                "name": name,
                "q_ticker": q,
                "probefenster": "2008-03",
                "bars_mitgliedschaftssymbol": bars,
                "bars_q_ticker": bars if q else None,
            }
        return {"ausscheider": aus, "kontrolle": kontrolle}

    def test_ohne_probe_kein_verdikt(self):
        assert P.verdikt(None)["status"] == "keine_probe"

    def test_lauter_fehler_ergibt_KEIN_verdikt(self):
        """Vorher: „der Endpunkt liefert Ausscheider — der Weg ist gangbar."""
        p = self._probe(["ERR:RuntimeError"] * 6, {"AMZN": 7000})
        v = P.verdikt(p)
        assert v["status"] == "unvollstaendig"
        assert len(v["fehler"]) >= 6

    def test_tote_kontrollgruppe_ergibt_KEIN_verdikt(self):
        """Ohne lieferende Kontrolle ist ein Negativbefund nicht deutbar."""
        v = P.verdikt(self._probe([0] * 6, {"AMZN": 0, "GILD": 0}))
        assert v["status"] == "unvollstaendig"
        assert v["n_kontrolle_lebt"] == 0

    def test_leere_kontrollgruppe_ebenso(self):
        assert P.verdikt(self._probe([0] * 6, {}))["status"] == "unvollstaendig"

    def test_teilweise_stumm_wird_NICHT_verallgemeinert(self):
        """Vorher: ein stummer von sechs -> „nicht zu haben"."""
        v = P.verdikt(self._probe([0] + [5000] * 5, {"AMZN": 7000}))
        assert v["status"] == "teilweise"
        assert v["n_stumm"] == 1 and v["n_ausscheider"] == 6

    def test_alle_stumm_bei_intakter_kontrolle_schliesst_den_weg(self):
        v = P.verdikt(self._probe([0] * 6, {"AMZN": 7000, "GILD": 7500}))
        assert v["status"] == "weg_zu"
        assert v["n_stumm"] == v["n_ausscheider"] == 6

    def test_alle_liefernd_haelt_den_weg_offen(self):
        v = P.verdikt(self._probe([5000] * 6, {"AMZN": 7000}))
        assert v["status"] == "weg_offen"
        assert v["n_stumm"] == 0

    def test_ein_einziger_fehler_entwertet_die_probe(self):
        """Ein Fehlerstring ist von einem Negativbefund nicht zu unterscheiden."""
        v = P.verdikt(self._probe([0, 0, 0, 0, 0, "ERR:HTTPError"], {"AMZN": 7000}))
        assert v["status"] == "unvollstaendig"

    def test_main_meldet_unvollstaendig_statt_gangbar(
        self, monkeypatch, capsys, tmp_path
    ):
        intraday = tmp_path / "i"
        intraday.mkdir()
        (intraday / "A.parquet").write_bytes(b"")

        class _D:
            membership = _membership({"A", "B"}, {"A"})

            def __repr__(self):
                return "CampaignData(TEST)"

        monkeypatch.setattr(P, "INTRADAY", intraday)
        monkeypatch.setattr(P, "OUT", tmp_path / "r")
        monkeypatch.setattr(P, "load_campaign", lambda: _D())
        monkeypatch.setattr("research.mandat2.intraday_pull.token", lambda: "x")

        def kaputt(sym, von, bis, tok):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "research.mandat2.intraday_pull.hole_fenster", kaputt, raising=True
        )
        monkeypatch.setattr(P.sys, "argv", ["p12g", "--probe"])
        assert P.main() == 0
        text = capsys.readouterr().out
        assert "die Probe ist unvollstaendig" in text
        assert "gangbar" not in text
