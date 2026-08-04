"""Tests für den Datenqualitäts-Befund-Renderer.

Warum ein Renderer Tests braucht: seine Verzweigungen sind keine Formatierung,
sondern **Aussagen**. „Der Datensatz kann die Frage nicht entscheiden" gegen
„die Verzerrung ist klein genug" ist der Unterschied zwischen einem Ergebnis
und keinem. Ein Zweig, der nie feuert oder beim Umschreiben still bricht, wäre
hier ein stiller Wechsel des Verdikts — genau die Klasse Fehler, gegen die
dieser ganze Strang läuft.

Getestet werden deshalb **beide** Seiten jeder Verzweigung, gegen synthetische
Ergebnisse mit bewusst gesetzten Zahlen.
"""

from __future__ import annotations

import json

import pytest

from research.mandat2 import render_befund_datenqualitaet as R

pytestmark = pytest.mark.fast


def _p12d(ueberhoehung_pp: float) -> dict:
    """Ein P12d-Ergebnis mit vorgegebener Überhöhung in Prozentpunkten."""
    return {
        "fenster": "2006-06-22..2016-12-30",
        "jahre": 10.5,
        "tote_ticker_im_pit_universum": ["EKDKQ"],
        "glitch_schwelle": 2.0,
        "ausgeschlossene_glitches": {},
        "zeilen": [
            {
                "universum": "intraday_p12",
                "n": 20,
                "halten": {"endwert": 3.0, "cagr": 0.11, "maxdd": -0.54},
                "umschichten": {"endwert": 3.0, "cagr": 0.11, "maxdd": -0.54},
                "diagnose": {},
            },
            {
                "universum": "pit_2004",
                "n": 382,
                "halten": {"endwert": 2.3, "cagr": 0.0816, "maxdd": -0.60},
                "umschichten": {"endwert": 2.4, "cagr": 0.0870, "maxdd": -0.60},
                "diagnose": {},
            },
        ],
        "spy": {"endwert": 2.18, "cagr": 0.077, "maxdd": -0.55},
        "ueberhoehung_cagr": {
            "cagr_halten": ueberhoehung_pp / 100.0,
            "cagr_umschichten": ueberhoehung_pp / 100.0 * 0.8,
        },
    }


def _p12e() -> dict:
    return {
        "fenster": "SUCHE",
        "top_in": 20,
        "momentum_fenster_handelstage": [21, 252],
        "n_auswahltermine": 252,
        "auswahlplaetze_gesamt": 5040,
        # Die Pflichtfelder des AKTUELLEN Detektors — fehlen sie, bricht der
        # Renderer ab, statt zwei Codegenerationen zu mischen (F-senior-1).
        "korrumpierte_namen": {
            "GPS": {
                "unaufloesbar": False,
                "unaufloesbar_grund": "",
                "n_tage_falsch": 253,
            },
            "CFC": {
                "unaufloesbar": False,
                "unaufloesbar_grund": "",
                "n_tage_falsch": 113,
            },
            "YRCW": {
                "unaufloesbar": True,
                "unaufloesbar_grund": "verschraenkt",
                "n_tage_falsch": 1697,
            },
            "WFT": {
                "unaufloesbar": True,
                "unaufloesbar_grund": "sentinel",
                "n_tage_falsch": 1375,
            },
        },
        "uebergangstage_gesamt": 246,
        "tage_auf_falscher_skala_gesamt": 32332,
        "halte_kanal": {
            "GPS": {
                "tage": ["1996-12-20"],
                "n_tage": 2,
                "portfolio_tagesrendite": {"1996-12-20": 0.1236},
                "groesste_wirkung": 0.1236,
                "groesste_wirkung_betrag": 0.1236,
                "groesste_wirkung_tag": "1996-12-20",
                "rang_unter_allen_tagen": 2,
                "n_handelstage": 5548,
            },
            "CFC": {
                "tage": ["2007-06-20"],
                "n_tage": 9,
                "portfolio_tagesrendite": {"2007-06-20": -0.0669},
                "groesste_wirkung": -0.0669,
                "groesste_wirkung_betrag": 0.0669,
                "groesste_wirkung_tag": "2007-06-20",
                "rang_unter_allen_tagen": 5532,
                "n_handelstage": 5548,
            },
        },
        "auswahl_kanal": {"GPS": ["1997-01-31", "1997-02-28"]},
        "auswahlplaetze_kanal_b": 22,
        "anteil_plaetze_kanal_b": 0.004365,
        "kontaminiert": True,
        "gemessene_konfiguration": {},
        "abdeckung": {"min": 0.84, "median": 0.913, "max": 0.962, "je_termin": []},
        "austritts_anreicherung": {
            "referenz_start": "1996-01-31",
            "referenz_ende": "2016-12-30",
            "n_mit_spalte": 409,
            "n_ohne_spalte": 78,
            "ueberlebensquote_mit_spalte": 0.462,
            "ueberlebensquote_ohne_spalte": 0.0897,
            "anreicherungsfaktor": 5.149,
        },
    }


def _zeile(**kw) -> dict:
    """Eine Gitterzeile mit allen Pflichtfeldern; ueberschreibbar per kw.

    Tests, die nur zwei Felder brauchen, sollen nicht die uebrigen vergessen —
    sonst faellt der Renderer erst auf, wenn eine neue Kennzahl sie liest.
    """
    return {
        "welt": "TEST",
        "haltetage": 730,
        "rank_out": 200,
        "hebel": 1.0,
        "endwert": 900_000.0,
        "median_kandidat": 2.0,
        "median_benchmark": 1.95,
        "schlimmster_maxdd": -0.62,
        "deckel_eingehalten": False,
        "schlaegt_bench": True,
        "bestanden": False,
    } | kw


def _zeilen(n_schlagen: int, *, dd_versatz: float) -> list[dict]:
    """24 Gitterzeilen mit den Feldern, die der Renderer wirklich liest.

    Vorher stand hier `[{}] * 24`. Das ging gut, solange nur gezaehlt wurde —
    `kipp_abstand` und `zeilen_mit_wechsel` lesen aber `schlimmster_maxdd` und
    `schlaegt_bench` je Zeile. Eine Fixture, die weniger traegt als die echten
    Daten, testet den Renderer nur bis zur ersten neuen Kennzahl.

    `dd_versatz` bildet die Wirkung der Bereinigung ab, `n_schlagen` steuert,
    wie viele Zeilen den Benchmark schlagen.
    """
    aus = []
    i = 0
    for haltetage in (0, 90, 365, 730):
        for rank_out in (30, 60, 200):
            for hebel in (1.0, 1.5):
                aus.append(
                    {
                        "welt": "TEST",
                        "haltetage": haltetage,
                        "rank_out": rank_out,
                        "hebel": hebel,
                        "endwert": 900_000.0 - i * 1_000.0,
                        "median_kandidat": 6.68 - i * 0.1,
                        "median_benchmark": 1.95,
                        "schlimmster_maxdd": -0.62 - i * 0.01 + dd_versatz,
                        "deckel_eingehalten": False,
                        "schlaegt_bench": i < n_schlagen,
                        "bestanden": False,
                    }
                )
                i += 1
    return aus


def _welt(
    n_bestanden_orig: int,
    n_bestanden_ber: int,
    optimum_wandert: bool,
    *,
    n_schlagen: int = 5,
) -> dict:
    """Eine Steuerwelt. Verdikt-Kriterium ist BESTANDEN, nicht schlaegt-Bench.

    Die beiden faellen im echten Lauf auseinander: 5 von 24 Parametrisierungen
    schlagen den Benchmark bei der Rendite, keine einzige haelt zusaetzlich den
    Drawdown-Deckel. Wer das erste misst, misst nicht die Kampagne.
    """
    bester_o = {
        "haltetage": 0,
        "rank_out": 60,
        "hebel": 1.0,
        "endwert": 900_000.0,
        "median_kandidat": 6.68,
        "median_benchmark": 1.95,
        "schlimmster_maxdd": -0.321,
    }
    bester_b = dict(bester_o)
    if optimum_wandert:
        bester_b["haltetage"] = 365
    return {
        "original": {
            "benchmark": 1_000_000.0,
            "bester": bester_o,
            "n_schlagen_bench": n_schlagen,
            "n_deckel_gehalten": 3,
            "n_bestanden": n_bestanden_orig,
            "zeilen": _zeilen(n_schlagen, dd_versatz=0.0),
        },
        "bereinigt": {
            "benchmark": 1_000_000.0,
            "bester": bester_b,
            "n_schlagen_bench": n_schlagen,
            "n_deckel_gehalten": 3,
            "n_bestanden": n_bestanden_ber,
            "zeilen": _zeilen(n_schlagen, dd_versatz=-0.02),
        },
        "optimum_wandert": optimum_wandert,
        "verdikt_dreht": (n_bestanden_orig == 0) != (n_bestanden_ber == 0),
        "schlaegt_dreht": False,
    }


def _p12f(welten: dict, unaufloesbar: list[str] | None = None) -> dict:
    return {
        "n_symbole_bereinigt": 13,
        "n_spannen": 40,
        "dd_deckel": -0.35,
        "gegenprobe": {
            "auffaellig_original": 458,
            "auffaellig_bereinigt": 430,
            "beseitigt": 28,
            "neu_entstanden": 0,
        },
        "dividendenrendite_max_abweichung": 1.388e-17,
        "unaufloesbar": unaufloesbar if unaufloesbar is not None else ["YRCW"],
        "unaufloesbar_grund": {
            "YRCW": "verschraenkt",
            "WFT": "sentinel",
            "CIN": "verschraenkt",
        },
        "protokoll": {},
        "welten": welten,
    }


def _rendere(tmp_path, monkeypatch, d: dict, e: dict, f: dict) -> str:
    res = tmp_path / "results"
    res.mkdir()
    for name, obj in (
        ("p12d_survivorship", d),
        ("p12e_panel_hygiene", e),
        ("p12f_neulauf_bereinigt", f),
    ):
        (res / f"{name}.json").write_text(json.dumps(obj), encoding="utf-8")
    ziel = tmp_path / "BEFUND.md"
    monkeypatch.setattr(R, "RES", res)
    monkeypatch.setattr(R, "ZIEL", ziel)
    assert R.main() == 0
    # Zeilenumbrueche sind Layout, keine Aussage: der Renderer bricht Saetze
    # nach Spaltenbreite um, und ein Test, der daran scheitert, testet die
    # Formatierung statt des Inhalts.
    return " ".join(ziel.read_text(encoding="utf-8").split())


class TestVerzweigungSurvivorship:
    def test_ueberhoehung_ueber_der_marge_sagt_datensatz_entscheidet_nicht(
        self, tmp_path, monkeypatch
    ):
        """Der reale Fall: 2,90 pp gegen eine Marge von 1,5 pp."""
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(2.90),
            _p12e(),
            _p12f({"ZERO": _welt(0, 0, False)}),
        )
        assert "kann die Frage nicht entscheiden" in text
        assert "2,90" in text
        assert "nicht groß genug" not in text

    def test_ueberhoehung_unter_der_marge_entwarnt(self, tmp_path, monkeypatch):
        """Die Gegenprobe — sonst ist unbelegt, dass der Zweig überhaupt greift."""
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(0.40),
            _p12e(),
            _p12f({"ZERO": _welt(0, 0, False)}),
        )
        assert "nicht groß genug" in text
        assert "kann die Frage nicht entscheiden" not in text

    def test_die_marge_ist_die_stellschraube(self, tmp_path, monkeypatch):
        """Mutationsprobe: hebt man die Marge an, kippt die Aussage.

        Damit ist belegt, dass die Wertung an der GEMESSENEN Zahl hängt und
        nicht an einem festgeschriebenen Satz.
        """
        monkeypatch.setattr(R, "ENTSCHEIDUNGSMARGE_PP", 99.0)
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(2.90),
            _p12e(),
            _p12f({"ZERO": _welt(0, 0, False)}),
        )
        assert "kann die Frage nicht entscheiden" not in text


class TestVerzweigungNeulauf:
    def test_kein_dreher_wird_als_robust_berichtet(self, tmp_path, monkeypatch):
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(2.90),
            _p12e(),
            _p12f({"ZERO": _welt(0, 0, False), "PRIVAT_DE": _welt(0, 0, False)}),
        )
        assert "dreht in keiner Steuerwelt" in text
        assert "Optimum wandert in keiner Welt" in text

    def test_dreher_wird_als_dreher_berichtet(self, tmp_path, monkeypatch):
        """Wenn im Original keine Zeile schlägt und bereinigt eine — Alarm."""
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(2.90),
            _p12e(),
            _p12f({"ZERO": _welt(0, 0, False), "GMBH+FK": _welt(0, 3, False)}),
        )
        assert "Das Verdikt dreht in GMBH+FK" in text
        assert "neu zu bewerten" in text
        assert "dreht in keiner Steuerwelt" not in text

    def test_wanderndes_optimum_zieht_den_P2_schluss_in_zweifel(
        self, tmp_path, monkeypatch
    ):
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(2.90),
            _p12e(),
            _p12f({"ZERO": _welt(0, 0, True)}),
        )
        assert "Optimum wandert in: ZERO" in text
        assert "tönernen Füßen" in text

    def test_unaufloesbare_namen_werden_genannt_nicht_verschwiegen(
        self, tmp_path, monkeypatch
    ):
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(2.90),
            _p12e(),
            _p12f({"ZERO": _welt(0, 0, False)}, unaufloesbar=["YRCW", "WFT"]),
        )
        assert "Nicht bereinigt: **2 Namen** (YRCW, WFT)" in text
        # Und der GRUND steht dabei — beide Gruppen getrennt (F-senior-6).
        assert "Bei 1 davon sind die Skalen **verschränkt**" in text
        assert "**1 Namen tragen den Sättigungswert des Datenlieferanten**" in text

    def test_ohne_unaufloesbare_faellt_der_absatz_weg(self, tmp_path, monkeypatch):
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(2.90),
            _p12e(),
            _p12f({"ZERO": _welt(0, 0, False)}, unaufloesbar=[]),
        )
        assert "Nicht bereinigt" not in text


class TestPflichtinhalte:
    def test_austritts_anreicherung_steht_drin(self, tmp_path, monkeypatch):
        """Der stärkste Einzelbefund darf nicht wegformatiert werden."""
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(2.90),
            _p12e(),
            _p12f({"ZERO": _welt(0, 0, False)}),
        )
        assert "5,15×" in text
        assert "bevorzugt die Verlierer" in text

    def test_fehlende_ergebnisse_brechen_ab_statt_zu_erfinden(
        self, tmp_path, monkeypatch, capsys
    ):
        res = tmp_path / "results"
        res.mkdir()
        (res / "p12d_survivorship.json").write_text(json.dumps(_p12d(2.9)), "utf-8")
        monkeypatch.setattr(R, "RES", res)
        monkeypatch.setattr(R, "ZIEL", tmp_path / "BEFUND.md")
        assert R.main() == 1
        assert "p12e" in capsys.readouterr().out
        assert not (tmp_path / "BEFUND.md").exists()

    def test_deutsche_zahlformatierung(self):
        assert R.dez(1234.5) == "1.234,50"
        assert R.pp(0.0290, 2) == "2,90"
        assert R.tsd(5548) == "5.548"


class TestGegenprobe:
    """Die Reparatur muss ihre eigenen Nebenwirkungen berichten (E-107)."""

    def test_null_neue_ausreisser_wird_als_untergrenze_berichtet(
        self, tmp_path, monkeypatch
    ):
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(2.90),
            _p12e(),
            _p12f({"ZERO": _welt(0, 0, False)}),
        )
        assert "Kein einziger Ausreißer ist durch die Bereinigung entstanden" in text
        # Und die ehrliche Einordnung: 28 von 458 ist keine Sauberkeit.
        assert "Untergrenze" in text
        assert "430 bleiben stehen" in text
        assert "nach oben" in text and "nach unten" in text

    def test_neue_ausreisser_entwerten_den_abschnitt(self, tmp_path, monkeypatch):
        """Der Zweig, der beim ersten Lauf hätte feuern müssen und nicht existierte."""
        f = _p12f({"ZERO": _welt(0, 0, False)})
        f["gegenprobe"]["neu_entstanden"] = 3
        text = _rendere(tmp_path, monkeypatch, _p12d(2.90), _p12e(), f)
        assert "3 neue Ausreißer entstanden durch die Bereinigung selbst" in text
        assert "nicht verwendbar" in text
        assert "Kein einziger Ausreißer" not in text

    def test_dividendeninvariante_wird_beziffert(self, tmp_path, monkeypatch):
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(2.90),
            _p12e(),
            _p12f({"ZERO": _welt(0, 0, False)}),
        )
        assert "Dividende je Kurseinheit" in text
        assert "26 % auf 274 %" in text


class TestVerdiktKriterium:
    """Das Verdikt hängt an BESTANDEN, nicht am Renditevergleich.

    Der echte Lauf zeigt beides auseinanderfallen: 5 von 24 Parametrisierungen
    schlagen den Benchmark bei der Rendite, keine hält zusätzlich den
    Drawdown-Deckel. Eine frühere Fassung dieses Renderers hätte daraus
    „keine Parametrisierung schlägt den passiven Vergleich ist robust"
    gemacht — eine Aussage, die die Daten nicht tragen und die die Kampagne
    nie gemacht hat (P2 hielt das Gegenteil fest).
    """

    def test_beide_zahlen_stehen_getrennt_im_dokument(self, tmp_path, monkeypatch):
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(2.90),
            _p12e(),
            _p12f({"ZERO": _welt(0, 0, False, n_schlagen=5)}),
        )
        assert "5/24" in text  # schlaegt Benchmark bei der Rendite
        assert "**0/24**" in text  # besteht Zielfunktion + Deckel
        assert "keine Parametrisierung schlägt" not in text
        # Der Satz nennt jetzt zusaetzlich den Abstand zum Kipppunkt — eine
        # blosse „robust"-Behauptung waere von einer gesaettigten Messung
        # getragen (F-auditor-1).
        assert "Das Verdikt dreht in keiner Steuerwelt" in text
        assert "wie weit der Ausgang vom Kippen entfernt war" in text

    def test_dreher_haengt_an_bestanden_nicht_an_schlaegt(self, tmp_path, monkeypatch):
        """Gleich viele schlagen, aber bereinigt besteht eine — das dreht."""
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(2.90),
            _p12e(),
            _p12f({"GMBH+FK": _welt(0, 1, False, n_schlagen=5)}),
        )
        assert "Das Verdikt dreht in GMBH+FK" in text

    def test_gleiche_bestanden_zahl_dreht_nicht_auch_wenn_schlaegt_variiert(
        self, tmp_path, monkeypatch
    ):
        """Und umgekehrt: mehr Rendite-Sieger allein drehen nichts."""
        f = _p12f({"ZERO": _welt(2, 2, False, n_schlagen=5)})
        f["welten"]["ZERO"]["bereinigt"]["n_schlagen_bench"] = 9
        text = _rendere(tmp_path, monkeypatch, _p12d(2.90), _p12e(), f)
        assert "dreht in keiner Steuerwelt" in text

    def test_deckel_wird_beziffert(self, tmp_path, monkeypatch):
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(2.90),
            _p12e(),
            _p12f({"ZERO": _welt(0, 0, False)}),
        )
        assert "MaxDD ≥ -35 %" in text or "MaxDD ≥ −35 %" in text


class TestWanderungsDimension:
    """„Das Optimum wandert" ist zu grob — es kommt darauf an, WAS wandert.

    P2 stützte seinen Schluss („nicht die Steuer bindet, sondern der Turnover")
    auf Mindesthaltedauer und `rank_out` und wies die Hebelwahl ausdrücklich
    als unterschiedlich aus. Wandert nur der Hebel, ist der Schluss unberührt.
    Im realen Lauf ist genau das der Fall — eine undifferenzierte Formulierung
    hätte einen intakten Befund als erschüttert dargestellt.
    """

    def _mit_hebelwanderung(self) -> dict:
        f = _p12f({"PRIVAT_DE": _welt(0, 0, False)})
        w = f["welten"]["PRIVAT_DE"]
        w["optimum_wandert"] = True
        w["bereinigt"]["bester"] = dict(w["original"]["bester"], hebel=1.5)
        w["original"]["zeilen"] = [
            _zeile(hebel=1.0, median_kandidat=2.1679),
            _zeile(hebel=1.5, median_kandidat=2.1560),
        ]
        return f

    def test_nur_hebel_laesst_den_P2_schluss_stehen(self, tmp_path, monkeypatch):
        text = _rendere(
            tmp_path, monkeypatch, _p12d(2.90), _p12e(), self._mit_hebelwanderung()
        )
        assert "ausschließlich in der Dimension Hebel" in text
        assert "überlebt die Bereinigung unverändert" in text
        assert "tönernen Füßen" not in text

    def test_enge_des_rennens_wird_beziffert(self, tmp_path, monkeypatch):
        """2,1679 gegen 2,1560 sind 0,55 % — das gehört in den Text."""
        text = _rendere(
            tmp_path, monkeypatch, _p12d(2.90), _p12e(), self._mit_hebelwanderung()
        )
        assert "0,55 %" in text
        assert "über die Auflösung der Messung" in text

    def test_wandernde_haltedauer_erschuettert_den_schluss_sehr_wohl(
        self, tmp_path, monkeypatch
    ):
        """Die Gegenprobe: bei der Handelsweise selbst ist der Alarm berechtigt."""
        f = self._mit_hebelwanderung()
        f["welten"]["PRIVAT_DE"]["bereinigt"]["bester"]["haltetage"] = 90
        text = _rendere(tmp_path, monkeypatch, _p12d(2.90), _p12e(), f)
        assert "tönernen Füßen" in text
        assert "Mindesthaltedauer" in text


class TestBezugsgroesseUndKonsistenz:
    """Stage-1-Findings F4, F8, F9, F11 — jeder ein stiller Fehlgriff."""

    def test_ueberhoehung_wird_gegen_das_PIT_universum_ausgewiesen(
        self, tmp_path, monkeypatch
    ):
        """F4: der Text nannte SPY, gerechnet wurde gegen das PIT-Universum.

        Beide Zahlen sind legitim, aber sie sind verschieden (2,90 gegen 3,36)
        und messen Verschiedenes. Der PIT-Vergleich isoliert die Auswahl; gegen
        SPY käme Gewichtung und Indexkonstruktion hinzu (E-079).
        """
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(2.90),
            _p12e(),
            _p12f({"ZERO": _welt(0, 0, False)}),
        )
        assert "gegenüber dem survivorship-freien **pit_2004**" in text
        # Die SPY-Zahl steht daneben, als Einordnung: 11,00 % - 7,70 % = 3,30 pp
        # (im echten Lauf 3,36 — die Fixture rundet die CAGR gröber).
        assert "Gegen **SPY** wäre die Zahl mit 3,30 pp noch größer" in text

    def test_ueberhoehung_ist_eine_spanne_keine_einzelzahl(self, tmp_path, monkeypatch):
        """F10: P12d rechnet zwei Delisting-Behandlungen, beide gehören hin."""
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(2.90),
            _p12e(),
            _p12f({"ZERO": _welt(0, 0, False)}),
        )
        assert "2,32 bis 2,90 Prozentpunkte" in text
        assert "Scheinpräzision" in text

    def test_kurzfassung_und_abschnitt_widersprechen_sich_nie(
        self, tmp_path, monkeypatch
    ):
        """F8: zwei Schwellen auf dieselbe Größe im selben Dokument.

        Mit halten = 1,0 pp und umschichten = 2,0 pp entwarnte die Kurzfassung
        (die nur `cagr_halten` las), während Abschnitt 1 Alarm schlug (der
        `max` las). Beide lesen jetzt denselben unteren Rand.
        """
        d = _p12d(2.0)
        d["ueberhoehung_cagr"] = {"cagr_halten": 0.010, "cagr_umschichten": 0.020}
        text = _rendere(
            tmp_path, monkeypatch, d, _p12e(), _p12f({"ZERO": _welt(0, 0, False)})
        )
        alarm = "kann die Frage nicht entscheiden" in text
        entwarnung = "nicht groß genug" in text
        assert alarm != entwarnung, "Dokument sagt beides gleichzeitig"
        # Unterer Rand 1,0 < Marge 1,5 -> Entwarnung ist die richtige Seite.
        assert entwarnung

    def test_satz_ueber_alle_steuerwelten_wird_berechnet(self, tmp_path, monkeypatch):
        """F9: der Satz stand unbedingt im else-Zweig, ohne je geprüft zu sein."""
        f = _p12f({"ZERO": _welt(0, 0, False), "PRIVAT_DE": _welt(0, 0, False)})
        w = f["welten"]["PRIVAT_DE"]
        w["optimum_wandert"] = True
        w["bereinigt"]["bester"] = dict(w["original"]["bester"], hebel=1.5)
        w["original"]["zeilen"] = [
            _zeile(hebel=1.0, median_kandidat=2.0),
            _zeile(hebel=1.5, median_kandidat=1.9),
        ]
        # ZERO bekommt eine ANDERE Haltedauer -> der Satz darf nicht mehr fallen
        f["welten"]["ZERO"]["original"]["bester"]["haltetage"] = 90
        f["welten"]["ZERO"]["bereinigt"]["bester"]["haltetage"] = 90
        text = _rendere(tmp_path, monkeypatch, _p12d(2.90), _p12e(), f)
        assert "sind **nicht** über alle Steuerwelten identisch" in text
        assert "2 verschiedene Kombinationen" in text
        assert "überlebt die Bereinigung unverändert" not in text

    def test_schlimmster_haltefall_wird_nach_BETRAG_gewaehlt(
        self, tmp_path, monkeypatch
    ):
        """F11: die alte Auswahl konnte nur positive Extremtage erreichen.

        Hier ist der negative Ausschlag der größere. Wird nach Rang statt nach
        Betrag gewählt, nennt das Dokument den falschen Namen.
        """
        e = _p12e()
        e["halte_kanal"]["CFC"]["groesste_wirkung"] = -0.30
        e["halte_kanal"]["CFC"]["groesste_wirkung_betrag"] = 0.30
        text = _rendere(
            tmp_path, monkeypatch, _p12d(2.90), e, _p12f({"ZERO": _welt(0, 0, False)})
        )
        assert "CFC lag an seinem Übergangstag" in text
        assert "-30,00 % Portfolio-Rendite" in text
        # Rang 5.532 von 5.548 ist der 17.-extremste vom naeheren Ende her.
        assert "17-extremste Tag" in text


class TestArtefaktAktualitaet:
    """F-senior-1 (BLOCKER): ein Dokument aus zwei Detektorgenerationen.

    `korruptions_spannen` wurde repariert; das Ergebnis-JSON des zweiten
    Konsumenten blieb vom alten Lauf stehen. Abschnitt 2 des Befunds kam aus dem
    alten Detektor (246 Übergangstage), Abschnitt 4 aus dem neuen (25) — beide
    Zahlen plausibel, das Dokument syntaktisch einwandfrei. Ein generiertes
    Dokument garantiert eben nur Konsistenz zwischen Zahl und Satz, nicht
    zwischen Artefakt und Code.
    """

    def test_veraltetes_p12e_bricht_ab_statt_zu_rendern(self, tmp_path, monkeypatch):
        e = _p12e()
        for v in e["korrumpierte_namen"].values():
            v.pop("unaufloesbar_grund", None)  # so sah das alte Artefakt aus
        res = tmp_path / "results"
        res.mkdir()
        for name, obj in (
            ("p12d_survivorship", _p12d(2.9)),
            ("p12e_panel_hygiene", e),
            ("p12f_neulauf_bereinigt", _p12f({"ZERO": _welt(0, 0, False)})),
        ):
            (res / f"{name}.json").write_text(json.dumps(obj), encoding="utf-8")
        monkeypatch.setattr(R, "RES", res)
        monkeypatch.setattr(R, "ZIEL", tmp_path / "BEFUND.md")
        assert R.main() == 1
        assert not (tmp_path / "BEFUND.md").exists()

    def test_auseinanderlaufende_laeufe_brechen_ab(self, tmp_path, monkeypatch):
        """p12f kennt einen unauflösbaren Namen, den p12e nicht meldet."""
        res = tmp_path / "results"
        res.mkdir()
        f = _p12f({"ZERO": _welt(0, 0, False)}, unaufloesbar=["GIBTESNICHT"])
        for name, obj in (
            ("p12d_survivorship", _p12d(2.9)),
            ("p12e_panel_hygiene", _p12e()),
            ("p12f_neulauf_bereinigt", f),
        ):
            (res / f"{name}.json").write_text(json.dumps(obj), encoding="utf-8")
        monkeypatch.setattr(R, "RES", res)
        monkeypatch.setattr(R, "ZIEL", tmp_path / "BEFUND.md")
        assert R.main() == 1

    def test_aktuelle_artefakte_rendern_normal(self, tmp_path, monkeypatch):
        """Die Gegenprobe — sonst wäre der Guard nur eine Bremse."""
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(2.90),
            _p12e(),
            _p12f({"ZERO": _welt(0, 0, False)}),
        )
        assert "Trägt der Datensatz die Verdikte?" in text


class TestKurzfassungFolgtDenDaten:
    """F-senior-3: die Kurzfassung war datenunabhängig formuliert.

    Sie ist die Stelle, die gelesen wird — und behauptete „drehen dennoch kein
    Verdikt", auch wenn Abschnitt 4 desselben Dokuments das Gegenteil zeigte.
    """

    def test_drehendes_verdikt_steht_auch_in_der_kurzfassung(
        self, tmp_path, monkeypatch
    ):
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(2.90),
            _p12e(),
            _p12f({"GMBH+FK": _welt(0, 3, False)}),
        )
        kurz = text.split("## 1.")[0]
        assert "drehen das Verdikt** in GMBH+FK" in kurz
        assert "drehen dennoch kein Verdikt" not in kurz

    def test_nicht_drehendes_verdikt_ebenso(self, tmp_path, monkeypatch):
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(2.90),
            _p12e(),
            _p12f({"ZERO": _welt(0, 0, False)}),
        )
        kurz = text.split("## 1.")[0]
        assert "drehen dennoch kein Verdikt" in kurz
        assert "drehen das Verdikt" not in kurz

    def test_unkontaminiertes_panel_wird_nicht_als_kontaminiert_gemeldet(
        self, tmp_path, monkeypatch
    ):
        e = _p12e()
        e["kontaminiert"] = False
        text = _rendere(
            tmp_path, monkeypatch, _p12d(2.90), e, _p12f({"ZERO": _welt(0, 0, False)})
        )
        kurz = text.split("## 1.")[0]
        assert "**nicht** in die Ergebnisse eingegangen" in kurz


class TestKippAbstand:
    """F-auditor-1: „robust" war von einer gesättigten Messung getragen.

    Keine der 24 Parametrisierungen hält den Drawdown-Deckel — in keinem Panel.
    Ein Test, dessen Ergebnis per Konstruktion nicht kippen kann, entwarnt
    nicht; er misst nichts. Erst der Abstand zum Kipppunkt macht die Aussage
    belastbar, und dann ist sie stärker als vorher.
    """

    def test_abstand_und_wirkung_werden_berechnet(self):
        f = _p12f({"ZERO": _welt(0, 0, False)})
        abstand, wirkung = R.kipp_abstand(f)
        # Bester DD im Original -0,62; bereinigt -0,64 (Versatz -0,02).
        assert abstand["bester_dd"] == pytest.approx(-0.62)
        assert abstand["deckel"] == pytest.approx(-0.35)
        assert abstand["abstand_pp"] == pytest.approx(27.0)
        assert wirkung == pytest.approx(0.02)
        # 27 pp Abstand gegen 2 pp Wirkung -> Faktor 13,5.
        assert abstand["faktor"] == pytest.approx(13.5)

    def test_abstand_ist_positiv_auch_wenn_alles_unter_dem_deckel_liegt(self):
        """Ohne abs() käme die Zahl negativ heraus — und der Faktor gleich mit.

        Eine Kennzahl, die ihre eigene Aussage umdreht, ist schlimmer als keine:
        „−26,2 Prozentpunkte Abstand" liest sich wie ein Überschreiten.
        """
        f = _p12f({"ZERO": _welt(0, 0, False)})
        abstand, _ = R.kipp_abstand(f)
        assert abstand["abstand_pp"] > 0
        assert abstand["faktor"] > 0

    def test_ohne_wirkung_ist_der_faktor_unendlich(self):
        """Bereinigung ohne jede Wirkung: dann kann nichts kippen."""
        f = _p12f({"ZERO": _welt(0, 0, False)})
        for welt in f["welten"].values():
            welt["bereinigt"]["zeilen"] = [dict(z) for z in welt["original"]["zeilen"]]
        abstand, wirkung = R.kipp_abstand(f)
        assert wirkung == 0.0
        assert abstand["faktor"] == float("inf")

    def test_dokument_nennt_abstand_und_faktor(self, tmp_path, monkeypatch):
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(2.90),
            _p12e(),
            _p12f({"ZERO": _welt(0, 0, False)}),
        )
        assert "wie weit der Ausgang vom Kippen entfernt war" in text
        assert "27,0 Prozentpunkte" in text
        assert "13,5-mal stärker" in text


class TestZeilenWechsel:
    """F-auditor-2: der 0-gegen-nicht-0-Schalter konnte nie feuern.

    Bei Startwerten von 6/2/2 geschlagenen Zeilen ist `n != 0` in beiden Panels
    wahr — der Absatz über die Empfindlichkeit des schwächeren Kriteriums blieb
    stumm, obwohl einzelne Parametrisierungen sehr wohl wechseln.
    """

    def test_zaehlt_einzelne_wechsel(self):
        f = _p12f({"ZERO": _welt(0, 0, False, n_schlagen=5)})
        # Im bereinigten Panel schlagen zwei Zeilen mehr.
        for i, z in enumerate(f["welten"]["ZERO"]["bereinigt"]["zeilen"]):
            z["schlaegt_bench"] = i < 7
        assert R.zeilen_mit_wechsel(f, "schlaegt_bench") == {"ZERO": 2}

    def test_ohne_wechsel_leeres_ergebnis(self):
        f = _p12f({"ZERO": _welt(0, 0, False, n_schlagen=5)})
        assert R.zeilen_mit_wechsel(f, "schlaegt_bench") == {}

    def test_dokument_meldet_die_empfindlichkeit(self, tmp_path, monkeypatch):
        f = _p12f({"ZERO": _welt(0, 0, False, n_schlagen=5)})
        for i, z in enumerate(f["welten"]["ZERO"]["bereinigt"]["zeilen"]):
            z["schlaegt_bench"] = i < 7
        text = _rendere(tmp_path, monkeypatch, _p12d(2.90), _p12e(), f)
        assert "**2 einzelne Parametrisierungen** ihren Status" in text
        assert "verschieben die Rangfolge messbar" in text

    def test_ohne_wechsel_schweigt_das_dokument(self, tmp_path, monkeypatch):
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(2.90),
            _p12e(),
            _p12f({"ZERO": _welt(0, 0, False, n_schlagen=5)}),
        )
        assert "einzelne Parametrisierungen" not in text


def _p12g(*, mit_probe: bool = True, ausscheider_stumm: bool = True) -> dict:
    """P12g-Artefakt. Kern ist die API-PROBE, nicht das Dateiverzeichnis."""
    bars_aus = 0 if ausscheider_stumm else 4200
    probe = {
        "ausscheider": {
            "LEH": {
                "name": "Lehman Brothers",
                "q_ticker": "LEHMQ",
                "probefenster": "2008-03",
                "bars_mitgliedschaftssymbol": bars_aus,
                "bars_q_ticker": bars_aus,
            },
            "BSC": {
                "name": "Bear Stearns",
                "q_ticker": None,
                "probefenster": "2007-09",
                "bars_mitgliedschaftssymbol": bars_aus,
                "bars_q_ticker": None,
            },
        },
        "kontrolle": {"AMZN": 7992, "GILD": 7521, "VRSN": 7381},
    }
    return {
        "bilanz": {
            "hinweis": "Anfrageliste, nicht Endpunkt.",
            "n_dateien": 298,
            "n_pit_mitglieder": 748,
            "n_mit_datei": 294,
            "abdeckung": 0.393,
            "ueberlebensquote_mit_datei": 0.861,
            "ueberlebensquote_ohne_datei": 0.281,
            "anreicherungsfaktor": 3.06,
            "maximal_verzerrt": False,
        },
        "api_probe": probe if mit_probe else None,
    }


class TestAbschnittPullBilanz:
    """Abschnitt 5 muss aus einer ABFRAGE stammen, nicht aus einem `ls`.

    Die erste Fassung las die Verfuegbarkeit aus fehlenden Dateien und lag
    falsch — acht von acht geprueften Namen der vermeintlichen Fehlgruppe
    lieferten Bars (E-112).
    """

    def _mit_g(self, tmp_path, monkeypatch, g: dict) -> str:
        res = tmp_path / "results"
        res.mkdir()
        for name, obj in (
            ("p12d_survivorship", _p12d(2.90)),
            ("p12e_panel_hygiene", _p12e()),
            ("p12f_neulauf_bereinigt", _p12f({"ZERO": _welt(0, 0, False)})),
            ("p12g_pull_bilanz", g),
        ):
            (res / f"{name}.json").write_text(json.dumps(obj), encoding="utf-8")
        ziel = tmp_path / "BEFUND.md"
        monkeypatch.setattr(R, "RES", res)
        monkeypatch.setattr(R, "ZIEL", ziel)
        assert R.main() == 0
        return " ".join(ziel.read_text(encoding="utf-8").split())

    def test_stumme_ausscheider_schliessen_den_weg(self, tmp_path, monkeypatch):
        text = self._mit_g(tmp_path, monkeypatch, _p12g())
        assert "Alle 2 geprüften Ausscheider liefern keine > einzige Bar" in text
        assert "Tagesdaten mit Delisting-Kursen" in text
        # Die Unterscheidung, die vorher fehlte:
        assert "erhöhen also die **Abdeckung**" in text
        assert "nicht seine **Unverzerrtheit**" in text

    def test_beide_symbole_stehen_in_der_tabelle(self, tmp_path, monkeypatch):
        """E-113: der Q-Ticker allein beweist nichts."""
        text = self._mit_g(tmp_path, monkeypatch, _p12g())
        assert "| Lehman Brothers | LEH | 0 | LEHMQ | 0 |" in text
        assert "unter dem sie **damals im Index standen**" in text

    def test_kontrollgruppe_belegt_dass_der_aufruf_geht(self, tmp_path, monkeypatch):
        text = self._mit_g(tmp_path, monkeypatch, _p12g())
        assert "3 von 3 liefern Bars (7.381–7.992" in text
        assert "kein Fehler der Abfrage" in text

    def test_liefernde_ausscheider_kehren_die_aussage_um(self, tmp_path, monkeypatch):
        text = self._mit_g(tmp_path, monkeypatch, _p12g(ausscheider_stumm=False))
        assert "der Weg über mehr Anfragen ist gangbar" in text
        assert "liefern keine einzige" not in text

    def test_ohne_probe_wird_KEINE_aussage_gemacht(self, tmp_path, monkeypatch):
        """Der wichtigste Test: kein Beleg -> kein Befund."""
        text = self._mit_g(tmp_path, monkeypatch, _p12g(mit_probe=False))
        assert "Ohne API-Probe ist hier nichts auszusagen" in text
        assert "liefern keine einzige" not in text
        assert "Tagesdaten mit Delisting-Kursen" not in text

    def test_abdeckung_wird_nicht_als_verzerrungsmass_ausgewiesen(
        self, tmp_path, monkeypatch
    ):
        """F-senior-1/7: die 39,3 % beschreiben die Anfrageliste."""
        text = self._mit_g(tmp_path, monkeypatch, _p12g())
        assert "beschreibt die Zusammensetzung der bisherigen Anfrageliste" in text
        assert "bewusst nicht als Verzerrungsmaß" in text
        # In ABSCHNITT 5 taucht der frueher prominente Faktor nicht mehr auf.
        # (In Abschnitt 3 steht er weiterhin — dort ist er korrekt, weil er
        # aus dem Tagespanel stammt und nicht aus einem Dateiverzeichnis.)
        abschnitt5 = text.split("## 5.")[1]
        assert "Anreicherungsfaktor" not in abschnitt5
        assert "3,06" not in abschnitt5

    def test_ohne_p12g_faellt_der_abschnitt_ersatzlos_weg(self, tmp_path, monkeypatch):
        text = _rendere(
            tmp_path,
            monkeypatch,
            _p12d(2.90),
            _p12e(),
            _p12f({"ZERO": _welt(0, 0, False)}),
        )
        assert "Kann der Intraday-Endpunkt" not in text
        assert "Trägt der Datensatz die Verdikte?" in text


class TestAbschnittFolgtDerEvidenz:
    """F-auditor-1/2 auf der Dokumentseite.

    Der Renderer entschied getrennt vom Skript und kam bei unvollständiger
    Evidenz zu einem Befund. Jetzt speist eine gemeinsame Funktion beide.
    """

    def _mit_g(self, tmp_path, monkeypatch, g: dict) -> str:
        res = tmp_path / "results"
        res.mkdir()
        for name, obj in (
            ("p12d_survivorship", _p12d(2.90)),
            ("p12e_panel_hygiene", _p12e()),
            ("p12f_neulauf_bereinigt", _p12f({"ZERO": _welt(0, 0, False)})),
            ("p12g_pull_bilanz", g),
        ):
            (res / f"{name}.json").write_text(json.dumps(obj), encoding="utf-8")
        ziel = tmp_path / "BEFUND.md"
        monkeypatch.setattr(R, "RES", res)
        monkeypatch.setattr(R, "ZIEL", ziel)
        assert R.main() == 0
        return " ".join(ziel.read_text(encoding="utf-8").split())

    def test_fehlgeschlagene_abfragen_ergeben_keinen_befund(
        self, tmp_path, monkeypatch
    ):
        g = _p12g()
        for v in g["api_probe"]["ausscheider"].values():
            v["bars_mitgliedschaftssymbol"] = "ERR:HTTPError"
            v["bars_q_ticker"] = "ERR:HTTPError"
        text = self._mit_g(tmp_path, monkeypatch, g)
        assert "Die Probe ist unvollständig" in text
        assert "Für die Survivorship-Korrektur ist dieser Weg zu" not in text

    def test_tote_kontrollgruppe_ergibt_keinen_befund(self, tmp_path, monkeypatch):
        g = _p12g()
        g["api_probe"]["kontrolle"] = {"AMZN": 0, "GILD": 0, "VRSN": 0}
        text = self._mit_g(tmp_path, monkeypatch, g)
        assert "Die Probe ist unvollständig" in text
        assert "liefert 0 von 3" in text

    def test_leere_kontrollgruppe_kracht_nicht(self, tmp_path, monkeypatch):
        """F-auditor-2: `min()` auf leerer Sequenz warf einen ValueError."""
        g = _p12g()
        g["api_probe"]["kontrolle"] = {}
        text = self._mit_g(tmp_path, monkeypatch, g)
        assert "Die Probe ist unvollständig" in text

    def test_teilweise_stumm_verallgemeinert_nicht(self, tmp_path, monkeypatch):
        g = _p12g()
        g["api_probe"]["ausscheider"]["BSC"]["bars_mitgliedschaftssymbol"] = 4200
        text = self._mit_g(tmp_path, monkeypatch, g)
        assert "1 von 2 geprüften Ausscheidern**" in text
        assert "nicht pauschal blind" in text
        assert "Für die Survivorship-Korrektur ist dieser Weg zu" not in text

    def test_minutenbars_werden_benannt(self, tmp_path, monkeypatch):
        """F-auditor-4: „Bars" neben Stundenbar-Zahlen war missverständlich."""
        text = self._mit_g(tmp_path, monkeypatch, _p12g())
        assert "Gezählt werden **Minutenbars**" in text

    def test_kontrollfenster_wird_als_argument_genannt(self, tmp_path, monkeypatch):
        text = self._mit_g(tmp_path, monkeypatch, _p12g())
        assert "eine reine Datumsgrenze scheidet damit als Erklärung aus" in text
