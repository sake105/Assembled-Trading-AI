"""Tests fuer den P13-Strang (SPY mit Trendfilter).

Schwerpunkt liegt auf den Stellen, an denen die Kampagne schon einmal falsche
Sicherheit erzeugt hat:

* eine Kennzahl, die wie ein Messwert aussieht und keiner ist (E-109),
* eine leere Gruppe, die als gemessene Null durchgeht (E-103),
* eine zweidimensionale Evidenz, die auf eine Achse verkuerzt wird (E-117),
* ein Befund, der ohne Artefakt entsteht (E-085).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.mandat2.metrics import FensterErgebnis
from research.mandat2.p13_spy_trend_robustheit import zaehle_buchungen
from research.mandat2.p13b_ausfuehrungsverzoegerung import (
    VERZOEGERUNG,
    verzoegertes_gate,
)
from research.mandat2.p13c_ereignisabhaengigkeit import KRISEN_DD, gruppiere
from research.mandat2.p13d_zufallstiming import bloecke, gemischtes_gate, p_wert
from research.mandat2.portfolio import Portfolio
from research.mandat2.render_befund_p13 import (
    MINDEST_KETTE,
    bestandene,
    ketten_je_definition,
    laden,
    laengster_block,
    tabelle,
)
from research.mandat2.tax_regimes import make_regime


# --------------------------------------------------------------- Buchungen
def _pf() -> Portfolio:
    return Portfolio(100_000.0, make_regime("ZERO"))


def test_zaehler_zaehlt_echte_buchungen() -> None:
    with zaehle_buchungen() as z:
        p = _pf()
        p.buy("SPY", 50_000.0, 100.0)
        p.sell("SPY", p.qty("SPY"), 110.0)
    assert z == {"kauf": 1, "verkauf": 1}


def test_zaehler_zaehlt_wirkungslose_aufrufe_nicht() -> None:
    """Der Grund fuer die Bestandspruefung statt eines reinen Aufrufzaehlers.

    `buy` kehrt bei `notional <= 0` frueh zurueck, ohne zu buchen. Wer die
    Aufrufe zaehlt, meldet Trades, die nie stattgefunden haben.
    """
    with zaehle_buchungen() as z:
        p = _pf()
        p.buy("SPY", 0.0, 100.0)
        p.buy("SPY", 50_000.0, 0.0)
        p.sell("SPY", 0.0, 100.0)
    assert z == {"kauf": 0, "verkauf": 0}


def test_zaehler_stellt_die_originalmethoden_wieder_her() -> None:
    original_buy, original_sell = Portfolio.buy, Portfolio.sell
    with zaehle_buchungen():
        assert Portfolio.buy is not original_buy
    assert Portfolio.buy is original_buy
    assert Portfolio.sell is original_sell


def test_zaehler_stellt_auch_nach_ausnahme_wieder_her() -> None:
    original_buy = Portfolio.buy
    with pytest.raises(RuntimeError):
        with zaehle_buchungen():
            raise RuntimeError("Lauf abgebrochen")
    assert Portfolio.buy is original_buy


def test_zaehler_ist_nach_verlassen_still() -> None:
    """Sonst zaehlen spaetere Laeufe in ein Dict eines frueheren."""
    with zaehle_buchungen() as z:
        pass
    p = _pf()
    p.buy("SPY", 50_000.0, 100.0)
    assert z == {"kauf": 0, "verkauf": 0}


# ------------------------------------------------------- Ereignisgruppierung
def _fenster(bench_dd: float, kand: float = 2.0, bench: float = 1.5) -> FensterErgebnis:
    return FensterErgebnis(
        start=pd.Timestamp("2000-01-03"),
        ende=pd.Timestamp("2010-01-04"),
        kandidat_faktor=kand,
        benchmark_faktor=bench,
        kandidat_maxdd=-0.2,
        benchmark_maxdd=bench_dd,
    )


def test_leere_gruppe_meldet_none_statt_null() -> None:
    """E-103: eine leere Gruppe darf nicht wie eine gemessene Null aussehen.

    Genau dieser Fall ist der interessante — im Suchfenster gibt es kein
    krisenfreies Fenster. Stuende dort 0.0 pp Vorsprung, laese sich das als
    'gemessen, kein Effekt' statt als 'nicht messbar'.
    """
    g = gruppiere([_fenster(-0.5), _fenster(-0.45)])
    assert g["ruhige_fenster"]["n"] == 0
    assert g["ruhige_fenster"]["median_vorsprung_pp"] is None
    assert g["ruhige_fenster"]["gewonnen"] is None


def test_schwelle_trennt_am_rand_korrekt() -> None:
    """Beide Raender pruefen, nicht die Mitte (E-104)."""
    genau = gruppiere([_fenster(KRISEN_DD)])
    assert genau["krisenfenster"]["n"] == 1, "Schwelle selbst zaehlt als Krise"
    knapp = gruppiere([_fenster(KRISEN_DD + 1e-9)])
    assert knapp["ruhige_fenster"]["n"] == 1


def test_gewonnen_zaehlt_nur_echte_siege() -> None:
    g = gruppiere(
        [
            _fenster(-0.5, kand=2.0, bench=1.5),
            _fenster(-0.5, kand=1.0, bench=1.5),
            _fenster(-0.5, kand=1.5, bench=1.5),
        ]
    )
    assert g["krisenfenster"]["gewonnen"] == 1
    assert g["krisenfenster"]["n"] == 3


# ------------------------------------------------------------ Zufallskontrolle
def _gate(muster: str) -> pd.Series:
    idx = pd.date_range("2000-01-03", periods=len(muster), freq="D")
    return pd.Series([float(c) for c in muster], index=idx)


def test_bloecke_zerlegen_laufweise() -> None:
    assert bloecke(_gate("1110011")) == [(1.0, 3), (0.0, 2), (1.0, 2)]


def test_bloecke_behandeln_nan_als_eigenen_wert() -> None:
    g = pd.Series([np.nan, np.nan, 1.0], index=pd.date_range("2000-01-03", periods=3))
    b = bloecke(g)
    assert len(b) == 2 and b[0][1] == 2 and np.isnan(b[0][0])


def test_mischen_erhaelt_anteil_anzahl_und_laengen() -> None:
    """Das ist die ganze Konstruktion: nur das Timing darf sich aendern.

    Wuerde das Mischen den investierten Anteil veraendern, waere die Kontrolle
    keine Kontrolle mehr, sondern eine andere Strategie.
    """
    g = _gate("111000110100011110")
    for seed in range(5):
        m = gemischtes_gate(g, seed)
        assert len(m) == len(g)
        assert m.sum() == g.sum(), "Anteil investierter Tage muss exakt gleich sein"
        assert sorted(x[1] for x in bloecke(m)) == sorted(x[1] for x in bloecke(g))
        assert m.index.equals(g.index)


def test_mischen_ist_je_seed_reproduzierbar_und_je_seed_verschieden() -> None:
    g = _gate("1110001101000111101010")
    assert gemischtes_gate(g, 7).equals(gemischtes_gate(g, 7))
    unterschiedlich = sum(
        1 for s in range(10) if not gemischtes_gate(g, s).equals(gemischtes_gate(g, 0))
    )
    assert unterschiedlich >= 5, "Seeds erzeugen praktisch identische Reihen"


# ------------------------------------------------------------------- Befund
def test_laengster_block_unterscheidet_breite_von_zusammenhang() -> None:
    """E-117: 10 von 12 mit einem Loch ist nicht dasselbe wie 2 von 12.

    Der erste Entwurf klassifizierte streng nach Lueckenlosigkeit und stellte
    beides gleich. Diese Funktion misst die zweite Achse getrennt.
    """
    mit_loch = [100, 120, 140, 160, 180, 200, 260, 280, 300, 320]
    assert len(laengster_block(mit_loch)) == 6
    assert len(laengster_block([200, 220])) == 2
    assert laengster_block([]) == []
    assert laengster_block([200]) == [200]


def test_laengster_block_nimmt_die_laengste_nicht_die_erste_kette() -> None:
    assert laengster_block([100, 200, 220, 240]) == [200, 220, 240]


def test_laden_bricht_ohne_artefakt_ab() -> None:
    """E-085: ein Befund ohne Artefakt ist genau der Fehler."""
    with pytest.raises(SystemExit):
        laden("p13_gibt_es_nicht.json")


def test_vorsprung_ist_in_prozentpunkten() -> None:
    """Die Einheit, nicht nur der Wert (Stage-1-Mutation M14).

    Ohne den Faktor 100 stuende im Befund `+0.6 pp` statt `+57.7 pp`, und die
    Funktion waere trotzdem 'getestet' gewesen.
    """
    g = gruppiere([_fenster(-0.5, kand=2.0, bench=1.5)])
    assert g["krisenfenster"]["median_vorsprung_pp"] == pytest.approx(50.0)


# ------------------------------------------------- Ableitung der Kopfzahlen
def _zeile(welt: str, defn: str, f: int, ok: bool) -> dict:
    return {"welt": welt, "definition": defn, "fenster": f, "bestanden": ok}


def test_bestandene_filtert_wirklich_nach_bestanden() -> None:
    """Ohne den Filter meldete die Kopfzeile 12/12 statt 9/12."""
    z = [
        _zeile("ZERO", "a", 100, True),
        _zeile("ZERO", "a", 120, False),
        _zeile("ZERO", "b", 140, True),
        _zeile("PRIVAT_DE", "a", 160, True),
    ]
    assert bestandene(z, "ZERO", "a") == [100]


def test_tabelle_bildet_die_schnittmenge_nicht_die_vereinigung() -> None:
    """Die Spalte heisst 'in beiden' — eine Vereinigung waere dort eine Luege."""
    ohne = {"zeilen": [_zeile("ZERO", "a", 100, True), _zeile("ZERO", "a", 120, True)]}
    mit = {"zeilen": [_zeile("ZERO", "a", 100, True), _zeile("ZERO", "a", 120, False)]}
    _, schnitt, n_fenster = tabelle(ohne, mit)
    assert schnitt["ZERO/a"] == [100]
    assert n_fenster == 2, (
        "Rastergroesse kommt aus den Zeilen, nicht aus einer Konstante"
    )


def test_ketten_je_definition_liefert_breite_und_zusammenhang() -> None:
    k = ketten_je_definition({"ZERO/a": [100, 120, 140, 200]}, ["a"])
    assert k["a"] == (4, 3), "vier bestanden, laengste lueckenlose Kette drei"


def test_mindest_kette_ist_zwei_drittel_des_rasters() -> None:
    """Verhindert das stille Absenken der Schwelle (Stage-1-Mutation N5)."""
    assert MINDEST_KETTE == 8


# ------------------------------------------------------------ p-Wert / Lag
def test_p_wert_hat_die_plus_eins_korrektur() -> None:
    """Ohne sie behauptete null Treffer p = 0 — mit 60 Ziehungen unmoeglich."""
    treffer, p = p_wert([1.0, 1.1, 1.2], echt=2.0, seeds=3)
    assert treffer == 0
    assert p == pytest.approx(1 / 4)


def test_p_wert_zaehlt_gleichstand_als_erreicht() -> None:
    treffer, _ = p_wert([1.0, 2.0], echt=2.0, seeds=2)
    assert treffer == 1, "wer den Kandidaten einholt, hat ihn erreicht"


def test_verzoegertes_gate_verschiebt_wirklich() -> None:
    """Ohne den shift waere die ganze Spalte 'mit Verz.' eine Kopie von P13."""
    idx = pd.date_range("2000-01-03", periods=5)
    roh = pd.Series([0.0, 1.0, 1.0, 0.0, 1.0], index=idx)
    close = pd.DataFrame({"SPY": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    g = verzoegertes_gate(lambda _close, _f: roh, close, fenster=200)
    assert np.isnan(g.iloc[0]), "erster Tag hat kein Signal von gestern"
    assert list(g.iloc[1:]) == list(roh.iloc[:-1])
    assert VERZOEGERUNG == 1


def test_verzoegerung_folgt_den_handelstagen_des_instruments() -> None:
    """F-senior-11: verschoben wird entlang SPY, nicht entlang des Panels.

    An Tag 2 handelt nur ein anderer Name; ein `shift` auf dem Panel-Index
    verzoegerte das Signal dort um zwei SPY-Tage statt um einen.
    """
    idx = pd.date_range("2000-01-03", periods=4)
    roh = pd.Series([0.0, 1.0, 0.0, 1.0], index=idx)
    close = pd.DataFrame(
        {"SPY": [1.0, 2.0, np.nan, 4.0], "AAA": [1.0, 1.0, 1.0, 1.0]}, index=idx
    )
    g = verzoegertes_gate(lambda _close, _f: roh, close, fenster=200)
    # SPY handelt an Tag 0, 1, 3 -> das Signal von Tag 1 wirkt an Tag 3.
    assert g.loc[idx[3]] == roh.loc[idx[1]]


def test_p13_artefakt_fuehrt_buchungen_und_nicht_n_trades() -> None:
    """E-109/E-119: `n_trades` ist in run_buy_and_hold eine harte 1.

    Faengt die Reintroduktion (Stage-1-Mutation N9) auf Artefakt-Ebene.
    """
    z = laden("p13_spy_trend_robustheit.json")["zeilen"][0]
    assert "kaeufe" in z and "verkaeufe" in z
    assert "n_trades" not in z


def test_befund_bandtabelle_stimmt_mit_den_artefakten() -> None:
    """Starker Drift-Guard: die Kopfzahlen werden nachgerechnet.

    Der erste Guard prueste drei Einzelwerte; ein Handedit der Bandtabelle
    ueberlebte ihn (Stage-1-Befund). Hier wird jede Zeile neu erzeugt und im
    Dokument gesucht.
    """
    from pathlib import Path

    befund = (
        Path(__file__).resolve().parents[1] / "research/mandat2/BEFUND_SPY_TREND.md"
    )
    if not befund.exists():
        pytest.skip("Befund noch nicht gerendert")
    text = befund.read_text(encoding="utf-8")
    zeilen, _, _ = tabelle(
        laden("p13_spy_trend_robustheit.json"),
        laden("p13b_ausfuehrungsverzoegerung.json"),
    )
    for z in zeilen[2:]:  # [0]/[1] sind Kopf und Trennlinie
        assert z in text, f"Bandzeile driftet gegen das Artefakt: {z}"


def test_befund_korrekturtabelle_stimmt_mit_dem_artefakt() -> None:
    """Die verdikt-tragende Tabelle braucht denselben Guard wie die Bandtabelle.

    Der erste Drift-Guard schuetzte die Bandtabelle — die wichtigere Tabelle
    (DSR je Schaetzer, PBO) war ungeschuetzt (F-auditor-6). Ein Handedit dort
    haette das Verdikt im Text drehen koennen, ohne dass ein Test anschlaegt.
    """
    from pathlib import Path

    befund = (
        Path(__file__).resolve().parents[1] / "research/mandat2/BEFUND_SPY_TREND.md"
    )
    if not befund.exists():
        pytest.skip("Befund noch nicht gerendert")
    text = befund.read_text(encoding="utf-8")
    k = laden("p13e_dsr_pbo_spy.json")
    for label in ("heterogen", "IID-Naeherung", "klonfamilie"):
        e = k["dsr"]["gewinner"][label]
        assert f"{e['sharpe_threshold']:.4f}" in text, f"Schwelle {label} driftet"
        assert f"{e['dsr_probability']:.4f}" in text, f"p-Wert {label} driftet"
    assert f"{k['pbo']:.1%}" in text
    # Und das Verdikt selbst darf nicht gegen das Artefakt stehen.
    assert k["verdikt"]["reif_fuer_holdout"] is False
    assert "kein Holdout-Schuss" in text


def test_befund_p_wert_stimmt_mit_dem_artefakt() -> None:
    from pathlib import Path

    befund = (
        Path(__file__).resolve().parents[1] / "research/mandat2/BEFUND_SPY_TREND.md"
    )
    if not befund.exists():
        pytest.skip("Befund noch nicht gerendert")
    z = laden("p13d_zufallstiming.json")["ZERO"]
    assert f"p = {z['p_wert']:.3f}" in befund.read_text(encoding="utf-8")


# --------------------------------------------------- Mehrfachtest-Korrektur
def test_familiengroesse_passt_zum_raster() -> None:
    """Eine kleinere Matrix wuerde PBO kuenstlich druecken (E-077)."""
    from research.mandat2.p5_gate_robustheit import DEFINITIONEN, FENSTER

    k = laden("p13e_dsr_pbo_spy.json")
    assert k["n_familie"] == len(DEFINITIONEN) * len(FENSTER) + 1


def test_entscheidungsgrundlage_ist_die_heterogene_varianz() -> None:
    """E-077: V aus der Klonfamilie ist NICHT entscheidungsfaehig.

    Der erste Entwurf war eine Kopie des dort verworfenen p7 und haette den
    Fehler in einen neuen Befund verlaengert.
    """
    d = laden("p13e_dsr_pbo_spy.json")["dsr"]["gewinner"]
    assert d["heterogen"]["entscheidungsfaehig"] is True
    assert d["klonfamilie"]["entscheidungsfaehig"] is False
    assert d["IID-Naeherung"]["entscheidungsfaehig"] is False
    # Die Klonvarianz senkt die Schwelle — genau der Mechanismus aus E-077.
    assert d["klonfamilie"]["sharpe_threshold"] < d["heterogen"]["sharpe_threshold"]


def test_verdikt_haengt_an_der_heterogenen_dsr_nicht_an_der_klonfamilie() -> None:
    k = laden("p13e_dsr_pbo_spy.json")
    assert (
        k["verdikt"]["dsr_kumuliert_bestanden"]
        == k["dsr"]["gewinner"]["heterogen"]["passes_5pct"]
    )
    assert k["verdikt"]["reif_fuer_holdout"] == (
        k["verdikt"]["dsr_kumuliert_bestanden"] and k["verdikt"]["pbo_bestanden"]
    )


def test_heterogene_varianz_stammt_aus_dem_p8_artefakt() -> None:
    """Derselbe Schaetzer wie beim Aktien-Kandidaten — sonst kein Vergleich."""
    import json
    from pathlib import Path

    res = Path(__file__).resolve().parents[1] / "research/mandat2/results"
    p8 = json.loads((res / "p8_dsr_heterogen.json").read_text(encoding="utf-8"))
    k = laden("p13e_dsr_pbo_spy.json")
    assert k["varianz_heterogen"] == pytest.approx(p8["varianz_heterogen"])
    assert k["n_strategien_p8"] == p8["n_strategien"]


def test_befund_zahlen_stammen_aus_den_artefakten() -> None:
    """Stichprobe gegen Drift zwischen Dokument und Lauf (E-085/E-116)."""
    from pathlib import Path

    befund = (
        Path(__file__).resolve().parents[1] / "research/mandat2/BEFUND_SPY_TREND.md"
    )
    if not befund.exists():
        pytest.skip("Befund noch nicht gerendert")
    text = befund.read_text(encoding="utf-8")
    zufall = laden("p13d_zufallstiming.json")
    ereignis = laden("p13c_ereignisabhaengigkeit.json")
    assert f"{zufall['ZERO']['echt_median']:.3f}x" in text
    assert f"{ereignis['ZERO']['benchmark_dd_verteilung']['mildester']:.1%}" in text
    assert (
        "Kein einziges krisenfreies Fenster" in text
        and ereignis["ZERO"]["ruhige_fenster"]["n"] == 0
    ), "Behauptung und Artefakt muessen zusammenpassen"
