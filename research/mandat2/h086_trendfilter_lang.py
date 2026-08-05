"""H-086 — Trägt der Trendfilter auch in krisenfreien Jahrzehnten?

Registriert als Welle 47 VOR diesem Lauf. Die Pass/Fail-Kriterien stehen dort
und werden hier nicht neu erfunden.

DIE FRAGE
---------
P13 hat gezeigt: der SPY-Trendfilter besteht die Zielfunktion breit — aber im
Suchfenster 1995–2016 ist **kein einziges** der 144 rollierenden
10-Jahres-Fenster krisenfrei. „Trendfolge wirkt" liess sich dort nicht von
„Trendfolge hat 2000–2002 und 2008 umgangen" trennen.

Die CRSP-Marktreihe ab 1926 enthält beides: 338 der 1.080 Fenster sind
krisenfrei, verteilt auf 4 disjunkte Blöcke.

WAS HIER GEMESSEN WIRD
----------------------
Genau eine Konfiguration — `preis > SMA200`, a priori — gegen **dieselbe Reihe
ohne Filter**. Kein ETF-Vergleich, damit E-079 gar nicht greifen kann.
Entscheidend ist die **Aufspaltung** nach Krisen- und krisenfreien Fenstern,
nicht der Gesamtmedian.

WAS DAS NICHT IST
-----------------
Kein Deployability-Test: vor den 1970ern gab es keine Indexfonds, und die
Kosten lagen um Größenordnungen über den hier angesetzten. Die Reihe ist ein
Mechanismus-Labor, kein handelbares Instrument. Kein Ersatz für den
gescheiterten PBO. Kein Holdout — die Kampagnendaten bleiben unberührt.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings(
    "ignore", message=".*Converting to PeriodArray.*", category=UserWarning
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.mandat2.campaign_data import CampaignData  # noqa: E402
from research.mandat2.data_gate import TrialCounter  # noqa: E402
from research.mandat2.engine import run_buy_and_hold, sma_gate  # noqa: E402
from research.mandat2.metrics import DD_DECKEL, auswerten  # noqa: E402
from research.mandat2.p13c_ereignisabhaengigkeit import KRISEN_DD  # noqa: E402
from research.mandat2.tax_regimes import make_regime  # noqa: E402

HIER = Path(__file__).resolve().parent
GRATIS = HIER / "data_gratis"
ZIEL = HIER / "results" / "h086_trendfilter_lang.json"

#: Der a-priori-Parameter aus P13c. NICHT aus einem Raster gewaehlt.
FENSTER = 200

#: Spaltenname im Panel. Der Name sagt, was die Reihe ist — CRSP
#: value-weighted, nicht SPY (E-079).
SYMBOL = "MKT_CRSP_VW"


def panel() -> CampaignData:
    """Die CRSP-Reihe in das Format bringen, das die Engine erwartet.

    Bewusst KEINE eigene Backtest-Logik: dieselbe `run_buy_and_hold`, dieselbe
    `auswerten`-Zielfunktion wie im Rest der Kampagne. Nur die Kursquelle ist
    eine andere (Rule 50).
    """
    p = GRATIS / "fama_french_daily.parquet"
    if not p.exists():
        raise SystemExit(
            "[ERROR] fama_french_daily.parquet fehlt — erst "
            "pull_gratis_quellen.py laufen lassen."
        )
    reihe = pd.read_parquet(p)["index_crsp_vw"]
    close = pd.DataFrame({SYMBOL: reihe})
    # Keine Dividenden separat: die CRSP-Marktrendite IST total return, die
    # Ausschuettung steckt schon in der Reihe. Ein zusaetzliches Div-Panel
    # wuerde sie doppelt zaehlen.
    leer = pd.DataFrame(index=close.index)
    return CampaignData(
        close=close,
        div_panel=leer,
        # Keine Index-Mitgliedschaft: es wird nichts ausgewaehlt, es gibt nur
        # dieses eine Instrument. Genau deshalb greift hier weder Survivorship
        # noch Befund 7.
        membership=pd.Series(dtype=object),
        fenster="CRSP-1926 (NICHT Kampagnen-Suchfenster)",
        von=close.index.min(),
        bis=close.index.max(),
    )


# KEIN eigenes Gate: `engine.sma_gate` tut bit-identisch dasselbe. Der erste
# Entwurf hatte es nachgebaut — eine zweite Wahrheit ohne Not (Rule 50), die
# der Stage-1-Review als `.equals() is True` nachgewiesen hat.


def excess_panel() -> CampaignData:
    """Dieselbe Reihe, aber als UEBERSCHUSSRENDITE ueber den risikolosen Satz.

    Antwort auf einen einseitigen Nachteil: der gefilterte Lauf haelt an 28 %
    der Auswertungstermine Cash und bekommt dafuer 0 %, waehrend der Benchmark
    immer investiert ist. Ueber 1926-2026 ist das nicht neutral — der RF-Median
    liegt bei 2,55 % p. a., das Maximum bei 14,66 %, und die Hochzinsphasen
    (1974er, 1987er) fallen in die krisenfreien Bloecke.

    Statt die Engine um eine Cash-Verzinsung zu erweitern — das waere eine
    Verhaltensaenderung fuer JEDES Kampagnenergebnis — laeuft derselbe Test auf
    `mkt_rf`. In Ueberschuss-Rechnung ist "nicht investiert" per Definition
    "zum risikolosen Satz angelegt". Beide Seiten liegen im selben Raum, die
    Engine bleibt unberuehrt (Rule 50).
    """
    p = GRATIS / "fama_french_daily.parquet"
    df = pd.read_parquet(p)
    ueberschuss = (1.0 + df["mkt_rf"]).cumprod() * 100.0
    close = pd.DataFrame({SYMBOL: ueberschuss})
    leer = pd.DataFrame(index=close.index)
    return CampaignData(
        close=close,
        div_panel=leer,
        membership=pd.Series(dtype=object),
        fenster="CRSP-1926 Ueberschuss (Cash = RF)",
        von=close.index.min(),
        bis=close.index.max(),
    )


def _taeglich_gegatet(d: CampaignData, gate: pd.Series):
    """Denselben Lauf mit TAEGLICHER Gate-Auswertung statt nur an Monatsenden.

    Die Engine liest `risk_off_gate` bewusst nur an Monatsenden — das ist die
    Konvention der ganzen Kampagne. Fuer die Sensitivitaet wird hier jeder
    Handelstag zum Auswertungstermin gemacht, indem die Engine auf einem
    Kalender laeuft, dessen Monatsenden alle Tage sind. Umgesetzt ueber ein
    temporaeres Umschreiben von `_monatsenden` — eine Messung, KEIN zweiter
    Backtest-Pfad: gebucht wird weiterhin von `run_buy_and_hold` (E-102).
    """
    from research.mandat2 import engine as _engine

    original = _engine._monatsenden
    _engine._monatsenden = lambda idx: set(idx)  # type: ignore[assignment]
    try:
        return run_buy_and_hold(
            d, make_regime("ZERO"), symbol=SYMBOL, risk_off_gate=gate
        )
    finally:
        _engine._monatsenden = original  # type: ignore[assignment]


def aufspalten(fenster_liste, label: str) -> dict:
    """Vorsprung getrennt nach Krisen- und krisenfreien Fenstern.

    Das ist die entscheidende Auswertung dieses Laufs — nicht der
    Gesamtmedian. Zusaetzlich die Zahl disjunkter Bloecke, weil die 338
    krisenfreien Fenster monatlich ueberlappen (E-078).
    """
    krise = [f for f in fenster_liste if f.benchmark_maxdd <= KRISEN_DD]
    ruhig = [f for f in fenster_liste if f.benchmark_maxdd > KRISEN_DD]

    def bloecke(gruppe) -> int:
        # Aufsteigend sortiert ist Vorbedingung: bei negativen Differenzen
        # greift die Abstandspruefung nie und Episoden verschmelzen still
        # ([1990, 1950, 1951] ergaebe 1 statt 2 Bloecke).
        starts = [f.start for f in gruppe]
        assert starts == sorted(starts), "Fenster muessen aufsteigend sortiert sein"
        n, letzter = 0, None
        for f in gruppe:
            if letzter is None or (f.start - letzter).days > 10 * 365:
                n += 1
            letzter = f.start
        return n

    def kennzahlen(gruppe, name: str) -> dict:
        if not gruppe:
            # Fail-loud: eine leere Gruppe ist der interessante Fall und darf
            # nicht wie eine gemessene Null aussehen (E-103).
            return {
                "name": name,
                "n": 0,
                "median_kandidat": None,
                "median_benchmark": None,
                "vorsprung_pp": None,
                "gewonnen": None,
                # None wie alle anderen Felder: eine 0 saehe wie ein
                # gemessener Wert aus. Das war die einzige Stelle, an der die
                # leere Gruppe eine Zahl lieferte (E-103).
                "disjunkte_bloecke": None,
                "schlimmster_kandidat_maxdd": None,
            }
        k = statistics.median(f.kandidat_faktor for f in gruppe)
        b = statistics.median(f.benchmark_faktor for f in gruppe)
        # Ergebnis JE BLOCK. Die Registrierung (Welle 47) macht die disjunkten
        # Bloecke zum Entscheidungskriterium ("nur in einem Block = Episode,
        # kein Mechanismus") — ohne diese Aufschluesselung ist genau das aus
        # dem Artefakt heraus nicht pruefbar (E-131).
        je_block: list[dict] = []
        letzter = None
        for f in gruppe:
            if letzter is None or (f.start - letzter).days > 10 * 365:
                je_block.append(
                    {
                        "start_jahr": f.start.year,
                        "n": 0,
                        "gewonnen": 0,
                        "kandidat": [],
                        "benchmark": [],
                    }
                )
            # NICHT `b` als Namen: das ueberschriebe den Benchmark-Median
            # dieser Funktion und liess `round(b, 4)` auf einem dict landen.
            blk = je_block[-1]
            blk["n"] += 1
            blk["gewonnen"] += int(f.kandidat_faktor > f.benchmark_faktor)
            blk["kandidat"].append(f.kandidat_faktor)
            blk["benchmark"].append(f.benchmark_faktor)
            letzter = f.start
        for blk in je_block:
            blk["median_kandidat"] = round(statistics.median(blk.pop("kandidat")), 4)
            blk["median_benchmark"] = round(statistics.median(blk.pop("benchmark")), 4)

        return {
            "name": name,
            "n": len(gruppe),
            "je_block": je_block,
            "median_kandidat": round(k, 4),
            "median_benchmark": round(b, 4),
            "vorsprung_pp": round((k - b) * 100.0, 2),
            "gewonnen": sum(
                1 for f in gruppe if f.kandidat_faktor > f.benchmark_faktor
            ),
            "disjunkte_bloecke": bloecke(gruppe),
            "schlimmster_kandidat_maxdd": round(
                min(f.kandidat_maxdd for f in gruppe), 4
            ),
        }

    return {
        "label": label,
        "krisenfenster": kennzahlen(krise, "mit Rueckgang <= 30 %"),
        "krisenfreie_fenster": kennzahlen(ruhig, "ohne solchen Rueckgang"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--regen",
        action="store_true",
        help=(
            "Artefakt neu erzeugen, OHNE den Trial-Zaehler zu erhoehen — fuer "
            "Wiederholungen nach einem Bugfix (E-090)."
        ),
    )
    args = ap.parse_args(argv)
    (HIER / "results").mkdir(exist_ok=True)
    d = panel()
    print(f"CRSP-Reihe: {d.close.index.min().date()} .. {d.close.index.max().date()}")
    # `args.regen` MUSS hier gelesen werden. Der erste Entwurf deklarierte das
    # Flag, band `parse_args` an `args` und las es nie — der Zaehler lief
    # unbedingt, und der Regenerationslauf zaehlte H-086 ein zweites Mal
    # (1566 -> 1567), waehrend die Commit-Message "KEIN zusaetzlicher Trial"
    # behauptete. Ein Flag, das im Hilfetext eine Garantie gibt und sie nicht
    # einloest, ist schlimmer als keins (E-129).
    if args.regen:
        print(
            f"[REGEN] Trial-Zaehler UNVERAENDERT bei {TrialCounter().total()} — "
            f"dieselbe Frage zaehlt nicht zweimal (E-090).\n",
            flush=True,
        )
    else:
        print(
            f"Trials kumuliert: "
            f"{TrialCounter().increment(1, label='H-086 Trendfilter lang')}\n",
            flush=True,
        )

    bench = run_buy_and_hold(d, make_regime("ZERO"), symbol=SYMBOL)
    r = run_buy_and_hold(
        d,
        make_regime("ZERO"),
        symbol=SYMBOL,
        risk_off_gate=sma_gate(d.close, symbol=SYMBOL, fenster=FENSTER),
    )
    a = auswerten(r.equity_netto, bench.equity_netto, label=f"CRSP/preis>sma{FENSTER}")
    spalt = aufspalten(a.fenster, f"preis>SMA{FENSTER} auf CRSP 1926-2026")

    # SENSITIVITAET — die Engine liest das Gate nur an Monatsenden. Das ist
    # die Konvention der gesamten Kampagne und deshalb entscheidend; sie ist
    # NICHT fuer dieses Ergebnis gewaehlt worden. Ein taeglich ausgewertetes
    # Gate bewegt die Effektgroesse aber erheblich, und das gehoert ins
    # Artefakt statt in eine Fussnote (Stage-1-Befund).
    #
    # KEIN zusaetzlicher Trial: hier wird nichts ausgewaehlt. Beide Konventionen
    # liefern dasselbe Verdikt (kein Vorsprung ohne Krise) — es gibt also keine
    # Variante, die man behalten koennte, und damit keine Suche (E-090).
    # P13b zaehlte, WEIL dort eine Auswahl moeglich gewesen waere.
    taeglich = _taeglich_gegatet(d, sma_gate(d.close, symbol=SYMBOL, fenster=FENSTER))
    a_tag = auswerten(
        taeglich.equity_netto, bench.equity_netto, label="CRSP/taeglich ausgewertet"
    )
    spalt_tag = aufspalten(a_tag.fenster, "dasselbe Gate, taeglich ausgewertet")

    # SENSITIVITAET 2 — Cash zum risikolosen Satz statt zu 0 %. Der gefilterte
    # Lauf ist an 28 % der Termine draussen; ueber 1926-2026 ist Nullzins ein
    # EINSEITIGER Nachteil (nur der Kandidat haelt Cash) und in den
    # Hochzinsbloecken 1974/1987 besonders teuer. Kein Trial: es wird nichts
    # ausgewaehlt, alle Konventionen fallen gleich aus.
    de = excess_panel()
    b_ex = run_buy_and_hold(de, make_regime("ZERO"), symbol=SYMBOL)
    r_ex = run_buy_and_hold(
        de,
        make_regime("ZERO"),
        symbol=SYMBOL,
        risk_off_gate=sma_gate(de.close, symbol=SYMBOL, fenster=FENSTER),
    )
    a_ex = auswerten(r_ex.equity_netto, b_ex.equity_netto, label="CRSP/Ueberschuss")
    spalt_ex = aufspalten(a_ex.fenster, "Cash zum risikolosen Satz")

    kf = spalt["krisenfreie_fenster"]
    kr = spalt["krisenfenster"]
    traegt = bool(kf["n"] and kf["vorsprung_pp"] is not None and kf["vorsprung_pp"] > 0)

    ergebnis = {
        "hypothese": "H-086",
        "registriert": "Welle 47, VOR dem Lauf",
        "reihe": "CRSP value-weighted (Ken French), taeglich — NICHT SPY",
        "konfiguration": f"preis>SMA{FENSTER} (a priori)",
        "n_fenster": a.n_fenster,
        "median_kandidat_gesamt": round(a.median_kandidat, 4),
        "median_benchmark_gesamt": round(a.median_benchmark, 4),
        "schlimmster_maxdd": round(a.schlimmster_maxdd, 4),
        "gerissene_fenster": len(a.gerissene_fenster),
        "dd_deckel": DD_DECKEL,
        "aufspaltung": spalt,
        "sensitivitaet_taegliche_auswertung": {
            "hinweis": (
                "Die Engine liest das Gate nur an Monatsenden (Kampagnen-"
                "Konvention, entscheidend). Taeglich ausgewertet aendert sich "
                "die Effektgroesse erheblich, das Verdikt nicht. Kein "
                "zusaetzlicher Trial: keine Auswahl moeglich, beide Konventionen "
                "fallen gleich aus (E-090)."
            ),
            "median_kandidat_gesamt": round(a_tag.median_kandidat, 4),
            "gerissene_fenster": len(a_tag.gerissene_fenster),
            "aufspaltung": spalt_tag,
        },
        "sensitivitaet_cash_zu_rf": {
            "hinweis": (
                "Gerechnet auf der UEBERSCHUSSREIHE (mkt_rf): dort bedeutet "
                "'nicht investiert' per Definition 'zum risikolosen Satz "
                "angelegt'. Die Engine bleibt unveraendert (Rule 50). Der "
                "Nullzins der Hauptrechnung ist ein EINSEITIGER Nachteil — nur "
                "der Kandidat haelt Cash, der Benchmark ist immer investiert."
            ),
            "median_kandidat_gesamt": round(a_ex.median_kandidat, 4),
            "median_benchmark_gesamt": round(a_ex.median_benchmark, 4),
            "gerissene_fenster": len(a_ex.gerissene_fenster),
            "aufspaltung": spalt_ex,
        },
        "verdikt": {
            "traegt_ohne_krise": traegt,
            "begruendung": (
                "Vorsprung in krisenfreien Fenstern > 0"
                if traegt
                else "kein Vorsprung in krisenfreien Fenstern — der P13-Effekt "
                "ist Krisenvermeidung"
            ),
        },
        "einschraenkung": (
            "Mechanismus-Labor, kein Deployability-Test: vor den 1970ern keine "
            "Indexfonds, Kosten um Groessenordnungen hoeher als angesetzt. "
            "Massgeblich sind die disjunkten Bloecke, nicht die Fensterzahl (E-078)."
        ),
    }

    print(
        f"Gesamt: Kandidat {a.median_kandidat:.3f}x vs Benchmark "
        f"{a.median_benchmark:.3f}x | MaxDD {a.schlimmster_maxdd:.1%} | "
        f"gerissen {len(a.gerissene_fenster)}/{a.n_fenster}\n"
    )
    for g in (kr, kf):
        if not g["n"]:
            print(f"  {g['name']:<26} GRUPPE LEER")
            continue
        print(
            f"  {g['name']:<26} n={g['n']:>4} in {g['disjunkte_bloecke']} Bloecken | "
            f"Kandidat {g['median_kandidat']:.3f}x vs {g['median_benchmark']:.3f}x | "
            f"Vorsprung {g['vorsprung_pp']:+.1f} pp | gewonnen "
            f"{g['gewonnen']}/{g['n']}"
        )
    print()
    print("=" * 70)
    print(
        "VERDIKT: der Filter traegt auch ohne Krise"
        if traegt
        else "VERDIKT: kein Vorsprung ohne Krise — der P13-Effekt ist Krisenvermeidung"
    )
    print("=" * 70, flush=True)

    # Artefakt als LETZTE Anweisung (E-116).
    ZIEL.write_text(
        json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n-> {ZIEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
