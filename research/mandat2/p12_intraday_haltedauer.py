"""P12 — Die Haltedauer-Frage, jetzt auch am kurzen Ende.

HANS' FRAGE, WOERTLICH
----------------------
„Du kannst doch gern mit der Haltezeit der Aktien rumexperimentieren. Du kannst
Aktie auch gern laenger als paar Jahre halten. Du kannst Aktie auch kuerzer
halten, nur mehrere wenige Monate oder wenige Stunden."

P2 hat das lange Ende beantwortet (Optimum bei ~730 Tagen Mindesthaltedauer).
Das kurze Ende hatte ich faelschlich fuer datenblockiert erklaert (E-080). Es
ist es nicht. Hier wird es beantwortet.

VIER KONSTRUKTIONSREGELN, DIE DER ERSTE ENTWURF VERLETZT HATTE
--------------------------------------------------------------
Die Stage-1-Review des ersten Laufs fand vier Fehler, die alle dasselbe Muster
hatten: die Tabelle verglich Dinge, die sich in MEHR als der Haltedauer
unterschieden. Die Korrekturen sind hier Konstruktionsprinzip:

1. **Gemeinsamer Start (``WARMUP``).** Ein Momentum-Score ist am Anfang des
   Panels NaN, und der fail-closed-Pfad haelt dann Cash. Im ersten Entwurf lag
   die 2-Jahres-Variante deshalb 4 Jahre in Cash, die 1-Stunden-Variante 20
   Bars — es variierte nicht nur die Haltedauer, sondern auch der effektive
   Startzeitpunkt. Jetzt startet JEDE Variante, auch Benchmark und
   Zufallskontrolle, am selben Bar.

2. **Positionen driften zwischen den Terminen.** Der erste Entwurf hielt feste
   GEWICHTE ueber das Segment — das ist stuendliches, kostenloses Rebalancing
   und damit selbst eine Strategie. Gerechnet wird jetzt in STUECKEN: am Termin
   gekauft, danach laeuft die Position. Nur so ist „Haltedauer" das, was das
   Wort sagt.

3. **Der Benchmark ist echtes Buy-and-Hold.** Aus demselben Grund. Zusaetzlich
   wird die monatlich rebalancierte Variante ausgewiesen — der Unterschied
   zwischen beiden ist selbst eine Information, kein Rauschen.

4. **Zwei Rueckblick-Familien, getrennt ausgewiesen.** „Rueckblick = 20x
   Haltedauer" laesst das Signal MITvariieren; bei drei Zeilen lief der
   Rueckblick ausserdem in eine Deckelung und war dort identisch. Familie A
   haelt den Rueckblick fest (echter Ein-Parameter-Sweep), Familie B skaliert
   ihn mit der Haltedauer (Signal-Zeitskala folgt der Haltedauer). Beide
   Fragen sind legitim, aber es sind zwei.

BRUTTO NEBEN NETTO
------------------
Bei einstuendigem Halten fallen ueber 18.000 Umschichtungen an; bei 10 bps je
Seite dominiert die Kostenseite. Nur wer beides zeigt, kann „kein Signal" von
„zu teuer" unterscheiden — sonst haelt man ein falsches Vorzeichen fuer ein
Reibungsproblem. Da beide Buecher identische GEWICHTE halten und Kosten nur
den Massstab aendern, gilt exakt ``brutto = netto / prod(1 - kostenanteil)``;
es genuegt eine Simulation.

WAS DIESER TEST NICHT KANN
--------------------------
Keine absolute Renditeaussage. Das Universum ist NICHT survivorship-frei (21
Namen gezogen, die 2004-2016 durchgehend im Index waren; 20 nach dem
Abdeckungsfilter). Verglichen wird deshalb
ausschliesslich INNERHALB des Universums, nie gegen SPY (E-079).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from research.mandat2.data_gate import TrialCounter  # noqa: E402
from research.mandat2.intraday_data import MIN_ABDECKUNG, load_intraday  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
KOSTEN_BPS = 10.0  # identisch zu Mandat I/II
TOP_K = 5  # von 20 Namen (21 gezogen, AON faellt am Abdeckungsfilter)
N_SEEDS = 5
BARS_PRO_TAG = 7  # regulaere Sitzung 09:30-16:00 ET -> Stundenlabel 9..15

# Gemeinsamer Vorlauf fuer ALLE Varianten. WARMUP ist GESETZT, nicht abgeleitet
# — ein frueherer Kommentar nannte ihn "den laengsten verwendeten Rueckblick",
# was seit dem Wegfall der Deckelung (E-084) nicht mehr stimmt: der laengste
# tatsaechlich verwendete Rueckblick ist 2.940 Bars (Familie B, 1 Monat).
# Der Wert kostet 4.355/7 = 622 Handelstage, also ~2,47 Jahre des Fensters.
#
# Er hat eine ZWEITE Wirkung, die leicht zu uebersehen ist: er entscheidet auch,
# WELCHE Haltedauern Familie B ueberhaupt enthaelt (nur solche mit 20x Rueckblick
# <= WARMUP). Beides ist im Befund offengelegt.
WARMUP = 4355

HALTEDAUERN = [
    (1, "1 Stunde"),
    (2, "2 Stunden"),
    (4, "4 Stunden"),
    (7, "1 Tag"),
    (35, "1 Woche"),
    (147, "1 Monat"),
    (441, "1 Quartal"),
    (1764, "1 Jahr"),
    (3528, "2 Jahre"),
]
RUECKBLICK_FEST = 882  # ~6 Monate, Familie A


def simuliere(
    close: pd.DataFrame,
    *,
    halte_bars: int,
    score: pd.DataFrame,
    top_k: int = TOP_K,
    kosten_bps: float = KOSTEN_BPS,
    start: int = 0,
) -> tuple[pd.Series, pd.Series, int, float]:
    """Top-K kaufen, ``halte_bars`` Bars HALTEN, dann umschichten.

    Gerechnet wird in Stuecken, nicht in Gewichten: zwischen zwei Terminen
    driften die Positionen wie bei einem echten Depot. Rueckgabe:
    (netto-Equity, brutto-Equity, Zahl der Umschichtungen, Cash-Anteil).

    Der Cash-Anteil ist Pflichtdiagnose: nur wenn er ueber alle Varianten
    gleich (idealerweise 0) ist, misst der Sweep die Haltedauer und nicht den
    effektiven Startzeitpunkt (E-082).

    Gehandelt wird zum Kurs der Entscheidungs-Bar, ohne Slippage ueber die
    Pauschale hinaus — zugunsten des kurzen Endes verzerrt. Wenn es SO nicht
    traegt, traegt es gar nicht.
    """
    px = close.to_numpy(dtype=float)
    spalten = list(close.columns)
    n = len(close)
    kosten = kosten_bps / 10_000.0

    equity = np.full(n, np.nan)
    kostenfaktor = np.full(n, np.nan)  # kumuliertes prod(1 - anteil)
    stuecke = np.zeros(len(spalten))
    cash = 1.0
    kf = 1.0
    n_um = 0
    bars_in_cash = 0

    for t in range(start, n, halte_bars):
        preise = px[t]
        pos_wert = np.where(np.isnan(preise), 0.0, stuecke * np.nan_to_num(preise))
        wert = cash + float(pos_wert.sum())
        if wert <= 0.0:  # Totalverlust: nichts mehr zu handeln
            ende = min(t + halte_bars, n - 1)
            equity[t : ende + 1] = 0.0
            kostenfaktor[t : ende + 1] = kf
            continue

        s = score.iloc[t]
        gueltig = ~np.isnan(preise)
        kand = s[pd.Series(gueltig, index=spalten)].dropna()

        ziel = np.zeros(len(spalten))
        if len(kand) >= top_k:
            gewaehlt = kand.nlargest(top_k).index
            idx = [spalten.index(c) for c in gewaehlt]
            ziel[idx] = wert / top_k

        anteil = float(np.abs(ziel - pos_wert).sum()) / wert
        if anteil > 1e-9:
            n_um += 1
        wert_netto = wert * (1.0 - anteil * kosten)
        kf *= 1.0 - anteil * kosten

        if ziel.sum() > 0.0:
            stuecke = np.divide(
                ziel * (wert_netto / wert),
                preise,
                out=np.zeros_like(ziel),
                where=(ziel > 0) & gueltig,
            )
            cash = 0.0
        else:
            stuecke = np.zeros(len(spalten))
            cash = wert_netto

        ende = min(t + halte_bars, n - 1)
        if ziel.sum() <= 0.0:
            bars_in_cash += ende - t + 1
        for i in range(t, ende + 1):
            p = px[i]
            equity[i] = cash + float(np.nansum(np.where(stuecke > 0, stuecke * p, 0.0)))
            kostenfaktor[i] = kf

    netto = pd.Series(equity, index=close.index)
    brutto = netto / pd.Series(kostenfaktor, index=close.index)
    anteil_cash = bars_in_cash / max(1, n - start)
    return netto.iloc[start:], brutto.iloc[start:], n_um, anteil_cash


def buy_and_hold(
    close: pd.DataFrame, *, start: int = 0, kosten_bps: float = KOSTEN_BPS
) -> pd.Series:
    """Echtes Buy-and-Hold: einmal gleichgewichtet kaufen, dann nichts mehr."""
    preise = close.iloc[start]
    gueltig = preise.dropna().index
    einsatz = (1.0 - kosten_bps / 10_000.0) / len(gueltig)
    stuecke = pd.Series(0.0, index=close.columns)
    stuecke[gueltig] = einsatz / preise[gueltig]
    return (close.iloc[start:] * stuecke).sum(axis=1)


def rebalanciert(
    close: pd.DataFrame,
    *,
    start: int = 0,
    alle_bars: int = 147,
    kosten_bps: float = KOSTEN_BPS,
) -> pd.Series:
    """Gleichgewichtet, monatlich rebalanciert — der Index-Vergleichsfall.

    ``top_k`` = Zahl der Spalten: mit konstantem Score waehlt ``nlargest`` dann
    ALLE verfuegbaren Namen. Ein bei ``start`` eingefrorener Wert haette spaeter
    startende Namen gegen frueh startende getauscht (Stage-2-Finding F-senior-5).
    """
    eins = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    n, _, _, _ = simuliere(
        close,
        halte_bars=alle_bars,
        score=eins,
        top_k=close.shape[1],
        kosten_bps=kosten_bps,
        start=start,
    )
    return n


def momentum(close: pd.DataFrame, rueckblick: int) -> pd.DataFrame:
    return close.pct_change(rueckblick, fill_method=None)


def zufall(
    close: pd.DataFrame, seed: int, wie: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Zufalls-Score. ``wie`` uebertraegt die NaN-Maske eines echten Signals.

    Ohne diese Maske waere die Kontrolle investiert, waehrend das Signal noch
    im Vorlauf steckt — dann misst der Vergleich Startzeitpunkte statt
    Signalqualitaet (Stage-1-Finding F-test-3).
    """
    rng = np.random.default_rng(seed)
    z = pd.DataFrame(
        rng.standard_normal(close.shape), index=close.index, columns=close.columns
    )
    return z.where(wie.notna()) if wie is not None else z


def kennzahlen(equity: pd.Series, jahre: float) -> dict:
    """MaxDD auf STUNDENbasis — die feinere und damit ehrlichere Aufloesung."""
    e = equity.dropna()
    if e.empty or e.iloc[0] <= 0:
        return {"endwert": 0.0, "cagr": -1.0, "maxdd": -1.0}
    norm = e / e.iloc[0]
    return {
        "endwert": float(norm.iloc[-1]),
        "cagr": float(norm.iloc[-1] ** (1.0 / jahre) - 1.0)
        if norm.iloc[-1] > 0
        else -1.0,
        "maxdd": float((norm / norm.cummax() - 1.0).min()),
    }


def main() -> int:
    OUT.mkdir(exist_ok=True)
    # --stufig: Gegenprobe mit erzwungen stufigem Tagesfaktor. Der Faktor SOLL
    # eine Treppe sein; gemessen enthaelt er ~11-31 bps/Tag reversierendes
    # Rauschen, das gleichgerichtet mit dem gesuchten Intraday-Effekt ist. Die
    # Differenz beider Laeufe ist die Artefaktschranke (E-083).
    stufig = "--stufig" in sys.argv
    d = load_intraday(stufig=stufig)
    print(d, flush=True)
    if d.verworfen:
        print(f"Verworfene Symbole: {d.verworfen}")
    n_trials = 2 * len(HALTEDAUERN) * (1 + N_SEEDS) + 2
    # --regen: reiner Wiederholungslauf zur Artefakt-Hygiene, KEINE neue
    # Hypothese. Der Zaehler steuert den DSR-Haircut und bedeutet "Zahl
    # gepruefter Hypothesen" — zaehlt er Regenerationen mit, verliert er
    # genau diese Bedeutung (Anti-Pattern E-090).
    regen = "--regen" in sys.argv
    if regen:
        print("[SKIP] Trial-Zaehler: --regen (Wiederholungslauf)", flush=True)
    else:
        print(
            f"Trials kumuliert: {TrialCounter().increment(n_trials, label='P12 Intraday-Haltedauer (v2)')}\n",
            flush=True,
        )

    close = d.close.ffill()
    if len(close) <= WARMUP + 100:
        raise RuntimeError(f"Panel zu kurz fuer WARMUP={WARMUP}")
    fenster = close.index[WARMUP:]
    jahre = (fenster[-1] - fenster[0]).days / 365.25
    print(
        f"Gemeinsames Fenster ab Bar {WARMUP}: {fenster[0]:%Y-%m-%d}..{fenster[-1]:%Y-%m-%d}"
        f"  ({jahre:.2f} Jahre, {len(fenster):,} Bars)\n"
    )

    bh = buy_and_hold(close, start=WARMUP)
    k_bh = kennzahlen(bh, jahre)
    reb = rebalanciert(close, start=WARMUP)
    k_reb = kennzahlen(reb, jahre)
    print(
        f"[Buy-and-Hold ]   {k_bh['endwert']:>7.3f}x  CAGR {k_bh['cagr']:+.2%}  "
        f"MaxDD {k_bh['maxdd']:.1%}"
    )
    print(
        f"[EW monatl. reb]  {k_reb['endwert']:>7.3f}x  CAGR {k_reb['cagr']:+.2%}  "
        f"MaxDD {k_reb['maxdd']:.1%}\n"
    )

    # Familie B nur dort, wo der skalierte Rueckblick OHNE Deckelung passt.
    # Gedeckelte Zeilen waeren untereinander identisch parametriert und hatten
    # im Vorlauf ausgerechnet die Bestwerte geliefert — ein Deckelungsartefakt,
    # kein Haltedauer-Effekt (Stage-2-Finding F-senior-4, Anti-Pattern E-084).
    familien = {
        "A_fester_rueckblick": (lambda bars: RUECKBLICK_FEST, HALTEDAUERN),
        "B_skalierter_rueckblick": (
            lambda bars: max(2, bars * 20),
            [(b, n) for b, n in HALTEDAUERN if b * 20 <= WARMUP],
        ),
    }
    alles: dict[str, list[dict]] = {}

    for fam, (rueck_fn, dauern) in familien.items():
        print(f"### Familie {fam} " + "#" * (58 - len(fam)))
        print(
            f"{'Haltedauer':<12}{'Rueckbl.':>9}{'Umsch.':>8}{'netto':>10}{'brutto':>10}"
            f"{'Zufall':>10}{'MaxDD':>9}{'Kostenlast':>12}"
        )
        zeilen = []
        for bars, name in dauern:
            rueck = rueck_fn(bars)
            sig = momentum(close, rueck)
            n_m, b_m, n_um, cash_m = simuliere(
                close, halte_bars=bars, score=sig, start=WARMUP
            )
            k_m, k_b = kennzahlen(n_m, jahre), kennzahlen(b_m, jahre)

            z_ends, z_brutto = [], []
            for seed in range(N_SEEDS):
                n_z, b_z, _, _ = simuliere(
                    close,
                    halte_bars=bars,
                    score=zufall(close, seed, wie=sig),
                    start=WARMUP,
                )
                z_ends.append(kennzahlen(n_z, jahre)["endwert"])
                z_brutto.append(kennzahlen(b_z, jahre)["endwert"])

            last = (
                k_b["endwert"] / k_m["endwert"] - 1.0
                if k_m["endwert"] > 0
                else float("inf")
            )
            zeilen.append(
                {
                    "halte_bars": bars,
                    "name": name,
                    "rueckblick_bars": rueck,
                    "umschichtungen": n_um,
                    "netto_end": k_m["endwert"],
                    "brutto_end": k_b["endwert"],
                    "cagr": k_m["cagr"],
                    "maxdd": k_m["maxdd"],
                    "zufall_end_mittel": float(np.mean(z_ends)),
                    "zufall_end_alle": z_ends,
                    "zufall_brutto_mittel": float(np.mean(z_brutto)),
                    "anteil_cash": cash_m,
                    "kostenlast": float(last),
                }
            )
            print(
                f"{name:<12}{rueck:>9,}{n_um:>8,}{k_m['endwert']:>9.3f}x"
                f"{k_b['endwert']:>9.3f}x{np.mean(z_ends):>9.3f}x"
                f"{k_m['maxdd']:>9.1%}{last:>11.1%}",
                flush=True,
            )
        alles[fam] = zeilen
        print()

    ergebnis = {
        "universum": list(close.columns),
        "verworfen": d.verworfen,
        "fenster": f"{fenster[0]:%Y-%m-%d}..{fenster[-1]:%Y-%m-%d}",
        "jahre": jahre,
        "warmup_bars": WARMUP,
        "bars_pro_tag": BARS_PRO_TAG,
        "min_abdeckung": MIN_ABDECKUNG,
        "stufiger_faktor": stufig,
        "kosten_bps": KOSTEN_BPS,
        "top_k": TOP_K,
        "roh_spruenge": d.roh_spruenge,
        "rest_spruenge": d.rest_spruenge,
        "split_diagnose": d.split_diagnose,
        "buy_and_hold": k_bh,
        "ew_rebalanciert": k_reb,
        "familien": alles,
    }
    ziel = "p12_intraday_stufig.json" if stufig else "p12_intraday_haltedauer.json"
    (OUT / ziel).write_text(json.dumps(ergebnis, indent=2), encoding="utf-8")
    print(f"-> {OUT / ziel}")

    flach = [dict(z, familie=f) for f, zs in alles.items() for z in zs]
    best_n = max(flach, key=lambda z: z["netto_end"])
    best_b = max(flach, key=lambda z: z["brutto_end"])
    kurz = [z for z in flach if z["halte_bars"] <= BARS_PRO_TAG]
    print("\n" + "=" * 78)
    print(
        f"Bester NETTO  : {best_n['name']:<11} {best_n['netto_end']:.3f}x"
        f"   ({best_n['familie']})"
    )
    print(
        f"Bester BRUTTO : {best_b['name']:<11} {best_b['brutto_end']:.3f}x"
        f"   ({best_b['familie']})"
    )
    print(f"Buy-and-Hold  : {close.shape[1]} Namen{'':<5} {k_bh['endwert']:.3f}x")
    print(
        f"Max Cash-Anteil ueber alle Zeilen: {max(z['anteil_cash'] for z in flach):.4%}"
    )
    print()
    if all(z["netto_end"] < k_bh["endwert"] for z in kurz):
        print("BEFUND: KEINE Haltedauer <= 1 Tag schlaegt schlichtes Halten")
        print("        desselben Universums. Das kurze Ende traegt nicht.")
    else:
        print("BEFUND: Mindestens eine kurze Haltedauer schlaegt das Halten —")
        print("        Gegenkontrollen zwingend, bevor das ein Befund heisst.")
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
