"""P13d — Zählt das Timing des Trendfilters, oder nur die Zeit außerhalb?

DIE LETZTE BILLIGE WIDERLEGUNG
------------------------------
Ein Gate tut zwei Dinge gleichzeitig: Es nimmt Zeit aus dem Markt heraus, und
es wählt aus, **wann**. Der Vorsprung in P13 könnte vollständig aus dem ersten
Teil stammen — im Suchfenster steckt in jedem 10-Jahres-Fenster ein Rückgang
von mindestens 47,5 % (P13c), und wer irgendwann pausiert, erwischt davon im
Mittel etwas.

Die Kontrollgruppe dieser Kampagne für genau diese Frage ist das Zufallssignal
(so wurde in P12b der Momentum-Befund erledigt: er lag unter dem Zufallsmedian).

WIE DIE KONTROLLE GEBAUT IST
----------------------------
Nicht „zufällig an/aus mit gleicher Wahrscheinlichkeit" — das erzeugt viele
kurze Episoden und ist eine leichtere Vergleichsgruppe. Stattdessen werden die
**Blocklängen des echten Gates** innerhalb ihrer Wertklasse gemischt, während
die Folge an/aus/an/aus unverändert bleibt. Auf **Signalebene** exakt erhalten:

* der Anteil investierter Tage,
* die Anzahl der Episoden,
* die Verteilung ihrer Längen.

Zerstört wird auf dieser Ebene ausschließlich, **wann** die Pausen liegen.

WAS AUF DER BUCHENDEN EBENE NICHT ERHALTEN BLEIBT
-------------------------------------------------
Die Engine liest das Gate nur an Monatsenden. Dort schaltet das echte Gate
18-mal, die gemischten 19- bis 37-mal (Median 27) — der Shuffle verteilt die
Wechsel anders auf die Rebalance-Termine. Jede Schaltung kostet `cost_bps`,
regimeunabhängig auch in ZERO. **Die Kontrollgruppe trägt damit systematisch
mehr Kostendrag als der Kandidat**, und das wirkt zu Gunsten des Kandidaten.

Der Abstand (2,525x gegen einen Zufallsmedian von 1,353x) ist zu groß, als dass
Kostendrag ihn erklären könnte — aber das ist eine Abschätzung, keine
Bereinigung. Beide Größen stehen deshalb im Artefakt und im Befund.

Wenn der Trendfilter den Zufallsmedian nicht schlägt, misst er keine
Trendinformation, sondern nur Abwesenheit vom Markt.

TRIAL-ZÄHLER
------------
Steigt **nicht**. Eine Kontrollgruppe ist keine Suche nach einer besseren
Konfiguration des Kandidaten (E-090); der Kandidat ist unverändert der
a-priori-Parameter aus P13c. Der Zufall wird nicht optimiert, sondern verteilt.
"""

from __future__ import annotations

import json
import statistics
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore", message=".*Converting to PeriodArray.*", category=UserWarning
)

from research.mandat2.campaign_data import load_campaign  # noqa: E402
from research.mandat2.engine import _monatsenden, run_buy_and_hold  # noqa: E402
from research.mandat2.metrics import auswerten  # noqa: E402
from research.mandat2.p13c_ereignisabhaengigkeit import FENSTER_APRIORI  # noqa: E402
from research.mandat2.p5_gate_robustheit import gate_preis_ueber_sma  # noqa: E402
from research.mandat2.portfolio import Portfolio  # noqa: E402
from research.mandat2.tax_regimes import make_regime  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"

#: Zahl der Zufallsvarianten. 60 wie in P12b — genug fuer einen stabilen
#: Median und ein Perzentil, ohne dass der Lauf Stunden braucht.
SEEDS = 60


def bloecke(gate: pd.Series) -> list[tuple[float, int]]:
    """Gate in (Wert, Laenge)-Bloecke zerlegen. NaN bleibt eigener Wert."""
    werte = gate.to_numpy(dtype=float)
    out: list[tuple[float, int]] = []
    start = 0
    for i in range(1, len(werte) + 1):
        ende = i == len(werte)
        gleich = not ende and (
            werte[i] == werte[start] or (np.isnan(werte[i]) and np.isnan(werte[start]))
        )
        if not gleich:
            out.append((werte[start], i - start))
            start = i
    return out


def gemischtes_gate(gate: pd.Series, seed: int) -> pd.Series:
    """Blocklaengen INNERHALB jeder Wertklasse mischen, Wertfolge behalten.

    Der erste Entwurf permutierte die Bloecke als Ganzes. Das zerstoert mehr
    als das Timing: zwei gleichwertige Bloecke koennen nebeneinander landen und
    **verschmelzen**, wodurch die Kontrollgruppe weniger und laengere Episoden
    bekommt als das echte Gate. Der Test wies das nach (Laengenmultimenge
    [1,1,1,2,3,3,...] wurde zu [1,3,3,3,4,4]). Damit waere die Kontrolle keine
    Kontrolle mehr, sondern eine andere Strategie — und der Vergleich haette
    zwei Ursachen statt einer gehabt.

    Diese Fassung laesst die Folge der Blockwerte unveraendert (an/aus wechseln
    sich weiterhin genauso oft ab) und permutiert nur die Laengen innerhalb
    jeder Wertklasse. Exakt erhalten bleiben damit: Anteil investierter Tage,
    Anzahl der Episoden je Klasse und deren Laengenverteilung. Veraendert wird
    ausschliesslich, WANN die langen und kurzen Episoden liegen.

    Bewusst `default_rng(seed)` statt globalem Zustand: der Lauf muss
    reproduzierbar sein, sonst ist die Kontrollgruppe nicht nachpruefbar.
    """
    bl = bloecke(gate)
    rng = np.random.default_rng(seed)

    # Laengen je Wertklasse einsammeln. NaN braucht einen eigenen Schluessel,
    # weil nan != nan und die Warmlaufphase sonst in keiner Klasse landet.
    def schluessel(w: float) -> str:
        return "nan" if np.isnan(w) else repr(w)

    nach_klasse: dict[str, list[int]] = {}
    for w, laenge in bl:
        nach_klasse.setdefault(schluessel(w), []).append(laenge)
    gemischt = {
        k: [v[i] for i in rng.permutation(len(v))] for k, v in nach_klasse.items()
    }
    zeiger = dict.fromkeys(gemischt, 0)

    teile = []
    for w, _ in bl:
        k = schluessel(w)
        teile.append(np.full(gemischt[k][zeiger[k]], w))
        zeiger[k] += 1
    werte = np.concatenate(teile)
    assert len(werte) == len(gate), "Mischen darf die Laenge nicht aendern"
    return pd.Series(werte, index=gate.index)


def wirksame_schaltungen(
    close: pd.DataFrame, gate: pd.Series, symbol: str = "SPY"
) -> int:
    """Zustandswechsel an den Terminen, an denen die Engine ueberhaupt handelt.

    Der Shuffle erhaelt die Blockstruktur auf TAGESebene. Gebucht wird aber nur
    an Monatsenden, und dort schaltet ein gemischtes Gate anders oft als das
    echte. Jede Schaltung kostet `cost_bps` — regimeunabhaengig, auch in ZERO
    (portfolio.py). Wer das nicht misst, laedt der Kontrollgruppe stillschweigend
    einen Kostennachteil auf und liest ihn als Effekt des Timings (F-auditor-1).

    Das ist eine MESSUNG, keine zweite Buchungswahrheit: gezaehlt wird die
    Gate-Lesung der Engine (`engine.py`, `t in monatsenden`), nichts wird
    gebucht. Dieselbe Groesse wie in `p9_gate_forensik.py`.
    """
    reihe = close[symbol].dropna()
    monatsenden = _monatsenden(pd.DatetimeIndex(reihe.index))
    stand: bool | None = None
    n = 0
    for t in reihe.index:
        if t not in monatsenden:
            continue
        g = gate.get(t)
        risk_on = True if g is None or not np.isfinite(float(g)) else bool(g)
        if stand is not None and risk_on != stand:
            n += 1
        stand = risk_on
    return n


def _investiert_anteil(d, regime, gate: pd.Series) -> float:
    """Anteil der Tage, an denen das PORTFOLIO wirklich investiert war.

    Nicht der An-Anteil des Signals: die Engine handelt nur an Monatsenden und
    haelt dazwischen durch. Gemessen wird ueber dieselbe Instrumentierung wie
    in P13 (Bestand statt Schaetzung, E-102) — ohne Nachbau der Engine.
    """
    gehalten = 0
    gesamt = 0
    original = Portfolio.set_date

    def set_date(self: Portfolio, datum) -> None:
        nonlocal gehalten, gesamt
        original(self, datum)
        gesamt += 1
        if self.qty("SPY") > 0:
            gehalten += 1

    Portfolio.set_date = set_date  # type: ignore[method-assign]
    try:
        run_buy_and_hold(d, regime, risk_off_gate=gate)
    finally:
        Portfolio.set_date = original  # type: ignore[method-assign]
    return gehalten / gesamt if gesamt else 0.0


def p_wert(medians: list[float], echt: float, seeds: int) -> tuple[int, float]:
    """Rangbasierter p-Wert plus die Zahl der Zufallslaeufe, die ihn erreichen.

    Als Funktion herausgezogen, weil die Zahl im Befund steht und in einem
    `main()`-Rumpf von keinem Test erreichbar waere (Stage-1-Befund N6/N7).

    Die `+1` in Zaehler UND Nenner ist keine Kosmetik: ohne sie kaeme bei null
    Treffern p = 0 heraus, also die Behauptung, ein solches Ergebnis sei mit
    endlich vielen Ziehungen ausgeschlossen. Der kleinstmoegliche Wert ist
    1/(seeds+1) — bei 60 Ziehungen 0,016.
    """
    besser = sum(1 for m in medians if m >= echt)
    return besser, (besser + 1) / (seeds + 1)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    d = load_campaign()
    print(d, flush=True)
    print(
        f"Kontrollgruppe zu preis>SMA{FENSTER_APRIORI}, {SEEDS} Seeds. "
        f"Kein Trial-Increment — keine Suche.\n",
        flush=True,
    )

    gate = gate_preis_ueber_sma(d.close, FENSTER_APRIORI)
    anteil = float(np.nanmean(gate.to_numpy(dtype=float)))
    bl = bloecke(gate)
    print(f"Echtes Gate: {anteil:.1%} der Tage investiert, {len(bl)} Bloecke\n")

    ergebnis: dict = {
        "fenster_apriori": FENSTER_APRIORI,
        "seeds": SEEDS,
        "anteil_investiert": anteil,
        "n_bloecke": len(bl),
    }
    for welt, name, kwargs in [("ZERO", "ZERO", {}), ("PRIVAT_DE", "PRIVAT_DE", {})]:
        bench = run_buy_and_hold(d, make_regime(name, **kwargs))
        echt = auswerten(
            run_buy_and_hold(
                d, make_regime(name, **kwargs), risk_off_gate=gate
            ).equity_netto,
            bench.equity_netto,
            label=f"{welt}/echt",
        )

        # Die realisierte Marktabwesenheit MUSS mitgemessen werden. Die
        # erhaltenen Groessen (Anteil, Blockzahl, Laengenverteilung) gelten
        # fuer das SIGNAL; die Engine liest das Gate aber nur an Monatsenden.
        # Zwischen Signal und Wirkung liegt also ein Sampling-Schritt, der die
        # Erhaltung nicht durchreicht — wer sie trotzdem fuer das Portfolio
        # behauptet, testet auf einer Ebene und behauptet auf einer anderen
        # (F-senior-4, dieselbe Klasse wie E-118).
        echt_anteil = _investiert_anteil(d, make_regime(name, **kwargs), gate)
        echt_schaltungen = wirksame_schaltungen(d.close, gate)
        zufall = []
        for s in range(SEEDS):
            g = gemischtes_gate(gate, s)
            r = run_buy_and_hold(d, make_regime(name, **kwargs), risk_off_gate=g)
            a = auswerten(r.equity_netto, bench.equity_netto, label=f"{welt}/zufall{s}")
            zufall.append(
                {
                    "seed": s,
                    "median": a.median_kandidat,
                    "maxdd": a.schlimmster_maxdd,
                    "gerissen": len(a.gerissene_fenster),
                    "bestanden": a.bestanden,
                    "investiert_anteil": _investiert_anteil(
                        d, make_regime(name, **kwargs), g
                    ),
                    "wirksame_schaltungen": wirksame_schaltungen(d.close, g),
                }
            )

        medians = sorted(z["median"] for z in zufall)
        # Rangbasiert statt Normalverteilungs-Annahme: der Anteil der
        # Zufallslaeufe, die den echten Filter erreichen, IST der p-Wert.
        besser, p = p_wert(medians, echt.median_kandidat, SEEDS)
        anteile = sorted(z["investiert_anteil"] for z in zufall)
        schalt = sorted(z["wirksame_schaltungen"] for z in zufall)
        ergebnis[welt] = {
            "echt_wirksame_schaltungen": echt_schaltungen,
            "zufall_schaltungen_min": schalt[0],
            "zufall_schaltungen_max": schalt[-1],
            "zufall_schaltungen_median": statistics.median(schalt),
            "echt_investiert_anteil": echt_anteil,
            "zufall_investiert_anteil_min": anteile[0],
            "zufall_investiert_anteil_max": anteile[-1],
            "zufall_investiert_anteil_median": statistics.median(anteile),
            "echt_median": echt.median_kandidat,
            "echt_maxdd": echt.schlimmster_maxdd,
            "echt_bestanden": echt.bestanden,
            "benchmark_median": echt.median_benchmark,
            "zufall_median": statistics.median(medians),
            "zufall_bestes": medians[-1],
            "zufall_schlechtestes": medians[0],
            "zufall_p95": medians[int(0.95 * (SEEDS - 1))],
            "zufall_bestanden": sum(1 for z in zufall if z["bestanden"]),
            "zufall_erreicht_echt": besser,
            "p_wert": p,
            "laeufe": zufall,
        }
        print(f"=== {welt} ===")
        print(
            f"  wirksame Schaltungen an Monatsenden: echt {echt_schaltungen} | "
            f"Zufall {schalt[0]}..{schalt[-1]} (Median {statistics.median(schalt):.0f})"
        )
        print(
            f"  realisiert investiert: echt {echt_anteil:.1%} | Zufall "
            f"{anteile[0]:.1%}..{anteile[-1]:.1%} "
            f"(Median {statistics.median(anteile):.1%})"
        )
        print(
            f"  echt      Median {echt.median_kandidat:.3f}x | MaxDD "
            f"{echt.schlimmster_maxdd:.1%} | "
            f"{'BESTANDEN' if echt.bestanden else '-'}"
        )
        print(
            f"  Zufall    Median {statistics.median(medians):.3f}x "
            f"(schlechtestes {medians[0]:.3f} | p95 "
            f"{medians[int(0.95 * (SEEDS - 1))]:.3f} | bestes {medians[-1]:.3f})"
        )
        print(f"  Benchmark Median {echt.median_benchmark:.3f}x")
        print(
            f"  {besser}/{SEEDS} Zufallslaeufe erreichen den echten Filter "
            f"-> p = {p:.3f}"
        )
        print(
            f"  {ergebnis[welt]['zufall_bestanden']}/{SEEDS} Zufallslaeufe "
            f"bestehen die Zielfunktion\n",
            flush=True,
        )

    # Artefakt als LETZTE Anweisung (E-116).
    (OUT / "p13d_zufallstiming.json").write_text(
        json.dumps(ergebnis, indent=2), encoding="utf-8"
    )
    print(f"-> {OUT / 'p13d_zufallstiming.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
