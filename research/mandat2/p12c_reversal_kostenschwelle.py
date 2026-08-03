"""P12c — Traegt das umgekehrte Vorzeichen am kurzen Ende die Reibung?

WAS P12 ZEIGT (und was eine fruehere Fassung dieses Docstrings falsch sagte)
---------------------------------------------------------------------------
Der erste Lauf hatte behauptet, Momentum verliere am kurzen Ende bereits
BRUTTO den Grossteil des Kapitals. Das war ein Artefakt eines fehlerhaften
Laufs (E-085) und ist zurueckgenommen: brutto gewinnt Momentum auch bei
einstuendiger Haltedauer.

Der belastbare Befund ist ein anderer und praeziser: **brutto liegt Momentum am
kurzen Ende UNTER der Zufallsauswahl** — bei 1 Stunde 2,360x gegen 3,104x im
Mittel, bei 4 Stunden 2,498x gegen 3,398x (Familie A, results/p12_intraday_
haltedauer.json). Die Rangfolge nach juengster Rendite waehlt dort also aktiv
schlechter als das Los, und zwar bevor eine einzige Gebuehr anfaellt.

DIE ANSCHLUSSFRAGE
------------------
Wenn die Rangfolge am kurzen Ende schadet, ist ihr Gegenteil die naechstliegende
Hypothese — der seit Jegadeesh (1990) und Lehmann (1990) dokumentierte
Short-Term-Reversal. Die Frage ist dann NICHT, ob es eine Bruttokante gibt,
sondern ob sie die Reibung traegt. Das ist Hans' Frage nach den kurzen
schnellen Gewinnen, und sie ist mit einer Zahl beantwortbar:

    Ab welchen Handelskosten (bps je Seite) kippt die Strategie ins Minus?

Diese **Break-even-Kostenschwelle** ist ehrlicher als ein Pass/Fail, weil sie
sagt, WIE WEIT etwas von der Umsetzbarkeit entfernt ist — nicht nur, DASS es
scheitert.

DISZIPLIN
---------
EINE theoriegeleitete Hypothese, vorab benannt, kein Absuchen eines
Parameterraums: Score = negatives Momentum, sonst identisch zu P12. Zaehlt im
Trial-Ledger. Ein positives Ergebnis waere hier NICHT deployierbar, sondern
ein Holdout-Kandidat — und selbst dann gilt die Survivorship-Warnung.

WARUM DIE KOSTENANNAHME KRITISCH IST
------------------------------------
10 bps je Seite ist die Kampagnen-Annahme (Mandat I) und enthaelt Spread und
Marktwirkung nur pauschal. Bei stuendlichem Umschichten in fuenf Namen ist
genau diese Pauschale die entscheidende Groesse — deshalb wird nicht ein Wert
gerechnet, sondern eine Kurve ueber 0 bis 30 bps.

ARTEFAKTSCHRANKE
----------------
Der Tagesfaktor der Bereinigung enthaelt ~11-31 bps/Tag reversierendes Rauschen
(E-083). Es ist gleichgerichtet mit dem hier gesuchten Effekt. Die
Break-even-Schwelle liegt in derselben Groessenordnung und ist deshalb NUR
zusammen mit der Stufen-Gegenprobe (`p12_intraday_stufig.json`) zu lesen.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from research.mandat2.data_gate import TrialCounter  # noqa: E402
from research.mandat2.intraday_data import load_intraday  # noqa: E402
from research.mandat2.p12_intraday_haltedauer import (  # noqa: E402
    BARS_PRO_TAG,
    WARMUP,
    buy_and_hold,
    kennzahlen,
    momentum,
    simuliere,
)

OUT = Path(__file__).resolve().parent / "results"
KURZ = [(1, "1 Stunde"), (2, "2 Stunden"), (4, "4 Stunden"), (BARS_PRO_TAG, "1 Tag")]
KOSTEN_RASTER = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0]


def main() -> int:
    OUT.mkdir(exist_ok=True)
    stufig = "--stufig" in sys.argv
    d = load_intraday(stufig=stufig)
    print(d, flush=True)
    n = len(KURZ) * len(KOSTEN_RASTER)
    print(
        f"Trials kumuliert: "
        f"{TrialCounter().increment(n, label='P12c Reversal-Kostenschwelle')}\n",
        flush=True,
    )

    close = d.close.ffill()
    fenster = close.index[WARMUP:]
    jahre = (fenster[-1] - fenster[0]).days / 365.25
    b_end = kennzahlen(buy_and_hold(close, start=WARMUP), jahre)["endwert"]
    print(f"[Benchmark] Buy-and-Hold, {jahre:.2f} Jahre: {b_end:.3f}x\n")

    zeilen = []
    for bars, name in KURZ:
        rueck = max(2, bars * 20)  # kurzfristiges Signal, Familie B
        score = -momentum(close, rueck)  # DAS umgekehrte Vorzeichen
        # Brutto/Umschlag EINMAL explizit bestimmen, nicht als Nebenwirkung
        # eines Schleifenzweigs (Stage-1-Finding F-test-14).
        _, b_null, umschichtungen, _ = simuliere(
            close, halte_bars=bars, score=score, kosten_bps=0.0, start=WARMUP
        )
        brutto = kennzahlen(b_null, jahre)["endwert"]
        kurve = {}
        for bps in KOSTEN_RASTER:
            n_e, _, _, _ = simuliere(
                close, halte_bars=bars, score=score, kosten_bps=bps, start=WARMUP
            )
            kurve[bps] = kennzahlen(n_e, jahre)["endwert"]

        # Break-even: hoechste Kostenstufe, bei der noch Vermoegen entsteht
        # (Endwert > 1,0), und die, bei der der Halte-Benchmark noch geschlagen wird.
        traegt = [b for b, v in kurve.items() if v > 1.0]
        schlaegt = [b for b, v in kurve.items() if v > b_end]
        be_kapital = max(traegt) if traegt else None
        be_bench = max(schlaegt) if schlaegt else None
        cagr_brutto = brutto ** (1 / jahre) - 1

        zeilen.append(
            {
                "halte_bars": bars,
                "name": name,
                "umschichtungen": umschichtungen,
                "brutto_end": brutto,
                "brutto_cagr": float(cagr_brutto),
                "kurve": {str(k): v for k, v in kurve.items()},
                "breakeven_kapitalerhalt_bps": be_kapital,
                "breakeven_schlaegt_benchmark_bps": be_bench,
                "bei_10bps": kurve[10.0],
                "bench_end": b_end,
            }
        )
        print(f"=== {name} — {umschichtungen:,} Umschichtungen ===")
        print(f"    brutto {brutto:>12,.1f}x  (CAGR {cagr_brutto:+.1%})")
        print(
            "    "
            + "  ".join(f"{b:g}bps:{kurve[b]:.3f}x" for b in (0.0, 1.0, 2.0, 5.0, 10.0))
        )
        print(
            f"    Break-even Kapitalerhalt: "
            f"{f'{be_kapital:g} bps' if be_kapital is not None else 'nie'}"
            f"  |  schlaegt Halten: "
            f"{f'{be_bench:g} bps' if be_bench is not None else 'nie'}",
            flush=True,
        )

    ziel = (
        "p12c_reversal_stufig.json" if stufig else "p12c_reversal_kostenschwelle.json"
    )
    (OUT / ziel).write_text(
        json.dumps(
            {
                "kosten_raster": KOSTEN_RASTER,
                "benchmark_end": b_end,
                "stufiger_faktor": stufig,
                "zeilen": zeilen,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n-> {OUT / ziel}")

    print("\n" + "=" * 78)
    print("Die entscheidende Zahl ist nicht der Bruttogewinn, sondern die Schwelle:")
    for z in zeilen:
        be = z["breakeven_schlaegt_benchmark_bps"]
        print(
            f"  {z['name']:<11} brutto {z['brutto_end']:>13,.1f}x | bei 10 bps "
            f"{z['bei_10bps']:>10.3f}x | schlaegt Halten bis "
            f"{f'{be:g} bps' if be is not None else 'nie'}"
        )
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
