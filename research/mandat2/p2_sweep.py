"""P2 — Haltedauer x Gewinnmitnahme x Hebel x Steuerwelt (Mandat II).

DIE FRAGE, um die es Hans geht
-------------------------------
Mandat I hat Strategien unter deutscher Privatanleger-Steuer bewertet. Dort war
Umschichten teuer, also gewann, wer wenig handelte. Faellt die Steuerbremse weg
(ZERO) oder wird sie klein (GmbH: 1,49 % auf Kursgewinne statt 26,375 %), darf
man ANDERS handeln — und genau das wurde bisher nie getestet: der P1-Erstlauf
fuhr eine einzige Parametrisierung durch vier Steuerwelten.

Hier wird die Strategie selbst variiert:

* ``min_haltetage``  — Gewinne laufen lassen (Sperre gegen frueh Verkaufen)
* ``rank_out``       — wie lange ein Gewinner im Depot bleibt, bevor das Signal
                       ihn rauswirft. 200 = praktisch „nie" bei 500 Namen.
* ``hebel``          — mit echten Finanzierungskosten
* Steuerwelt         — ZERO / PRIVAT_DE / GMBH+Fixkosten

Wenn die Steuerfreiheit einen Unterschied macht, muss das OPTIMUM im
Parametergitter zwischen den Steuerwelten WANDERN: unter PRIVAT_DE zu langer
Haltedauer und wenig Turnover, unter ZERO zu schnellerem Handel. Wandert es
nicht, ist die Steuer nicht die bindende Restriktion — und dann ist die
Antwort auf Hans' Frage ein belegtes Nein statt einer Vermutung.

DISZIPLIN
---------
Laeuft ausschliesslich auf dem SUCH-Fenster. Jede Parameterkombination ist ein
Trial und wird gezaehlt (Start 1.964 aus Mandat I) — der DSR-Haircut wird
dadurch haerter, nicht weicher.

EHRLICHE GRENZE: „wenige Stunden Haltedauer" ist mit diesem EOD-Panel NICHT
testbar. Die feinste Aufloesung ist ein Handelstag. Intraday braucht das
EODHD-Intraday-Paket (ab ca. Okt 2020) und ist ein eigener Schritt.
"""

from __future__ import annotations

import json
import warnings
from itertools import product
from pathlib import Path

warnings.filterwarnings(
    "ignore", message=".*Converting to PeriodArray.*", category=UserWarning
)

from research.mandat2.campaign_data import load_campaign  # noqa: E402
from research.mandat2.data_gate import TrialCounter  # noqa: E402
from research.mandat2.engine import run_buy_and_hold, run_momentum  # noqa: E402
from research.mandat2.metrics import DD_DECKEL, auswerten  # noqa: E402
from research.mandat2.tax_regimes import make_regime  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
FIXKOSTEN_PA = 3_500.0

#: (Label, Regime-Name, Kwargs)
WELTEN = [
    ("ZERO", "ZERO", {}),
    ("PRIVAT_DE", "PRIVAT_DE", {}),
    ("GMBH+FK", "GMBH_THESAURIEREND", {"fixkosten_pa": FIXKOSTEN_PA}),
]

#: Gewinne laufen lassen: Mindesthaltedauer in Kalendertagen.
HALTETAGE = [0, 90, 365, 730]
#: Wie spaet ein Gewinner rausfliegt. 200 = praktisch nie (bei ~500 Namen).
RANK_OUT = [30, 60, 200]
#: Hebel inkl. Finanzierung.
HEBEL = [1.0, 1.5]


def main() -> int:
    OUT.mkdir(exist_ok=True)
    d = load_campaign()
    print(d, flush=True)

    kombis = list(product(HALTETAGE, RANK_OUT, HEBEL))
    zaehler = TrialCounter()
    gesamt = zaehler.increment(
        len(kombis), label=f"P2-Sweep {len(kombis)} Kombis x {len(WELTEN)} Welten"
    )
    print(f"Trials kumuliert (inkl. Mandat I): {gesamt}\n", flush=True)

    # Benchmark je Welt einmal.
    benchmarks = {}
    for label, name, kwargs in WELTEN:
        b = run_buy_and_hold(d, make_regime(name, **kwargs))
        benchmarks[label] = b
        print(f"[BENCH] {label}: {b.equity.iloc[-1]:,.0f}", flush=True)

    zeilen = []
    for label, name, kwargs in WELTEN:
        bench = benchmarks[label]
        print(f"\n=== {label} ===", flush=True)
        for haltetage, rank_out, hebel in kombis:
            kand = run_momentum(
                d,
                make_regime(name, **kwargs),
                top_in=20,
                rank_out=rank_out,
                min_haltetage=haltetage,
                hebel=hebel,
            )
            a = auswerten(
                kand.equity_netto,
                bench.equity_netto,
                label=f"{label} hold{haltetage} out{rank_out} x{hebel}",
            )
            zeile = {
                "welt": label,
                "min_haltetage": haltetage,
                "rank_out": rank_out,
                "hebel": hebel,
                "endwert": float(kand.equity.iloc[-1]),
                "benchmark_endwert": float(bench.equity.iloc[-1]),
                "median_kandidat": a.median_kandidat,
                "median_benchmark": a.median_benchmark,
                "anteil_fenster_geschlagen": a.anteil_fenster_geschlagen,
                "schlimmster_maxdd": a.schlimmster_maxdd,
                "deckel_eingehalten": a.deckel_eingehalten,
                "bestanden": a.bestanden,
                "n_trades": kand.n_trades,
                "steuer": kand.portfolio.tax_paid,
                "kosten": kand.portfolio.costs_paid,
                "finanzierung": kand.finanzierung_gezahlt,
            }
            zeilen.append(zeile)
            print(
                f"  hold{haltetage:>4}d out{rank_out:>4} x{hebel}: "
                f"End {zeile['endwert']:>10,.0f} | Median {a.median_kandidat:.3f} "
                f"vs {a.median_benchmark:.3f} | DD {a.schlimmster_maxdd:>7.1%} | "
                f"Trades {kand.n_trades:>6} | "
                f"{'BESTANDEN' if a.bestanden else 'durchgefallen'}",
                flush=True,
            )

    (OUT / "p2_sweep.json").write_text(
        json.dumps(
            {"dd_deckel": DD_DECKEL, "trials_kumuliert": gesamt, "zeilen": zeilen},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n-> {OUT / 'p2_sweep.json'}", flush=True)

    # Wandert das Optimum zwischen den Steuerwelten?
    print("\n=== BESTE KOMBINATION JE STEUERWELT (nach Median-Faktor) ===", flush=True)
    for label, _, _ in WELTEN:
        w = [z for z in zeilen if z["welt"] == label]
        best = max(w, key=lambda z: z["median_kandidat"])
        print(
            f"  {label:<10} hold{best['min_haltetage']}d out{best['rank_out']} "
            f"x{best['hebel']} -> Median {best['median_kandidat']:.3f} "
            f"(Benchmark {best['median_benchmark']:.3f}), "
            f"End {best['endwert']:,.0f}, Trades {best['n_trades']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
