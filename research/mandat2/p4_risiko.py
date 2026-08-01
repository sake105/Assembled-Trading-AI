"""P4 — Risiko-Sleeve: die einzige Achse, die den DD-Deckel erreichen kann.

BEFUNDLAGE
----------
72 von 72 Kombinationen aus P2 scheitern am Deckel (-35 %), bester Drawdown
-64,6 %. Auch SPY selbst reisst ihn in 144/144 Fenstern (-47,5 % bis -55,2 %).
Renditeoptimierung ist damit sinnlos, solange nichts den Drawdown adressiert:
die Nebenbedingung greift vorher.

Getestet wird der EINFACHSTE Mechanismus, der das kann — ein SMA-Trendfilter
auf den Index. Bewusst der einfachste: gelingt es damit, ist es schwer als
Overfitting abzutun; gelingt es damit nicht, sagt das mehr ueber die Aufgabe
als ein weiterer komplizierter Versuch.

Varianten:
  * ohne Gate (Referenz aus P2)
  * SMA200-Gate auf SPY (Klassiker)
  * SMA100 / SMA300 (Robustheit gegen die Fensterwahl — ein Gate, das nur bei
    genau 200 funktioniert, ist ein gefundenes Fenster, kein Mechanismus)
  * Gate auf den reinen Benchmark (schafft SPY MIT Gate den Deckel?)

Die letzte Zeile ist die wichtigste: wenn nicht einmal der gefilterte Index
unter -35 % bleibt, ist der Deckel auf diesem Suchfenster fuer JEDE
Aktienstrategie unerreichbar — und das waere ein Ergebnis, kein Zwischenstand.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore", message=".*Converting to PeriodArray.*", category=UserWarning
)

from research.mandat2.campaign_data import load_campaign  # noqa: E402
from research.mandat2.data_gate import TrialCounter  # noqa: E402
from research.mandat2.engine import (  # noqa: E402
    run_buy_and_hold,
    run_strategy,
    sma_gate,
)
from research.mandat2.metrics import DD_DECKEL, auswerten  # noqa: E402
from research.mandat2.tax_regimes import make_regime  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
PARAMS = dict(top_in=20, rank_out=200, min_haltetage=730, hebel=1.0)
FENSTER = [None, 100, 200, 300]
WELTEN = [("ZERO", {}), ("PRIVAT_DE", {}), ("GMBH+FK", {"fixkosten_pa": 3_500.0})]
REGIME_NAME = {
    "ZERO": "ZERO",
    "PRIVAT_DE": "PRIVAT_DE",
    "GMBH+FK": "GMBH_THESAURIEREND",
}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    d = load_campaign()
    print(d, flush=True)
    zaehler = TrialCounter()
    n = len(FENSTER) * len(WELTEN)
    print(f"Trials kumuliert: {zaehler.increment(n, label='P4 Risiko-Sleeve')}\n")

    zeilen = []
    for welt, kwargs in WELTEN:
        name = REGIME_NAME[welt]
        bench = run_buy_and_hold(d, make_regime(name, **kwargs))
        print(f"=== {welt} ===", flush=True)
        for fenster in FENSTER:
            gate = None if fenster is None else sma_gate(d.close, fenster=fenster)
            kand = run_strategy(
                d, make_regime(name, **kwargs), risk_off_gate=gate, **PARAMS
            )
            a = auswerten(
                kand.equity_netto,
                bench.equity_netto,
                label=f"{welt} SMA{fenster or '-'}",
            )
            zeile = {
                "welt": welt,
                "sma": fenster,
                "endwert": float(kand.equity.iloc[-1]),
                "benchmark_endwert": float(bench.equity.iloc[-1]),
                "median_kandidat": a.median_kandidat,
                "median_benchmark": a.median_benchmark,
                "schlimmster_maxdd": a.schlimmster_maxdd,
                "gerissene_fenster": len(a.gerissene_fenster),
                "n_fenster": a.n_fenster,
                "deckel_eingehalten": a.deckel_eingehalten,
                "bestanden": a.bestanden,
                "n_trades": kand.n_trades,
            }
            zeilen.append(zeile)
            print(
                f"  SMA{str(fenster or '---'):>4}: End {zeile['endwert']:>11,.0f} | "
                f"Median {a.median_kandidat:.3f} vs {a.median_benchmark:.3f} | "
                f"DD {a.schlimmster_maxdd:>7.1%} | gerissen "
                f"{len(a.gerissene_fenster):>3}/{a.n_fenster} | "
                f"{'BESTANDEN' if a.bestanden else 'durchgefallen'}",
                flush=True,
            )

    # Der entscheidende Kontrollfall: der reine Index MIT Gate.
    print("\n=== Kontrolle: SPY selbst, mit und ohne Gate (ZERO) ===", flush=True)
    bench = run_buy_and_hold(d, make_regime("ZERO"))
    kontrolle = []
    for fenster in FENSTER:
        gate = None if fenster is None else sma_gate(d.close, fenster=fenster)
        r = run_buy_and_hold(
            d, make_regime("ZERO"), risk_off_gate=gate, label=f"SPY SMA{fenster}"
        )
        a = auswerten(r.equity_netto, bench.equity_netto, label=f"SPY SMA{fenster}")
        kontrolle.append(
            {
                "sma": fenster,
                "schlimmster_maxdd": a.schlimmster_maxdd,
                "gerissene_fenster": len(a.gerissene_fenster),
                "median": a.median_kandidat,
                "endwert": float(r.equity.iloc[-1]),
            }
        )
        print(
            f"  SMA{str(fenster or '---'):>4}: DD {a.schlimmster_maxdd:>7.1%} | "
            f"gerissen {len(a.gerissene_fenster):>3}/{a.n_fenster} | "
            f"Median {a.median_kandidat:.3f} | End {r.equity.iloc[-1]:>11,.0f}",
            flush=True,
        )

    (OUT / "p4_risiko.json").write_text(
        json.dumps(
            {"dd_deckel": DD_DECKEL, "zeilen": zeilen, "spy_kontrolle": kontrolle},
            indent=2,
        ),
        encoding="utf-8",
    )
    besteht = [z for z in zeilen if z["bestanden"]]
    print(f"\n-> {OUT / 'p4_risiko.json'}")
    print(f"Kombinationen die BESTEHEN: {len(besteht)} von {len(zeilen)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
