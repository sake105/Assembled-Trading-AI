"""P10 — Ab welchem Kapital traegt die vermoegensverwaltende GmbH?

DIE FRAGE
---------
P1/P8 haben gezeigt: die Steuerasymmetrie der GmbH ist real und gross
(Kursgewinn 1,49 % statt 26,375 %), aber bei 100.000 EUR Startkapital frisst
die Rechtsform ihren eigenen Vorteil auf — +141.742 EUR ohne, +4.079 EUR mit
Fixkosten, und auf der Zielfunktion sogar schlechter als privat.

Der Grund ist arithmetisch, nicht strategisch: **Rechtsformkosten sind ein
FIXbetrag.** 3.500 EUR/Jahr sind bei 100.000 EUR 3,5 % p. a., bei 1 Mio nur
0,35 %. Der Steuervorteil dagegen skaliert mit dem Kapital. Es muss also einen
Break-even geben — und der ist die eigentliche Antwort auf Hans' Strukturfrage.

DIES IST KEINE ALPHA-SUCHE
--------------------------
Es wird keine Strategie ausgewaehlt und kein Signal optimiert. Gerechnet wird
EIN fixierter Kandidat (der P2-Gewinner) in zwei Steuerwelten ueber ein
Kapital- und Kostenraster. Der Trial-Zaehler steigt trotzdem, weil neue
Backtests laufen — aber die Selektion findet nicht ueber Strategien statt,
sondern die Frage ist eine strukturelle.

Zusaetzlich wird der Benchmark (SPY Buy-and-Hold) in beiden Welten mitgefuehrt:
fuer die Strukturentscheidung zaehlt nicht nur „GmbH gegen privat beim
Kandidaten", sondern auch „GmbH gegen privat beim reinen ETF-Sparer" — das ist
der Fall, der Hans tatsaechlich betrifft, wenn am Ende doch der ETF gewinnt.
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
from research.mandat2.engine import run_buy_and_hold, run_strategy  # noqa: E402
from research.mandat2.tax_regimes import make_regime  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
PARAMS = dict(top_in=20, rank_out=200, min_haltetage=730, hebel=1.0)
KAPITAL = [100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000]
FIXKOSTEN = [2_000.0, 3_500.0, 5_000.0]


def main() -> int:
    OUT.mkdir(exist_ok=True)
    d = load_campaign()
    print(d, flush=True)
    n = len(KAPITAL) * len(FIXKOSTEN) * 2  # Kandidat + Benchmark
    print(
        f"Trials kumuliert: {TrialCounter().increment(n, label='P10 GmbH-Break-even')}\n"
    )

    zeilen = []
    for kap in KAPITAL:
        # Privat ist von den Fixkosten unberuehrt -> einmal je Kapital rechnen.
        p_kand = run_strategy(
            d, make_regime("PRIVAT_DE"), startkapital=float(kap), **PARAMS
        )
        p_bench = run_buy_and_hold(d, make_regime("PRIVAT_DE"), startkapital=float(kap))
        print(
            f"=== {kap:>9,.0f} EUR ===  privat: Kandidat {p_kand.equity.iloc[-1]:>13,.0f}"
            f" | ETF {p_bench.equity.iloc[-1]:>13,.0f}",
            flush=True,
        )
        for fk in FIXKOSTEN:
            g_kand = run_strategy(
                d,
                make_regime("GMBH_THESAURIEREND", fixkosten_pa=fk),
                startkapital=float(kap),
                **PARAMS,
            )
            g_bench = run_buy_and_hold(
                d,
                make_regime("GMBH_THESAURIEREND", fixkosten_pa=fk),
                startkapital=float(kap),
            )
            d_kand = float(g_kand.equity.iloc[-1] - p_kand.equity.iloc[-1])
            d_bench = float(g_bench.equity.iloc[-1] - p_bench.equity.iloc[-1])
            zeilen.append(
                {
                    "startkapital": kap,
                    "fixkosten_pa": fk,
                    "privat_kandidat": float(p_kand.equity.iloc[-1]),
                    "gmbh_kandidat": float(g_kand.equity.iloc[-1]),
                    "delta_kandidat": d_kand,
                    "delta_kandidat_pct": d_kand / float(p_kand.equity.iloc[-1]),
                    "privat_etf": float(p_bench.equity.iloc[-1]),
                    "gmbh_etf": float(g_bench.equity.iloc[-1]),
                    "delta_etf": d_bench,
                    "delta_etf_pct": d_bench / float(p_bench.equity.iloc[-1]),
                    "fixkosten_gezahlt": g_kand.portfolio.fixed_costs_paid,
                }
            )
            print(
                f"    FK {fk:>6,.0f}/J: GmbH Kandidat {g_kand.equity.iloc[-1]:>13,.0f}"
                f" ({d_kand:>+12,.0f} = {d_kand / float(p_kand.equity.iloc[-1]):>+6.1%})"
                f"  |  ETF {g_bench.equity.iloc[-1]:>13,.0f} ({d_bench:>+12,.0f}"
                f" = {d_bench / float(p_bench.equity.iloc[-1]):>+6.1%})",
                flush=True,
            )

    (OUT / "p10_gmbh_breakeven.json").write_text(
        json.dumps({"params": PARAMS, "zeilen": zeilen}, indent=2), encoding="utf-8"
    )
    print(f"\n-> {OUT / 'p10_gmbh_breakeven.json'}")

    print("\n" + "=" * 78)
    print("BREAK-EVEN: ab welchem Startkapital ist die GmbH besser als privat?")
    for fk in FIXKOSTEN:
        for feld, was in (("delta_kandidat", "Kandidat"), ("delta_etf", "reiner ETF")):
            positiv = [
                z["startkapital"]
                for z in zeilen
                if z["fixkosten_pa"] == fk and z[feld] > 0
            ]
            grenze = (
                f"ab {min(positiv):,.0f} EUR"
                if positiv
                else "in keinem getesteten Kapital"
            )
            print(f"  FK {fk:>6,.0f}/J | {was:<11}: {grenze}")
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
