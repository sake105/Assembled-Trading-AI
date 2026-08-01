"""P1-Erstlauf: Momentum gegen SPY unter allen vier Regimen (Mandat II).

Beantwortet die Frage, mit der Hans das Mandat neu geoeffnet hat: kippt das
Urteil, wenn die deutsche Privatanleger-Steuer nicht mehr die Bremse ist?

WICHTIG: laeuft ausschliesslich auf dem SUCH-Fenster (bis 2016-12-31). Der
Holdout wird hier nicht angefasst.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

# Eng begrenzt statt global: ein pauschales filterwarnings("ignore") im
# Runner, der die entscheidungstragenden Zahlen erzeugt, wuerde genau die
# Signale schlucken, die bei Panel-Arbeit auf Datenprobleme zeigen
# (CLAUDE.md: Datenprobleme nicht still verschlucken).
warnings.filterwarnings(
    "ignore", message=".*Converting to PeriodArray.*", category=UserWarning
)

from research.mandat2.campaign_data import load_campaign  # noqa: E402
from research.mandat2.engine import run_buy_and_hold, run_momentum  # noqa: E402
from research.mandat2.metrics import DD_DECKEL, auswerten  # noqa: E402
from research.mandat2.tax_regimes import make_regime  # noqa: E402

REGIME = ["ZERO", "PRIVAT_DE", "GMBH_THESAURIEREND", "GMBH_AUSSCHUETTUNG"]
OUT = Path(__file__).resolve().parent / "results"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    d = load_campaign()
    print(d, flush=True)
    ergebnisse = {}

    for name in REGIME:
        bench = run_buy_and_hold(d, make_regime(name))
        kand = run_momentum(d, make_regime(name), top_in=20, rank_out=60)
        # Ausschuettungsebene erst am Ende, damit die Kurve sie nicht sieht.
        if name == "GMBH_AUSSCHUETTUNG":
            for lauf in (bench, kand):
                netto = lauf.portfolio.settle_terminal(float(lauf.equity.iloc[-1]))
                lauf.equity.iloc[-1] = netto
                lauf.equity_netto.iloc[-1] = netto
        # Auf der NETTO-Kurve auswerten: sonst traegt der umschichtende
        # Kandidat seine Steuer laufend und der Buy-and-Hold-Benchmark nie
        # (E-071).
        a = auswerten(
            kand.equity_netto, bench.equity_netto, label=f"Momentum vs SPY [{name}]"
        )
        print(f"\n=== {name} ===", flush=True)
        print("  " + bench.kurz(), flush=True)
        print("  " + kand.kurz(), flush=True)
        print("  " + a.bericht(), flush=True)
        ergebnisse[name] = {
            "benchmark_end": float(bench.equity.iloc[-1]),
            "kandidat_end": float(kand.equity.iloc[-1]),
            "n_fenster": a.n_fenster,
            "median_kandidat": a.median_kandidat,
            "median_benchmark": a.median_benchmark,
            "anteil_fenster_geschlagen": a.anteil_fenster_geschlagen,
            "schlimmster_maxdd": a.schlimmster_maxdd,
            "deckel_eingehalten": a.deckel_eingehalten,
            "bestanden": a.bestanden,
            "steuer_kandidat": kand.portfolio.tax_paid,
            "steuer_benchmark": bench.portfolio.tax_paid,
            "trades_kandidat": kand.n_trades,
            # Befund 1 muss aus dem Artefakt reproduzierbar sein, nicht nur
            # aus dem Prosatext (F-senior-9).
            "benchmark_maxdd_schlimmster": min(f.benchmark_maxdd for f in a.fenster),
            "benchmark_maxdd_bester": max(f.benchmark_maxdd for f in a.fenster),
            "benchmark_fenster_deckel_gerissen": sum(
                1 for f in a.fenster if f.benchmark_maxdd < DD_DECKEL
            ),
            "kandidat_fenster_deckel_gerissen": len(a.gerissene_fenster),
            "div_steuer_kandidat": kand.portfolio.tax_on_dividends,
            "kursgewinn_steuer_kandidat": kand.portfolio.tax_on_gains,
            "nicht_ausfuehrbare_auftraege": kand.nicht_ausfuehrbar,
            "fixkosten_pa": 0.0,
        }

    (OUT / "p1_baseline.json").write_text(
        json.dumps(ergebnisse, indent=2), encoding="utf-8"
    )
    print(f"\n-> {OUT / 'p1_baseline.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
