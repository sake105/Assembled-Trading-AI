"""P8 — DSR/PBO mit HETEROGENER Familie: der Test, der ueber den Holdout entscheidet.

WARUM DIESER LAUF
-----------------
P7 hat den Kandidaten nicht durchgelassen, aber aus einem Grund, der behebbar
ist: die Varianz der Sharpes wurde aus 37 Fast-Klonen geschaetzt (nur das
Gate-Fenster verschoben). Das misst Klon-Aehnlichkeit, nicht Such-Streuung —
und ein grosses N mit einem zu kleinen V senkt die DSR-Schwelle kuenstlich
(E-077). Dasselbe entwertete die PBO-Zahl: CSCV braucht heterogene Spalten.

Hier wird die Familie so gebaut, wie sie sein muss: **strukturell
verschiedene Strategien**, deren Sharpes tatsaechlich streuen.

  * Buy-and-Hold des Index (der Nullpunkt jeder Suche)
  * Momentum ungegatet, ueber das ganze Haltedauer/Turnover-Gitter
  * Momentum gegatet, mehrere Fenster und Trend-Definitionen
  * Zufallsauswahl, gegatet und ungegatet, mehrere Seeds
  * verschiedene Portfoliobreiten (10 / 20 / 50 Namen)

Das ist genau der Raum, den die Kampagne abgesucht hat. Die Streuung DARIN ist
die ehrliche Bezugsgroesse fuer V.

ENTSCHEIDUNGSREGEL, VOR DEM LAUF FESTGELEGT
-------------------------------------------
Der Kandidat bekommt den EINEN Holdout-Schuss nur, wenn er BEIDE besteht:
  * DSR > 0,95 mit N = kumuliertem Trial-Zaehler UND heterogen geschaetztem V
  * PBO < 50 % ueber die heterogene Matrix
Faellt er durch, ist die Kampagne auf diesem Suchraum beendet — kein weiteres
Nachjustieren, weil jedes weitere N die Schwelle nur hoeher legt.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore", message=".*Converting to PeriodArray.*", category=UserWarning
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.mandat.h011_kandidat_a import cscv_pbo  # noqa: E402
from research.mandat2.campaign_data import load_campaign  # noqa: E402
from research.mandat2.data_gate import TrialCounter  # noqa: E402
from research.mandat2.engine import (  # noqa: E402
    run_buy_and_hold,
    run_strategy,
    sma_gate,
    zufalls_score,
)
from research.mandat2.metrics import auswerten  # noqa: E402
from research.mandat2.tax_regimes import make_regime  # noqa: E402
from src.assembled_core.qa.deflated_sharpe import deflated_sharpe  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
WELT = "ZERO"
KANDIDAT = ("Momentum gegatet SMA140", dict(top_in=20, rank_out=200, min_haltetage=730))


def familie(d) -> list[tuple[str, dict, object | None, int | None]]:
    """(Label, Params, Score-Panel|None fuer Momentum, Gate-Fenster|None)."""
    aus: list[tuple[str, dict, object | None, int | None]] = []
    # 1) Momentum ungegatet ueber das Gitter — die schwachen Varianten gehoeren
    #    dazu, sie sind Teil der Suche und erzeugen die Streuung.
    for hold in (0, 365, 730):
        for out in (30, 60, 200):
            aus.append(
                (
                    f"mom h{hold} o{out}",
                    dict(top_in=20, rank_out=out, min_haltetage=hold),
                    None,
                    None,
                )
            )
    # 2) Portfoliobreite
    for top in (10, 50):
        aus.append(
            (
                f"mom top{top}",
                dict(top_in=top, rank_out=200, min_haltetage=730),
                None,
                None,
            )
        )
    # 3) Momentum gegatet, mehrere Fenster
    for f in (140, 200, 260):
        aus.append(
            (f"mom gate{f}", dict(top_in=20, rank_out=200, min_haltetage=730), None, f)
        )
    # 4) Zufall ungegatet und gegatet
    for seed in range(6):
        aus.append(
            (
                f"zufall s{seed}",
                dict(top_in=20, rank_out=200, min_haltetage=730),
                zufalls_score(d.close, seed),
                None,
            )
        )
        aus.append(
            (
                f"zufall s{seed} gate200",
                dict(top_in=20, rank_out=200, min_haltetage=730),
                zufalls_score(d.close, seed),
                200,
            )
        )
    # 5) Zufall mit schnellem Turnover — der schlechte Rand des Suchraums
    for seed in range(3):
        aus.append(
            (
                f"zufall churn s{seed}",
                dict(top_in=20, rank_out=30, min_haltetage=0),
                zufalls_score(d.close, 100 + seed),
                None,
            )
        )
    return aus


def main() -> int:
    OUT.mkdir(exist_ok=True)
    d = load_campaign()
    print(d, flush=True)
    n_gesamt = TrialCounter().total()
    print(f"Trial-Zaehler kumuliert: {n_gesamt}\n", flush=True)

    bench = run_buy_and_hold(d, make_regime(WELT))
    kurven: dict[str, pd.Series] = {
        "buy&hold SPY": bench.equity_netto.pct_change().dropna()
    }
    kennzahlen: dict[str, dict] = {}

    varianten = familie(d)
    print(f"Heterogene Familie: {len(varianten) + 1} Strategien\n", flush=True)
    for label, params, score, fenster in varianten:
        gate = None if fenster is None else sma_gate(d.close, fenster=fenster)
        r = run_strategy(
            d, make_regime(WELT), score=score, risk_off_gate=gate, **params
        )
        ret = r.equity_netto.pct_change().dropna()
        kurven[label] = ret
        sr = ret.mean() / ret.std() if ret.std() > 0 else np.nan
        a = auswerten(r.equity_netto, bench.equity_netto, label=label)
        kennzahlen[label] = {
            "sharpe_taeglich": float(sr),
            "median": a.median_kandidat,
            "maxdd": a.schlimmster_maxdd,
            "bestanden": a.bestanden,
        }
        print(
            f"  {label:<22} Sharpe {sr:>7.4f} | Median {a.median_kandidat:>6.3f} | "
            f"DD {a.schlimmster_maxdd:>7.1%} | {'BESTANDEN' if a.bestanden else '-'}",
            flush=True,
        )

    rm = pd.DataFrame(kurven).dropna()
    sharpes = rm.apply(lambda x: x.mean() / x.std() if x.std() > 0 else np.nan).dropna()
    var_het = float(sharpes.var(ddof=1))
    print(f"\nRenditematrix: {rm.shape[0]} Tage x {rm.shape[1]} Strategien")
    print(f"Sharpe-Spannweite: {sharpes.min():.4f} .. {sharpes.max():.4f}")
    print(f"HETEROGENE Varianz der Sharpes: {var_het:.3e}")
    print("  (P7-Klonfamilie war 3.723e-05, IID-Naeherung 1.803e-04)")

    kand_label = "mom gate140"
    if kand_label not in rm.columns:
        raise RuntimeError(f"Kandidat {kand_label} fehlt in der Matrix")

    pbo = cscv_pbo(rm, n_blocks=8)
    print(f"\nPBO (CSCV, heterogene Matrix): {pbo:.1%}")

    ergebnis = {
        "welt": WELT,
        "kandidat": kand_label,
        "n_strategien": int(rm.shape[1]),
        "sharpe_min": float(sharpes.min()),
        "sharpe_max": float(sharpes.max()),
        "varianz_heterogen": var_het,
        "pbo": pbo,
        "n_kumuliert": n_gesamt,
        "kennzahlen": kennzahlen,
        "dsr": {},
    }
    print()
    for label, V in (("heterogen", var_het), ("IID-Naeherung", None)):
        res = deflated_sharpe(
            rm[kand_label], n_trials=n_gesamt, variance_across_trials=V
        )
        ergebnis["dsr"][label] = {
            "sharpe_observed": float(res.sharpe_observed),
            "sharpe_threshold": float(res.sharpe_threshold),
            "dsr_probability": float(res.deflated_sharpe_probability),
            "passes_5pct": bool(res.passes_5pct),
        }
        print(
            f"  DSR ({label:<14} N={n_gesamt}): Schwelle {res.sharpe_threshold:.4f} | "
            f"p = {res.deflated_sharpe_probability:.4f} | "
            f"{'BESTANDEN' if res.passes_5pct else 'DURCHGEFALLEN'}"
        )

    (OUT / "p8_dsr_heterogen.json").write_text(
        json.dumps(ergebnis, indent=2), encoding="utf-8"
    )
    print(f"\n-> {OUT / 'p8_dsr_heterogen.json'}")

    dsr_ok = ergebnis["dsr"]["heterogen"]["passes_5pct"]
    dsr_iid_ok = ergebnis["dsr"]["IID-Naeherung"]["passes_5pct"]
    pbo_ok = pbo < 0.5
    print("\n" + "=" * 70)
    print(f"DSR heterogen : {'BESTANDEN' if dsr_ok else 'DURCHGEFALLEN'}")
    print(
        f"DSR IID       : {'BESTANDEN' if dsr_iid_ok else 'DURCHGEFALLEN'} (Gegenprobe)"
    )
    print(f"PBO < 50 %    : {'BESTANDEN' if pbo_ok else 'DURCHGEFALLEN'} ({pbo:.1%})")
    if dsr_ok and dsr_iid_ok and pbo_ok:
        print("\n-> ALLE Kriterien bestanden. Der Holdout-Schuss ist gerechtfertigt.")
    else:
        print("\n-> NICHT alle Kriterien bestanden. Kein Holdout-Schuss.")
        print("   Die Kampagne ist auf diesem Suchraum beendet — jedes weitere")
        print("   Nachjustieren erhoeht N und damit die Schwelle.")
    print("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
