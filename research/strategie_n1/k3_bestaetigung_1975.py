"""K3-Bestaetigung: a100_g100 ("Gold statt Cash") auf 1975-2004 — EIN Versuch.

REGISTRIERUNG (+1 Trial, vorab): Die einzige bestehende K2/K3-Config
(100% Aktien; bei Trendbruch wandern 60% des Aktiengewichts VOLL in Gold)
wird auf disjunkten Daten geprueft: FF-Marktindex (Total Return) + LBMA-
Gold-PM-Fixing USD, 1975-01..2004-10 (freier Goldmarkt; endet vor dem
K3-Fenster 2005-2016). Haelften 1975-1989 / 1990-2004. Kein Cash-Bein
-> Total-Return-Vergleich zulaessig. Kosten 5 bps je Umschichtungseinheit.
OFFENGELEGT: Fixing ist kein handelbares Produkt (keine Lager-/Spread-
kosten), Marktindex vor 1993 nicht direkt handelbar.
KRITERIUM (identisch K1/K3): vs 100% Aktien-Buy-and-Hold: MDD >= 25 %
besser UND CAGR-Abgabe <= 1.0 pp p.a., in gesamt + BEIDEN Haelften.
FAIL = K3-Kandidat gilt als nicht bestaetigt; kein weiterer Versuch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HIER = Path(__file__).resolve().parent
ROOT = HIER.parents[1]
sys.path.insert(0, str(ROOT))

from research.mandat2.data_gate import TrialCounter  # noqa: E402
from research.strategie_n1.k2_k3_dial_raster import kennzahlen, pruefe  # noqa: E402

ZIEL = HIER / "k3_bestaetigung_1975.json"


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args()
    if args.regen:
        print(f"[REGEN] Trials unveraendert: {TrialCounter().total()}", flush=True)
    else:
        print(
            "Trials kumuliert: "
            + str(
                TrialCounter().increment(1, label="K3-Bestaetigung a100_g100 1975-2004")
            ),
            flush=True,
        )
    ff = pd.read_parquet(
        ROOT / "research" / "mandat2" / "data_gratis" / "fama_french_daily.parquet"
    )
    aktien = (1 + ff["mkt"]).cumprod().rename("AKT")
    roh = json.load(open(HIER / "data_geo" / "lbma_gold_pm.json", encoding="utf-8"))
    gold = pd.Series(
        {pd.Timestamp(r["d"]): r["v"][0] for r in roh if r["v"] and r["v"][0]},
        name="GOLD",
    ).sort_index()
    beide = pd.concat([aktien, gold], axis=1).dropna().loc["1973-06-01":"2004-10-31"]

    ma = beide["AKT"].rolling(200).mean()
    r = (beide["AKT"] < ma).astype(float).shift(1).fillna(0)
    frei = 0.6 * r
    ws, wg = 1 - frei, frei
    rt_a, rt_g = beide["AKT"].pct_change(), beide["GOLD"].pct_change()
    umschichtung = ws.diff().abs().fillna(0) * 2
    strat = (1 + (ws * rt_a + wg * rt_g - umschichtung * 0.0005).fillna(0)).cumprod()
    bh = beide["AKT"]

    fenster = {
        "gesamt": ("1975-01-01", "2004-10-31"),
        "h1": ("1975-01-01", "1989-12-31"),
        "h2": ("1990-01-01", "2004-10-31"),
    }
    checks, ok = pruefe(strat, bh, fenster)
    for f, c in checks.items():
        print(
            f"{f}: Strat CAGR {c['strategie']['cagr_pct']}% MDD {c['strategie']['mdd_pct']}% | "
            f"BH {c['benchmark']['cagr_pct']}%/{c['benchmark']['mdd_pct']}% | "
            f"Verb. {c['mdd_verbesserung_pct']}% Abgabe {c['cagr_abgabe_pp']}pp -> {'ok' if c['besteht'] else 'FAIL'}",
            flush=True,
        )
    ZIEL.write_text(
        json.dumps(
            {"registriert": __doc__, "checks": checks, "bestaetigt": ok},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("VERDIKT:", {"bestaetigt": ok})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
