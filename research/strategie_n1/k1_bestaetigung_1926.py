"""K1-Bestaetigung: E1 (gradueller Trend-Dial) auf 1926-1995 — EIN Versuch.

REGISTRIERUNG (+1 Trial, vorab): Das K1-Beinahe-Ergebnis (E1 nur_trend:
MDD -30..39 % fuer 0.3-1.2 pp CAGR auf SPY 1996-2016) wird auf VOELLIG
DISJUNKTEN Daten geprueft: CRSP-VW-Index 1926-01..1995-12 (fama_french_
daily.parquet; kein Ueberlapp mit K1-Fenster oder Holdout). E-130-konform
auf UEBERSCHUSSRENDITEN (mkt_rf): nicht investiert = risikoloser Satz.
Regel identisch K1/E1: R = 1 wenn Kurs < 200T-Schnitt sonst 0;
Exposure = 1 - 0.6*R, 1 Tag Lag, 5 bps je Umschichtungseinheit.
KRITERIUM (identisch K1): MDD >= 25 % besser als Buy-and-Hold UND
CAGR-Abgabe <= 1.0 pp p.a., in GESAMT und BEIDEN Haelften
(1926-1960 / 1961-1995). Ein FAIL hier = K1/E1 gilt als nicht bestaetigt;
KEIN weiterer Versuch auf diesen Daten.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HIER = Path(__file__).resolve().parent
ROOT = HIER.parents[1]
sys.path.insert(0, str(ROOT))

from research.mandat2.data_gate import TrialCounter  # noqa: E402

ZIEL = HIER / "k1_bestaetigung_1926.json"


def kennzahlen(kurve: pd.Series) -> dict:
    r = kurve.pct_change().dropna()
    jahre = (kurve.index[-1] - kurve.index[0]).days / 365.25
    cagr = float((kurve.iloc[-1] / kurve.iloc[0]) ** (1 / jahre) - 1)
    dd = float((kurve / kurve.cummax() - 1).min())
    return {"cagr_pct": round(cagr * 100, 2), "mdd_pct": round(dd * 100, 2)}


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
            + str(TrialCounter().increment(1, label="K1-Bestaetigung E1 1926-1995")),
            flush=True,
        )
    df = pd.read_parquet(
        ROOT / "research" / "mandat2" / "data_gratis" / "fama_french_daily.parquet"
    )
    df = df.loc["1926-07-01":"1995-12-31"]
    px = (1 + df["mkt"]).cumprod()  # Preisreihe nur fuer das Trendsignal
    risiko = (px < px.rolling(200).mean()).astype(float)
    expo = (1 - 0.6 * risiko).shift(1)
    kosten = expo.diff().abs().fillna(0) * 0.0005
    strat_x = (expo * df["mkt_rf"] - kosten).fillna(0)
    bh_x = df["mkt_rf"].fillna(0)
    fenster = {
        "gesamt": ("1926-07-01", "1995-12-31"),
        "h1": ("1926-07-01", "1960-12-31"),
        "h2": ("1961-01-01", "1995-12-31"),
    }
    ergebnis: dict = {"registriert": __doc__, "fenster": {}}
    alle_ok = True
    for f, (a, b) in fenster.items():
        ks = kennzahlen((1 + strat_x.loc[a:b]).cumprod())
        kb = kennzahlen((1 + bh_x.loc[a:b]).cumprod())
        mdd_impr = 1 - ks["mdd_pct"] / kb["mdd_pct"]
        abgabe = kb["cagr_pct"] - ks["cagr_pct"]
        ok = bool(mdd_impr >= 0.25 and abgabe <= 1.0)
        alle_ok &= ok
        ergebnis["fenster"][f] = {
            "strategie": ks,
            "buy_and_hold": kb,
            "mdd_verbesserung_pct": round(mdd_impr * 100, 1),
            "cagr_abgabe_pp": round(abgabe, 2),
            "besteht": ok,
        }
        print(
            f"{f}: Strat CAGRx {ks['cagr_pct']}% MDD {ks['mdd_pct']}% | "
            f"BH CAGRx {kb['cagr_pct']}% MDD {kb['mdd_pct']}% | "
            f"MDD-Verb. {round(mdd_impr * 100, 1)}% Abgabe {round(abgabe, 2)}pp -> {'ok' if ok else 'FAIL'}",
            flush=True,
        )
    ergebnis["verdikt"] = {"bestaetigt": bool(alle_ok)}
    ZIEL.write_text(
        json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("VERDIKT:", ergebnis["verdikt"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
