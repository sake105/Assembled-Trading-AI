"""Kampagne K1 — Komposit-Risiko-Dial auf dem SPY-Kern (defensive Fusion).

AUFTRAG (Hans, 2026-08-09): weg von Einsignal-Strategien — Mischungen mit
unterschiedlichen Gewichtungen, aggressiv testen, aber diszipliniert.

MOTIVATION: Die einzige je belegte positive Wirkung im Haus (H-087) war
KRISENVERMEIDUNG — als binaeres Alles-oder-nichts-Timing aber zu teuer
(0/338 CAGR-Siege). K1 testet die Fusion-These in dieser Richtung: drei
Risiko-Komponenten, GEMISCHT und GRADUELL, als Exposure-Regler.

REGISTRIERUNG (vor Datenkontakt fixiert; +8 Trials = 8 Gewichts-Configs;
alle berichtet):
KOMPONENTEN (taeglich, SPY, PIT via shift(1)):
  R1 Trend    = 1 wenn Close < 200T-Schnitt, sonst 0
  R2 Vol      = clip(z(20T-Realvol vs 180T), 0, 3) / 3
  R3 Drawdown = clip(DD-Tiefe / 20 %, 0, 1)
RISIKO = w1*R1 + w2*R2 + w3*R3;  EXPOSURE = 1 - 0.6*RISIKO  (in [0.4, 1]).
Anwendung mit 1 Tag Verzoegerung auf SPY-Tagesrenditen; Kosten 5 bps je
Einheit Exposure-Umschichtung.
GEWICHTS-RASTER (fix): E1 nur_trend(1,0,0) E2 nur_vol(0,1,0)
  E3 nur_dd(0,0,1) E4 gleich(1/3) E5 trend50(.5,.25,.25)
  E6 vol50(.25,.5,.25) E7 dd50(.25,.25,.5) E8 trend_vol(.5,.5,0).
  E1-E3 sind bewusst die Einsignal-Referenzen, E4-E8 die Fusion.
FENSTER: 1996-01-01..2016-12-31 (vor dem versiegelten Holdout).
ERFOLG (vorab, DEFENSIV — nicht CAGR-Sieg): gegen Buy-and-Hold SPY:
  (a) MaxDrawdown um >= 25 % verbessert UND (b) CAGR-Abgabe <= 1.0 pp
  p.a. UND (c) beides haelt in BEIDEN Haelften (1996-2006 / 2007-2016)
  separat. Bonferroni-Hinweis: 8 Configs — Sharpe-Aussagen nur mit
  t > 2.74 zitierfaehig.
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

ZIEL = HIER / "k1_risiko_dial.json"
START, ENDE = "1996-01-01", "2016-12-31"
DIAL_MAX = 0.6
KOSTEN_UMSCHICHTUNG = 0.0005

CONFIGS = {
    "E1_nur_trend": (1.0, 0.0, 0.0),
    "E2_nur_vol": (0.0, 1.0, 0.0),
    "E3_nur_dd": (0.0, 0.0, 1.0),
    "E4_gleich": (1 / 3, 1 / 3, 1 / 3),
    "E5_trend50": (0.50, 0.25, 0.25),
    "E6_vol50": (0.25, 0.50, 0.25),
    "E7_dd50": (0.25, 0.25, 0.50),
    "E8_trend_vol": (0.50, 0.50, 0.0),
}


def lade_spy() -> pd.Series:
    pv = pd.read_parquet(
        ROOT / "research" / "mandat" / "data" / "prices_verdict.parquet"
    )
    s = (
        pv[pv["symbol"] == "SPY"]
        .set_index("timestamp")["close"]
        .sort_index()
        .loc["1994-06-01":ENDE]
    )
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s.astype(float)


def kennzahlen(kurve: pd.Series) -> dict:
    r = kurve.pct_change().dropna()
    jahre = (kurve.index[-1] - kurve.index[0]).days / 365.25
    cagr = float((kurve.iloc[-1] / kurve.iloc[0]) ** (1 / jahre) - 1)
    dd = float((kurve / kurve.cummax() - 1).min())
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
    return {
        "cagr_pct": round(cagr * 100, 2),
        "mdd_pct": round(dd * 100, 2),
        "sharpe": round(sharpe, 3),
    }


def simuliere(px: pd.Series, gewichte: tuple[float, float, float]) -> pd.Series:
    w1, w2, w3 = gewichte
    r1 = (px < px.rolling(200).mean()).astype(float)
    vol = px.pct_change().rolling(20).std()
    mu, sd = vol.rolling(180).mean(), vol.rolling(180).std()
    r2 = (((vol - mu) / sd).clip(0, 3) / 3).fillna(0)
    r3 = ((1 - px / px.cummax()).clip(0, 0.20) / 0.20).fillna(0)
    risiko = (w1 * r1 + w2 * r2 + w3 * r3).clip(0, 1)
    expo = (1 - DIAL_MAX * risiko).shift(1)  # PIT: gestriges Risiko steuert heute
    rt = px.pct_change()
    kosten = expo.diff().abs().fillna(0) * KOSTEN_UMSCHICHTUNG
    strat = (expo * rt - kosten).fillna(0)
    kurve = (1 + strat).cumprod()
    kurve.name = "kurve"
    return kurve


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args(argv)
    if args.regen:
        print(f"[REGEN] Trials unveraendert: {TrialCounter().total()}", flush=True)
    else:
        print(
            "Trials kumuliert: "
            + str(
                TrialCounter().increment(8, label="K1 Risiko-Dial E1-E8 SPY 1996-2016")
            ),
            flush=True,
        )
    px = lade_spy()
    fenster = {
        "gesamt": (START, ENDE),
        "h1": (START, "2006-12-31"),
        "h2": ("2007-01-01", ENDE),
    }
    bh = {f: kennzahlen(px.loc[a:b]) for f, (a, b) in fenster.items()}
    ergebnis: dict = {"registriert": __doc__, "buy_and_hold": bh, "configs": {}}
    bestehende = []
    for name, w in CONFIGS.items():
        kurve = simuliere(px, w)
        werte = {
            f: kennzahlen(kurve.loc[a:b] / kurve.loc[a:b].iloc[0])
            for f, (a, b) in fenster.items()
        }
        checks = {}
        for f in fenster:
            mdd_impr = 1 - werte[f]["mdd_pct"] / bh[f]["mdd_pct"]
            cagr_abgabe = bh[f]["cagr_pct"] - werte[f]["cagr_pct"]
            checks[f] = {
                "mdd_verbesserung_pct": round(mdd_impr * 100, 1),
                "cagr_abgabe_pp": round(cagr_abgabe, 2),
                "besteht": bool(mdd_impr >= 0.25 and cagr_abgabe <= 1.0),
            }
        besteht = all(checks[f]["besteht"] for f in ("gesamt", "h1", "h2"))
        if besteht:
            bestehende.append(name)
        ergebnis["configs"][name] = {
            "gewichte": w,
            "kennzahlen": werte,
            "kriterium": checks,
            "besteht": besteht,
        }
        g = werte["gesamt"]
        print(
            f"{name}: CAGR {g['cagr_pct']}% (BH {bh['gesamt']['cagr_pct']}%) "
            f"MDD {g['mdd_pct']}% (BH {bh['gesamt']['mdd_pct']}%) "
            f"Sharpe {g['sharpe']} -> {'BESTEHT' if besteht else 'fail'}",
            flush=True,
        )
    ergebnis["verdikt"] = {
        "bestehende_configs": bestehende,
        "kriterium": "MDD >= 25% besser UND CAGR-Abgabe <= 1pp, in gesamt+h1+h2 (vorab fixiert)",
    }
    ZIEL.write_text(
        json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("VERDIKT:", ergebnis["verdikt"])
    print(f"-> {ZIEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
