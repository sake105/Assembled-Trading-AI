"""K2+K3 — Feinstruktur des bestaetigten E1-Dials + Cross-Asset-Variante.

REGISTRIERUNG (vor Datenkontakt fixiert; +12 Trials = 12 Configs; alle
berichtet; Auftrag Hans 2026-08-10 "na dann los geht's" auf Basis
INDIKATOR_AUDIT.md K2/K3):

K2 — E1-DIAL-FEINSTRUKTUR (SPY, 1996-2016, Haelften 1996-2006/2007-2016):
  Signalform: (a) binaer  R = 1 wenn Close < MA200 (wie E1)
              (b) graduell R = clip((MA200-Close)/MA200, 0, 0.10)/0.10
  Dial-Tiefe D in {0.4, 0.6, 0.8}; EXPOSURE = 1 - D*R; 1 Tag Lag;
  5 bps je Umschichtungseinheit. -> 6 Configs (K2_bin40..K2_grad80).
  K2_bin60 = E1 aus K1 (Referenz, bewusst enthalten).
KRITERIUM (identisch K1, vorab): vs Buy-and-Hold SPY: MDD >= 25 % besser
  UND CAGR-Abgabe <= 1.0 pp p.a., in GESAMT und BEIDEN Haelften.
BESTAETIGUNG: jede bestehende K2-Config bekommt EINEN Versuch auf
  1926-1995 (CRSP-VW, Ueberschussrechnung, wie k1_bestaetigung) — je +1
  Trial, im Skript mitgebucht NUR falls Bestehen (Ansage vorab: maximal
  +6 zusaetzlich; tatsaechliche Zahl wird ausgewiesen).

K3 — CROSS-ASSET-DIAL (SPY+GLD; GLD erst ab 2004-11 -> Fenster
  2005-07-01..2016-12-31, Haelften 2005-07..2011-03 / 2011-04..2016-12;
  OFFENGELEGT: kuerzer und nur ein Boersenzyklus):
  Basis-Allokation A in {100/0, 70/30} (SPY/GLD, taeglich rebalanciert
  ueber die Gewichte — vereinfachend, Kosten auf Gewichtsdrift).
  Risiko R = binaerer SPY-Trend (wie E1). Freigesetztes SPY-Gewicht
  (D=0.6 fix) geht zu Anteil g in {0 (Cash), 0.5, 1.0} nach GLD.
  -> 6 Configs (K3_a100_g0.. K3_a70_g100). K3_a100_g0 = E1 im kurzen
  Fenster (Referenz).
KRITERIUM: vs statisches Buy-and-Hold DERSELBEN Basis-Allokation:
  MDD >= 25 % besser UND CAGR-Abgabe <= 1.0 pp, gesamt + beide Haelften.
KEINE historische Bestaetigung moeglich (Gold-Preisgeschichte vor 1975
  reguliert) — Bestehen hiesse nur "Kandidat fuer Forward-Shadow".
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

ZIEL = HIER / "k2_k3_dial_raster.json"
KOSTEN = 0.0005


def lade(symbol: str) -> pd.Series:
    pv = pd.read_parquet(
        ROOT / "research" / "mandat" / "data" / "prices_verdict.parquet"
    )
    s = (
        pv[pv["symbol"] == symbol]
        .set_index("timestamp")["close"]
        .sort_index()
        .astype(float)
    )
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s


def kennzahlen(kurve: pd.Series) -> dict:
    jahre = (kurve.index[-1] - kurve.index[0]).days / 365.25
    cagr = float((kurve.iloc[-1] / kurve.iloc[0]) ** (1 / jahre) - 1)
    dd = float((kurve / kurve.cummax() - 1).min())
    return {"cagr_pct": round(cagr * 100, 2), "mdd_pct": round(dd * 100, 2)}


def pruefe(kurve: pd.Series, bh: pd.Series, fenster: dict) -> tuple[dict, bool]:
    checks = {}
    for f, (a, b) in fenster.items():
        ks = kennzahlen(kurve.loc[a:b] / kurve.loc[a:b].iloc[0])
        kb = kennzahlen(bh.loc[a:b] / bh.loc[a:b].iloc[0])
        mdd_impr = 1 - ks["mdd_pct"] / kb["mdd_pct"]
        abgabe = kb["cagr_pct"] - ks["cagr_pct"]
        checks[f] = {
            "strategie": ks,
            "benchmark": kb,
            "mdd_verbesserung_pct": round(mdd_impr * 100, 1),
            "cagr_abgabe_pp": round(abgabe, 2),
            "besteht": bool(mdd_impr >= 0.25 and abgabe <= 1.0),
        }
    return checks, all(c["besteht"] for c in checks.values())


def k2_kurve(px: pd.Series, form: str, tiefe: float) -> pd.Series:
    ma = px.rolling(200).mean()
    if form == "bin":
        r = (px < ma).astype(float)
    else:
        r = ((ma - px) / ma).clip(0, 0.10) / 0.10
    expo = (1 - tiefe * r).shift(1)
    rt = px.pct_change()
    kosten = expo.diff().abs().fillna(0) * KOSTEN
    return (1 + (expo * rt - kosten).fillna(0)).cumprod()


def k3_kurve(spy: pd.Series, gld: pd.Series, w_spy: float, g: float) -> pd.Series:
    ma = spy.rolling(200).mean()
    r = (spy < ma).astype(float).shift(1).fillna(0)
    frei = 0.6 * w_spy * r  # freigesetztes SPY-Gewicht bei Risiko
    ws = w_spy - frei
    wg = (1 - w_spy) + g * frei  # Rest von frei bleibt Cash
    rt_s, rt_g = spy.pct_change(), gld.pct_change()
    umschichtung = ws.diff().abs().fillna(0) + wg.diff().abs().fillna(0)
    strat = (ws * rt_s + wg * rt_g - umschichtung * KOSTEN).fillna(0)
    return (1 + strat).cumprod()


def bestaetigung_1926(form: str, tiefe: float) -> dict:
    df = pd.read_parquet(
        ROOT / "research" / "mandat2" / "data_gratis" / "fama_french_daily.parquet"
    )
    df = df.loc["1926-07-01":"1995-12-31"]
    px = (1 + df["mkt"]).cumprod()
    ma = px.rolling(200).mean()
    if form == "bin":
        r = (px < ma).astype(float)
    else:
        r = ((ma - px) / ma).clip(0, 0.10) / 0.10
    expo = (1 - tiefe * r).shift(1)
    kosten = expo.diff().abs().fillna(0) * KOSTEN
    strat = (1 + (expo * df["mkt_rf"] - kosten).fillna(0)).cumprod()
    bh = (1 + df["mkt_rf"].fillna(0)).cumprod()
    fenster = {
        "gesamt": ("1926-07-01", "1995-12-31"),
        "h1": ("1926-07-01", "1960-12-31"),
        "h2": ("1961-01-01", "1995-12-31"),
    }
    checks, ok = pruefe(strat, bh, fenster)
    return {"checks": checks, "bestaetigt": ok}


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
            + str(TrialCounter().increment(12, label="K2+K3 Dial-Raster (12 Configs)")),
            flush=True,
        )
    ergebnis: dict = {"registriert": __doc__, "K2": {}, "K3": {}}

    # --- K2 ---
    spy = lade("SPY")
    px = spy.loc["1994-06-01":"2016-12-31"]
    fenster2 = {
        "gesamt": ("1996-01-01", "2016-12-31"),
        "h1": ("1996-01-01", "2006-12-31"),
        "h2": ("2007-01-01", "2016-12-31"),
    }
    bh2 = px
    k2_bestanden = []
    for form in ("bin", "grad"):
        for tiefe in (0.4, 0.6, 0.8):
            name = f"K2_{form}{int(tiefe * 100)}"
            kurve = k2_kurve(px, form, tiefe)
            checks, ok = pruefe(kurve, bh2, fenster2)
            ergebnis["K2"][name] = {"checks": checks, "besteht_1996_2016": ok}
            if ok:
                k2_bestanden.append((name, form, tiefe))
            g = checks["gesamt"]
            print(
                f"{name}: MDD-Verb. {g['mdd_verbesserung_pct']}% "
                f"Abgabe {g['cagr_abgabe_pp']}pp "
                f"(h1 {checks['h1']['mdd_verbesserung_pct']}%/{checks['h1']['cagr_abgabe_pp']}pp, "
                f"h2 {checks['h2']['mdd_verbesserung_pct']}%/{checks['h2']['cagr_abgabe_pp']}pp) "
                f"-> {'BESTEHT' if ok else 'fail'}",
                flush=True,
            )
    # Bestaetigungen (je +1 Trial, nur fuer Bestehende — vorab angekuendigt)
    if k2_bestanden and not args.regen:
        print(
            "Trials kumuliert: "
            + str(
                TrialCounter().increment(
                    len(k2_bestanden),
                    label=f"K2-Bestaetigungen 1926-1995 ({len(k2_bestanden)}x)",
                )
            ),
            flush=True,
        )
    for name, form, tiefe in k2_bestanden:
        b = bestaetigung_1926(form, tiefe)
        ergebnis["K2"][name]["bestaetigung_1926_1995"] = b
        print(
            f"{name} Bestaetigung 1926-95: {'BESTANDEN' if b['bestaetigt'] else 'FAIL'}",
            flush=True,
        )

    # --- K3 ---
    gld = lade("GLD")
    beide = pd.concat([spy.rename("SPY"), gld.rename("GLD")], axis=1).dropna()
    beide = beide.loc["2004-11-18":"2016-12-31"]
    fenster3 = {
        "gesamt": ("2005-07-01", "2016-12-31"),
        "h1": ("2005-07-01", "2011-03-31"),
        "h2": ("2011-04-01", "2016-12-31"),
    }
    for w_spy in (1.0, 0.7):
        # statischer Benchmark derselben Basis-Allokation
        rt = beide.pct_change()
        bh_r = (w_spy * rt["SPY"] + (1 - w_spy) * rt["GLD"]).fillna(0)
        bh3 = (1 + bh_r).cumprod()
        for g_anteil in (0.0, 0.5, 1.0):
            name = f"K3_a{int(w_spy * 100)}_g{int(g_anteil * 100)}"
            kurve = k3_kurve(beide["SPY"], beide["GLD"], w_spy, g_anteil)
            checks, ok = pruefe(kurve, bh3, fenster3)
            ergebnis["K3"][name] = {"checks": checks, "besteht": ok}
            gg = checks["gesamt"]
            print(
                f"{name}: MDD-Verb. {gg['mdd_verbesserung_pct']}% "
                f"Abgabe {gg['cagr_abgabe_pp']}pp "
                f"(h1 {checks['h1']['mdd_verbesserung_pct']}%/{checks['h1']['cagr_abgabe_pp']}pp, "
                f"h2 {checks['h2']['mdd_verbesserung_pct']}%/{checks['h2']['cagr_abgabe_pp']}pp) "
                f"-> {'BESTEHT' if ok else 'fail'}",
                flush=True,
            )

    ergebnis["verdikt"] = {
        "K2_bestanden_1996_2016": [n for n, _, _ in k2_bestanden],
        "K2_bestaetigt_1926_1995": [
            n
            for n, _, _ in k2_bestanden
            if ergebnis["K2"][n].get("bestaetigung_1926_1995", {}).get("bestaetigt")
        ],
        "K3_bestanden": [n for n, v in ergebnis["K3"].items() if v["besteht"]],
        "kriterium": "MDD>=25% besser UND CAGR-Abgabe<=1pp, gesamt+h1+h2 (vorab fixiert)",
    }
    ZIEL.write_text(
        json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("VERDIKT:", ergebnis["verdikt"])
    print(f"-> {ZIEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
