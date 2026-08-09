"""Strategie N1 — KOMBINATIONS-Backtest (Geo + Finanz + TA) mit Gewichtsraster.

AUFTRAG (Hans, 2026-08-09): nicht ein Strang, sondern das ZUSAMMENSPIEL —
und an den Gewichten drehen (Beispiel: TA 50 %, Geo/Finanz abgesenkt).

REGISTRIERUNG (vor Datenkontakt fixiert; +6 Trials = 6 Gewichts-Configs;
ALLE sechs werden berichtet, die beste muss Bonferroni-korrigiert t > 2,64
schlagen — sonst gilt das Raster als FAIL):

TEILSCORES (taeglich, PIT wie geo_backtest: Signal T nutzt Daten bis T-1):
  GEO = max(z_me, z_gl) der QuadClass-4-Artikel (wie geo_backtest.py).
  FIN = -z(globaler AvgTone-Mittelwert)  — Finanz-Stress-Proxy aus dem
        GDELT-Korpus (FAZ/WSJ etc. haben keine Historie; OFFENGELEGT:
        das ist Nachrichten-Ton, kein reiner Finanz-Feed).
  TA  = z(20-Handelstage-Momentum des jeweiligen Instrument-Baskets,
        rollierend 180T) — TA hier als KOMPONENTE (Auftrag), nicht nur Veto.
  SOC = vor 2022 nicht existent -> Gewicht 0, OFFENGELEGT; volle 4er-Fusion
        nur im Forward-Shadow moeglich.
  z-Normierung je Teilscore: rolling 180 T (min 120), shift(1).

KOMPOSIT je Instrument: K = w_geo*GEO + w_fin*FIN + w_ta*TA.
GEWICHTS-RASTER (fix, keine weiteren Kombinationen):
  W1 gleich        (1/3, 1/3, 1/3)
  W2 ta50          (0.25, 0.25, 0.50)   <- Hans' Beispiel
  W3 ta60          (0.20, 0.20, 0.60)
  W4 geo_dominant  (0.50, 0.25, 0.25)
  W5 fin_dominant  (0.25, 0.50, 0.25)
  W6 ohne_ta       (0.50, 0.50, 0.00)
ENTRY: K(T-1) >= rollierendes 98%-Perzentil von K (252 T, min 120,
  shift(1)) -> Kauf zum naechsten Close (wie geo_backtest, close-basiert).
INSTRUMENTE/EXITS/KOSTEN/KONTROLLE: identisch zu geo_backtest.py
  (GLD, Defense-Basket, Oel-Majors-Basket; Zerfall K < halbes Entry-
  Niveau ersetzt den Score-Zerfall; 15 T; -5 %; Trailing +3 %/2 %;
  10 bps RT; 100 Zufalls-Entries je Instrument, Seed 52).
FENSTER: 2015-01-01..2016-12-31 (Holdout versiegelt).
VERDIKT (vorab): Raster besteht nur, wenn MINDESTENS EINE Config gepoolt
  netto > 0 UND t > 2,64 UND PF > 1 UND klar ueber der Kontrolle liegt.
"""

from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HIER = Path(__file__).resolve().parent
ROOT = HIER.parents[1]
sys.path.insert(0, str(ROOT))

from research.mandat2.data_gate import TrialCounter  # noqa: E402
from research.strategie_n1.geo_backtest import (  # noqa: E402
    ENDE,
    FEE_SEITE,
    INSTRUMENTE,
    KONTROLLEN,
    MAX_TAGE,
    NAHOST,
    START,
    STOP,
    TRAIL_ABSTAND,
    TRAIL_AKTIV,
    lade_basket,
    statistik,
)

DATEN = HIER / "data_geo"
ZIEL = HIER / "composite_backtest.json"
UA = "Mozilla/5.0 Forschung-N1 (hans.oertel2@gmail.com)"
SEED = 52
PERZENTIL = 0.98

GEWICHTE = {
    "W1_gleich": (1 / 3, 1 / 3, 1 / 3),
    "W2_ta50": (0.25, 0.25, 0.50),
    "W3_ta60": (0.20, 0.20, 0.60),
    "W4_geo_dominant": (0.50, 0.25, 0.25),
    "W5_fin_dominant": (0.25, 0.50, 0.25),
    "W6_ohne_ta": (0.50, 0.50, 0.00),
}


def lade_gdelt_tag_v2(datum: str) -> dict:
    """me/gl (QuadClass-4-Artikel) + tone_summe/tone_n (alle Events)."""
    cache = DATEN / f"g2_{datum}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    url = f"http://data.gdeltproject.org/events/{datum}.export.CSV.zip"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for versuch in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as h:  # noqa: S310
                roh = h.read()
            break
        except Exception:
            if versuch == 2:
                raise
            time.sleep(5)
    me = gl = tone_summe = 0.0
    tone_n = 0
    with zipfile.ZipFile(io.BytesIO(roh)) as z:
        with z.open(z.namelist()[0]) as fh:
            for zeile in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
                f = zeile.rstrip("\n").split("\t")
                if len(f) < 55:
                    continue
                try:
                    tone = float(f[34])
                except ValueError:
                    continue
                tone_summe += tone
                tone_n += 1
                if f[29] == "4":
                    try:
                        artikel = float(f[33])
                    except ValueError:
                        continue
                    gl += artikel
                    if f[51] in NAHOST:
                        me += artikel
    d = {"me": me, "gl": gl, "tone_summe": tone_summe, "tone_n": tone_n}
    cache.write_text(json.dumps(d))
    return d


def z_roll(s: pd.Series, fenster: int = 180) -> pd.Series:
    mu = s.rolling(fenster, min_periods=120).mean().shift(1)
    sd = s.rolling(fenster, min_periods=120).std().shift(1)
    return (s - mu) / sd


def baue_teilscores() -> pd.DataFrame:
    tage = pd.date_range("2014-06-01", ENDE, freq="D")
    zeilen = []
    for i, t in enumerate(tage):
        d = lade_gdelt_tag_v2(t.strftime("%Y%m%d"))
        zeilen.append(
            {
                "tag": t,
                "me": d["me"],
                "gl": d["gl"],
                "tone": d["tone_summe"] / d["tone_n"] if d["tone_n"] else np.nan,
            }
        )
        if i % 100 == 0:
            print(f"[GDELT-v2] {t.date()} ({i}/{len(tage)})", flush=True)
        time.sleep(0.2)
    df = pd.DataFrame(zeilen).set_index("tag")
    aus = pd.DataFrame(index=df.index)
    aus["GEO"] = pd.concat([z_roll(df["me"]), z_roll(df["gl"])], axis=1).max(axis=1)
    aus["FIN"] = -z_roll(df["tone"])
    return aus


def simuliere_k(px: pd.Series, k: pd.Series, schwelle: pd.Series, i_entry: int):
    c = px.to_numpy()
    if i_entry + 1 >= len(px):
        return None
    entry = c[i_entry] * (1 + FEE_SEITE)
    basis = c[i_entry]
    entry_k = k.iloc[i_entry - 1] if i_entry >= 1 else np.nan
    peak_close = -np.inf
    for n, j in enumerate(range(i_entry + 1, min(i_entry + 1 + MAX_TAGE, len(px) - 1))):
        peak_close = max(peak_close, c[j])
        stop_riss = c[j] <= basis * (1 + STOP)
        trail_aktiv = peak_close >= basis * (1 + TRAIL_AKTIV)
        trail_riss = trail_aktiv and c[j] <= peak_close * (1 - TRAIL_ABSTAND)
        kj = k.iloc[j]
        zerfall = bool(np.isfinite(kj) and np.isfinite(entry_k) and kj < 0.5 * entry_k)
        if stop_riss or trail_riss or zerfall or n == MAX_TAGE - 1:
            return c[j + 1] * (1 - FEE_SEITE) / entry - 1, j + 1
    j = min(i_entry + MAX_TAGE, len(px) - 1)
    return c[j] * (1 - FEE_SEITE) / entry - 1, j


def lauf_k(px, k, schwelle, entries):
    aus, frei_ab = [], -1
    for i in entries:
        if i <= frei_ab:
            continue
        r = simuliere_k(px, k, schwelle, i)
        if r is not None:
            aus.append(r[0])
            frei_ab = r[1]
    return aus


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args(argv)
    DATEN.mkdir(parents=True, exist_ok=True)
    if args.regen:
        print(f"[REGEN] Trials unveraendert: {TrialCounter().total()}", flush=True)
    else:
        print(
            "Trials kumuliert: "
            + str(
                TrialCounter().increment(
                    6, label="N1 Komposit Gewichtsraster W1-W6 GDELT 2015-16"
                )
            ),
            flush=True,
        )
    teil = baue_teilscores()
    rng = np.random.default_rng(SEED)
    ergebnis: dict = {"registriert": __doc__, "configs": {}}
    for wname, (wg, wf, wt) in GEWICHTE.items():
        pool_s, pool_k = [], []
        je_instrument = {}
        for sym, symbole in INSTRUMENTE.items():
            px = lade_basket(symbole)
            mom = px.pct_change(20)
            ta = z_roll(mom).reindex(px.index)
            t_geo = teil["GEO"].reindex(px.index).ffill(limit=3)
            t_fin = teil["FIN"].reindex(px.index).ffill(limit=3)
            k = (wg * t_geo + wf * t_fin + wt * ta).rename("K")
            schwelle = k.rolling(252, min_periods=120).quantile(PERZENTIL).shift(1)
            px_w = px.loc[START:ENDE]
            k_w = k.reindex(px_w.index)
            s_w = schwelle.reindex(px_w.index)
            sig = [
                i
                for i in range(1, len(px_w))
                if np.isfinite(k_w.iloc[i - 1])
                and np.isfinite(s_w.iloc[i - 1])
                and k_w.iloc[i - 1] >= s_w.iloc[i - 1]
            ]
            zufall = sorted(
                rng.choice(
                    np.arange(1, len(px_w) - MAX_TAGE - 1),
                    size=KONTROLLEN,
                    replace=False,
                )
            )
            s_tr = lauf_k(px_w, k_w, s_w, sig)
            k_tr = lauf_k(px_w, k_w, s_w, [int(i) for i in zufall])
            je_instrument[sym] = {
                "n_signaltage": len(sig),
                "signal": statistik(s_tr),
                "kontrolle": statistik(k_tr),
            }
            pool_s += s_tr
            pool_k += k_tr
        ergebnis["configs"][wname] = {
            "gewichte_geo_fin_ta": [wg, wf, wt],
            "instrumente": je_instrument,
            "gepoolt_signal": statistik(pool_s),
            "gepoolt_kontrolle": statistik(pool_k),
        }
        print(
            f"{wname}: {ergebnis['configs'][wname]['gepoolt_signal']}",
            flush=True,
        )
    besteht = []
    for wname, cfg in ergebnis["configs"].items():
        g = cfg["gepoolt_signal"]
        kk = cfg["gepoolt_kontrolle"]
        ok = bool(
            g.get("n_trades", 0) > 2
            and (g.get("mittel_pp") or 0) > 0
            and (g.get("t") or 0) > 2.64
            and (g.get("profit_faktor") or 0) > 1
            and (g.get("mittel_pp") or 0) > (kk.get("mittel_pp") or 0)
        )
        if ok:
            besteht.append(wname)
    ergebnis["verdikt"] = {
        "bestehende_configs": besteht,
        "raster_besteht": bool(besteht),
        "kriterium": (
            "je Config: netto>0 & t>2.64 (Bonferroni, 6 Tests) & PF>1 & "
            "ueber Kontrolle; ALLE Configs berichtet (vorab fixiert)"
        ),
    }
    ZIEL.write_text(
        json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("VERDIKT:", ergebnis["verdikt"])
    print(f"-> {ZIEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
