"""H-083 — Einheitliche OOS-Re-Evaluation aller Strategien (Welle 43, keine neuen Trials).

Teil A: Ernte aller gespeicherten oos_sharpe-Metriken aus results/*.json (familienweise).
Teil B: Einheitliches Rezenz-Holdout 2021-07-01→2026-07 für ~20 kanonische Strategien —
        CAGR/Sharpe (brutto inkl. Kosten) vs SPY im selben Fenster. PIT-sauber (Signale nutzen
        Historie, Bewertung nur im Holdout; IV-Gewichte aus Prä-Holdout-Daten).
"""

from __future__ import annotations

import glob
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from h011_kandidat_a import OUTD  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
H0 = pd.Timestamp("2021-07-01")


# ---------------------------------------------------------------- Teil A
def harvest() -> dict:
    fam = {}
    for f in glob.glob(str(OUTD / "*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        name = Path(f).stem

        def walk(obj):
            if isinstance(obj, dict):
                if "oos_sharpe" in obj and isinstance(
                    obj.get("oos_sharpe"), (int, float)
                ):
                    yield obj
                else:
                    for v in obj.values():
                        yield from walk(v)

        vals = [o for o in walk(d)]
        if vals:
            oos = [o["oos_sharpe"] for o in vals]
            surv = [o for o in vals if o.get("survives") or o.get("beats")]
            fam[name] = {
                "n": len(vals),
                "oos_median": round(float(np.median(oos)), 3),
                "oos_max": round(float(max(oos)), 3),
                "n_flagged": len(surv),
            }
    return fam


# ---------------------------------------------------------------- Teil B
def load_series() -> dict:
    S = {}
    oc = pd.read_parquet(DATA / "prices_overnight_oc.parquet")
    for sym in ("SPY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"):
        s = oc[oc["symbol"] == sym].set_index("date")["close"].sort_index()
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        S[sym] = s
    pc = pd.read_parquet(DATA / "prices_crisis.parquet")
    g = pc[pc["symbol"] == "GLD"].set_index("timestamp")["close"].sort_index()
    S["GLD"] = pd.Series(g.values, index=pd.DatetimeIndex(g.index).tz_localize(None))
    for name, f in (("TLT", "bond_TLT.parquet"),):
        s = pd.read_parquet(DATA / f).set_index("timestamp")["close"]
        S[name] = pd.Series(s.values, index=pd.DatetimeIndex(s.index).tz_localize(None))
    for name, f in (
        ("BTC", "crypto_BTCUSDCC.parquet"),
        ("ETH", "crypto_ETHUSDCC.parquet"),
    ):
        s = pd.read_parquet(DATA / f)["close"]
        S[name] = pd.Series(s.values, index=pd.DatetimeIndex(s.index).tz_localize(None))
    cb = pd.read_parquet(DATA / "prices_cboe_buywrite.parquet")
    w = cb.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    w.index = pd.DatetimeIndex(w.index).tz_localize(None)
    for c in ("BXM", "BXMD", "PUT", "PPUT", "CNDR", "SP500TR"):
        S[c] = w[c].dropna()
    return S


def ev(ret: pd.Series, label: str, spy_hold: pd.Series) -> dict:
    r = ret[ret.index >= H0].dropna()
    if len(r) < 200:
        return {"label": label, "skip": "zu wenig Holdout-Daten"}
    e = (1 + r).cumprod()
    years = (r.index[-1] - r.index[0]).days / 365.25
    sh = float(r.mean() / r.std() * sqrt(252)) if r.std() > 0 else 0.0
    cagr = (float(e.iloc[-1]) ** (1 / years) - 1) * 100
    ssh = float(spy_hold.mean() / spy_hold.std() * sqrt(252))
    scagr = (float((1 + spy_hold).prod()) ** (1 / years) - 1) * 100
    return {
        "label": label,
        "cagr_pct": round(cagr, 2),
        "sharpe": round(sh, 3),
        "maxdd": round(float((e / e.cummax() - 1).min()), 3),
        "beats_SPY": bool(cagr > scagr and sh > ssh),
    }


def main() -> int:
    out = {"holdout": str(H0.date()) + " -> 2026-07", "A_harvest": harvest()}
    print("[A] Familien mit gespeicherten OOS-Metriken:", flush=True)
    for k, v in sorted(out["A_harvest"].items()):
        print(
            f"  {k:28s} n={v['n']:>4} oos_med={v['oos_median']:>7} oos_max={v['oos_max']:>7} flagged={v['n_flagged']}",
            flush=True,
        )

    S = load_series()
    spy_r = S["SPY"].pct_change().dropna()
    spy_hold = spy_r[spy_r.index >= H0]
    rows = []

    def add(name, ret):
        rows.append(ev(ret, name, spy_hold))

    add("SPY B&H", spy_r)
    add("GLD HODL (§23)", S["GLD"].pct_change().dropna())
    add("BTC HODL (§23)", S["BTC"].pct_change().dropna())
    add("ETH HODL (§23)", S["ETH"].pct_change().dropna())

    def sig_ret(s, pos, cost=5e-4):
        r = s.pct_change()
        p = pos.shift(1).fillna(0)
        return (p * r - p.diff().abs().fillna(0) * cost).dropna()

    spy = S["SPY"]
    add("SPY SMA200 L/C", sig_ret(spy, (spy > spy.rolling(200).mean()).astype(float)))
    add("SPY TSmom252 L/C", sig_ret(spy, (spy > spy.shift(252)).astype(float)))
    s200 = (spy.rolling(50).mean() > spy.rolling(200).mean()).astype(float)
    m12 = (spy > spy.shift(252)).astype(float)
    add("SPY OR(SMA200,Mom12)", sig_ret(spy, ((s200 > 0) | (m12 > 0)).astype(float)))
    btc = S["BTC"]
    add(
        "BTC TSmom252 L/C",
        sig_ret(btc, (btc > btc.shift(252)).astype(float), cost=10e-4),
    )

    def blend(weights):
        rr = sum(w * S[a].pct_change() for a, w in weights.items())
        return rr.dropna()

    add("70/30 SPY/GLD", blend({"SPY": 0.7, "GLD": 0.3}))
    add(
        "65/25/5/5 +BTC/ETH",
        blend({"SPY": 0.65, "GLD": 0.25, "BTC": 0.05, "ETH": 0.05}),
    )
    add("60/40 SPY/TLT", blend({"SPY": 0.6, "TLT": 0.4}))
    # PIT-IV-Gewichte (nur Prä-Holdout-Daten)
    pre = pd.concat(
        {a: S[a].pct_change() for a in ("SPY", "GLD", "TLT")}, axis=1
    ).dropna()
    pre = pre[pre.index < H0]
    iv = 1.0 / pre.std()
    ivw = dict((iv / iv.sum()).round(4))
    add(f"RP-IV3 (PIT: {ivw})", blend(ivw))

    for c in ("BXMD", "PUT", "BXM", "PPUT", "CNDR"):
        add(f"CBOE {c}", S[c].pct_change().dropna())

    # Kalender: TOM 4+3 / Halloween
    idx = spy.index
    months = idx.to_period("M")
    pos = np.zeros(len(idx))
    arr = np.arange(len(idx))
    for m in months.unique():
        ii = arr[months == m]
        pos[ii[-4:]] = 1.0
        nxt = arr[months > m]
        if len(nxt):
            pos[nxt[:3]] = 1.0
    add("TOM 4+3", sig_ret(spy, pd.Series(pos, index=idx)))
    hall = pd.Series(
        [1.0 if m in (11, 12, 1, 2, 3, 4) else 0.0 for m in idx.month], index=idx
    )
    add("Halloween Nov-Apr", sig_ret(spy, hall))

    # Sektor-Rotation top1 (monatlich, brutto inkl. 10 bps Wechsel)
    sec = pd.concat(
        {
            s: S[s]
            for s in ("XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB")
        },
        axis=1,
    ).dropna()
    me = sec.groupby(sec.index.to_period("M")).last()
    mom = me.shift(1) / me.shift(12) - 1.0
    fwd = me.pct_change().shift(-1)
    rot = {}
    prev_pick = None
    for i in range(12, len(me) - 1):
        pick = mom.iloc[i].idxmax()
        r = float(fwd.iloc[i][pick])
        if prev_pick is not None and pick != prev_pick:
            r -= 10e-4
        prev_pick = pick
        rot[me.index[i].to_timestamp(how="end")] = r
    rot = pd.Series(rot)
    rot.index = pd.DatetimeIndex(rot.index)
    rh = rot[rot.index >= H0]
    e = (1 + rh).cumprod()
    years = (rh.index[-1] - rh.index[0]).days / 365.25
    rows.append(
        {
            "label": "SektorRot top1 (monatl.)",
            "cagr_pct": round((float(e.iloc[-1]) ** (1 / years) - 1) * 100, 2),
            "sharpe": round(float(rh.mean() / rh.std() * sqrt(12)), 3),
            "maxdd": round(float((e / e.cummax() - 1).min()), 3),
            "beats_SPY": None,
        }
    )

    out["B_holdout"] = rows
    spy_row = rows[0]
    print(
        f"\n[B] Holdout {H0.date()}→2026-07 | SPY: CAGR {spy_row['cagr_pct']} % Sharpe {spy_row['sharpe']} DD {spy_row['maxdd']}",
        flush=True,
    )
    for r in rows[1:]:
        if "skip" in r:
            print(f"  {r['label']:28s} SKIP ({r['skip']})", flush=True)
        else:
            b = "✓" if r.get("beats_SPY") else " "
            print(
                f"  {r['label']:28s} cagr={r['cagr_pct']:>7} sh={r['sharpe']:>6} dd={r['maxdd']:>6} beatsSPY={b}",
                flush=True,
            )
    n_beat = sum(1 for r in rows[1:] if r.get("beats_SPY"))
    out["B_n_beat_SPY"] = n_beat
    print(
        f"\n[B] Strategien > SPY (CAGR UND Sharpe) im Holdout: {n_beat}/{len(rows) - 1}",
        flush=True,
    )
    (OUTD / "h083_unified_oos.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
