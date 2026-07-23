"""H-079 — Stufe-2-Validierung des Options-Clusters unter ADVERSARIALEN Annahmen.

Die W39-Options-Survivors (CSP/CC) werden gegen die konservativsten Modell-Annahmen gestresst:
- CC: OTM-Call-IV = VIX × (1−25 %) (Skew maximal GEGEN den Verkäufer)
- CSP: Put-IV = VIX flat (KEIN Skew-Bonus — real sind Puts reicher, also konservativ)
- Prämien-Haircut 10 % (Bid/Ask — Verkäufer bekommt Geldseite)
- Kosten 3 bps/Monat auf Underlying-Notional
Überlebt der Cluster DAS, ist der Modell-Fall stark (Verdict weiter nur mit echten Daten).
"""

from __future__ import annotations

import json
import sys
from math import erf, exp, log, sqrt
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h011_kandidat_a import OUTD  # noqa: E402
from h077_mega_search import screen_eval  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
START = 100_000.0


def phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def bs(S, K, T, sig, put=False):
    if sig <= 0 or T <= 0:
        return max((K - S) if put else (S - K), 0.0)
    d1 = (log(S / K) + (0.02 + sig * sig / 2) * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    c = S * phi(d1) - K * exp(-0.02 * T) * phi(d2)
    return c - S + K * exp(-0.02 * T) if put else c


def main() -> int:
    w = pd.read_parquet(DATA / "prices_w28.parquet")
    px = w.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    px.index = pd.DatetimeIndex(px.index)
    oc = pd.read_parquet(DATA / "prices_overnight_oc.parquet")
    spy = oc[oc["symbol"] == "SPY"].set_index("date")["close"].sort_index()
    spy.index = pd.DatetimeIndex(spy.index)
    d = pd.DataFrame({"SPY": spy}).join(px[["VIX"]], how="inner").dropna()
    me = d.groupby(d.index.to_period("M")).tail(1)
    T = 21 / 252
    HAIRCUT = 0.90  # Verkäufer erhält 90 % der Modell-Prämie (Bid/Ask)

    g = START * float(me["SPY"].iloc[-1] / me["SPY"].iloc[0])
    bench = round(START + (g - START) * (1 - 0.1846))
    print(
        f"[BENCH] SPY B&H (Fenster {me.index[0].date()}–{me.index[-1].date()}): {bench:,}",
        flush=True,
    )

    res = {}
    for kind, otms in (("CC", (0.02, 0.03, 0.05)), ("CSP", (0.0, 0.02, 0.03))):
        for otm in otms:
            for frac in (0.5, 1.0):
                mrets = []
                for i in range(len(me) - 1):
                    S0, S1 = float(me["SPY"].iloc[i]), float(me["SPY"].iloc[i + 1])
                    iv = float(me["VIX"].iloc[i]) / 100
                    base = S1 / S0 - 1
                    if kind == "CC":
                        K = S0 * (1 + otm)
                        prem = (
                            bs(S0, K, T, iv * 0.75) * HAIRCUT
                        )  # skew −25 % GEGEN Verkäufer
                        pnl = frac * (prem - max(S1 - K, 0)) / S0
                        mrets.append(base + pnl - 3e-4)
                    else:
                        K = S0 * (1 - otm)
                        prem = bs(S0, K, T, iv, put=True) * HAIRCUT  # KEIN Skew-Bonus
                        pnl = (prem - max(K - S1, 0)) / S0
                        mrets.append(frac * pnl + (1 - frac) * base - 3e-4)
                mr = pd.Series(mrets, index=me.index[:-1])
                key = f"ADV_{kind}_otm{int(otm * 100)}_f{frac}"
                res[key] = screen_eval(mr, 0.26375, bench)
                print(
                    f"[H079] {key:22s} net={res[key].get('net', 0):>10,} "
                    f"oos={res[key].get('oos_sharpe', 0)} survives={res[key].get('survives')}",
                    flush=True,
                )
    n_surv = sum(1 for v in res.values() if v.get("survives"))
    print(f"\n[STAGE2] {n_surv}/{len(res)} überleben ADVERSARIALE Annahmen", flush=True)
    (OUTD / "h079_options_stage2.json").write_text(
        json.dumps(res, indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
