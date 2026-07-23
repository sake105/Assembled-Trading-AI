"""H-037 — Krypto §23-EStG-Steuer-Keil (Registry Welle 12). Spot only, kein Trade.

§23: privates Veraeusserungsgeschaeft — Haltefrist >= 365 T => 0 % Steuer;
< 365 T => persoenlicher ESt-Satz (Annahme Spitzensatz+Soli 44 %). FIFO.
Vergleich: HODL (nie verkaufen => steuerfrei) vs. aktiv (monatliches SMA200-Gate,
Switches realisieren kurzfristige Gewinne). N->106.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from h011_kandidat_a import OUTD  # noqa: E402

TOK = os.environ["EODHD_API_TOKEN"]
TAX_SHORT = 0.44
COST = 0.002
START = 100000.0
DATA = Path(__file__).resolve().parent / "data"


def load_crypto(sym: str) -> pd.Series:
    p = DATA / f"crypto_{sym.replace('-', '').replace('.', '')}.parquet"
    if p.exists():
        return pd.read_parquet(p)["close"]
    url = f"https://eodhd.com/api/eod/{sym}?api_token={TOK}&fmt=json&from=2010-01-01"
    rows = json.loads(
        urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "research"}), timeout=60
        )
        .read()
        .decode()
    )
    df = pd.DataFrame(rows)[["date", "adjusted_close"]]
    df.columns = ["date", "close"]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    s = df.set_index("date")["close"].sort_index()
    s.to_frame("close").to_parquet(p)
    return s


class CryptoPF:
    def __init__(self, cash):
        self.cash = cash
        self.lots: list[list] = []  # [qty, px, date]
        self.tax_paid = 0.0
        self.loss_pot = 0.0

    def qty(self):
        return sum(q for q, _, _ in self.lots)

    def buy(self, notional, px, date):
        spend = min(notional, self.cash)
        if spend <= 0:
            return
        cost = spend * COST
        q = (spend - cost) / px
        self.cash -= spend
        self.lots.append([q, px, date])

    def sell_all(self, px, date, *, taxed=True):
        q_tot = self.qty()
        if q_tot <= 0:
            return
        proceeds = q_tot * px
        cost = proceeds * COST
        gain_short = 0.0
        for q, lpx, ld in self.lots:
            g = q * (px - lpx)
            if taxed and (date - ld).days < 365:
                gain_short += g
        gain_short -= cost
        tax = 0.0
        if taxed:
            if gain_short >= 0:
                offset = min(gain_short, self.loss_pot)
                self.loss_pot -= offset
                tax = (gain_short - offset) * TAX_SHORT
            else:
                self.loss_pot += -gain_short
        self.cash += proceeds - cost - tax
        self.tax_paid += tax
        self.lots = []

    def value(self, px):
        return self.cash + self.qty() * px


def run(close: pd.Series, *, active: bool, taxed: bool):
    sma = close.rolling(200).mean()
    month_ends = set(
        pd.Series(close.index, index=close.index)
        .groupby(close.index.to_period("M"))
        .max()
    )
    pf = CryptoPF(START)
    invested = False
    eq = []
    for t in close.index:
        px = float(close.at[t])
        eq.append((t, pf.value(px)))
        if t not in month_ends or not np.isfinite(sma.at[t]):
            continue
        if active:
            want = px >= sma.at[t]
        else:
            want = True  # HODL
        if want and not invested:
            pf.buy(pf.cash, px, t)
            invested = True
        elif not want and invested:
            pf.sell_all(px, t, taxed=taxed)
            invested = False
    e = pd.Series(dict(eq)).sort_index()
    e = e[sma.notna()]
    ret = e.pct_change().dropna()
    years = (e.index[-1] - e.index[0]).days / 365.25
    return {
        "final_value": float(e.iloc[-1] / e.iloc[0] * START),
        "cagr": float((e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1),
        "sharpe": float(ret.mean() / ret.std() * np.sqrt(365)),
        "maxdd": float((e / e.cummax() - 1).min()),
        "tax_paid": float(pf.tax_paid),
        "years": float(years),
    }


def main() -> int:
    results = {}
    for sym in ("BTC-USD.CC", "ETH-USD.CC"):
        close = load_crypto(sym)
        hodl = run(close, active=False, taxed=True)
        act_net = run(close, active=True, taxed=True)
        act_gross = run(close, active=True, taxed=False)
        results[sym] = {"HODL": hodl, "active_net": act_net, "active_gross": act_gross}
        print(
            f"[RUN] {sym}: HODL {hodl['final_value']:.0f} | active_net {act_net['final_value']:.0f} (tax {act_net['tax_paid']:.0f}) | active_gross {act_gross['final_value']:.0f}",
            flush=True,
        )

    verd = {}
    for sym, r in results.items():
        c1 = r["HODL"]["final_value"] > r["active_net"]["final_value"] * 1.20
        c2 = r["active_net"]["final_value"] < r["active_gross"]["final_value"] * 0.85
        verd[sym] = {
            "hodl_gt_active_x120": c1,
            "tax_wedge_ge_15pct": c2,
            "keil_pct": round(
                100
                * (
                    1
                    - r["active_net"]["final_value"] / r["active_gross"]["final_value"]
                ),
                1,
            ),
        }
    results["_verdict"] = {
        "per_asset": verd,
        "PASS": all(
            v["hodl_gt_active_x120"] and v["tax_wedge_ge_15pct"] for v in verd.values()
        ),
    }
    (OUTD / "h037_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print("[VERDICT]", json.dumps(results["_verdict"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
