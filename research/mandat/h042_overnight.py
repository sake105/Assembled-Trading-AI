"""H-042 — Overnight- vs Intraday-Return-Dekomposition + deployable Overnight-Buch.

Test 1 (Dekomposition, KEIN Trial): mittlere Overnight- (close->open) vs Intraday-
(open->close) Rendite für SPY + SPDR-Sektoren. Bekannter Befund (Cooper/Cliff/Gulen).
Test 2 (Trial): deployable Overnight-Buch (kaufe Close, verkaufe Open) nach Kosten +
deutscher Steuer (Jahres-Netting, Verlusttopf) vs ETF-Netto-Pfad.

Zieht eigene Mini-Daten (12 Symbole, RAW open/close). Steuer-Prior hart: 252 Round-
trips/J -> jeder Gewinn <1J -> voller 26,375%-Keil.
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

from h011_kandidat_a import OUTD, START_CAPITAL  # noqa: E402

load_dotenv(ROOT / ".env")
TOK = os.environ["EODHD_API_TOKEN"]
DATA = Path(__file__).resolve().parent / "data"
CACHE = DATA / "prices_overnight_oc.parquet"
ETF_TAX = 0.185
TAX = 0.26375
SYMS = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]


def pull() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    frames = []
    for s in SYMS:
        url = (
            f"https://eodhd.com/api/eod/{s}.US?api_token={TOK}&fmt=json&from=2000-01-01"
        )
        rows = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "research"}),
                timeout=60,
            )
            .read()
            .decode()
        )
        df = pd.DataFrame(rows)[["date", "open", "close"]]
        df["symbol"] = s
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], utc=True)
    out.to_parquet(CACHE, index=False)
    return out


def overnight_book(oc: pd.DataFrame, sym: str, *, cost_bps: float = 3.0) -> dict:
    """Buy at close, sell at next open. Annual-netting German tax (Verlusttopf)."""
    d = oc[oc["symbol"] == sym].set_index("date").sort_index()
    on = (d["open"] / d["close"].shift(1) - 1.0).dropna()  # overnight return
    cap = START_CAPITAL
    year_pnl: dict[int, float] = {}
    eq = []
    for t, r in on.items():
        gross = cap * r
        cost = cap * cost_bps / 1e4 * 2.0  # round trip (buy close + sell open)
        net = gross - cost
        year_pnl[t.year] = year_pnl.get(t.year, 0.0) + net
        cap += net
        eq.append((t, cap))
    # apply annual tax on positive net annual P&L (loss years carry nothing forward here; conservative)
    tax_total = sum(max(p, 0.0) * TAX for p in year_pnl.values())
    e = pd.Series(dict(eq))
    r = e.pct_change().dropna()
    years = (e.index[-1] - e.index[0]).days / 365.25
    return {
        "final_pretax": float(e.iloc[-1]),
        "final_net": float(e.iloc[-1] - tax_total),
        "tax_total": float(tax_total),
        "sharpe_pretax": float(r.mean() / r.std() * np.sqrt(252))
        if r.std() > 0
        else float("nan"),
        "years": float(years),
    }


def main() -> int:
    oc = pull()
    # Test 1: decomposition
    decomp = {}
    for s in SYMS:
        d = oc[oc["symbol"] == s].set_index("date").sort_index()
        on = (d["open"] / d["close"].shift(1) - 1.0).dropna()
        intra = (d["close"] / d["open"] - 1.0).dropna()
        decomp[s] = {
            "overnight_mean_bps": round(float(on.mean()) * 1e4, 3),
            "intraday_mean_bps": round(float(intra.mean()) * 1e4, 3),
            "n": int(len(on)),
        }
    print("[DECOMP]", json.dumps(decomp, indent=2), flush=True)

    # Test 2: deployable overnight book on SPY + EW-sectors
    spy_book = overnight_book(oc, "SPY")
    # ETF-net path SPY over same span (buy&hold total return via close)
    d = oc[oc["symbol"] == "SPY"].set_index("date").sort_index()
    gross_gain = START_CAPITAL * (d["close"].iloc[-1] / d["close"].iloc[0] - 1)
    etf_net = START_CAPITAL + gross_gain * (1 - ETF_TAX)
    spy_r = d["close"].pct_change().dropna()

    verdict = {
        "decomposition": decomp,
        "SPY_overnight_book": spy_book,
        "ETF_net_path": round(float(etf_net)),
        "SPY_buyhold_sharpe": round(
            float(spy_r.mean() / spy_r.std() * np.sqrt(252)), 3
        ),
        "PASS": bool(
            spy_book["final_net"] > etf_net
            and spy_book["sharpe_pretax"]
            > float(spy_r.mean() / spy_r.std() * np.sqrt(252))
        ),
    }
    (OUTD / "h042_results.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8"
    )
    print(
        "[VERDICT]",
        json.dumps(
            {k: v for k, v in verdict.items() if k != "decomposition"},
            indent=2,
            default=str,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
