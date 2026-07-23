"""H-077b — Whale/13F-Strang mit CUSIP-Mapping (Nachzügler Welle 39)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h011_kandidat_a import OUTD  # noqa: E402
from h077_mega_search import basket_returns, month_panel, report, screen_eval, spy_bench  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


def main() -> int:
    mclose = month_panel()
    bench = spy_bench(mclose, tax=0.26375)
    w = pd.read_parquet(DATA / "13f_top100.parquet")
    cmap = pd.read_parquet(DATA / "cusip_ticker_map.parquet")
    ccols = {x.lower(): x for x in cmap.columns}
    cu, tk = ccols.get("cusip"), (ccols.get("ticker") or ccols.get("symbol"))
    m = dict(zip(cmap[cu].astype(str).str[:8], cmap[tk]))
    w["symbol"] = w["CUSIP"].astype(str).str[:8].map(m)
    w["avail"] = pd.to_datetime(w["FILING_DATE"], utc=True, errors="coerce")
    w = w.dropna(subset=["symbol", "avail"])
    w = (
        w[w["PUTCALL"].isna() | (w["PUTCALL"].astype(str).str.strip() == "")]
        if "PUTCALL" in w.columns
        else w
    )
    print(
        f"[WHALE] {len(w)} Positionen, {w['symbol'].nunique()} Symbole, {w['CIK'].nunique()} Manager",
        flush=True,
    )

    res = {}
    months = list(mclose.index)
    for cons in (3, 5, 10, 15, 20):
        for hold in (3, 6, 12):
            for ex_mega in (False, True):
                for win in (3, 6):
                    sig = {}
                    for t in months:
                        r = w[
                            (w["avail"] <= t)
                            & (w["avail"] > t - pd.DateOffset(months=win))
                        ]
                        if not len(r):
                            continue
                        vc = r.groupby("symbol")["CIK"].nunique()
                        s = set(vc[vc >= cons].index)
                        if ex_mega:
                            s -= {
                                "AAPL",
                                "MSFT",
                                "NVDA",
                                "AMZN",
                                "GOOGL",
                                "GOOG",
                                "META",
                                "TSLA",
                                "BRK.B",
                            }
                        if s:
                            sig[t] = s
                    res[
                        f"WHL_c{cons}_w{win}_h{hold}_{'xmega' if ex_mega else 'all'}"
                    ] = screen_eval(basket_returns(mclose, sig, hold), 0.26375, bench)
    out = {}
    report("WHALE_13F", res, out)
    (OUTD / "h077b_whale.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
