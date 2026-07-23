"""H-080 — Rest-Dimensionen (Welle 40): Insider-Verkäufe als Filter, Event×Technik, EU-Momentum."""

from __future__ import annotations

import glob
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


def load_form4(code: str):
    frames = [
        pd.read_parquet(f)
        for f in glob.glob(str(DATA / "form4_broad" / "tranche_*.parquet"))
    ]
    frames.append(
        pd.read_parquet(
            ROOT / "data" / "raw" / "insider_congress" / "form4_insider_full.parquet"
        )
    )
    f = pd.concat(frames, ignore_index=True)
    f = f[(f["transaction_code"] == code) & (~f["is_derivative"].astype(bool))]
    f["available_at"] = pd.to_datetime(f["available_at"], utc=True, errors="coerce")
    return f.dropna(subset=["available_at", "symbol"])


def main() -> int:
    mclose = month_panel()
    bench = spy_bench(mclose, tax=0.26375)
    months = list(mclose.index)
    out: dict = {}

    # (a) Insider-SALES als Negativ-Filter: EW über alle handelbaren Titel MINUS Verkaufs-Cluster
    sells = load_form4("S")
    res_a = {}
    tradable_all = {
        t: set(mclose.columns[mclose.loc[t].notna()]) - {"SPY"} for t in months
    }
    for win in (1, 3):
        for mins in (2, 3, 5):
            for hold in (1, 3):
                sig = {}
                for t in months:
                    r = sells[
                        (sells["available_at"] <= t)
                        & (sells["available_at"] > t - pd.DateOffset(months=win))
                    ]
                    vc = r.groupby("symbol")["reporting_owner_cik"].nunique()
                    avoid = set(vc[vc >= mins].index)
                    keep = tradable_all[t] - avoid
                    if keep:
                        sig[t] = keep
                res_a[f"AVOID_S_w{win}_n{mins}_h{hold}"] = screen_eval(
                    basket_returns(mclose, sig, hold), 0.26375, bench
                )
    report("INSIDER_SELL_FILTER", res_a, out)

    # (b) Event×Technik: Insider-Kauf UND Titel über eigener SMA200 (Monatsproxy: > 10M-Schnitt)
    buys = load_form4("P")
    sma10m = mclose.rolling(10).mean()
    res_b = {}
    for win in (1, 2, 3):
        for mins in (1, 2):
            for hold in (3, 6):
                sig = {}
                for t in months:
                    r = buys[
                        (buys["available_at"] <= t)
                        & (buys["available_at"] > t - pd.DateOffset(months=win))
                    ]
                    vc = r.groupby("symbol")["reporting_owner_cik"].nunique()
                    cand = set(vc[vc >= mins].index)
                    if t in sma10m.index:
                        above = set(
                            mclose.columns[
                                (mclose.loc[t] > sma10m.loc[t]).fillna(False)
                            ]
                        )
                        cand &= above
                    if cand:
                        sig[t] = cand
                res_b[f"INSxTA_w{win}_n{mins}_h{hold}"] = screen_eval(
                    basket_returns(mclose, sig, hold), 0.26375, bench
                )
    report("EVENT_x_TECHNIK", res_b, out)

    # (c) EU-Querschnitts-Momentum: Top-N nach 12-1 aus 20 EU-Blue-Chips
    eu = pd.read_parquet(DATA / "prices_eu_stocks.parquet")
    epx = eu.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    em = epx.groupby(epx.index.to_period("M")).last()
    em.index = em.index.to_timestamp(how="end").tz_localize("UTC")
    mom = em.shift(1) / em.shift(12) - 1.0
    fwd = em.pct_change().shift(-1)
    res_c = {}
    g_eu = 100_000 * float((em.mean(axis=1).iloc[-1] / em.mean(axis=1).iloc[13]))
    bench_eu = round(100_000 + (g_eu - 100_000) * (1 - 0.26375))
    for topn in (3, 5, 8):
        for skip_neg in (False, True):
            rets = {}
            for i in range(13, len(em) - 1):
                t = em.index[i]
                m = mom.iloc[i].dropna()
                if skip_neg:
                    m = m[m > 0]
                sel = m.nlargest(topn).index
                if len(sel):
                    rets[t] = float(fwd.loc[t, sel].mean()) - 10e-4
                else:
                    rets[t] = 0.0
            res_c[f"EUMOM_top{topn}_{'posonly' if skip_neg else 'all'}"] = screen_eval(
                pd.Series(rets), 0.26375, bench_eu
            )
    print(f"[EU] Bench EW-EU-B&H: {bench_eu:,}", flush=True)
    report("EU_MOMENTUM", res_c, out)

    n = sum(v["n"] for v in out.values() if isinstance(v, dict) and "n" in v)
    out["_total"] = {"configs": n, "N_cumulative": 1934 + n}
    (OUTD / "h080_results.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n[TOTAL] {n} Configs, N={1934 + n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
