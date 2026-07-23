"""H-073 — WELT-Indikator-Sweep (Welle 35): Batterie über ~38 Assets weltweit.

Regionen-ETFs (18,46 %), Sektor-ETFs (18,46 %), TLT (26,375 % Bond), SLV/ETH (§23),
20 EU-Einzelaktien (26,375 %). 25 Configs je Asset (h071-Signale). Familien-Auswertung:
je Asset-Klasse Anteil der Configs > B&H; globale Top-Tabelle; DSR des Besten bei neuem N.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h011_kandidat_a import OUTD  # noqa: E402
from h071_indicator_lab import backtest, bh_ref, build_signals  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


def load_series():
    out = {}  # name -> (series, tax_kind, klass)
    w = pd.read_parquet(DATA / "prices_world_etf.parquet")
    for s in ("URTH", "ACWI", "EFA", "VGK", "EWJ", "EEM"):
        ser = w[w["symbol"] == s].set_index("timestamp")["close"].sort_index()
        ser.index = pd.DatetimeIndex(ser.index)
        out[s] = (ser, "etf", "region")
    oc = pd.read_parquet(DATA / "prices_overnight_oc.parquet")
    for s in ("XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"):
        ser = oc[oc["symbol"] == s].set_index("date")["close"].sort_index()
        ser.index = pd.DatetimeIndex(ser.index)
        out[s] = (ser, "etf", "sector")
    tlt = (
        pd.read_parquet(DATA / "bond_TLT.parquet")
        .set_index("timestamp")["close"]
        .sort_index()
    )
    tlt.index = pd.DatetimeIndex(tlt.index)
    out["TLT"] = (tlt, "stock26", "bond")
    slv = (
        pd.read_parquet(DATA / "bond_SLV.parquet")
        .set_index("timestamp")["close"]
        .sort_index()
    )
    slv.index = pd.DatetimeIndex(slv.index)
    out["SLV"] = (slv, "p23", "metal")
    eth = pd.read_parquet(DATA / "crypto_ETHUSDCC.parquet")["close"]
    eth.index = pd.DatetimeIndex(eth.index)
    out["ETH"] = (
        eth[eth.index >= pd.Timestamp("2017-01-01", tz="UTC")],
        "p23",
        "crypto",
    )
    eu = pd.read_parquet(DATA / "prices_eu_stocks.parquet")
    for s in sorted(eu["symbol"].unique()):
        ser = eu[eu["symbol"] == s].set_index("timestamp")["close"].sort_index()
        ser.index = pd.DatetimeIndex(ser.index)
        out[s] = (ser, "stock26", "eu_stock")
    return out


def main() -> int:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    assets = load_series()
    results, rets, klass_stats = {}, {}, {}
    n_total = 0
    for name, (ser, kind, klass) in assets.items():
        ser = ser.dropna()
        if len(ser) < 1000:
            continue
        eff_kind = kind if kind != "stock26" else "etf"  # rate handled below
        ref = bh_ref(ser, "etf" if kind in ("etf", "stock26") else kind)
        if kind == "stock26":  # correct B&H net for 26.375 %
            g = 100_000 * (ser.iloc[-1] / ser.iloc[0])
            ref["net"] = round(100_000 + (g - 100_000) * (1 - 0.26375))
        sigs = build_signals(ser)
        n_beat = 0
        best = None
        for sname, pos in sigs.items():
            res, r = backtest(ser, pos, tax_kind="etf" if kind == "stock26" else kind)
            if (
                kind == "stock26" and res["tax"] > 0
            ):  # scale tax from 18.46 to 26.375 (approx, conservative pro-strategy skipped)
                extra = res["tax"] * (0.26375 / 0.1846 - 1)
                res["net"] = round(res["net"] - extra)
                res["tax"] = round(res["tax"] + extra)
            key = f"{name}:{sname}"
            results[key] = res
            beats = res["net"] > ref["net"] and res["oos_sharpe"] > ref["oos_sharpe"]
            if beats:
                n_beat += 1
                rets[key] = r
            if best is None or res["net"] > best[1]:
                best = (sname, res["net"])
            n_total += 1
        ks = klass_stats.setdefault(klass, {"assets": 0, "configs": 0, "beat": 0})
        ks["assets"] += 1
        ks["configs"] += len(sigs)
        ks["beat"] += n_beat
        print(
            f"[{klass:8s}] {name:10s} B&H={ref['net']:>10,} | beat B&H: {n_beat:2d}/25 | best {best[0]} {best[1]:,}",
            flush=True,
        )

    print("\n[KLASSEN]", json.dumps(klass_stats, indent=2), flush=True)
    N_NEW = 238 + 7 + n_total  # nach W33(238) + W34(7)
    summary = {"n_configs": n_total, "N_cumulative": N_NEW, "by_class": klass_stats}
    if rets:
        best_key = max(rets, key=lambda k: results[k]["net"])
        dsr = deflated_sharpe(rets[best_key], n_trials=N_NEW)
        summary["best_beating"] = {
            best_key: results[best_key],
            "DSR_prob": round(float(dsr.deflated_sharpe_probability), 3),
            "DSR_pass": bool(dsr.passes_5pct),
        }
    results["_summary"] = summary
    (OUTD / "h073_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print("\n[SUMMARY]", json.dumps(summary, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
