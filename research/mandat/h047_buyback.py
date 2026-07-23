"""H-047 — Net-Share-Issuance / Buyback-Anomalie (Pontiff-Woodgate / Daniel-Titman).

Long-only oberstes Terzil der Netto-Rueckkaeufer (groesster YoY-Rueckgang der verwaesserten
Aktienzahl, PIT aus XBRL) via EW-Band. Reuse verdict_engine + h040_h041-Helfer.
Signal low-turnover (Jahres-Share-Count) -> passt ins ueberlebende Muster.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import h011_kandidat_a as eng  # noqa: E402
from h011_kandidat_a import OUTD  # noqa: E402
from h040_h041_lowvol_quality import consistency, etf_and_spy, sub_membership  # noqa: E402
from verdict_engine import (  # noqa: E402
    load_div_panel,
    load_membership,
    load_verdict_prices,
)
from verdict_engine import run_verdict  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


def load_issuance_panel(
    close_index: pd.DatetimeIndex, symbols: list[str]
) -> pd.DataFrame:
    """PIT net-issuance score per (month-end, symbol): -(shares_FY_t/shares_FY_{t-1}-1).

    Higher score = share count fell (net buyback = buy). FY diluted-share values,
    visible only after available_at.
    """
    f = pd.read_parquet(DATA / "fundamentals_sp500.parquet")
    f = f[
        (f["symbol"].isin(symbols))
        & (f["tag"] == "WeightedAverageNumberOfDilutedSharesOutstanding")
    ].copy()
    f["available_at"] = pd.to_datetime(f["available_at"], utc=True)
    f["period_end"] = pd.to_datetime(f["period_end"], utc=True)
    f["period_start"] = pd.to_datetime(f["period_start"], utc=True)
    f["dur"] = (f["period_end"] - f["period_start"]).dt.days
    fy = f[(f["dur"] >= 330) & (f["dur"] <= 400) & (f["val"] > 0)]

    month_ends = close_index.to_series().groupby(close_index.to_period("M")).max()
    rows = []
    for as_of in month_ends:
        v = fy[fy["available_at"] <= as_of]
        v = (
            v.sort_values("available_at").groupby(["symbol", "period_end"]).tail(1)
        )  # latest restatement
        # two most recent distinct FY period_ends per symbol
        v = v.sort_values("period_end")
        last2 = v.groupby("symbol").tail(2)
        for sym, g in last2.groupby("symbol"):
            if len(g) < 2:
                continue
            prior, latest = g["val"].iloc[0], g["val"].iloc[1]
            if prior > 0 and latest > 0:
                rows.append(
                    (as_of, sym, -(latest / prior - 1.0))
                )  # buyback -> positive
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "score"]).pivot(
        index="timestamp", columns="symbol", values="score"
    )


def main() -> int:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    eng.COST_BPS = 10.0
    close = load_verdict_prices()
    membership = load_membership(close.index)
    div = load_div_panel(close.index)
    universe = [c for c in close.columns if c != "SPY"]
    iss = load_issuance_panel(close.index, universe)
    cov = int(iss.notna().sum(axis=1).median())
    print(
        f"[DATA] {len(membership)} rebalances; issuance coverage median {cov} symbols/month",
        flush=True,
    )

    res_base, _, ret_base = run_verdict(
        close, membership, label="EWband_fullSP", mode="ew", ew_band=0.5, div_panel=div
    )
    print(
        f"[BASE] EWband_fullSP final={res_base['final_value']:.0f} sharpe={res_base['sharpe_net']:.3f}",
        flush=True,
    )

    results = {"base": res_base}
    rets = {}
    for pk in (0.33, 0.50):
        mem = sub_membership(
            membership, iss, pick=pk, low_is_good=False
        )  # high score = buyback = buy
        lab = f"H047_buyback_{int(pk * 100)}"
        res, _, ret = run_verdict(
            close, mem, label=lab, mode="ew", ew_band=0.5, div_panel=div
        )
        results[lab] = res
        rets[lab] = ret
        print(
            f"[RUN] {lab}: final={res['final_value']:.0f} sharpe={res['sharpe_net']:.3f} "
            f"maxdd={res['maxdd_net']:.3f}",
            flush=True,
        )

    v_lab = "H047_buyback_33"
    v, vr = results[v_lab], rets[v_lab]
    bench = etf_and_spy(close, vr.index)
    dsr = deflated_sharpe(vr, n_trials=128)
    c = consistency(vr, ret_base)

    crit = {
        "sharpe_gt_base_x105": bool(v["sharpe_net"] > res_base["sharpe_net"] * 1.05),
        "final_gt_base_x105": bool(v["final_value"] > res_base["final_value"] * 1.05),
        "final_gt_ETF": bool(v["final_value"] > bench["ETF_net_final"]),
        "DSR_passes": bool(dsr.passes_5pct),
    }
    out = {
        "base_final": res_base["final_value"],
        "base_sharpe": res_base["sharpe_net"],
        "v_final": v["final_value"],
        "v_sharpe": v["sharpe_net"],
        "v_maxdd": v["maxdd_net"],
        "ETF_net": bench["ETF_net_final"],
        "SPY_sharpe": bench["SPY_sharpe"],
        "DSR_p": float(dsr.deflated_sharpe_probability),
        "consistency": c,
        "criteria": crit,
        "PASS": bool(all(crit.values())),
    }
    (OUTD / "h047_results.json").write_text(
        json.dumps(
            {**out, "all": {k: results[k] for k in results}}, indent=2, default=str
        ),
        encoding="utf-8",
    )
    print("[VERDICT]", json.dumps(out, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
