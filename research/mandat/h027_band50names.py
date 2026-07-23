"""H-027 — Band-Rebalancing auf 50-Namen-Subset (Registry Welle 5, VOR Lauf).

Top 50 nach trailing 252d-Dollar-Volumen, jaehrlich im Januar bestimmt
(Liquiditaets-, kein Alpha-Kriterium), geschnitten mit PIT-S&P-Membership.
3 Laeufe: EW50 monatlich (Referenz) | Band 25 % | Band 50 %. N->84.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h011_kandidat_a import OUTD, cscv_pbo  # noqa: E402
from verdict_engine import DATA, load_membership, load_verdict_prices, run_verdict  # noqa: E402

N_TRIALS = 84


def build_top50_membership(close: pd.DataFrame, membership: pd.Series) -> pd.Series:
    raw = pd.read_parquet(DATA / "prices_verdict.parquet")
    vol = raw.pivot(index="timestamp", columns="symbol", values="volume").sort_index()
    vol = vol.reindex(close.index)
    dv = (close * vol).rolling(252, min_periods=200).mean()

    out = {}
    current: frozenset | None = None
    for me in membership.index:
        if me.month == 1 or current is None:
            members = [
                s
                for s in membership.loc[me]
                if s in dv.columns and np.isfinite(dv.at[me, s])
            ]
            ranked = sorted(members, key=lambda s: -dv.at[me, s])
            current = frozenset(ranked[:50])
        # intersect with live PIT membership each month
        out[me] = frozenset(current & membership.loc[me]) or current
    return pd.Series(out)


def main() -> int:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    close = load_verdict_prices()
    membership = load_membership(close.index)
    m50 = build_top50_membership(close, membership)
    sizes = pd.Series({k: len(v) for k, v in m50.items()})
    print(
        f"[DATA] top50 membership: median {sizes.median():.0f} names/month", flush=True
    )

    results, rets = {}, {}
    for name, band in (
        ("H027_ew50_monthly", None),
        ("H027_band25", 0.25),
        ("H027_band50", 0.50),
    ):
        res, _eq, ret = run_verdict(close, m50, label=name, mode="ew", ew_band=band)
        results[name] = res
        rets[name] = ret
        print(
            f"[RUN] {name}: final={res['final_value']:.0f} sharpe={res['sharpe_net']:.3f} maxdd={res['maxdd_net'] * 100:.1f}% tax={res['tax_paid']:.0f} costs={res['costs_paid']:.0f}",
            flush=True,
        )

    ref = results["H027_ew50_monthly"]["final_value"]
    ref_sharpe = results["H027_ew50_monthly"]["sharpe_net"]
    crit_final = all(
        results[k]["final_value"] > ref for k in ("H027_band25", "H027_band50")
    )
    crit_sharpe = all(
        results[k]["sharpe_net"] >= ref_sharpe - 0.05
        for k in ("H027_band25", "H027_band50")
    )
    pbo = float(
        cscv_pbo(pd.DataFrame({k: rets[k] for k in ("H027_band25", "H027_band50")}))
    )
    best = max(("H027_band25", "H027_band50"), key=lambda k: results[k]["final_value"])
    dsr = deflated_sharpe(rets[best], n_trials=N_TRIALS)
    results["_verdict"] = {
        "crit_final_both_gt_ref": crit_final,
        "crit_sharpe_within_005": crit_sharpe,
        "PBO": pbo,
        "crit_pbo": pbo <= 0.5,
        "selected": best,
        "DSR_prob_info": float(dsr.deflated_sharpe_probability),
        "PASS": all([crit_final, crit_sharpe, pbo <= 0.5]),
    }
    out = OUTD / "h027_results.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(
        "[VERDICT]", json.dumps(results["_verdict"], indent=2, default=str), flush=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
