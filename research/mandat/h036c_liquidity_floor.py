"""H-036c — Liquiditäts-Floor-Test: ist die „Size-Prämie" real oder Micro-Cap-Artefakt?

Fixe Kosten killen den EW-Band-Small-PASS nicht (50%-Band = moderater Turnover). Verdacht:
Illiquiditäts-/Bid-Ask-Bounce-Artefakt (EW-Rebalancing erntet Phantom-Returns aus
verrauschten Micro-Cap-Closes; −96% DD = uninvestierbar). Diskriminator: ADV-Floor hochziehen.
Wenn die Prämie bei ECHT liquiden Small Caps (ADV≥$10M/$50M) kollabiert → Artefakt bestätigt.

Small-Band EW-Band bei FLOOR_ADV ∈ {$1M, $10M, $50M}, 60 bps. KEIN neuer Trial.
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
import smallcap_data as scd  # noqa: E402
from h011_kandidat_a import OUTD, START_CAPITAL  # noqa: E402
from smallcap_data import band_membership, load_smallcap  # noqa: E402
from verdict_engine import run_verdict  # noqa: E402

ETF_TAX = 0.185


def main() -> int:
    close, adv = load_smallcap()
    spy = close["SPY"].dropna()
    etf_net = START_CAPITAL + START_CAPITAL * (spy.iloc[-1] / spy.iloc[0] - 1) * (
        1 - ETF_TAX
    )
    eng.COST_BPS = 60.0

    out = {"ETF_net_path": round(float(etf_net))}
    for floor in (1e6, 10e6, 50e6):
        scd.FLOOR_ADV = floor
        mem = band_membership(close, adv, side="small")
        med = int(pd.Series([len(s) for s in mem]).median()) if len(mem) else 0
        res, _, _ = run_verdict(
            close, mem, label=f"small_adv{floor:.0e}", mode="ew", ew_band=0.5
        )
        out[f"adv_{int(floor / 1e6)}M"] = {
            "median_names": med,
            "final": round(res["final_value"]),
            "sharpe": round(res["sharpe_net"], 3),
            "maxdd": round(res["maxdd_net"], 3),
            "gt_ETF": bool(res["final_value"] > etf_net),
        }
        print(
            f"[ADV>={int(floor / 1e6)}M] names~{med} final={res['final_value']:,.0f} "
            f"ETF={etf_net:,.0f} gt_ETF={res['final_value'] > etf_net} "
            f"Sharpe={res['sharpe_net']:.3f} DD={res['maxdd_net']:.3f}",
            flush=True,
        )

    (OUTD / "h036c_liquidity_floor.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    print("[DONE]", json.dumps(out, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
