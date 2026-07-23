"""H-036b — Kosten-Sensitivität der EW-Band-Small „Size-Prämie" (§2.5-Realismus-Check).

Der H-036-PASS (Small 5,97M > Large 3,53M > ETF 736k) steht unter Verdacht:
maxDD −0,958 + 2.288 Micro/Small-Namen @30 bps = unrealistisch. Micro-Cap-Realkosten
(Bid/Ask + Impact) sind 100e bps, nicht 30. Dieser Lauf zeigt, ob der „PASS" ein
Liquiditäts-/Rebalancing-Artefakt ist: EW-Band Small vs Large bei 30/60/100/150 bps
vs ETF-Pfad. KEIN neuer Hypothesen-Trial — Robustheit von H-036.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import h011_kandidat_a as eng  # noqa: E402
from h011_kandidat_a import OUTD, START_CAPITAL  # noqa: E402
from smallcap_data import band_membership, load_smallcap  # noqa: E402
from verdict_engine import run_verdict  # noqa: E402

ETF_TAX = 0.185


def main() -> int:
    close, adv = load_smallcap()
    mem_small = band_membership(close, adv, side="small")
    mem_large = band_membership(close, adv, side="large")

    spy = close["SPY"].dropna()
    etf_net = START_CAPITAL + START_CAPITAL * (spy.iloc[-1] / spy.iloc[0] - 1) * (
        1 - ETF_TAX
    )

    out = {"ETF_net_path": round(float(etf_net))}
    for cost in (30.0, 60.0, 100.0, 150.0):
        eng.COST_BPS = cost
        rs, _, _ = run_verdict(
            close, mem_small, label=f"small_{cost}", mode="ew", ew_band=0.5
        )
        rl, _, _ = run_verdict(
            close, mem_large, label=f"large_{cost}", mode="ew", ew_band=0.5
        )
        out[f"{int(cost)}bps"] = {
            "small_final": round(rs["final_value"]),
            "large_final": round(rl["final_value"]),
            "small_sharpe": round(rs["sharpe_net"], 3),
            "small_maxdd": round(rs["maxdd_net"], 3),
            "small_gt_ETF": bool(rs["final_value"] > etf_net),
            "small_gt_large_x110": bool(rs["final_value"] > rl["final_value"] * 1.10),
        }
        print(
            f"[COST {int(cost)}bps] small={rs['final_value']:,.0f} large={rl['final_value']:,.0f} "
            f"ETF={etf_net:,.0f} small>ETF={out[f'{int(cost)}bps']['small_gt_ETF']} "
            f"DD={rs['maxdd_net']:.3f}",
            flush=True,
        )

    (OUTD / "h036b_cost_sensitivity.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    print("[DONE]", json.dumps(out, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
