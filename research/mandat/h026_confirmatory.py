"""H-026 — Confirmatory-Test gate_both via Parameter-Störungen (Registry Welle 5).

6 Nachbar-Kombis {P70,P80,P90} x {SMA150,SMA250} — die Original-Kombi (P80,S200)
ist bewusst ausgeschlossen. PASS nur wenn ALLE 6 > gate-los UND > ETF-Pfad UND
Hälften-Konsistenz UND DSR(N=81) UND PBO<=0.5. Eine Verfehlung -> Thema ZU.
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
from verdict_engine import load_membership, load_verdict_prices, run_verdict  # noqa: E402

GATELESS_FINAL = 1214720.0
ETF_FINAL = 1610149.0
N_TRIALS = 81


def main() -> int:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    close = load_verdict_prices()
    membership = load_membership(close.index)
    results, rets = {}, {}
    for pct in (0.7, 0.8, 0.9):
        for sma in (150, 250):
            if (pct, sma) == (0.8, 200):
                continue
            name = f"H026_p{int(pct * 100)}_s{sma}"
            res, _eq, ret = run_verdict(
                close,
                membership,
                label=name,
                mode="momentum",
                top_out=60,
                use_gate=True,
                gate_mode="both",
                gate_vol_pct=pct,
                gate_sma_win=sma,
            )
            results[name] = res
            rets[name] = ret
            print(
                f"[RUN] {name}: final={res['final_value']:.0f} sharpe={res['sharpe_net']:.3f} maxdd={res['maxdd_net'] * 100:.1f}%",
                flush=True,
            )

    finals = {k: results[k]["final_value"] for k in rets}
    crit1 = all(v > GATELESS_FINAL for v in finals.values())
    crit2 = all(v > ETF_FINAL for v in finals.values())
    best = max(finals, key=finals.get)
    v = rets[best]
    mid = v.index[len(v) // 2]
    halves = []
    for sl in (v[v.index < mid], v[v.index >= mid]):
        halves.append(float(sl.mean() / sl.std() * np.sqrt(252)))
    # gateless halves for comparison
    res_gl, _eq, ret_gl = run_verdict(
        close, membership, label="gateless_ref", mode="momentum", top_out=60
    )
    gl_halves = []
    for sl in (ret_gl[ret_gl.index < mid], ret_gl[ret_gl.index >= mid]):
        gl_halves.append(float(sl.mean() / sl.std() * np.sqrt(252)))
    crit3 = halves[0] > gl_halves[0] and halves[1] > gl_halves[1]
    dsr = deflated_sharpe(v, n_trials=N_TRIALS)
    crit4 = bool(dsr.passes_5pct)
    pbo = float(cscv_pbo(pd.DataFrame(rets)))
    crit5 = pbo <= 0.5

    results["_verdict"] = {
        "crit1_all_gt_gateless": crit1,
        "crit2_all_gt_etf": crit2,
        "crit3_halves": {"best": halves, "gateless": gl_halves, "pass": crit3},
        "crit4_dsr": {"prob": float(dsr.deflated_sharpe_probability), "pass": crit4},
        "crit5_pbo": {"value": pbo, "pass": crit5},
        "selected": best,
        "PASS": all([crit1, crit2, crit3, crit4, crit5]),
    }
    out = OUTD / "h026_results.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(
        "[VERDICT]", json.dumps(results["_verdict"], indent=2, default=str), flush=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
