"""Welle 3 — H-020/021/022: No-Signal-Steuer-Designs (Registry, VOR Lauf registriert).

Survivorship kürzt sich im Paar No-Signal-vs-No-Signal heraus: gemessen wird der
deutsche Steuer-/Turnover-Effekt der Umsetzung. 4 Läufe, N→67.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h011_kandidat_a import OUTD, cscv_pbo, load_prices, run_variant  # noqa: E402

EW_MONTHLY_FINAL = 416611.81
EW_MONTHLY_SHARPE = 0.851
H012_OUT60_FINAL = 614905.0


def main() -> int:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    close, high, low = load_prices()
    close = close.dropna(axis=1, thresh=int(len(close) * 0.5))
    high, low = high[close.columns], low[close.columns]
    print(f"[DATA] {close.shape[1] - 1} symbols", flush=True)

    results, rets = {}, {}

    def go(name, **kw):
        res, _eq, ret = run_variant(close, high, low, None, label=name, **kw)
        results[name] = res
        rets[name] = ret
        print(
            f"[RUN] {name}: final={res['final_value']:.0f} sharpe={res['sharpe_net']:.3f} tax={res['tax_paid']:.0f} costs={res['costs_paid']:.0f}",
            flush=True,
        )

    ewbase = dict(
        use_quality=False,
        use_gate=False,
        use_atr_backstop=False,
        top_out=40,
        ew_baseline=True,
    )
    go("H020_band25", ew_band=0.25, **ewbase)
    go("H020_band50", ew_band=0.50, **ewbase)
    go(
        "H021_annual",
        use_quality=False,
        use_gate=False,
        use_atr_backstop=False,
        no_retrim=True,
        top_out=60,
        rebal_months={1},
    )
    go("H022_buyhold", ew_band=1e9, **ewbase)

    n_map = {
        "H020_band25": 65,
        "H020_band50": 65,
        "H021_annual": 66,
        "H022_buyhold": 67,
    }
    best20 = max(
        ("H020_band25", "H020_band50"), key=lambda k: results[k]["final_value"]
    )
    for name, fam_n in ((best20, 65), ("H021_annual", 66), ("H022_buyhold", 67)):
        dsr = deflated_sharpe(rets[name], n_trials=fam_n)
        results[f"_dsr_{name}"] = {
            "prob": float(dsr.deflated_sharpe_probability),
            "passes": bool(dsr.passes_5pct),
            "n": fam_n,
        }
    results["_pbo_H020"] = float(
        cscv_pbo(pd.DataFrame({k: rets[k] for k in ("H020_band25", "H020_band50")}))
    )
    results["_refs"] = {
        "EW_monthly_final": EW_MONTHLY_FINAL,
        "EW_monthly_sharpe": EW_MONTHLY_SHARPE,
        "H012_out60_final": H012_OUT60_FINAL,
    }

    out = OUTD / "welle3_results.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"[DONE] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
