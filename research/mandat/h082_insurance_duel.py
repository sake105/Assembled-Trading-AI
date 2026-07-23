"""H-082 — Versicherungs-Duell: Protective Put (PPUT, echte CBOE-Preise) vs §23-Gold-Sleeve.

(1) Brutto 1988–2026: PPUT/CNDR (und CLL 2008+) vs SPXTR — Kosten der Versicherung mit eigenen Augen.
(2) DE-Steuer-Overlay (wie H-081, Monats-Dekomposition, Topf) auf 2005+-Fenster.
(3) Duell-Tabelle vs Gold-Sleeve-Incumbents (H-061-Referenz: 70/30 biennial 890k/DD −0,36;
    100 % SPY 767k/DD −0,55; Fenster 2005–2026, tagesbasiert — Quellen-Note im JSON).
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
from h081_cboe_buywrite import german_overlay, monthly, stats  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


def main() -> int:
    px = pd.read_parquet(DATA / "prices_cboe_buywrite.parquet")
    w = px.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    w.index = pd.DatetimeIndex(w.index).tz_localize(None)
    out = {}
    for name in ("PPUT", "CNDR", "CLL"):
        d = w[[name, "SP500TR"]].dropna()
        im, sm = monthly(d[name]), monthly(d["SP500TR"])
        ri, rs = im.pct_change(fill_method=None), sm.pct_change(fill_method=None)
        both = ri.notna() & rs.notna()
        ri, rs = ri[both], rs[both]
        res = {"gross_idx": stats(ri, name), "gross_spxtr": stats(rs, "SPXTR")}
        dec = {}
        for d0 in (1990, 2000, 2010, 2020):
            m = (ri.index.year >= d0) & (ri.index.year < d0 + 10)
            if m.sum() > 24:
                dec[f"{d0}s"] = {
                    "idx": round(
                        (float((1 + ri[m]).prod()) ** (12 / m.sum()) - 1) * 100, 2
                    ),
                    "spx": round(
                        (float((1 + rs[m]).prod()) ** (12 / m.sum()) - 1) * 100, 2
                    ),
                }
        res["decades"] = dec
        # DE-Overlay auf 2005+-Fenster (Duell-Fenster mit Gold-Sleeve)
        s05 = pd.Timestamp("2005-01-01")
        res["german_overlay_2005"] = german_overlay(
            im[im.index >= s05], sm[sm.index >= s05]
        )
        out[name] = res
        print(f"[{name}] {res['gross_idx']} vs {res['gross_spxtr']}", flush=True)
        print(f"  Dekaden: {json.dumps(dec)}", flush=True)
        print(
            f"  DE-Overlay 2005+: {json.dumps(res['german_overlay_2005'])}", flush=True
        )

    out["_incumbents_2005_2026_ref"] = {
        "quelle": "H-061/W26 (tagesbasiert, SPY+GLD, End-Liq)",
        "100% SPY": {"net": 767247, "maxdd": -0.552},
        "70/30 Gold biennial": {"net": 889650, "maxdd": -0.361, "sharpe": 0.818},
    }
    (OUTD / "h082_insurance_duel.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("[DONE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
