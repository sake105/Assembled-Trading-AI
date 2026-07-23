"""Sensitivitaets-Rerun der Kern-Verdicts mit Dividendensteuer (Modell-Korrektur).

Ersetzt die div-freien Laeufe (keine neuen Trials, Registry-Konvention).
Reruns: EW_PIT_monthly, H023_out100 (selected), H024_band25/50.
Frage: bleiben die Welle-4-Verdicts stabil?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h011_kandidat_a import OUTD  # noqa: E402
from verdict_engine import (  # noqa: E402
    load_div_panel,
    load_membership,
    load_verdict_prices,
    run_verdict,
)

OLD = {
    "EW_PIT_monthly": 1305849,
    "H023_out100": 1475985,
    "H024_band25": 1616344,
    "H024_band50": 1690822,
}


def main() -> int:
    close = load_verdict_prices()
    membership = load_membership(close.index)
    divp = load_div_panel(close.index)
    print(
        f"[DIV] panel: {divp.shape[1]} symbols, {divp.notna().sum().sum():.0f} events",
        flush=True,
    )

    runs = {
        "EW_PIT_monthly": dict(mode="ew"),
        "H023_out100": dict(mode="momentum", top_out=100),
        "H024_band25": dict(mode="ew", ew_band=0.25),
        "H024_band50": dict(mode="ew", ew_band=0.50),
    }
    results = {}
    for name, kw in runs.items():
        res, _eq, _ret = run_verdict(
            close, membership, label=name + "_divtax", div_panel=divp, **kw
        )
        res["old_final"] = OLD[name]
        res["delta_pct"] = round(100 * (res["final_value"] / OLD[name] - 1), 1)
        results[name] = res
        print(
            f"[RUN] {name}: {res['final_value']:.0f} (was {OLD[name]}, {res['delta_pct']}%) tax={res['tax_paid']:.0f}",
            flush=True,
        )

    # verdict stability
    ew = results["EW_PIT_monthly"]["final_value"]
    results["_stability"] = {
        "H023_still_below_ETF": results["H023_out100"]["final_value"] < 1610149,
        "H024_both_still_above_EW": (
            results["H024_band25"]["final_value"] > ew
            and results["H024_band50"]["final_value"] > ew
        ),
    }
    (OUTD / "divtax_sensitivity.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print("[STABILITY]", results["_stability"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
