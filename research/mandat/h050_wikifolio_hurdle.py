"""wikifolio-Hürde — deterministische Gebühren-/Steuer-Rechnung (KEIN Trial, kein Signal).

Frage: Wieviel BRUTTO-Alpha/Jahr muss die wikifolio-Underlying-Strategie erzeugen, nur um
den passiven Aktien-ETF NETTO-NACH-STEUER zu erreichen? Strukturvergleich, verifizierbar.

ETF:       TER 0,20 %/J; Endsteuer 18,46 % (Teilfreistellung 30 %).
wikifolio: 0,95 %/J Zertifikategebühr + Performancegebühr p auf jährlichen Gewinn (Jahres-
           HWM-Reset 31.12.); Endsteuer 26,375 % (KEINE Teilfreistellung); Emittentenrisiko
           (nicht bepreist -> zugunsten wikifolio).
Interne Umschichtung im Zertifikat ist steuergestundet (wie hier modelliert: nur Endsteuer).
"""

from __future__ import annotations

import json
from pathlib import Path

P = 100_000.0
H = 20  # Jahre
ETF_TER = 0.0020
ETF_TAX = 0.1846
WIKI_FEE = 0.0095
WIKI_TAX = 0.26375
OUTD = Path(__file__).resolve().parent / "results"
OUTD.mkdir(exist_ok=True)


def etf_net(market_gross: float) -> float:
    v = P * (1 + market_gross - ETF_TER) ** H
    return P + (v - P) * (1 - ETF_TAX)


def wiki_net(underlying_gross: float, perf_fee: float) -> float:
    v = P
    for _ in range(H):
        start = v
        v = (
            v * (1 + underlying_gross) * (1 - WIKI_FEE)
        )  # underlying + Zertifikategebühr
        gain = v - start
        if gain > 0:
            v -= gain * perf_fee  # Performancegebühr auf Jahresgewinn (HWM-Reset)
    return P + (v - P) * (1 - WIKI_TAX)


def breakeven_gross(market_gross: float, perf_fee: float) -> float:
    """Underlying-Brutto, das wikifolio-netto == ETF-netto macht (Bisektion)."""
    target = etf_net(market_gross)
    lo, hi = market_gross, market_gross + 0.15
    for _ in range(60):
        mid = (lo + hi) / 2
        if wiki_net(mid, perf_fee) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> int:
    out = {"horizon_years": H, "principal": P, "scenarios": []}
    for mkt in (0.06, 0.08, 0.10):
        for pf in (0.10, 0.20, 0.30):
            be = breakeven_gross(mkt, pf)
            # Parität: wikifolio-Underlying == Markt -> wieviel weniger endet der Anleger?
            parity_gap = wiki_net(mkt, pf) / etf_net(mkt) - 1
            out["scenarios"].append(
                {
                    "market_gross_pct": round(mkt * 100, 1),
                    "perf_fee_pct": round(pf * 100),
                    "ETF_net": round(etf_net(mkt)),
                    "wiki_net_at_parity_gross": round(wiki_net(mkt, pf)),
                    "parity_shortfall_pct": round(parity_gap * 100, 1),
                    "breakeven_underlying_gross_pct": round(be * 100, 2),
                    "required_annual_gross_alpha_pct": round((be - mkt) * 100, 2),
                }
            )
    (OUTD / "h050_wikifolio_hurdle.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("[HURDLE]", json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
