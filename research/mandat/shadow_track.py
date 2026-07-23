"""Shadow-Tracking (Echtzeit-OOS, Guardrail-2-konform — KEINE Trades, nur Aufzeichnung).

Baskets (fixiert am Start, Rebalance nur bei neuem Quartals-/Monatssignal):
  - 13F_k10:   H-029-k10-Konsens ex-ETF (Snapshot 2026-07-08)
  - INSIDER:   H-031-officer_10k-Basket (aktuelle 12M-Halteliste zum Startdatum)
Jeder Aufruf: holt aktuelle EODHD-Kurse, schreibt NAV-Zeile je Basket nach
results/shadow_log.jsonl (append-only). Benchmarks: SPY. Aufruf: manuell/Session.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
TOK = os.environ["EODHD_API_TOKEN"]
RES = Path(__file__).resolve().parent / "results"
STATE = RES / "shadow_state.json"
LOG = RES / "shadow_log.jsonl"

K10 = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "AVGO",
    "META",
    "GOOG",
    "TSLA",
    "JPM",
    "LLY",
]


def px(sym: str) -> float | None:
    url = f"https://eodhd.com/api/real-time/{sym}.US?api_token={TOK}&fmt=json"
    try:
        d = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "research"}),
                timeout=30,
            )
            .read()
            .decode()
        )
        v = d.get("close") or d.get("previousClose")
        return float(v) if v not in (None, "NA") else None
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    today = dt.date.today().isoformat()
    if not STATE.exists():
        # initialize equal-weight virtual portfolios at today's prices
        state = {"start_date": today, "baskets": {}}
        for name, syms in (("13F_k10", K10), ("SPY_bench", ["SPY"])):
            prices = {s: px(s) for s in syms}
            prices = {s: p for s, p in prices.items() if p}
            qty = {s: (100000.0 / len(prices)) / p for s, p in prices.items()}
            state["baskets"][name] = {"qty": qty, "start_prices": prices}
        STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"[INIT] shadow baskets anchored at {today}", flush=True)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    row = {"date": today}
    for name, b in state["baskets"].items():
        nav = 0.0
        for s, q in b["qty"].items():
            p = px(s)
            if p:
                nav += q * p
        row[name] = round(nav, 2)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"[OK] {row}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
