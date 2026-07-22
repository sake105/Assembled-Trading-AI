"""One-off Ops: Externe Broker-Positionen (manuelle Käufe) ins Paper-Ledger adoptieren.

Kontext 2026-07-14: Reconcile-Halt seit 2026-07-07 — GLD/TLT liegen im Alpaca-Paper-Konto
(manuell gekauft), Ledger kennt sie nicht (cash_diff $14.941,52). Operator-Entscheidung Hans:
Adoption ins Ledger (Option a), danach ack_halt.

Verhalten: DRY-RUN per Default (zeigt Fills + Cash-Abgleich). Schreiben NUR mit --apply.
Nutzt ausschließlich offizielle APIs (ops.paper_ledger, execution.broker_adapter read-only);
save_ledger_state schreibt atomar mit Backups. Adoptiert werden NUR Symbole, die im Broker
existieren und im Ledger fehlen — zu deren Broker-avg_entry_price (BUY-Fill-Semantik).

Usage:
  python scripts/ops_adopt_external_positions.py           # Dry-Run
  python scripts/ops_adopt_external_positions.py --apply   # schreibt Ledger
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from src.assembled_core.execution.broker_adapter import AlpacaAdapter  # noqa: E402
from src.assembled_core.ops.paper_ledger import (  # noqa: E402
    apply_fills_to_ledger,
    load_ledger_state,
    save_ledger_state,
)

LEDGER_PATH = ROOT / "output" / "runs" / "_paper_ledger" / "ledger_state.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--apply", action="store_true", help="Ledger wirklich schreiben (sonst Dry-Run)"
    )
    args = ap.parse_args()

    # Cold-Start-Guard (Review TR-M1/F-senior-1): NIE gegen einen frisch erzeugten
    # 10k-Default-State adoptieren — wenn die Ledger-Datei fehlt, laut abbrechen.
    if not LEDGER_PATH.exists():
        print(
            f"[ABBRUCH] Ledger-Datei fehlt: {LEDGER_PATH} — Adoption gegen Fresh-State verweigert."
        )
        return 1

    adapter = AlpacaAdapter()
    acct = adapter.get_account()
    broker_cash = float(acct.get("cash", 0))
    positions = adapter.get_positions()
    print(
        f"[BROKER] cash={broker_cash:,.2f} positions={[(p.symbol, p.qty, p.avg_entry_price) for p in positions]}"
    )

    state = load_ledger_state(LEDGER_PATH)
    led_cash = float(state.get("cash") or 0)
    led_pos = state.get("positions") or {}
    print(
        f"[LEDGER] cash={led_cash:,.2f} positions={ {k: v.get('qty') for k, v in led_pos.items()} }"
    )
    print(f"[DIFF ] cash ledger−broker = {led_cash - broker_cash:+,.2f}")

    fills = []
    for p in positions:
        if p.symbol in led_pos and float(led_pos[p.symbol].get("qty") or 0) > 0:
            continue  # existiert schon — nicht adoptieren
        if p.qty <= 0 or p.avg_entry_price <= 0:
            continue
        fills.append(
            {
                "symbol": p.symbol,
                "side": "BUY",
                "qty": float(p.qty),
                "price": float(p.avg_entry_price),
            }
        )
    if not fills:
        print("[OK] Nichts zu adoptieren — Ledger kennt alle Broker-Positionen.")
        return 0
    notional = sum(f["qty"] * f["price"] for f in fills)
    print(f"[PLAN] Adoption-Fills (BUY @ broker avg_entry): {fills}")
    print(
        f"[PLAN] Notional gesamt: {notional:,.2f} -> Ledger-Cash danach ≈ {led_cash - notional:,.2f} "
        f"(Broker: {broker_cash:,.2f}, Rest-Diff ≈ {led_cash - notional - broker_cash:+,.2f})"
    )

    if not args.apply:
        print("[DRY-RUN] Nichts geschrieben. Mit --apply ausführen.")
        return 0

    new_state = apply_fills_to_ledger(state, fills)
    out_path = save_ledger_state(new_state, LEDGER_PATH)
    new_cash = float(new_state.get("cash") or 0)
    print(f"[APPLIED] Ledger gespeichert: {out_path}")
    print(
        f"[VERIFY] Ledger-Cash neu = {new_cash:,.2f} vs Broker {broker_cash:,.2f} "
        f"(Diff {new_cash - broker_cash:+,.2f}) — Schwelle $100/10bps"
    )
    print("[NEXT ] scripts/ack_halt.py ausführen, um den Halt zu quittieren.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
