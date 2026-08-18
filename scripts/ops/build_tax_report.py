# -*- coding: utf-8 -*-
"""Standalone-Backfill der Steuer-Sicht (Audit-Plan 5.5, 2026-08-17).

Liest eine Trade-/Fill-Historie (Parquet oder CSV im Ledger-Schema:
timestamp, symbol, side, fill_qty, fill_price, status[, commission_cash])
und schreibt dieselben ``tax_view``-Artefakte wie der Orchestrator-Step 4c
(``--write-tax-view``-Pfad). Rein lesend gegenueber dem Input.

Usage:
    python scripts/ops/build_tax_report.py --trades output/orders_1d.csv
    python scripts/ops/build_tax_report.py --trades trades.parquet --output-dir output
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trades",
        required=True,
        help="Trade-/Fill-Historie (.parquet oder .csv, Ledger-Schema)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO / "output"),
        help="Zielverzeichnis (default: output/)",
    )
    args = parser.parse_args()

    from src.assembled_core.accounting.tax_view import (
        build_tax_view_from_trades,
        write_tax_view_json,
    )

    trades_path = Path(args.trades)
    if not trades_path.exists():
        print(f"[ERROR] Trades-Datei fehlt: {trades_path}")
        return 1
    if trades_path.suffix.lower() == ".parquet":
        trades = pd.read_parquet(trades_path)
    else:
        trades = pd.read_csv(trades_path)

    result = build_tax_view_from_trades(trades)
    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = write_tax_view_json(
        result,
        Path(args.output_dir),
        run_id,
        trades_source=f"file:{trades_path.name}",
    )

    if result.n_fills_replayed == 0:
        # F-senior-3 (E-187): 0 verarbeitete Fills ist KEIN Erfolg. Die im
        # Docstring genannte Beispiel-Invocation traf genau das: falsches
        # Spaltenschema -> frueher Ausstieg -> "[OK] 0 Fills", Exit 0 und ein
        # Artefakt mit Qualitaetssiegel. Jetzt laut und rot.
        print(
            f"[ERROR] 0 Fills verarbeitet — Eingabe {trades_path.name} hat "
            f"vermutlich nicht das Ledger-Schema (timestamp, symbol, side, "
            f"fill_qty, fill_price[, status, commission_cash]). Notes: "
            f"{result.notes}"
        )
        return 1
    print(f"[OK] {result.n_fills_replayed} Fills replayed -> {out}")
    for y, s in sorted(result.years.items()):
        print(
            f"  {y}: Gewinne {s.gains_eur:.2f} EUR, Verluste {s.losses_eur:.2f}, "
            f"Verlusttopf {s.loss_pot_start:.2f}->{s.loss_pot_end:.2f}, "
            f"Pauschbetrag {s.pauschbetrag_used:.2f}, Steuer {s.tax_eur:.2f}"
        )
    if result.over_close_qty:
        print(f"[WARN] over_close (Shorts/Luecken?): {result.over_close_qty}")
    print(f"[INFO] fx_source={result.fx_source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
