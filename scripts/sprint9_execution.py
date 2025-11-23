#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sprint9_execution.py
--------------------
Minimal-invasive Ausführungsschicht, die bestehende Orders nach freq-spezifischem
Schema wegschreibt, damit Backtests/Reports sich nicht gegenseitig überschreiben.

Verwendung:
    python scripts/sprint9_execution.py --freq 5min
    python scripts/sprint9_execution.py --freq 1d

Wirkung:
    - Prüft auf output/orders.csv (Legacy)
    - Schreibt nach output/orders_{freq}.csv
"""

from __future__ import annotations
import argparse
import os
import sys
import pandas as pd

OUT_DIR = os.path.join("output")


def _load_legacy_orders(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Orders-Datei nicht gefunden: {path}")
    df = pd.read_csv(path)
    # Minimal-Sanity: Spalten normalisieren, falls nötig
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--freq", type=str, default="5min", help="Zeitauflösung, z.B. 5min oder 1d")
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    legacy = os.path.join(OUT_DIR, "orders.csv")
    outpath = os.path.join(OUT_DIR, f"orders_{args.freq}.csv")

    try:
        df = _load_legacy_orders(legacy)
        df.to_csv(outpath, index=False)
        print(f"[EXEC] START Execution | freq={args.freq}")
        print(f"[EXEC] [OK] written: {outpath}")
        print("[EXEC] DONE Execution")
        return 0
    except FileNotFoundError:
        # Kein Legacy? Dann freundlich abbrechen, ohne Seiteneffekte.
        print(f"[EXEC] WARN: {legacy} nicht gefunden. Nichts zu tun.", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[EXEC] FEHLER: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
