# -*- coding: utf-8 -*-
"""Producer fuer /monitoring/regime (Audit-Follow-up, 2026-08-17).

WOZU: Der Endpoint suchte seit jeher regime_state_*.json, die NIEMAND schrieb
(Nutzungsaudit §3 Punkt 10 — Consumer ohne Producer). Dieses Script schreibt
den Zustand mit EXAKT demselben Detector, den die Live-Pipeline nutzt
(risk.regime_models.build_regime_state_hmm — Rule 50: keine zweite
Regime-Wahrheit), auf dem operativen Preis-Panel.

Aufrufer: scripts/daily_paper_trading.bat (non-fatal, nach dem Pilot-Lauf)
oder manuell. Read-only gegenueber dem Panel.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd

PANEL_PATH = _REPO / "output" / "aggregates" / "daily.parquet"
OUT_DIR = _REPO / "output" / "regime"


def main() -> int:
    if not PANEL_PATH.exists():
        print(f"[ERROR] Panel fehlt: {PANEL_PATH}")
        return 1

    df = pd.read_parquet(PANEL_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    cutoff = df["timestamp"].max() - pd.Timedelta(days=400)
    df = df[df["timestamp"] >= cutoff]

    from src.assembled_core.risk.regime_models import build_regime_state_hmm

    hmm_df = build_regime_state_hmm(prices=df, n_regimes=3, benchmark_symbol="SPY")
    if hmm_df.empty:
        print("[ERROR] Regime-Detector lieferte leeres Ergebnis")
        return 1
    sort_col = "date" if "date" in hmm_df.columns else "timestamp"
    hmm_df = hmm_df.sort_values(sort_col, kind="mergesort")
    last = hmm_df.iloc[-1]
    # F-TR-2/F-senior-4: fehlendes ODER NaN-Label ist "unknown" — NaN ist
    # truthy, ein or-Guard reicht fuer pandas-Werte nicht (E-175).
    _label = last.get("regime_label")
    regime = str(_label) if pd.notna(_label) and str(_label) else "unknown"

    now = datetime.now(timezone.utc)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": now.isoformat(),
        "data_as_of": str(df["timestamp"].max()),
        "regime": regime,
        # F-senior-3: der Detector liefert regime_confidence, nie
        # regime_score — das Confidence-Feld IST der Score (E-174).
        "regime_score": (
            float(last["regime_confidence"])
            if "regime_confidence" in hmm_df.columns
            and pd.notna(last.get("regime_confidence"))
            else None
        ),
        "detector": "risk.regime_models.build_regime_state_hmm (same as pipeline)",
        "producer": "scripts/ops/write_regime_state.py",
    }
    out = OUT_DIR / f"regime_state_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"[OK] regime='{regime}' (as of {payload['data_as_of']}) -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
