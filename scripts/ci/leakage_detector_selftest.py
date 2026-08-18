# -*- coding: utf-8 -*-
"""CI-Selbsttest des Leakage-Detektors (Audit-Plan 3.2, 2026-08-17).

ERSETZT den historischen No-op-Step in daily-diagnostics.yml, der ein NICHT
existierendes Script (validate_altdata.py --check-leakage) mit `|| echo`
aufrief — es gab dadurch nie einen laufenden Leakage-Check in CI (Audit §4
Punkt 6). Der Runner HAT keine Produktionsdaten (output/ ist gitignored);
ein "Daten-Check" hier waere Theater. Der echte Daten-Check laeuft seit
Audit-Plan 3.1 taeglich lokal im Orchestrator (Gate 8 mit echtem Frame).

WAS DIESER SELBSTTEST BEWEIST (fail-drill-Muster, wie weekly-drills):
  1. POSITIVE Probe: ein praepariertes Leak (Feature-Beobachtung VOR
     disclosure_date) muss BLOCK ergeben — der Detektor ist scharf.
  2. NEGATIVE Probe: ein sauberes Frame muss OK ergeben — kein Fehlalarm.
  3. SKIP-Ehrlichkeit: ohne Frame muss SKIPPED herauskommen, nie OK (E-066).

Exit 0 nur, wenn alle drei stimmen. Exit 1 = Detektor defekt -> Job rot.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd

from src.assembled_core.qa.qa_gates import QAResult, check_leakage


def main() -> int:
    ok = True

    # 1. Praepariertes Leak: Beobachtung 2 Tage VOR Offenlegung.
    leaky = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-05", "2026-01-06"], utc=True),
            "disclosure_date": pd.to_datetime(["2026-01-07", "2026-01-06"], utc=True),
            "eps_surprise_pct": [4.2, 1.1],
        }
    )
    r1 = check_leakage(
        feature_df=leaky,
        feature_col="eps_surprise_pct",
        disclosure_col="disclosure_date",
        timestamp_col="timestamp",
    )
    if r1.result is QAResult.BLOCK:
        print("[OK] positive probe: prepared leak -> BLOCK (detector fires)")
    else:
        print(f"[ERROR] positive probe: expected BLOCK, got {r1.result.value}")
        ok = False

    # 2. Sauberes Frame: Beobachtung fruehestens am Offenlegungstag.
    clean = leaky.copy()
    clean["timestamp"] = clean["disclosure_date"]
    r2 = check_leakage(
        feature_df=clean,
        feature_col="eps_surprise_pct",
        disclosure_col="disclosure_date",
        timestamp_col="timestamp",
    )
    if r2.result is QAResult.OK:
        print("[OK] negative probe: clean frame -> OK (no false alarm)")
    else:
        print(f"[ERROR] negative probe: expected OK, got {r2.result.value}")
        ok = False

    # 3. Kein Frame: SKIPPED, niemals OK (E-066: nicht geprueft != sauber).
    r3 = check_leakage(feature_df=None)
    if r3.result is QAResult.SKIPPED:
        print("[OK] skip probe: no frame -> SKIPPED (honest, not green)")
    else:
        print(f"[ERROR] skip probe: expected SKIPPED, got {r3.result.value}")
        ok = False

    print(f"Selftest verdict: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
