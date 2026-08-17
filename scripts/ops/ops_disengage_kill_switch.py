# -*- coding: utf-8 -*-
"""Operator-Tool: Kill-Switch kontrolliert loesen (Disengage).

WOZU: Der Kill-Switch ist seit 2026-08-09 engaged (actor
trading_cycle_v2_auto_dd, "drawdown=-90.00%" — nachweislich
Test-Kontamination, kein echter Drawdown; Memory-Befund 2026-08-15).
Der Pilot ist seither orderlos. Deaktivierung erfordert
OPERATOR_KILL_TOKEN und ist bewusst NICHT agent-ausfuehrbar
(Auto-Mode-Klassifizierer blockt Token-Zugriff) — dieses Script
fuehrt der Operator selbst aus:

    python scripts/ops/ops_disengage_kill_switch.py [--reason "..."]

Das Token wird aus .env geladen und NIE ausgegeben (Rule 20).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

STATE_PATH = _REPO / "output" / "ops" / "kill_switch_state.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reason",
        default="operator_disengage_test_contamination_2026-08-09",
        help="Audit-Grund fuer die Deaktivierung",
    )
    args = parser.parse_args()

    # M1 (Stage 1, 2026-08-17): die Kill-Switch-Engine loest ihre Pfade
    # CWD-relativ auf (kill_switch._DEFAULT_STATE_FILE) — aus fremdem CWD
    # gestartet saehe sie keinen State und dieses Tool wuerde faelschlich
    # "[SKIP] bereits disengaged" melden (E-146-Klasse). Deshalb hart ins
    # Repo-Root wechseln, bevor die Engine angefasst wird.
    os.chdir(_REPO)

    from dotenv import load_dotenv

    load_dotenv(_REPO / ".env")

    from src.assembled_core.execution.kill_switch import (
        deactivate_kill_switch,
        is_kill_switch_engaged,
    )

    if not is_kill_switch_engaged():
        print("[SKIP] Kill-Switch ist bereits disengaged — nichts zu tun.")
        return 0

    # F-senior-7: nur Info-Ausgabe — ein korruptes State-File wertet die
    # Engine fail-closed als engaged; genau dann darf der einzige
    # Wiederherstellungspfad nicht an json.loads sterben.
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            print(
                f"[START] engaged seit {state.get('activated_at')} "
                f"(actor={state.get('actor')}, reason={str(state.get('reason'))[:60]})"
            )
        except (OSError, ValueError) as exc:
            print(f"[WARN] State-File unlesbar ({exc}) — fahre mit Disengage fort.")

    token = os.environ.get("OPERATOR_KILL_TOKEN")
    if not token:
        print(
            "[ERROR] OPERATOR_KILL_TOKEN nicht in der Umgebung/.env — "
            "Disengage nicht moeglich."
        )
        return 1

    try:
        deactivate_kill_switch(
            reason=args.reason,
            actor="operator_manual",
            operator_token=token,
        )
    except PermissionError as exc:
        print(f"[ERROR] Deaktivierung abgelehnt: {exc}")
        return 1

    if is_kill_switch_engaged():
        print("[ERROR] State meldet weiterhin engaged — bitte manuell pruefen.")
        return 1
    print("[OK] Kill-Switch disengaged — Pilot kann beim naechsten Lauf ordern.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
