# -*- coding: utf-8 -*-
"""Pilot-Manifest aus den Run-Artefakten rekonstruieren (E-190, 2026-08-18).

WOZU: ``output/pilot/pilot_manifest.json`` ist die Bewertungsgrundlage des
30-Tage-Pilots und ein Watchdog-Input (zero_orders_unexpected). Bis zum
Pfad-Split in ``run_paper_pilot.py`` schrieben CI-Runner und lokaler Betrieb
DIESELBE Datei, und der Workflow committete sie mit ``git add -f``: ein
``git pull`` ersetzte die echte Betriebshistorie durch die kurze
CI-Variante (gemessen 27 Tage -> 1 Tag).

Dieses Script baut die Historie aus dem wieder auf, was der Betrieb ohnehin
je Lauf schreibt: ``output/runs/live_paper_*/`` (orders_latest.json,
run_kpis.json). Es RECHNET NICHTS HINZU — was in den Artefakten nicht steht,
steht auch nicht im Ergebnis, und jeder Tag traegt seine Quelle.

Usage:
    python scripts/ops/rebuild_pilot_manifest.py            # Dry-Run
    python scripts/ops/rebuild_pilot_manifest.py --write    # schreiben
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "output" / "runs"
MANIFEST = ROOT / "output" / "pilot" / "pilot_manifest.json"


def _run_day(run_dir: Path) -> dict | None:
    """Einen Lauf auf einen Manifest-Tag abbilden (nur belegte Felder)."""
    orders_path = run_dir / "orders_latest.json"
    kpis_path = run_dir / "run_kpis.json"
    if not orders_path.exists() and not kpis_path.exists():
        return None

    n_orders = None
    if orders_path.exists():
        try:
            n_orders = len(
                json.loads(orders_path.read_text(encoding="utf-8")).get("items", [])
            )
        except (OSError, ValueError):
            n_orders = None

    ts = None
    if kpis_path.exists():
        try:
            ts = json.loads(kpis_path.read_text(encoding="utf-8")).get("generated_utc")
        except (OSError, ValueError):
            ts = None
    if ts is None:
        # Aus dem Verzeichnisnamen: live_paper_YYYYMMDD_HHMMSS[Z]_hash
        parts = run_dir.name.split("_")
        if len(parts) >= 4:
            try:
                ts = (
                    datetime.strptime(parts[2] + parts[3].rstrip("Z"), "%Y%m%d%H%M%S")
                    .replace(tzinfo=timezone.utc)
                    .isoformat()
                )
            except ValueError:
                ts = None
    if ts is None:
        return None

    return {
        "timestamp": ts,
        "date": ts[:10],
        "rc": 0,  # ein geschriebenes Run-Verzeichnis impliziert einen Lauf
        "crashed": False,
        "n_orders_detected": n_orders if n_orders is not None else 0,
        "source": f"reconstructed:{run_dir.name}",
        "orders_artifact": orders_path.exists(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Manifest schreiben")
    parser.add_argument(
        "--since",
        default="2026-07-02",
        help=(
            "Nur Laeufe ab diesem Datum (Default: Relaunch der aktuellen "
            "Pilot-Phase, Baseline 87874.90). Aeltere Laeufe gehoeren zur "
            "abgebrochenen v1-Phase und duerfen NICHT in dieselbe "
            "30-Tage-Bewertung fliessen."
        ),
    )
    args = parser.parse_args()

    if not RUNS_DIR.exists():
        print(f"[ERROR] {RUNS_DIR} fehlt")
        return 1

    days: list[dict] = []
    for run_dir in sorted(RUNS_DIR.glob("live_paper_*")):
        d = _run_day(run_dir)
        if d and d["date"] >= args.since:
            days.append(d)

    if not days:
        print("[ERROR] keine rekonstruierbaren Laeufe gefunden")
        return 1

    # Ein Kalendertag = ein Pilot-Tag (der letzte Lauf des Tages gewinnt).
    by_date: dict[str, dict] = {}
    for d in sorted(days, key=lambda x: x["timestamp"]):
        by_date[d["date"]] = d
    ordered = [by_date[k] for k in sorted(by_date)]
    for i, d in enumerate(ordered, start=1):
        d["day"] = i

    with_orders = sum(1 for d in ordered if d["n_orders_detected"] > 0)
    payload = {
        "started_at": ordered[0]["timestamp"],
        "reconstructed_at": datetime.now(tz=timezone.utc).isoformat(),
        "reconstructed_from": f"output/runs/live_paper_* (since {args.since})",
        "phase_start": args.since,
        "note": (
            "Aus Run-Artefakten rekonstruiert (E-190). UNTERGRENZE, keine "
            "vollstaendige Historie: nur Laeufe MIT Run-Verzeichnis sind "
            "abbildbar — ein Lauf, der frueh abbrach (z. B. Kill-Switch-Abort "
            "mit rc=1, 10.-17.08.), hinterlaesst keine Artefakte und fehlt "
            "hier. Die vor dem Ueberschreiben gezaehlten 27 Tage sind daher "
            "NICHT wiederherstellbar; diese Datei belegt die Tage, die "
            "Artefakte tragen. rc wird als 0 gefuehrt, weil ein geschriebenes "
            "Run-Verzeichnis einen durchgelaufenen Zyklus belegt; "
            "n_orders_detected stammt aus orders_latest.json des Laufs."
        ),
        "is_lower_bound": True,
        "days": ordered,
    }

    print(
        f"[OK] {len(days)} Laeufe -> {len(ordered)} Pilot-Tage "
        f"({ordered[0]['date']} bis {ordered[-1]['date']}), "
        f"{with_orders} Tage mit Orders"
    )
    if not args.write:
        print("[DRY-RUN] nichts geschrieben — mit --write ausfuehren")
        return 0

    if MANIFEST.exists():
        backup = MANIFEST.with_suffix(
            f".pre_rebuild_{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        backup.write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[OK] Backup: {backup.name}")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] geschrieben: {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
