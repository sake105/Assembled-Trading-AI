# -*- coding: utf-8 -*-
"""Retention fuer wachsende Ops-Artefakt-Familien (KNOWN_ISSUES §0.06 f2).

Bewusst KONSERVATIV: geloescht wird nur, was aelter als die Familien-Frist
ist UND wovon eine juengere Datei derselben Familie existiert (das jeweils
neueste Artefakt bleibt IMMER, egal wie alt — ein leeres Verzeichnis waere
der naechste stille Informationsverlust). Append-only-Stores (SQLite,
JSONL-Audits) werden hier NICHT angefasst.

Familien (Glob -> Frist in Tagen):
  output/ops/pull_log_*.json          -> 30
  output/signals/signal_scores_*.json -> 30
  output/attribution/attribution_report_*.json -> 90
  output/regime/regime_state_*.json   -> 30

Aufrufer: scripts/daily_paper_trading.bat (non-fatal) oder manuell.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

FAMILIES: list[tuple[str, int]] = [
    ("output/ops/pull_log_*.json", 30),
    ("output/signals/signal_scores_*.json", 30),
    ("output/attribution/attribution_report_*.json", 90),
    ("output/regime/regime_state_*.json", 30),
]


def prune(dry_run: bool = False) -> int:
    now = time.time()
    total = 0
    for pattern, days in FAMILIES:
        files = sorted(_REPO.glob(pattern), key=lambda p: p.stat().st_mtime)
        if len(files) <= 1:
            continue  # das Neueste bleibt immer
        cutoff = now - days * 86400
        # alle ausser der juengsten Datei sind Kandidaten
        for f in files[:-1]:
            # F-senior-8: ein kaputter Symlink / eine Race-geloeschte Datei
            # darf nicht die GESAMTE Retention abbrechen.
            try:
                if not f.is_file():
                    continue
                if f.stat().st_mtime < cutoff:
                    if dry_run:
                        print(f"[DRY] wuerde loeschen: {f.relative_to(_REPO)}")
                    else:
                        f.unlink()
                    total += 1
            except OSError as exc:
                print(f"[WARN] prune uebersprungen ({f.name}): {exc}")
    print(
        f"[OK] prune: {total} Datei(en) {'markiert' if dry_run else 'entfernt'} "
        f"ueber {len(FAMILIES)} Familien"
    )
    return 0


def main() -> int:
    dry = "--dry-run" in sys.argv[1:]
    return prune(dry_run=dry)


if __name__ == "__main__":
    sys.exit(main())
