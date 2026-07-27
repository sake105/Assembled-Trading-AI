"""Backfill entry_ts on existing paper-ledger positions (operator tool).

Context (2026-07-27): the zombie-killer hold-time check reads ``entry_ts``
from ledger positions. The field is stamped automatically since commit
c7f5af64 for NEW opens/flips — positions that predate the schema (e.g. the
GLD/TLT adoption of 2026-07-14) carry none and are skipped loudly by
``risk/zombie_killer`` (warn-once). This tool sets an HONEST entry_ts for
such legacy positions: the operator-supplied timestamp (typically the
adoption date — never an invented/backdated "real purchase" time).

Safety model (mirrors ops_adopt_external_positions.py):
- DRY-RUN by default; ``--apply`` performs the write.
- Writes go through paper_ledger.save_ledger_state (file lock, atomic
  replace, .1/.2/.3 backup rotation).
- Refuses to overwrite an existing entry_ts without ``--force``.
- Refuses unknown symbols, zero-qty positions, unparseable or future
  timestamps, and a missing ledger file (never creates a fresh ledger).
- Refuses a corrupt MAIN ledger file (load_ledger_state would silently fall
  back to an older backup — a write tool must never promote that state).
- Post-write verify covers the UNCHANGED fields too (cash/qty/avg_price/hwm).
- Run OUTSIDE the scheduler window (21:10 Europe/Berlin): the save is
  file-locked, but load→save is not atomic against a concurrent cycle write.

Usage:
  python scripts/ops/backfill_position_entry_ts.py --set GLD=2026-07-14T00:00:00+00:00 --set TLT=2026-07-14T00:00:00+00:00
  python scripts/ops/backfill_position_entry_ts.py --set ... --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.assembled_core.ops.paper_ledger import (  # noqa: E402
    load_ledger_state,
    save_ledger_state,
)

DEFAULT_LEDGER = REPO_ROOT / "output" / "runs" / "_paper_ledger" / "ledger_state.json"


def _parse_iso_utc(raw: str) -> datetime:
    """Parse an ISO timestamp; REQUIRE an explicit UTC offset (no naive input)."""
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"timestamp must carry an explicit UTC offset: {raw!r}")
    return dt.astimezone(timezone.utc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--set",
        dest="sets",
        action="append",
        required=True,
        metavar="SYMBOL=ISO_TS",
        help="Symbol and honest entry_ts, e.g. GLD=2026-07-14T00:00:00+00:00 "
        "(repeatable).",
    )
    ap.add_argument(
        "--ledger-path",
        type=Path,
        default=DEFAULT_LEDGER,
        help=f"Ledger state JSON (default: {DEFAULT_LEDGER})",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Perform the write (default: dry-run).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an EXISTING entry_ts.",
    )
    args = ap.parse_args(argv)

    # Parse and validate requested changes up front.
    changes: dict[str, str] = {}
    now_utc = datetime.now(tz=timezone.utc)
    for item in args.sets:
        if "=" not in item:
            print(f"[ERROR] --set expects SYMBOL=ISO_TS, got: {item!r}")
            return 2
        sym, raw_ts = item.split("=", 1)
        sym = sym.strip().upper()
        try:
            dt = _parse_iso_utc(raw_ts.strip())
        except ValueError as exc:
            print(f"[ERROR] {sym}: {exc}")
            return 2
        if dt > now_utc:
            print(f"[ERROR] {sym}: entry_ts {dt.isoformat()} lies in the future")
            return 2
        changes[sym] = dt.isoformat()

    ledger_path = args.ledger_path
    if not ledger_path.exists():
        print(f"[ERROR] ledger file not found: {ledger_path} — refusing to create one")
        return 2

    # Backup-fallback guard (review F-senior-1): load_ledger_state silently
    # falls back to an OLDER .1/.2/.3 backup when the MAIN file is corrupt —
    # a write tool must never persist that older state as the new truth.
    # Pre-validate the main file itself; abort on any parse problem.
    try:
        _main_raw = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"[ERROR] main ledger file is unreadable/corrupt ({exc}) — refusing "
            "to write (a save now could silently promote an older backup state)"
        )
        return 2
    if not isinstance(_main_raw, dict) or not isinstance(
        _main_raw.get("positions"), dict
    ):
        print("[ERROR] main ledger file has no positions dict — refusing to write")
        return 2

    try:
        state = load_ledger_state(ledger_path, start_capital=0.0)
    except Exception as exc:  # LedgerCorruptionError etc. (review F-senior-2)
        print(f"[ERROR] ledger load failed: {exc} — refusing to write")
        return 2
    positions = state.get("positions") or {}
    if not positions:
        print("[ERROR] ledger has no positions — nothing to backfill")
        return 2

    # Validate every requested symbol BEFORE touching anything (all-or-nothing).
    problems: list[str] = []
    for sym, ts in changes.items():
        pos = positions.get(sym)
        if pos is None:
            problems.append(f"{sym}: not in ledger (have: {sorted(positions)})")
            continue
        if not float(pos.get("qty", 0)):
            problems.append(f"{sym}: qty is 0 — no open position to stamp")
            continue
        existing = pos.get("entry_ts")
        if existing and not args.force:
            problems.append(
                f"{sym}: already has entry_ts={existing!r} — use --force to overwrite"
            )
    if problems:
        for p in problems:
            print(f"[ERROR] {p}")
        return 2

    print(f"[START] backfill_position_entry_ts ledger={ledger_path}")
    for sym, ts in changes.items():
        pos = positions[sym]
        print(
            f"[PLAN] {sym}: qty={pos.get('qty')} avg_price={pos.get('avg_price')} "
            f"entry_ts {pos.get('entry_ts')!r} -> {ts!r}"
        )

    if not args.apply:
        print("[SKIP] dry-run (no write). Re-run with --apply to persist.")
        return 0

    # Snapshot of everything the tool must NOT change (review F-senior-1:
    # verify the unchanged fields too, not only the stamped one).
    _pre_cash = state.get("cash")
    _pre_positions = {
        s: {k: v for k, v in p.items() if k != "entry_ts"} for s, p in positions.items()
    }

    for sym, ts in changes.items():
        positions[sym]["entry_ts"] = ts
    save_ledger_state(state, ledger_path)

    # Verify by re-reading through the canonical loader (round-trip proof):
    # stamped entry_ts present AND cash/qty/avg_price/hwm byte-identical.
    reloaded = load_ledger_state(ledger_path, start_capital=0.0)
    ok = True
    for sym, ts in changes.items():
        actual = (reloaded.get("positions") or {}).get(sym, {}).get("entry_ts")
        status = "OK" if actual == ts else "ERROR"
        if actual != ts:
            ok = False
        print(f"[{status}] {sym}: entry_ts={actual!r}")
    if reloaded.get("cash") != _pre_cash:
        ok = False
        print(f"[ERROR] cash drifted: {_pre_cash!r} -> {reloaded.get('cash')!r}")
    _post_positions = {
        s: {k: v for k, v in p.items() if k != "entry_ts"}
        for s, p in (reloaded.get("positions") or {}).items()
    }
    if _post_positions != _pre_positions:
        ok = False
        print("[ERROR] non-entry_ts position fields drifted during write")
    if not ok:
        print("[ERROR] post-write verification failed — inspect ledger + backups")
        return 1
    print(
        "[OK] backfill applied. audit: "
        + json.dumps({"changes": changes, "applied_utc": now_utc.isoformat()})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
