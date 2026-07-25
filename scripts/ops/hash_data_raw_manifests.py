"""Integrity sidecars for data/raw pulls (GESAMTBEWERTUNG Nice-to-have, 2026-07-25).

The EDGAR/fundamentals pulls under data/raw/ carry strong PIT manifests but
NO file hashes — bitrot or a partial overwrite would be undetectable (audit
finding §2.1). This tool writes one ``manifest_integrity.json`` sidecar per
pull directory (sha256 + size per data file) and can verify against it.

The sidecar lives NEXT to the data (data/raw is gitignored — integrity
travels with the pull, not with git). Deterministic output (sorted paths).

Verify semantics (deliberate asymmetry): a missing pull dir is [SKIP]/rc 0
in write mode but [MISS]/rc 1 in --verify mode — for an integrity checker a
vanished directory IS a finding. On a fresh machine without data/raw, run
the pull scripts first.

Usage:
    python scripts/ops/hash_data_raw_manifests.py            # write/update sidecars
    python scripts/ops/hash_data_raw_manifests.py --verify   # check, exit 1 on mismatch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Pull directories that carry a manifest and deserve an integrity sidecar.
PULL_DIRS = [
    ROOT / "data" / "raw" / "fundamentals",
    ROOT / "data" / "raw" / "insider_congress",
]

# Data files worth protecting (not logs/readmes/the sidecar itself).
DATA_SUFFIXES = {".parquet", ".json", ".jsonl", ".csv"}
SIDECAR_NAME = "manifest_integrity.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _inventory(pull_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(pull_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in DATA_SUFFIXES:
            continue
        if p.name == SIDECAR_NAME:
            continue
        rel = p.relative_to(pull_dir).as_posix()
        out[rel] = {"sha256": _sha256(p), "size": p.stat().st_size}
    return out


def write_sidecars() -> int:
    for pull_dir in PULL_DIRS:
        if not pull_dir.exists():
            print(f"[SKIP] {pull_dir} (missing)")
            continue
        inv = _inventory(pull_dir)
        sidecar = pull_dir / SIDECAR_NAME
        payload = {
            "schema": "data_raw_integrity.v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "n_files": len(inv),
            "files": inv,
        }
        tmp = sidecar.with_name(sidecar.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(sidecar)
        total_mb = sum(v["size"] for v in inv.values()) / 1e6
        print(
            f"[OK] {pull_dir.name}: {len(inv)} files, {total_mb:.1f} MB -> {sidecar.name}"
        )
    return 0


def verify_sidecars() -> int:
    rc = 0
    for pull_dir in PULL_DIRS:
        sidecar = pull_dir / SIDECAR_NAME
        if not sidecar.exists():
            print(f"[MISS] {pull_dir.name}: no sidecar — run without --verify first")
            rc = 1
            continue
        try:
            recorded = json.loads(sidecar.read_text(encoding="utf-8"))["files"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            # A corrupt sidecar is the checker's own core scenario — report
            # it as a clean FAIL, not a traceback (still rc 1, fail-closed).
            print(f"[FAIL] {pull_dir.name}: sidecar corrupt/unreadable ({exc})")
            rc = 1
            continue
        current = _inventory(pull_dir)
        missing = sorted(set(recorded) - set(current))
        added = sorted(set(current) - set(recorded))
        changed = sorted(
            k
            for k in set(recorded) & set(current)
            if recorded[k]["sha256"] != current[k]["sha256"]
        )
        if missing or added or changed:
            rc = 1
            print(
                f"[FAIL] {pull_dir.name}: missing={len(missing)} added={len(added)} "
                f"changed={len(changed)}"
            )
            for k in (missing + changed)[:10]:
                print(f"       ! {k}")
        else:
            print(f"[OK] {pull_dir.name}: {len(current)} files verified")
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    return verify_sidecars() if args.verify else write_sidecars()


if __name__ == "__main__":
    sys.exit(main())
