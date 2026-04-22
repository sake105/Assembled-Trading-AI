"""
validate_system_map.py — Schema-Check + Orphan/Stale/Circular report

Usage:
  python scripts/architecture/validate_system_map.py [PATH]

Exit: 0 OK, 1 schema error
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
DEFAULT_MAP = REPO_ROOT / "docs/architecture/system_map/data/system_map.json"
STALE_DAYS  = 30

VALID_STATUSES = {"green", "yellow", "orange", "red", "gray"}
VALID_TYPES    = {"galaxy", "domain", "module", "external_api", "script", "workflow", "entry_point"}
VALID_KINDS    = {"import", "api_call", "data_flow", "trigger"}


def validate(path: Path) -> int:
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        return 1

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON: {e}", file=sys.stderr)
        return 1

    errors: list[str] = []
    warnings: list[str] = []

    # ── Meta ─────────────────────────────────────────────────
    meta = data.get("meta", {})
    for field in ("generated_at", "generator_version", "source_commit", "node_count", "edge_count"):
        if field not in meta:
            errors.append(f"meta.{field} missing")

    # Stale check
    if "generated_at" in meta:
        try:
            gen_dt = datetime.fromisoformat(meta["generated_at"].replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - gen_dt).days
            if age_days > STALE_DAYS:
                warnings.append(f"Map is {age_days} days old (threshold: {STALE_DAYS}d) — run generator")
        except Exception:
            warnings.append("Could not parse meta.generated_at")

    # ── Nodes ────────────────────────────────────────────────
    nodes = data.get("nodes", [])
    node_ids: set[str] = set()
    for i, n in enumerate(nodes):
        nid = n.get("id", "")
        if not nid:
            errors.append(f"nodes[{i}]: missing id")
            continue
        if nid in node_ids:
            errors.append(f"Duplicate node id: {nid}")
        node_ids.add(nid)

        if n.get("type") not in VALID_TYPES:
            errors.append(f"{nid}: invalid type '{n.get('type')}'")
        if n.get("status") not in VALID_STATUSES:
            errors.append(f"{nid}: invalid status '{n.get('status')}'")

    # ── Edges ────────────────────────────────────────────────
    edges = data.get("edges", [])
    edge_ids: set[str] = set()
    for i, e in enumerate(edges):
        eid = e.get("id", "")
        if not eid:
            errors.append(f"edges[{i}]: missing id")
        if eid in edge_ids:
            errors.append(f"Duplicate edge id: {eid}")
        edge_ids.add(eid)

        if e.get("kind") not in VALID_KINDS:
            errors.append(f"Edge {eid}: invalid kind '{e.get('kind')}'")
        if e.get("source") not in node_ids:
            warnings.append(f"Edge {eid}: source '{e.get('source')}' not in nodes")
        if e.get("target") not in node_ids:
            warnings.append(f"Edge {eid}: target '{e.get('target')}' not in nodes")

    # ── Orphan Report ────────────────────────────────────────
    orphans = [n for n in nodes if n.get("orphan") and n.get("type") == "module"]
    if orphans:
        warnings.append(f"{len(orphans)} orphan modules detected:")
        for o in orphans[:10]:
            warnings.append(f"  - {o['id']}")

    # ── Circular Import Report ───────────────────────────────
    circular_edges = [e for e in edges if e.get("circular")]
    circular_nodes = [n for n in nodes if n.get("in_cycle")]
    if circular_nodes:
        warnings.append(f"{len(circular_nodes)} nodes in circular imports:")
        for n in circular_nodes[:10]:
            warnings.append(f"  - {n['id']}")

    # ── God Module Report (top 5 fan-in) ─────────────────────
    god_modules = sorted(
        [n for n in nodes if n.get("type") == "module"],
        key=lambda n: n.get("fan_in", 0), reverse=True
    )[:5]
    if god_modules and god_modules[0].get("fan_in", 0) > 10:
        warnings.append("High fan-in modules (potential god modules):")
        for gm in god_modules:
            warnings.append(f"  - {gm['id']}  fan_in={gm.get('fan_in', 0)}")

    # ── Output ───────────────────────────────────────────────
    for w in warnings:
        sys.stdout.buffer.write((f"[WARN]  {w}\n").encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()
    for e in errors:
        sys.stderr.buffer.write((f"[ERROR] {e}\n").encode("utf-8", errors="replace"))
        sys.stderr.buffer.flush()

    if errors:
        print(f"\nValidation FAILED: {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1

    print(f"[OK] Validation passed — {len(nodes)} nodes, {len(edges)} edges, {len(warnings)} warning(s)")
    return 0


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MAP
    return validate(path)


if __name__ == "__main__":
    sys.exit(main())
