"""
generate_system_map.py — AST-Scan + API-Grep → system_map.json

Usage:
  python scripts/architecture/generate_system_map.py [OPTIONS]

Options:
  --output PATH     Output JSON  (default: docs/architecture/system_map/data/system_map.json)
  --overrides PATH  Overrides YAML (default: docs/architecture/system_map/data/system_map_overrides.yaml)
  --prev PATH       Previous JSON for changelog delta
  --no-diff         Skip changelog update
  --report          Print circular imports + god-module report to stdout
  --dry-run         Print JSON to stdout, do not write file

Exit codes: 0 OK, 1 schema error, 2 generator error
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ── Root detection ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_OUTPUT = REPO_ROOT / "docs/architecture/system_map/data/system_map.json"
DEFAULT_OVERRIDES = (
    REPO_ROOT / "docs/architecture/system_map/data/system_map_overrides.yaml"
)
DEFAULT_CHANGELOG = REPO_ROOT / "docs/architecture/system_map/data/changelog.json"

# ── Known external API markers ──────────────────────────────────────────────
API_MARKERS: dict[str, str] = {
    "yfinance": "api:yfinance",
    "fredapi": "api:fredapi",
    "alpaca_trade_api": "api:alpaca",
    "alpaca": "api:alpaca",
    "ib_insync": "api:ibkr",
    "finnhub": "api:finnhub",
    "sec_edgar_downloader": "api:sec_edgar",
    "edgar": "api:sec_edgar",
    "feedparser": "api:rss_feeds",
    "gdelt": "api:gdelt",
    "newsapi": "api:newsapi",
    "polygon": "api:polygon",
    "alpha_vantage": "api:alpha_vantage",
    "bls": "api:bls",
    "worldbank": "api:worldbank",
    "websocket": "api:websocket",
}

API_URL_PATTERN = re.compile(r'requests\.(get|post|put|delete)\s*\(\s*["\']https?://')


# ── Helpers ─────────────────────────────────────────────────────────────────


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def loc_count(source: str) -> int:
    return sum(
        1
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def module_id(path: Path) -> str:
    """Convert src/assembled_core/features/momentum.py → module:features.momentum"""
    try:
        rel = path.relative_to(REPO_ROOT / "src" / "assembled_core")
        parts = list(rel.parts)
        # Build dotted path from all parts, replacing __init__ with parent folder
        name_parts = [p.replace(".py", "") for p in parts]
        if not name_parts:
            return "module:root.__init__"
        if name_parts[-1] == "__init__":
            name_parts = name_parts[:-1]
        if not name_parts:
            return "module:root.__init__"
        domain = name_parts[0]
        if len(name_parts) == 1:
            return f"module:{domain}.__init__"
        # Join remaining parts with dot to make unique nested IDs
        sub = ".".join(name_parts[1:])
        return f"module:{domain}.{sub}"
    except ValueError:
        pass
    # scripts/
    try:
        path.relative_to(REPO_ROOT / "scripts")
        return f"entry_point:{path.stem}"
    except ValueError:
        pass
    return f"module:{path.stem}"


def domain_id(domain_name: str) -> str:
    return f"domain:{domain_name}"


def build_galaxy_map(overrides: dict) -> tuple[list[dict], dict[str, str]]:
    """Return (galaxy_nodes, domain_name → galaxy_id). Empty if no galaxies defined."""
    galaxies = overrides.get("galaxies", []) if overrides else []
    if not galaxies:
        return [], {}
    nodes: list[dict] = []
    domain_to_galaxy: dict[str, str] = {}
    for g in galaxies:
        nodes.append(
            {
                "id": g["id"],
                "type": "galaxy",
                "label": g.get("label", g["id"]),
                "parent": None,
                "purpose": g.get("purpose", ""),
                "status": "gray",
                "tests_count": 0,
                "loc": 0,
                "orphan": False,
                "in_cycle": False,
                "duplicate_group": None,
            }
        )
        for d in g.get("domains", []):
            domain_to_galaxy[d] = g["id"]
    return nodes, domain_to_galaxy


def parse_module(path: Path) -> dict[str, Any]:
    """AST-scan a Python file, return node data dict."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}

    node_data: dict[str, Any] = {
        "loc": loc_count(source),
        "functions": [],
        "imports": [],  # internal: list of module_ids
        "api_calls": [],  # internal: list of api_ids
        "in_cycle": False,
        "raises_not_implemented": False,
    }

    # Docstring
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return node_data

    docstring = ast.get_docstring(tree)
    if docstring:
        node_data["purpose"] = " ".join(docstring.split()[:30])

    # Determine this module's package path so that relative imports
    # (from .baseline import ...) can be resolved to absolute ones.
    self_pkg_parts: list[str] = []
    try:
        rel = path.relative_to(REPO_ROOT / "src" / "assembled_core")
        parts = [p.replace(".py", "") for p in rel.parts]
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        elif parts:
            parts = parts[:-1]
        self_pkg_parts = ["assembled_core"] + parts
    except ValueError:
        self_pkg_parts = []

    # Walk AST
    annotated_args = 0
    total_args = 0
    for node in ast.walk(tree):
        # Functions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                node_data["functions"].append(node.name)
            for arg in node.args.args + node.args.kwonlyargs:
                total_args += 1
                if arg.annotation is not None:
                    annotated_args += 1

        # Imports
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom):
                level = getattr(node, "level", 0) or 0
                base = node.module or ""
                if level and self_pkg_parts:
                    # Resolve relative import to absolute dotted path.
                    if level > len(self_pkg_parts):
                        anchor: list[str] = []
                    else:
                        anchor = self_pkg_parts[: len(self_pkg_parts) - level + 1]
                    resolved = ".".join(
                        [p for p in anchor if p] + ([base] if base else [])
                    )
                    if resolved:
                        # Emit one import per imported symbol so we can resolve
                        # either to a sub-module (.models → events.news.models)
                        # or fall back to the package.
                        for alias in node.names:
                            sym = alias.name
                            if sym and sym != "*":
                                names.append(f"{resolved}.{sym}")
                        if not names:
                            names = [resolved]
                    else:
                        names = [base] if base else []
                else:
                    # Absolute import: "from assembled_core.pipeline import backtest_legacy"
                    # → emit both the base and base.alias so deeper modules get resolved.
                    if base:
                        names.append(base)
                        for alias in node.names:
                            sym = alias.name
                            if sym and sym != "*":
                                names.append(f"{base}.{sym}")
            else:
                names = [alias.name for alias in node.names]
            for name in names:
                if not name:
                    continue
                top = name.split(".")[0]
                if top in API_MARKERS:
                    api = API_MARKERS[top]
                    if api not in node_data["api_calls"]:
                        node_data["api_calls"].append(api)
                elif name.startswith("assembled_core.") or name.startswith(
                    "src.assembled_core."
                ):
                    node_data["imports"].append(name)

        # NotImplementedError detection
        elif isinstance(node, ast.Raise):
            if node.exc and isinstance(node.exc, ast.Call):
                func = node.exc.func
                if isinstance(func, ast.Name) and func.id == "NotImplementedError":
                    node_data["raises_not_implemented"] = True
            elif node.exc and isinstance(node.exc, ast.Name):
                if node.exc.id == "NotImplementedError":
                    node_data["raises_not_implemented"] = True

    # URL-based API calls
    if API_URL_PATTERN.search(source):
        if "api:http_generic" not in node_data["api_calls"]:
            node_data["api_calls"].append("api:http_generic")

    node_data["type_annotation_ratio"] = round(annotated_args / max(total_args, 1), 2)
    node_data["complexity_score"] = min(
        5, max(1, math.ceil(len(node_data["functions"]) / 10))
    )
    return node_data


def resolve_import_to_id(import_name: str) -> str | None:
    """Resolve an import like assembled_core.events.news.pipeline to a module id.

    Walks from longest-prefix to shortest and returns the first prefix that
    points to a real .py file or package __init__.py under src/assembled_core/.
    This keeps sub-package modules reachable instead of collapsing every
    deep import onto the 2-level domain node.
    """
    parts = import_name.replace("assembled_core.", "").replace("src.", "").split(".")
    parts = [p for p in parts if p]
    if not parts:
        return None
    core_root = REPO_ROOT / "src" / "assembled_core"

    def id_for(prefix: list[str]) -> str:
        domain = prefix[0]
        if len(prefix) == 1:
            return f"module:{domain}.__init__"
        return f"module:{domain}.{'.'.join(prefix[1:])}"

    # Walk from longest prefix down: prefer deepest real match.
    for depth in range(len(parts), 0, -1):
        prefix = parts[:depth]
        candidate_file = core_root.joinpath(*prefix[:-1], prefix[-1] + ".py")
        candidate_pkg = core_root.joinpath(*prefix, "__init__.py")
        if candidate_file.exists() or candidate_pkg.exists():
            return id_for(prefix)

    # Fallback: 2-level domain grouping, matches historic behaviour.
    if len(parts) >= 2:
        return f"module:{parts[0]}.{parts[1]}"
    return f"module:{parts[0]}.__init__"


def count_tests(py_path: Path, tests_root: Path) -> int:
    """Count test functions for a given module path."""
    stem = py_path.stem
    if stem == "__init__":
        return 0
    pattern = f"test_{stem}*.py"
    count = 0
    for test_file in tests_root.rglob(pattern):
        try:
            src = test_file.read_text(encoding="utf-8", errors="replace")
            count += src.count("\ndef test_")
        except OSError:
            pass
    return count


# ── Cycle Detection ─────────────────────────────────────────────────────────


def detect_cycles(edges: list[dict]) -> set[str]:
    """DFS-based cycle detection. Returns set of node IDs involved in cycles."""
    graph: dict[str, list[str]] = {}
    for e in edges:
        if e.get("kind") == "import":
            graph.setdefault(e["source"], []).append(e["target"])

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {}
    cyclic: set[str] = set()

    def dfs(node: str) -> bool:
        color[node] = GRAY
        for nb in graph.get(node, []):
            c = color.get(nb, WHITE)
            if c == GRAY:
                cyclic.add(node)
                cyclic.add(nb)
                return True
            if c == WHITE and dfs(nb):
                cyclic.add(node)
        color[node] = BLACK
        return False

    import signal

    class Timeout(Exception):
        pass

    def _handler(signum, frame):
        raise Timeout()

    # Timeout: skip if DFS takes too long (Windows: signal not reliable, skip)
    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(5)
    except AttributeError:
        pass  # Windows

    try:
        for node in list(graph.keys()):
            if color.get(node, WHITE) == WHITE:
                dfs(node)
    except Exception:
        pass
    finally:
        try:
            signal.alarm(0)
        except AttributeError:
            pass

    return cyclic


# ── Overrides ───────────────────────────────────────────────────────────────


def load_overrides(path: Path) -> dict:
    if not path.exists() or not HAS_YAML:
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[WARN] Could not load overrides: {e}", file=sys.stderr)
        return {}


def apply_overrides(nodes: list[dict], overrides: dict) -> None:
    node_map = {n["id"]: n for n in nodes}

    for ov in overrides.get("status_overrides", []):
        if ov["id"] in node_map:
            node_map[ov["id"]]["status"] = ov["status"]
            node_map[ov["id"]]["status_reason"] = ov.get("reason", "")

    for grp in overrides.get("duplicate_groups", []):
        for member_id in grp.get("members", []):
            if member_id in node_map:
                node_map[member_id]["duplicate_group"] = grp["id"]

    for ov in overrides.get("orphan_overrides", []):
        if ov["id"] in node_map:
            node_map[ov["id"]]["orphan"] = ov.get("orphan", True)


# ── Diff / Changelog ────────────────────────────────────────────────────────


def build_changelog_entry(
    prev_path: Path | None, new_nodes: list[dict], new_edges: list[dict]
) -> dict | None:
    if not prev_path or not prev_path.exists():
        return None
    try:
        prev = json.loads(prev_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    prev_ids = {n["id"] for n in prev.get("nodes", [])}
    new_ids = {n["id"] for n in new_nodes}
    prev_status = {n["id"]: n.get("status") for n in prev.get("nodes", [])}
    new_status = {n["id"]: n.get("status") for n in new_nodes}

    added = sorted(new_ids - prev_ids)
    removed = sorted(prev_ids - new_ids)
    changed = sorted(
        nid for nid in new_ids & prev_ids if prev_status.get(nid) != new_status.get(nid)
    )

    return {
        "date": datetime.now(timezone.utc).isoformat(),
        "added": added,
        "removed": removed,
        "status_changed": [
            {"id": nid, "from": prev_status.get(nid), "to": new_status.get(nid)}
            for nid in changed
        ],
    }


def update_changelog(entry: dict, changelog_path: Path) -> None:
    try:
        data = (
            json.loads(changelog_path.read_text(encoding="utf-8"))
            if changelog_path.exists()
            else {"entries": []}
        )
        data["entries"].insert(0, entry)
        data["entries"] = data["entries"][:50]  # keep last 50
        changelog_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Changelog update failed: {e}", file=sys.stderr)


# ── Main Generator ───────────────────────────────────────────────────────────


def generate(args: argparse.Namespace) -> int:
    core_root = REPO_ROOT / "src" / "assembled_core"
    scripts_dir = REPO_ROOT / "scripts"
    wf_dir = REPO_ROOT / ".github" / "workflows"
    tests_root = REPO_ROOT / "tests"

    if not core_root.exists():
        print(f"[ERROR] src/assembled_core not found at {core_root}", file=sys.stderr)
        return 2

    nodes: list[dict] = []
    edges: list[dict] = []
    api_nodes_seen: set[str] = set()
    edge_counter = 0

    # Load overrides up-front so galaxy mapping can set domain.parent from the start.
    override_path = Path(args.overrides) if args.overrides else DEFAULT_OVERRIDES
    overrides = load_overrides(override_path)

    # ── 0. Emit galaxy compound nodes (if defined in overrides) ──
    galaxy_nodes, domain_to_galaxy = build_galaxy_map(overrides)
    nodes.extend(galaxy_nodes)

    # ── 1. Discover domains ──────────────────────────────────
    domains: list[str] = []
    unmapped_domains: list[str] = []
    for item in sorted(core_root.iterdir()):
        if item.is_dir() and not item.name.startswith("_"):
            domains.append(item.name)
            parent = domain_to_galaxy.get(item.name)
            if domain_to_galaxy and parent is None:
                unmapped_domains.append(item.name)
            nodes.append(
                {
                    "id": domain_id(item.name),
                    "type": "domain",
                    "label": item.name,
                    "parent": parent,
                    "status": "gray",
                    "tests_count": 0,
                    "loc": 0,
                    "orphan": False,
                    "in_cycle": False,
                    "duplicate_group": None,
                }
            )
    if unmapped_domains:
        print(
            f"[WARN] {len(unmapped_domains)} domain(s) not mapped to a galaxy: "
            f"{', '.join(unmapped_domains)}",
            file=sys.stderr,
        )

    # ── 2. Scan Python modules ───────────────────────────────
    py_files = sorted(core_root.rglob("*.py"))
    module_raw: dict[str, dict] = {}

    for py_path in py_files:
        mid = module_id(py_path)
        rel = py_path.as_posix().replace(REPO_ROOT.as_posix() + "/", "")
        parsed = parse_module(py_path)
        tc = count_tests(py_path, tests_root)

        # Status
        if parsed.get("raises_not_implemented"):
            status = "red"
        elif tc > 5:
            status = "green"
        elif tc >= 1:
            status = "yellow"
        else:
            status = "orange"

        # Determine domain
        try:
            rel_to_core = py_path.relative_to(core_root)
            domain_name = rel_to_core.parts[0] if rel_to_core.parts else ""
        except ValueError:
            domain_name = ""

        node = {
            "id": mid,
            "type": "module",
            "label": py_path.name,
            "parent": domain_id(domain_name) if domain_name else None,
            "purpose": parsed.get("purpose", ""),
            "status": status,
            "tests_count": tc,
            "loc": parsed.get("loc", 0),
            "fan_in": 0,
            "fan_out": 0,
            "complexity_score": parsed.get("complexity_score", 1),
            "type_annotation_ratio": parsed.get("type_annotation_ratio", 0.0),
            "functions": parsed.get("functions", [])[:40],
            "orphan": False,
            "in_cycle": False,
            "duplicate_group": None,
            "path": rel,
        }
        nodes.append(node)
        module_raw[mid] = parsed

        # Import edges
        for imp in parsed.get("imports", []):
            target_id = resolve_import_to_id(imp)
            if target_id and target_id != mid:
                edge_counter += 1
                edges.append(
                    {
                        "id": f"e{edge_counter}:{mid}→{target_id}",
                        "source": mid,
                        "target": target_id,
                        "kind": "import",
                        "weight": 1,
                        "circular": False,
                    }
                )

        # API-call edges
        for api_id in parsed.get("api_calls", []):
            if api_id not in api_nodes_seen:
                api_nodes_seen.add(api_id)
                nodes.append(
                    {
                        "id": api_id,
                        "type": "external_api",
                        "label": api_id.replace("api:", ""),
                        "parent": None,
                        "status": "gray",
                        "tests_count": 0,
                        "loc": 0,
                        "orphan": False,
                        "in_cycle": False,
                        "duplicate_group": None,
                    }
                )
            edge_counter += 1
            edges.append(
                {
                    "id": f"e{edge_counter}:{mid}→{api_id}",
                    "source": mid,
                    "target": api_id,
                    "kind": "api_call",
                    "weight": 1,
                    "circular": False,
                }
            )

    # ── 3. Scripts (entry points) ────────────────────────────
    for py_path in sorted(scripts_dir.glob("run_*.py")):
        eid = f"entry_point:{py_path.stem}"
        rel = py_path.as_posix().replace(REPO_ROOT.as_posix() + "/", "")
        nodes.append(
            {
                "id": eid,
                "type": "entry_point",
                "label": py_path.name,
                "parent": None,
                "status": "gray",
                "tests_count": 0,
                "loc": (
                    loc_count(py_path.read_text(encoding="utf-8", errors="replace"))
                    if py_path.exists()
                    else 0
                ),
                "orphan": False,
                "in_cycle": False,
                "duplicate_group": None,
                "path": rel,
            }
        )
        # Scan entry-point imports so connected modules are not orphaned.
        parsed_ep = parse_module(py_path)
        for imp in parsed_ep.get("imports", []):
            target_id = resolve_import_to_id(imp)
            if target_id and target_id != eid:
                edge_counter += 1
                edges.append(
                    {
                        "id": f"e{edge_counter}:{eid}→{target_id}",
                        "source": eid,
                        "target": target_id,
                        "kind": "import",
                        "weight": 1,
                        "circular": False,
                    }
                )

    # ── 4. Workflows ─────────────────────────────────────────
    if wf_dir.exists():
        for wf in sorted(wf_dir.glob("*.yml")):
            wid = f"workflow:{wf.stem}"
            nodes.append(
                {
                    "id": wid,
                    "type": "workflow",
                    "label": wf.name,
                    "parent": None,
                    "status": "gray",
                    "tests_count": 0,
                    "loc": 0,
                    "orphan": False,
                    "in_cycle": False,
                    "duplicate_group": None,
                    "path": wf.as_posix().replace(REPO_ROOT.as_posix() + "/", ""),
                }
            )

    # ── 5. Fan-in / fan-out ───────────────────────────────────
    node_map = {n["id"]: n for n in nodes}
    for e in edges:
        if e["kind"] == "import":
            if e["source"] in node_map:
                node_map[e["source"]]["fan_out"] = (
                    node_map[e["source"]].get("fan_out", 0) + 1
                )
            if e["target"] in node_map:
                node_map[e["target"]]["fan_in"] = (
                    node_map[e["target"]].get("fan_in", 0) + 1
                )

    # ── 6. Orphan detection ───────────────────────────────────
    has_connection: set[str] = set()
    for e in edges:
        has_connection.add(e["source"])
        has_connection.add(e["target"])
    for n in nodes:
        if n["type"] == "module" and n["id"] not in has_connection:
            n["orphan"] = True

    # ── 7. Cycle detection ────────────────────────────────────
    cyclic = detect_cycles(edges)
    for e in edges:
        if e["source"] in cyclic and e["target"] in cyclic and e["kind"] == "import":
            e["circular"] = True
    for n in nodes:
        if n["id"] in cyclic:
            n["in_cycle"] = True

    # ── 8. Overrides (loaded earlier for galaxy map, apply now) ──
    apply_overrides(nodes, overrides)

    # ── 9. Domain status rollup ───────────────────────────────
    STATUS_RANK = {"red": 0, "orange": 1, "yellow": 2, "green": 3, "gray": 4}
    domain_statuses: dict[str, list[str]] = {}
    for n in nodes:
        if n["type"] == "module" and n.get("parent"):
            d = n["parent"]
            domain_statuses.setdefault(d, []).append(n["status"])
    for did, statuses in domain_statuses.items():
        if did in node_map:
            worst = min(statuses, key=lambda s: STATUS_RANK.get(s, 4))
            node_map[did]["status"] = worst
            node_map[did]["tests_count"] = sum(
                node_map.get(n["id"], {}).get("tests_count", 0)
                for n in nodes
                if n.get("parent") == did and n["type"] == "module"
            )

    # ── 9a. Galaxy status rollup (worst-of-domains) ──────────
    galaxy_statuses: dict[str, list[str]] = {}
    for n in nodes:
        if n["type"] == "domain" and n.get("parent"):
            galaxy_statuses.setdefault(n["parent"], []).append(n.get("status", "gray"))
    for gid, statuses in galaxy_statuses.items():
        if gid in node_map:
            worst = min(statuses, key=lambda s: STATUS_RANK.get(s, 4))
            node_map[gid]["status"] = worst
            node_map[gid]["tests_count"] = sum(
                node_map[n["id"]]["tests_count"]
                for n in nodes
                if n.get("parent") == gid
                and n["type"] == "domain"
                and n["id"] in node_map
            )

    # ── 9b. Deduplicate nodes (keep first occurrence) ────────
    seen_ids: set[str] = set()
    deduped_nodes: list[dict] = []
    for n in nodes:
        if n["id"] not in seen_ids:
            seen_ids.add(n["id"])
            deduped_nodes.append(n)
    nodes = deduped_nodes
    node_map = {n["id"]: n for n in nodes}

    # ── 9c. Referential cleanup (must run before JSON is written) ─
    # Cytoscape throws on edges pointing to missing nodes and on
    # parent references to non-existent compounds. Filter those out.
    dropped_edges = 0
    cleaned_edges: list[dict] = []
    for e in edges:
        if e.get("source") in node_map and e.get("target") in node_map:
            cleaned_edges.append(e)
        else:
            dropped_edges += 1
    edges = cleaned_edges

    reparented = 0
    for n in nodes:
        p = n.get("parent")
        if p and p not in node_map:
            n["parent"] = None
            reparented += 1

    if dropped_edges or reparented:
        print(
            f"[CLEAN] Dropped {dropped_edges} dangling edges, "
            f"reparented {reparented} nodes with missing parent"
        )

    # ── 10. Meta ──────────────────────────────────────────────
    status_summary: dict[str, int] = {
        "green": 0,
        "yellow": 0,
        "orange": 0,
        "red": 0,
        "gray": 0,
    }
    for n in nodes:
        if n["type"] == "module":
            status_summary[n.get("status", "gray")] = (
                status_summary.get(n.get("status", "gray"), 0) + 1
            )

    meta = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator_version": "0.2.0",
        "source_commit": get_git_commit(),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "circular_import_count": len(cyclic),
        "honest_status_summary": status_summary,
    }

    output = {"meta": meta, "nodes": nodes, "edges": edges}

    # ── 11. Changelog ─────────────────────────────────────────
    if not args.no_diff:
        prev_path = Path(args.prev) if args.prev else None
        entry = build_changelog_entry(prev_path, nodes, edges)
        if entry:
            update_changelog(entry, DEFAULT_CHANGELOG)

    # ── 12. Report ────────────────────────────────────────────
    if args.report:
        print(f"\n[REPORT] {meta['node_count']} nodes, {meta['edge_count']} edges")
        print(f"[REPORT] Circular imports: {len(cyclic)} nodes in cycles")
        if cyclic:
            print("         " + ", ".join(sorted(cyclic)[:10]))
        god_modules = sorted(
            [n for n in nodes if n["type"] == "module"],
            key=lambda n: n.get("fan_in", 0),
            reverse=True,
        )[:5]
        print("[REPORT] Top fan-in (god modules):")
        for gm in god_modules:
            print(f"         {gm['id']:60s}  fan_in={gm.get('fan_in', 0)}")
        print(f"[REPORT] Status: {status_summary}")

    # ── 13. Write / dry-run ────────────────────────────────────
    if args.dry_run:
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 0

    out_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_str = json.dumps(output, indent=2, ensure_ascii=False)
    out_path.write_text(json_str, encoding="utf-8")

    # Also write JS-embedded version for file:// offline use
    js_path = out_path.parent / "system_map_data.js"
    js_path.write_text(f"window.SYSTEM_MAP_DATA = {json_str};\n", encoding="utf-8")

    print(f"[OK] Written {out_path}  ({len(nodes)} nodes, {len(edges)} edges)")
    print(f"[OK] Written {js_path}  (embedded JS for file:// offline use)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate system_map.json from codebase AST"
    )
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--overrides", default=None, help="Overrides YAML path")
    parser.add_argument("--prev", default=None, help="Previous JSON for changelog")
    parser.add_argument("--no-diff", action="store_true", help="Skip changelog update")
    parser.add_argument("--report", action="store_true", help="Print summary report")
    parser.add_argument("--dry-run", action="store_true", help="Print JSON to stdout")
    args = parser.parse_args()
    try:
        return generate(args)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
