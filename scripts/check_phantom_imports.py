"""Phantom import checker — finds imports of archived/missing modules.

Scans all Python files in src/ and scripts/ for imports that reference modules
that no longer exist in the codebase (e.g. moved to archive/).

Usage:
    python scripts/check_phantom_imports.py
    python scripts/check_phantom_imports.py --fix-report output/phantom_imports.json
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SCAN_DIRS = ["src", "scripts"]

# Modules known to be archived or intentionally missing
# Restored from archive (now importable — remove from this set when fixed):
KNOWN_ARCHIVED = {
    # portfolio sizing alternatives — all in try/except, degrade gracefully
    "src.assembled_core.portfolio.covariance",
    "src.assembled_core.portfolio.risk_budgeting",
    "src.assembled_core.portfolio.hrp_sizing",
    "src.assembled_core.portfolio.bl_sizing",
    "src.assembled_core.portfolio.barbell_strategy",
    # risk overlays — in try/except, degrade gracefully
    "src.assembled_core.risk.disclosures_confirm",
    "src.assembled_core.risk.short_risk",
    "src.assembled_core.risk.evt_tail_var",
    "src.assembled_core.risk.factor_risk_model",
    # signals — in try/except, degrade gracefully
    "src.assembled_core.signals.short_signals",
    "src.assembled_core.signals.mean_reversion",
    "src.assembled_core.signals.earnings_integration",
    # ml — all in try/except; advanced features not in base install
    "src.assembled_core.ml.conformal",
    "src.assembled_core.ml.evt_models",
    "src.assembled_core.ml.feedback_loop",
    "src.assembled_core.ml.lime_explainer",
    "src.assembled_core.ml.news_ml_bridge",
    "src.assembled_core.ml.nlp_sentiment",
    "src.assembled_core.ml.quantile_models",
    "src.assembled_core.ml.retraining_scheduler",
    # ops
    "src.assembled_core.ops.shadow_recorder",
}


def _extract_imports(filepath: Path) -> list[str]:
    """Return all module names imported in a Python file."""
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []

    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
    return modules


def _check_importable(module: str) -> bool:
    """Return True if the module is importable."""
    try:
        importlib.import_module(module)
        return True
    except (ImportError, ModuleNotFoundError):
        return False
    except Exception:
        return True  # other errors (e.g. config missing) → module exists


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan for phantom imports.")
    parser.add_argument(
        "--fix-report", metavar="PATH", help="Write JSON report to PATH."
    )
    parser.add_argument(
        "--check-importable",
        action="store_true",
        help="Actually try to import each suspect module (slower, more accurate).",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))

    findings: list[dict] = []
    scanned_files = 0

    for scan_dir in SCAN_DIRS:
        base = REPO_ROOT / scan_dir
        if not base.exists():
            continue
        for pyfile in sorted(base.rglob("*.py")):
            scanned_files += 1
            imports = _extract_imports(pyfile)
            rel = pyfile.relative_to(REPO_ROOT)

            for mod in imports:
                # Only flag src.assembled_core.* imports
                if not mod.startswith("src.assembled_core."):
                    continue

                is_phantom = mod in KNOWN_ARCHIVED
                if not is_phantom and args.check_importable:
                    is_phantom = not _check_importable(mod)

                if is_phantom:
                    findings.append(
                        {
                            "file": str(rel),
                            "module": mod,
                            "in_known_archived": mod in KNOWN_ARCHIVED,
                        }
                    )

    # Deduplicate and sort
    seen: set[tuple] = set()
    unique: list[dict] = []
    for f in findings:
        key = (f["file"], f["module"])
        if key not in seen:
            seen.add(key)
            unique.append(f)
    unique.sort(key=lambda x: (x["module"], x["file"]))

    # Print report
    print(f"\nScanned {scanned_files} files in {SCAN_DIRS}")
    print(f"Found {len(unique)} phantom import references\n")

    by_module: dict[str, list[str]] = {}
    for f in unique:
        by_module.setdefault(f["module"], []).append(f["file"])

    for mod, files in sorted(by_module.items()):
        tag = "[ARCHIVED]" if mod in KNOWN_ARCHIVED else "[MISSING]"
        print(f"  {tag} {mod}")
        for fp in files:
            print(f"           -> {fp}")

    if args.fix_report:
        out = Path(args.fix_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(unique, indent=2), encoding="utf-8")
        print(f"\nReport written to {args.fix_report}")

    if unique:
        print(
            "\nRecommendation per module:"
            "\n  A) Restore from archive/ if the feature is needed"
            "\n  B) Remove the import if the feature is deprecated"
            "\n  C) Wrap in try/except ImportError with clear [WARN] log"
        )
        sys.exit(1)
    else:
        print("All imports OK.")
        sys.exit(0)


if __name__ == "__main__":
    main()
