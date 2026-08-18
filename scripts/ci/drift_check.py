"""CI: Feature drift and leakage smoke check.

Exit codes:
  0  happy path (module imports, sanity passes)
  1  real regression: import broken or the smoke check raised (FAIL)

A blanket ``except Exception -> sys.exit(0)`` was previously masking genuine
import/runtime regressions as a green CI pass (null enforcement).

FIX 2026-08-17 (ci-debugger F1, Klasse Audit-Plan 3.2): das Script lief in CI
als ``python scripts/ci/drift_check.py`` OHNE sys.path-Anker — sys.path[0] =
scripts/ci, ``src`` nie importierbar, und der ImportError wurde als
"optional dependency missing" mit Exit 0 maskiert: der Step war seit jeher
ein stiller No-op. Jetzt: _REPO-Anker (wie leakage_detector_selftest) UND
ImportError von ``src.*`` ist ein FAIL, kein SKIP — das Kernpaket ist keine
optionale Dependency.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    try:
        from src.assembled_core.qa.drift_detection import compute_psi  # noqa: F401

        print("[drift] compute_psi available — endpoint handles full check")
    except (ImportError, ModuleNotFoundError) as exc:
        print(
            f"[drift] FAIL — core package not importable (was silently "
            f"masked as SKIP before 2026-08-17): {exc!r}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"[drift] FAIL — smoke check raised: {exc!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
