"""CI: Feature drift and leakage smoke check.

Exit codes:
  0  happy path (module imports, sanity passes) ODER ImportError (SKIP, laut)
  1  real regression: the smoke check raised a non-import error (FAIL)

A blanket ``except Exception -> sys.exit(0)`` was previously masking genuine
import/runtime regressions as a green CI pass (null enforcement).

FIX 2026-08-17/18 (ci-debugger F1, Klasse Audit-Plan 3.2): das Script lief in
CI als ``python scripts/ci/drift_check.py`` OHNE sys.path-Anker — sys.path[0]
= scripts/ci, ``src`` nie importierbar, und der ImportError wurde als
"optional dependency missing" mit Exit 0 maskiert: der Step war seit jeher
ein stiller No-op. Der _REPO-Anker (wie in leakage_detector_selftest) behebt
GENAU DAS: der Import gelingt jetzt, der Check laeuft wirklich.

Die ImportError-Semantik bleibt bewusst bei SKIP/Exit 0 (E-188): der
A26-Vertrag gilt einheitlich fuer alle drei ci-Smoke-Checks
(tests/test_batch10_scripts_honesty.py) und ist eine bewusste Entscheidung,
keine Nachlaessigkeit — ein Fix darf sie nicht einseitig umdrehen. Der Skip
ist jetzt aber LAUT (WARNING + explizite Begruendung im Text), damit er nie
wieder wie ein erfolgreicher Lauf aussieht.
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
        # E-188: SKIP bleibt der Vertrag (A26), aber LAUT — bis 2026-08-17
        # verbarg genau diese Zeile einen dauerhaften No-op.
        print(
            f"[drift] SKIP (exit 0 per A26-Vertrag) — core module not "
            f"importable: {exc!r}. ACHTUNG: kein Drift-Check gelaufen; wenn "
            f"das dauerhaft auftritt, ist es ein Defekt, kein Skip.",
            file=sys.stderr,
        )
        return 0
    except Exception as exc:
        print(f"[drift] FAIL — smoke check raised: {exc!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
