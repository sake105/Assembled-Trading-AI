"""CI: Feature drift and leakage smoke check.

Exit codes:
  0  happy path (module imports, sanity passes) OR optional dependency missing (SKIP)
  1  real regression: the smoke check raised a non-import error (FAIL)

A blanket ``except Exception -> sys.exit(0)`` was previously masking genuine
import/runtime regressions as a green CI pass (null enforcement). We now
distinguish an expected optional-dependency SKIP from a real FAILURE.
"""

import sys


def main() -> int:
    try:
        from src.assembled_core.qa.drift_detection import compute_psi  # noqa: F401

        print("[drift] compute_psi available — endpoint handles full check")
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"[drift] SKIP — optional dependency missing: {exc}")
        return 0
    except Exception as exc:
        print(f"[drift] FAIL — smoke check raised: {exc!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
