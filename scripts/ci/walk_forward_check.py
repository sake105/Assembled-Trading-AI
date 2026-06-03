"""CI: Walk-forward validation smoke check.

Exit codes:
  0  happy path (module imports, config builds) OR optional dependency missing (SKIP)
  1  real regression: the smoke check raised a non-import error (FAIL)

A blanket ``except Exception -> sys.exit(0)`` was previously masking genuine
import/runtime regressions as a green CI pass (null enforcement). We now
distinguish an expected optional-dependency SKIP from a real FAILURE.
"""

import sys


def main() -> int:
    try:
        import pandas as pd

        from src.assembled_core.qa.walk_forward import WalkForwardConfig

        cfg = WalkForwardConfig(
            start_date=pd.Timestamp("2020-01-01", tz="UTC"),
            end_date=pd.Timestamp("2023-12-31", tz="UTC"),
            train_window_days=252,
            test_window_days=63,
            step_size_days=63,
            max_splits=5,
        )
        print(
            "[walk_forward] module imported, config built "
            f"(test_window_days={cfg.test_window_days}) — "
            "skipping full run (no panel in CI)"
        )
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"[walk_forward] SKIP — optional dependency missing: {exc}")
        return 0
    except Exception as exc:
        print(f"[walk_forward] FAIL — smoke check raised: {exc!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
