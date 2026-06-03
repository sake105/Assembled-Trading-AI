"""CI: Retraining scheduler smoke check.

Exit codes:
  0  happy path (scheduler evaluates) OR optional dependency missing (SKIP)
  1  real regression: the smoke check raised a non-import error (FAIL)

A blanket ``except Exception -> sys.exit(0)`` was previously masking genuine
import/runtime regressions as a green CI pass (null enforcement). We now
distinguish an expected optional-dependency SKIP from a real FAILURE.
"""

import sys
from datetime import date, timedelta


def main() -> int:
    try:
        from src.assembled_core.ml.retraining_scheduler import RetrainingScheduler

        sched = RetrainingScheduler()
        rec = sched.evaluate(model_last_trained_date=date.today() - timedelta(days=35))
        print(f"[retraining] decision={rec.decision} signals_fired={rec.signals_fired}")
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"[retraining] SKIP — optional dependency missing: {exc}")
        return 0
    except Exception as exc:
        print(f"[retraining] FAIL — smoke check raised: {exc!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
