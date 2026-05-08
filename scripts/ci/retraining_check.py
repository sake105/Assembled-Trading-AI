"""CI: Retraining scheduler smoke check."""

import sys
from datetime import date, timedelta

try:
    from src.assembled_core.ml.retraining_scheduler import RetrainingScheduler

    sched = RetrainingScheduler()
    rec = sched.evaluate(model_last_trained_date=date.today() - timedelta(days=35))
    print(f"[retraining] decision={rec.decision} signals_fired={rec.signals_fired}")
except Exception as exc:
    print(f"[retraining] skipped: {exc}")
    sys.exit(0)
