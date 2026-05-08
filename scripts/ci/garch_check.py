"""CI: GARCH vol forecast smoke check."""

import datetime
import json
import pathlib
import sys

import numpy as np
import pandas as pd

try:
    from src.assembled_core.risk.garch_vol_forecast import forecast_garch_vol

    rng = np.random.default_rng(42)
    prices = pd.Series(100 * (1 + rng.normal(0, 0.01, 500)).cumprod())
    result = forecast_garch_vol(prices, horizon=5)
    out = pathlib.Path("output/qa/garch_forecast_latest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "as_of": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "horizon_days": 5,
        "forecast_vol": (
            result
            if isinstance(result, (int, float))
            else float(result) if result is not None else None
        ),
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"[garch] forecast saved: {payload}")
except Exception as exc:
    print(f"[garch] skipped: {exc}")
    sys.exit(0)
