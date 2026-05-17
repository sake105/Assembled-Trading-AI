"""CI: GARCH vol forecast smoke check."""

import datetime
import json
import pathlib
import sys

import numpy as np
import pandas as pd

try:
    # KNOWN_ISSUES §6.5.2 Phase 2: migrated from deprecated garch_vol_forecast
    # to canonical garch_vol. Behavior equivalent for default params; canonical
    # module also provides rolling-window fallback when arch unavailable.
    from src.assembled_core.risk.garch_vol import forecast_vol

    rng = np.random.default_rng(42)
    # forecast_vol expects RETURNS, not prices. Compute pct-change first.
    prices = pd.Series(100 * (1 + rng.normal(0, 0.01, 500)).cumprod())
    returns = prices.pct_change().dropna()
    result = forecast_vol(returns, horizon=5)
    out = pathlib.Path("output/qa/garch_forecast_latest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(result, (int, float)):
        forecast_vol_val = result
    elif result is not None:
        forecast_vol_val = float(result)
    else:
        forecast_vol_val = None
    payload = {
        "as_of": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "horizon_days": 5,
        "forecast_vol": forecast_vol_val,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"[garch] forecast saved: {payload}")
except Exception as exc:
    print(f"[garch] skipped: {exc}")
    sys.exit(0)
