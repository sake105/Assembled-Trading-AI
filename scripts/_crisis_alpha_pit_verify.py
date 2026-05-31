"""One-shot script: verify market_stress PIT fix for 2022-03-07."""

import logging
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(message)s")

from src.assembled_core.data.prices_ingest import load_eod_prices
from src.assembled_core.config.policy_loader import load_policy
from src.assembled_core.risk.market_stress import compute_market_stress

prices = load_eod_prices(price_file="output/backtest_crisis_test.parquet")
prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
mask = (prices["timestamp"] >= "2022-01-01") & (prices["timestamp"] <= "2023-12-31")
prices = prices[mask].reset_index(drop=True)
syms = prices["symbol"].nunique()
dr0 = prices["timestamp"].min().date()
dr1 = prices["timestamp"].max().date()
print(f"Prices: {len(prices)} rows, {syms} symbols, {dr0} to {dr1}")

policy = load_policy()
as_of = pd.Timestamp("2022-03-07", tz="UTC")

# WITHOUT fix: full 2-year dataset (tail = Dec 2023)
ms_no = compute_market_stress(prices, policy)
vz = ms_no["details"]["vol_z"]
dd = ms_no["details"]["min_dd"]
print(
    f"\nWITHOUT PIT fix (tail=2023-12): stress_ok={ms_no['stress_ok']}  vol_z={vz:.3f}  min_dd={dd:.3f}"
)

# WITH fix: slice to as_of
prices_pit = prices[prices["timestamp"] <= as_of]
ms_yes = compute_market_stress(prices_pit, policy)
vz2 = ms_yes["details"]["vol_z"]
dd2 = ms_yes["details"]["min_dd"]
npit = len(prices_pit)
tmax = prices_pit["timestamp"].max().date()
print(
    f"WITH PIT fix (as_of 2022-03-07): stress_ok={ms_yes['stress_ok']}  vol_z={vz2:.3f}  min_dd={dd2:.3f}"
)
print(f"  PIT slice: {npit} rows up to {tmax}")

print("\nExpected: WITH PIT fix stress_ok=True (Ukraine invasion vol spike)")
