"""One-off: Reconstruct Cash + Mark-to-Market Equity from trades and prices."""

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

r = ROOT / "output/system_run/debug_relaxed_1y"
trades = pd.read_csv(r / "trades_1d.csv")
price_path = r / "price_slice.parquet"
if not price_path.exists():
    price_path = (
        ROOT / "output/system_run/benchmark/trend_baseline/1y/price_slice.parquet"
    )
prices = pd.read_parquet(price_path)

trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")
prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True, errors="coerce")

fills = (
    trades[(trades["fill_qty"].fillna(0) > 0)].copy()
    if "fill_qty" in trades.columns
    else trades.copy()
)
fills["side_sign"] = fills["side"].map({"BUY": 1, "SELL": -1}).fillna(0)
fills["signed_qty"] = fills["fill_qty"] * fills["side_sign"]

qty_by_ts = (
    fills.pivot_table(
        index="timestamp", columns="symbol", values="signed_qty", aggfunc="sum"
    )
    .fillna(0)
    .sort_index()
)
holdings = qty_by_ts.cumsum()

px = prices.pivot_table(
    index="timestamp", columns="symbol", values="close", aggfunc="last"
).sort_index()
holdings = holdings.reindex(px.index).ffill().fillna(0)

if "cash_delta" in trades.columns:
    trades["cash_delta"] = pd.to_numeric(trades["cash_delta"], errors="coerce").fillna(
        0
    )
    cash_delta = (
        trades.groupby("timestamp")["cash_delta"].sum().reindex(px.index).fillna(0)
    )
else:
    fills["notional"] = fills["fill_qty"] * fills["fill_price"]
    fills["cash_delta_est"] = fills["notional"] * (-fills["side_sign"])
    cash_delta = (
        fills.groupby("timestamp")["cash_delta_est"].sum().reindex(px.index).fillna(0)
    )

start_cash = 10000.0
cash = start_cash + cash_delta.cumsum()

pos_value = (holdings * px).sum(axis=1)
equity_mtM = cash + pos_value

print("cash_min", float(cash.min()), "cash_max", float(cash.max()))
print("pos_value_min", float(pos_value.min()), "pos_value_max", float(pos_value.max()))
print(
    "equity_mtm_min", float(equity_mtM.min()), "equity_mtm_max", float(equity_mtM.max())
)
print("last equity_mtm", float(equity_mtM.iloc[-1]))
