"""Close the data gap before the big test round.
1) Inventory news-sentiment file coverage (for the 'news on/off' experiments).
2) Backfill FULL survivor price history via yfinance (auto_adjust=True ~ total-return,
   matching the project's TR-adjusted convention) for the whole 195-name universe +
   SPY/AGG/QQQ, so the cross-section widens from ~94 to ~all survivors. Delisted names
   remain unobtainable for free (survivorship caveat stays, breadth improves).
Writes research/fable_exploration/data/prices_enriched.parquet (uniform yfinance source)."""

from __future__ import annotations
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
OUT = os.path.join("research", "fable_exploration", "data")

# ---- 1) news inventory ----
print("=== NEWS FILE COVERAGE ===")
import glob

for f in sorted(glob.glob("output/news_sentiment_*.parquet")) + [
    "output/news_raw.parquet"
]:
    if not os.path.exists(f):
        continue
    try:
        df = pd.read_parquet(f)
        dcol = next(
            (
                c
                for c in df.columns
                if c.lower()
                in ("date", "timestamp", "day", "published_utc", "time_published")
            ),
            None,
        )
        rng = ""
        if dcol:
            t = pd.to_datetime(df[dcol], utc=True, errors="coerce")
            rng = f"{t.min()}..{t.max()}" if t.notna().any() else "no-dates"
        scol = next((c for c in df.columns if c.lower() in ("symbol", "ticker")), None)
        ns = df[scol].nunique() if scol else "?"
        print(
            f"  {os.path.basename(f):38} rows={len(df):>7} syms={ns} cols={list(df.columns)[:6]} {rng}"
        )
    except Exception as e:
        print(f"  {os.path.basename(f)}: ERR {e}")

# ---- 2) price backfill via yfinance ----
print("\n=== PRICE BACKFILL (yfinance, auto_adjust) ===")
uni = sorted(
    pd.read_csv("data/universe/master_universe_panel.csv")["symbol"]
    .str.upper()
    .unique()
)
extra = ["SPY", "AGG", "QQQ", "TLT"]
tickers = sorted(set(uni) | set(extra))
print(f"requesting {len(tickers)} tickers, period=max")

import yfinance as yf

closes = []
CH = 40
for i in range(0, len(tickers), CH):
    chunk = tickers[i : i + CH]
    df = yf.download(
        chunk, period="max", auto_adjust=True, progress=False, threads=True
    )
    if df is None or df.empty:
        print(f"  chunk {i // CH}: empty")
        continue
    # df columns MultiIndex (field, ticker) when multiple; ('Close', tkr)
    if isinstance(df.columns, pd.MultiIndex):
        cl = (
            df["Close"]
            if "Close" in df.columns.get_level_values(0)
            else df.xs("Close", axis=1, level=0)
        )
    else:
        cl = df[["Close"]].rename(columns={"Close": chunk[0]})
    closes.append(cl)
    print(
        f"  chunk {i // CH}: {cl.shape[1]} tickers, {cl.dropna(how='all').index.min().date()}..{cl.dropna(how='all').index.max().date()}",
        flush=True,
    )

wide = pd.concat(closes, axis=1)
wide = wide.loc[:, ~wide.columns.duplicated()]
wide.index = pd.to_datetime(wide.index).tz_localize(None)
tall = wide.reset_index().melt(
    id_vars=wide.index.name or "Date", var_name="symbol", value_name="close"
)
tall.columns = ["date", "symbol", "close"]
tall = tall.dropna(subset=["close"])
tall["date"] = pd.to_datetime(tall["date"]).dt.normalize()
tall.to_parquet(os.path.join(OUT, "prices_enriched.parquet"), index=False)

# coverage report
g = tall.groupby("symbol")["date"].agg(["min", "max", "count"])
since2018 = tall[tall["date"] >= "2018-01-01"]
g18 = since2018.groupby("symbol")["date"].count()
print(
    f"\nsaved prices_enriched.parquet: {tall.shape}, {tall['symbol'].nunique()} symbols"
)
print(
    f"symbols with >=1000 rows since 2018: {(g18 >= 1000).sum()} (was ~94 in daily.parquet)"
)
print(
    f"universe symbols present: {len(set(uni) & set(tall['symbol'].unique()))}/{len(uni)}"
)
missing = sorted(set(uni) - set(tall["symbol"].unique()))
if missing:
    print(f"  missing (no yf data): {missing}")
print("\n[DONE] enrich")
