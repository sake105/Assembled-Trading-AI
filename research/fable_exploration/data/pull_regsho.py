"""Pull FINRA RegSHO daily short-volume for OUR universe (free, ToS-clean CDN).
Resumable: skips dates already in the output parquet. Trading days taken from the
price index so we don't hammer weekends/holidays. Run with START/END env or args.

Columns kept: date, symbol, short_volume, short_exempt, total_volume, market.
Signal downstream = short_volume / total_volume (daily short-flow ratio, BJZ 2008).
"""

from __future__ import annotations
import io
import os
import sys
import time
import urllib.request
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
OUT = os.path.join("research", "fable_exploration", "data")
os.makedirs(OUT, exist_ok=True)

start = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp("2023-01-01")
end = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else pd.Timestamp("2024-12-31")
outfile = os.path.join(OUT, f"short_volume_{start.date()}_{end.date()}.parquet")

uni = set(pd.read_csv("data/universe/master_universe_panel.csv")["symbol"].str.upper())

# trading days from the price store
px = pd.read_parquet("output/aggregates/daily.parquet", columns=["timestamp"])
tdays = (
    pd.to_datetime(px["timestamp"], utc=True)
    .dt.normalize()
    .dt.tz_localize(None)
    .drop_duplicates()
)
tdays = sorted(d for d in tdays if start <= d <= end)

done_dates = set()
parts = []
if os.path.exists(outfile):
    prev = pd.read_parquet(outfile)
    done_dates = set(pd.to_datetime(prev["date"]).dt.normalize())
    parts.append(prev)
    print(f"[resume] {len(done_dates)} dates already pulled")

UA = "Assembled-Trading-AI research hans.oertel2@gmail.com"
got = miss = 0
buf = []
for i, d in enumerate(tdays):
    if d in done_dates:
        continue
    url = (
        f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d.strftime('%Y%m%d')}.txt"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
        df = pd.read_csv(io.StringIO(raw), sep="|")
        if "Symbol" not in df.columns:
            miss += 1
            continue
        df = df[df["Symbol"].isin(uni)].copy()
        df["date"] = d
        df = df.rename(
            columns={
                "Symbol": "symbol",
                "ShortVolume": "short_volume",
                "ShortExemptVolume": "short_exempt",
                "TotalVolume": "total_volume",
                "Market": "market",
            }
        )
        buf.append(
            df[
                [
                    "date",
                    "symbol",
                    "short_volume",
                    "short_exempt",
                    "total_volume",
                    "market",
                ]
            ]
        )
        got += 1
    except Exception as e:
        miss += 1
        if "404" not in str(e):
            print(f"  {d.date()}: {type(e).__name__}: {e}")
    if i % 100 == 0:
        print(f"  {i}/{len(tdays)} got={got} miss={miss}", flush=True)
    time.sleep(0.12)

if buf:
    parts.append(pd.concat(buf, ignore_index=True))
full = (
    pd.concat(parts, ignore_index=True).drop_duplicates(["date", "symbol"])
    if parts
    else pd.DataFrame()
)
full.to_parquet(outfile, index=False)
print(
    f"[DONE] got={got} miss={miss} rows={len(full)} symbols={full['symbol'].nunique() if len(full) else 0} -> {outfile}"
)
