"""Pull minute bars around earnings releases (precise XBRL acceptance timestamps) for
the first-minutes post-earnings drift test. Resumable: skips events already pulled.
Writes earnings_minute.parquet (tagged by event_id) + earnings_events.parquet."""

from __future__ import annotations
import os
import sys
import time
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, os.path.join("research", "fable_exploration", "data"))
from pull_polygon_intraday import fetch_minute_bars, KEY  # noqa: E402

OUTDIR = os.path.join("research", "fable_exploration", "data", "intraday")
os.makedirs(OUTDIR, exist_ok=True)
EV_PATH = os.path.join(OUTDIR, "earnings_events.parquet")
MIN_PATH = os.path.join(OUTDIR, "earnings_minute.parquet")

# high earnings-reaction tech names (intersected with our universe below)
CAND = [
    "AAPL",
    "AMZN",
    "GOOGL",
    "AMD",
    "AVGO",
    "CRM",
    "ADBE",
    "INTC",
    "NVDA",
    "AMAT",
    "LRCX",
    "KLAC",
    "ANET",
    "CRWD",
    "DDOG",
    "PANW",
    "FTNT",
    "DELL",
    "ISRG",
    "IDXX",
]
# Polygon key is a limited-history tier (~2y) + 5 req/min: pre-~2024-06 dates 403.
START, END = pd.Timestamp("2024-06-15", tz="UTC"), pd.Timestamp("2026-06-01", tz="UTC")

uni = set(pd.read_csv("data/universe/master_universe_panel.csv")["symbol"].str.upper())
syms = [s for s in CAND if s in uni]
print(f"key set: {bool(KEY)} | symbols: {syms}")

# ---- build earnings events from XBRL (earliest acceptance per symbol-quarter) ----
xb = pd.read_parquet(
    "data/raw/fundamentals/fundamentals_xbrl_full.parquet",
    columns=["symbol", "tag", "period_start", "period_end", "available_at", "form"],
)
xb["symU"] = xb["symbol"].astype(str).str.upper()
xb = xb[xb["symU"].isin(syms)]
# quarterly EPS-bearing rows -> earliest acceptance per (sym, period_end) ~ release time
dur = (pd.to_datetime(xb["period_end"]) - pd.to_datetime(xb["period_start"])).dt.days
q = xb[
    (xb["tag"].isin(["EarningsPerShareDiluted", "EarningsPerShareBasic"]))
    & dur.between(80, 100)
]
ev = (
    q.dropna(subset=["available_at"])
    .groupby(["symU", q["period_end"].dt.normalize()])["available_at"]
    .min()
    .reset_index()
)
ev.columns = ["symbol", "period_end", "release_ts"]
ev["release_ts"] = pd.to_datetime(ev["release_ts"], utc=True)
ev = ev[(ev["release_ts"] >= START) & (ev["release_ts"] <= END)].reset_index(drop=True)
ev["event_id"] = ev["symbol"] + "_" + ev["release_ts"].dt.strftime("%Y%m%d%H%M")
ev.to_parquet(EV_PATH, index=False)
print(
    f"earnings events: {len(ev)} ({ev['symbol'].nunique()} symbols, "
    f"{ev['release_ts'].min().date()}..{ev['release_ts'].max().date()})"
)

# ---- resume ----
done = set()
parts = []
if os.path.exists(MIN_PATH):
    prev = pd.read_parquet(MIN_PATH)
    done = set(prev["event_id"].unique())
    parts.append(prev)
    print(f"[resume] {len(done)} events already pulled")

buf = []
for i, r in ev.iterrows():
    if r["event_id"] in done:
        continue
    d0 = r["release_ts"].tz_convert("UTC").normalize()
    start = (d0 - pd.Timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )  # day before for pre-release baseline
    end = (d0 + pd.Timedelta(days=2)).strftime("%Y-%m-%d")  # release day + next session
    try:
        df = fetch_minute_bars(r["symbol"], start, end)
        if len(df):
            df["event_id"] = r["event_id"]
            df["release_ts"] = r["release_ts"]
            buf.append(df)
    except Exception as e:
        print(f"  [FAIL] {r['event_id']}: {type(e).__name__}: {e}", flush=True)
    time.sleep(13)  # respect free-tier ~5 req/min (avoids 429 thrash)
    if i % 25 == 0:
        print(
            f"  {i}/{len(ev)} pulled (buf={sum(len(b) for b in buf)} rows)", flush=True
        )

if buf:
    parts.append(pd.concat(buf, ignore_index=True))
full = (
    pd.concat(parts, ignore_index=True).drop_duplicates(["event_id", "ts"])
    if parts
    else pd.DataFrame()
)
full.to_parquet(MIN_PATH, index=False)
print(
    f"[DONE] events_with_bars={full['event_id'].nunique() if len(full) else 0} rows={len(full)} -> {MIN_PATH}"
)
