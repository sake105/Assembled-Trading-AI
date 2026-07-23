"""Reachability: FINRA RegSHO daily short-volume (free, breadth-rich, large-cap-fit)
+ EFTS retry (confirm not hard-blocked)."""

from __future__ import annotations
import io
import json
import os
import sys
import time
import urllib.request
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
try:
    from src.assembled_core.data.edgar_form4_ingest import resolve_user_agent

    UA = resolve_user_agent()
except Exception:
    UA = "Assembled-Trading-AI research hans.oertel2@gmail.com"


def fetch(url, ua, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    return (
        urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")
    )


print("=== FINRA RegSHO daily short-volume ===")
uni = set(pd.read_csv("data/universe/master_universe_panel.csv")["symbol"].str.upper())
for d in ("20240102", "20240705", "20260605"):
    url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt"
    try:
        raw = fetch(url, "Mozilla/5.0 research")
        df = pd.read_csv(io.StringIO(raw), sep="|")
        df = df[df.get("Symbol").notna()] if "Symbol" in df.columns else df
        ncov = df["Symbol"].isin(uni).sum() if "Symbol" in df.columns else 0
        print(f"  {d}: rows={len(df)} cols={list(df.columns)} universe_covered={ncov}")
        if "Symbol" in df.columns and "ShortVolume" in df.columns:
            ex = df[df["Symbol"].isin(uni)].head(3)
            print(
                f"    sample: {ex[['Symbol', 'ShortVolume', 'TotalVolume']].to_dict('records')}"
            )
    except Exception as e:
        print(f"  {d}: ERR {type(e).__name__}: {e}")
    time.sleep(0.3)

print("\n=== EFTS retry (single month, backoff) ===")
for attempt in range(3):
    try:
        url = "https://efts.sec.gov/LATEST/search-index?forms=SC%2013D&startdt=2024-01-01&enddt=2024-01-31&from=0"
        j = json.loads(fetch(url, UA))
        print(f"  OK total={j['hits']['total']['value']} (attempt {attempt + 1})")
        break
    except Exception as e:
        print(f"  attempt {attempt + 1}: {type(e).__name__}: {e}")
        time.sleep(2 * (attempt + 1))
print("\n[DONE]")
