"""Measure 13D-subject overlap with OUR universe before any heavy pull.
Paginate one full month of SC 13D; count subjects that are in our 195 tickers."""

from __future__ import annotations
import json
import os
import re
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

uni = set(pd.read_csv("data/universe/master_universe_panel.csv")["symbol"].str.upper())
print(f"universe size: {len(uni)}")


def page(frm, start="2024-01-01", end="2024-03-31"):
    url = (
        f"https://efts.sec.gov/LATEST/search-index?forms=SC%2013D"
        f"&startdt={start}&enddt={end}&from={frm}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return json.loads(urllib.request.urlopen(req, timeout=30).read().decode())


first = page(0)
total = first["hits"]["total"]["value"]
print(f"SC 13D filings 2024-Q1: {total}")

subj_tickers = []
filers = []
frm = 0
while frm < min(total, 1000):
    j = first if frm == 0 else page(frm)
    hits = j["hits"]["hits"]
    if not hits:
        break
    for h in hits:
        names = h["_source"].get("display_names", [])
        if names:
            m = re.search(r"\(([A-Z.\-]{1,6})\)\s*\(CIK", names[0])
            if m:
                subj_tickers.append(m.group(1))
        if len(names) > 1:
            filers.append(names[1])
    frm += 10
    time.sleep(0.15)

print(f"parsed {len(subj_tickers)} subject tickers ({len(set(subj_tickers))} unique)")
inuni = [t for t in subj_tickers if t in uni]
print(f"subjects IN our universe: {len(inuni)} -> {sorted(set(inuni))}")
print(f"overlap rate: {len(inuni) / max(len(subj_tickers), 1):.1%}")
print(f"sample subjects: {sorted(set(subj_tickers))[:30]}")
print("\n[DONE] overlap probe")
