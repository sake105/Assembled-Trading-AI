"""Verify EDGAR EFTS ciks= filters by SUBJECT company for SC 13D, and that we can
parse (date, subject, filer). Test on known activist targets before the full pull."""

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

fund = pd.read_parquet(
    "data/raw/fundamentals/fundamentals_xbrl_full.parquet", columns=["symbol", "cik"]
).drop_duplicates()
sym2cik = {r.symbol.upper(): str(int(r.cik)).zfill(10) for r in fund.itertuples()}


def efts(cik, forms="SC 13D,SC 13D/A", start="2018-01-01", end="2026-06-13", frm=0):
    url = (
        f"https://efts.sec.gov/LATEST/search-index?forms={forms.replace(' ', '%20')}"
        f"&ciks={cik}&startdt={start}&enddt={end}&from={frm}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return json.loads(urllib.request.urlopen(req, timeout=30).read().decode())


def parse_names(names):
    """display_names: ['Company (TICK) (CIK n)', 'Filer (CIK n)', ...]; first=subject."""
    subj = names[0] if names else ""
    filers = names[1:] if len(names) > 1 else []
    tick = re.search(r"\(([A-Z.\-]+)\)\s*\(CIK", subj)
    return subj, (tick.group(1) if tick else None), "; ".join(filers)


for sym in ("DIS", "CVS", "INTC", "AAPL"):
    cik = sym2cik.get(sym)
    if not cik:
        print(f"{sym}: no cik")
        continue
    try:
        j = efts(cik)
        total = j["hits"]["total"]["value"]
        print(f"\n{sym} (cik {cik}): {total} SC 13D(/A) hits 2018-2026")
        for h in j["hits"]["hits"][:4]:
            s = h["_source"]
            subj, tick, filers = parse_names(s.get("display_names", []))
            print(
                f"   {s.get('file_date')} form={s.get('root_form')} subj_tick={tick} filer={filers[:50]}"
            )
    except Exception as e:
        print(f"{sym}: ERR {type(e).__name__}: {e}")
    time.sleep(0.2)
print("\n[DONE] efts test")
