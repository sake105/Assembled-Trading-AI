"""Diagnose + fetch what's reachable & ToS-clean for survivorship + new edges.
1) Wikipedia S&P-500 current + 'selected changes' (survivorship universe metadata).
2) Stooq retry with RAW diagnostic (is it blocked or rate-limited here?).
3) EDGAR full-text search (efts) for SC 13D (new activist-edge source) reachability.
Saves anything useful under research/fable_exploration/data/. Read-only on the repo."""

from __future__ import annotations
import json
import os
import sys
import urllib.request
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
OUT = os.path.join("research", "fable_exploration", "data")
os.makedirs(OUT, exist_ok=True)

try:
    from src.assembled_core.data.edgar_form4_ingest import resolve_user_agent

    SEC_UA = resolve_user_agent()
except Exception:
    SEC_UA = "Assembled-Trading-AI research hans.oertel2@gmail.com"
print(f"SEC_UA = {SEC_UA[:60]}")


def get(url, ua, timeout=30):
    req = urllib.request.Request(
        url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
    )
    r = urllib.request.urlopen(req, timeout=timeout)
    raw = r.read()
    enc = r.headers.get("Content-Encoding", "")
    if enc == "gzip":
        import gzip

        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


# ---- 1) Wikipedia S&P 500 ----
print("\n=== [1] Wikipedia S&P 500 (current + changes) ===")
try:
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    cur = tables[0]
    print(f"current members table: {cur.shape} cols={list(cur.columns)[:6]}")
    cur.to_csv(os.path.join(OUT, "sp500_current.csv"), index=False)
    # the changes table (additions/removals with dates) — survivorship gold
    chg = None
    for t in tables[1:]:
        cols = " ".join(str(c) for c in t.columns).lower()
        if "added" in cols or "removed" in cols or "date" in cols:
            chg = t
            break
    if chg is not None:
        print(f"changes table: {chg.shape} cols={list(chg.columns)}")
        chg.to_csv(os.path.join(OUT, "sp500_changes.csv"), index=False)
        print("  saved sp500_current.csv + sp500_changes.csv")
    else:
        print("  no changes table identified")
except Exception as e:
    print(f"  WIKI FAIL: {type(e).__name__}: {e}")


# ---- 2) Stooq raw diagnostic ----
print("\n=== [2] Stooq raw diagnostic ===")
for t in ("aapl", "atvi", "twtr"):
    try:
        raw = get(
            f"https://stooq.com/q/d/l/?s={t}.us&i=d", "Mozilla/5.0 research", timeout=20
        )
        head = raw[:120].replace("\n", " | ")
        print(f"  {t}: len={len(raw)} head='{head}'")
    except Exception as e:
        print(f"  {t}: ERR {type(e).__name__}: {e}")


# ---- 3) EDGAR full-text search for SC 13D ----
print("\n=== [3] EDGAR EFTS — SC 13D reachability ===")
try:
    url = "https://efts.sec.gov/LATEST/search-index?q=%22&forms=SC+13D&startdt=2024-01-01&enddt=2024-03-31"
    # documented endpoint is /LATEST/search-index ; try the standard one
    url = "https://efts.sec.gov/LATEST/search-index?forms=SC%2013D&startdt=2024-01-01&enddt=2024-01-31"
    raw = get(url, SEC_UA, timeout=30)
    print(f"  raw len={len(raw)} head='{raw[:160]}'")
    try:
        j = json.loads(raw)
        total = j.get("hits", {}).get("total", {}).get("value")
        print(f"  total SC 13D hits (2024-01): {total}")
        hits = j.get("hits", {}).get("hits", [])[:3]
        for h in hits:
            src = h.get("_source", {})
            print(f"    {src.get('file_date')} {src.get('display_names')}")
    except Exception:
        print("  (not JSON — endpoint shape differs)")
except Exception as e:
    print(f"  EFTS FAIL: {type(e).__name__}: {e}")

print("\n[DONE] source diagnostic")
