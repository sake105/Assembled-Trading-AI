"""H-084b — Odd-Lot Stufe 2: ECHTER Capture (Tender-Preis aus Filing vs Marktpreis).

Harvest mit Accession/CIK -> Primärdokument -> Preis-Regex (fix + Dutch-Range, konservativ LOW).
Capture = tender_low / Close(Filing-Tag) − 1 für ≤99-Aktien-Position (Odd-Lot-Priorität = ~sichere
Annahme). Statistik: Verteilung, Median-Capture, Events/Jahr, Jahres-Yield auf Odd-Lot-Kapital.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv  # noqa: E402

from h011_kandidat_a import OUTD  # noqa: E402

load_dotenv(ROOT / ".env")
TOK = os.environ["EODHD_API_TOKEN"]
DATA = Path(__file__).resolve().parent / "data"
UA = {"User-Agent": "research hans.oertel2@gmail.com"}
TICK_RE = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,5})\)\s*\(CIK\s*(\d+)\)")
PRICE_FIXED = re.compile(
    r"(?:purchase\s+price\s+of|at\s+a\s+price\s+of|price\s+of)\s*\$\s*([0-9]+(?:\.[0-9]{1,4})?)\s*per\s+share",
    re.I,
)
PRICE_DUTCH = re.compile(
    r"not\s+less\s+than\s*\$\s*([0-9]+(?:\.[0-9]{1,4})?)\s*(?:nor|or|and\s+not)\s*(?:more|greater|higher)\s*than\s*\$\s*([0-9]+(?:\.[0-9]{1,4})?)",
    re.I,
)
TAG_RE = re.compile(r"<[^>]+>")


def harvest() -> pd.DataFrame:
    rows = []
    for frm in range(0, 600, 100):
        params = urllib.parse.urlencode(
            {
                "q": '"odd lot"',
                "forms": "SC TO-I",
                "startdt": "2015-01-01",
                "enddt": "2026-07-01",
                "from": frm,
            }
        )
        req = urllib.request.Request(
            f"https://efts.sec.gov/LATEST/search-index?{params}", headers=UA
        )
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] from={frm}: {str(e)[:60]}", flush=True)
            break
        for h in d.get("hits", {}).get("hits", []):
            s = h.get("_source", {})
            name = (s.get("display_names") or [""])[0]
            m = TICK_RE.search(name)
            if not m:
                continue
            acc_file = h.get("_id", "")
            acc, _, fname = acc_file.partition(":")
            rows.append(
                {
                    "date": s.get("file_date"),
                    "ticker": m.group(1),
                    "cik": int(m.group(2)),
                    "accession": acc,
                    "fname": fname,
                }
            )
        time.sleep(0.3)
    df = pd.DataFrame(rows).drop_duplicates(subset=["accession"])
    df["date"] = pd.to_datetime(df["date"])
    print(f"[HARVEST] {len(df)} Filings mit Accession", flush=True)
    return df


def fetch_doc(cik: int, acc: str, fname: str) -> str | None:
    accn = acc.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/{fname}"
    try:
        raw = urllib.request.urlopen(
            urllib.request.Request(url, headers=UA), timeout=30
        ).read()
        return TAG_RE.sub(" ", raw[:400_000].decode("utf-8", errors="ignore"))
    except Exception:  # noqa: BLE001
        return None


def entry_price(tkr: str, d0: pd.Timestamp) -> float | None:
    frm = (d0 - pd.Timedelta(days=7)).date()
    to = (d0 + pd.Timedelta(days=7)).date()
    url = f"https://eodhd.com/api/eod/{tkr}.US?api_token={TOK}&fmt=json&from={frm}&to={to}"
    try:
        rows = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "research"}),
                timeout=30,
            )
            .read()
            .decode()
        )
        s = pd.Series(
            {pd.Timestamp(r["date"]): float(r["close"]) for r in rows}
        ).sort_index()
        after = s[s.index >= d0]
        return float(after.iloc[0]) if len(after) else None
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ev = harvest()
    captures = []
    n_doc = n_price = 0
    for _, r in ev.iterrows():
        doc = fetch_doc(r["cik"], r["accession"], r["fname"])
        time.sleep(0.15)
        if not doc:
            continue
        n_doc += 1
        tender_low = None
        md = PRICE_DUTCH.search(doc)
        kind = None
        if md:
            tender_low, kind = float(md.group(1)), "dutch_low"
        else:
            mf = PRICE_FIXED.search(doc)
            if mf:
                tender_low, kind = float(mf.group(1)), "fixed"
        if tender_low is None or tender_low <= 0:
            continue
        n_price += 1
        p0 = entry_price(r["ticker"], r["date"])
        if p0 is None or p0 <= 0 or p0 < 0.2 * tender_low or p0 > 5 * tender_low:
            continue  # Preis-Mismatch (Splits/andere Klasse) rausfiltern
        cap = tender_low / p0 - 1
        captures.append(
            {
                "ticker": r["ticker"],
                "date": str(r["date"].date()),
                "kind": kind,
                "tender_low": tender_low,
                "entry": p0,
                "capture_pct": round(cap * 100, 2),
            }
        )
        if len(captures) % 25 == 0:
            print(f"[..] {len(captures)} Captures berechnet", flush=True)

    df = pd.DataFrame(captures)
    out = {
        "n_filings": int(len(ev)),
        "n_docs": n_doc,
        "n_priced": n_price,
        "n_captures": int(len(df)),
    }
    if len(df) >= 20:
        a = df["capture_pct"].values
        pos = df[df["capture_pct"] > 0]
        yrs = 2026.5 - 2015.0
        out.update(
            {
                "capture_mean_pct": round(float(np.mean(a)), 2),
                "capture_median_pct": round(float(np.median(a)), 2),
                "share_positive_pct": round(float((a > 0).mean()) * 100, 1),
                "pos_only_median_pct": round(float(pos["capture_pct"].median()), 2),
                "events_per_year_positive": round(len(pos) / yrs, 1),
                "hinweis": "Capture = Dutch-LOW/Fix-Preis vs Close am Filing-Tag; Fill via Odd-Lot-"
                "Priorität ~sicher; VOR Steuern/Spreads; Completion-Risk nicht modelliert.",
            }
        )
        df.to_parquet(DATA / "oddlot_captures.parquet", index=False)
    (OUTD / "h084b_oddlot_stage2.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    print("[RESULT]", json.dumps(out, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
