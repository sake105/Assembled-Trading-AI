"""H-084 — Odd-Lot-Tender Event-Study (Stufe 1). EDGAR SC TO-I + EODHD-Preise.

Harvest: Volltextsuche '"odd lot" forms=SC TO-I' 2015–2026 (paginierte hits, Ticker aus
display_names). Event-Study: Kauf zum Close des Filing-Tages, Bewertung +10/+30/+45 BD,
Excess vs SPY. Stufe-1-Proxy (ohne geparste Tender-Preise) — ehrlich benannt.
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
TICK_RE = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,5})\)\s*\(CIK")


def harvest() -> pd.DataFrame:
    rows = []
    for frm in range(0, 1000, 100):  # API max ~10k, wir nehmen 1.000 jüngste
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
            print(f"[WARN] from={frm}: {str(e)[:80]}", flush=True)
            break
        hits = d.get("hits", {}).get("hits", [])
        if not hits:
            break
        for h in hits:
            s = h.get("_source", {})
            name = (s.get("display_names") or [""])[0]
            m = TICK_RE.search(name)
            if m:
                rows.append(
                    {
                        "date": s.get("file_date"),
                        "ticker": m.group(1),
                        "name": name[:50],
                    }
                )
        time.sleep(0.3)
    df = pd.DataFrame(rows).drop_duplicates(subset=["date", "ticker"])
    df["date"] = pd.to_datetime(df["date"])
    print(
        f"[HARVEST] {len(df)} Events mit Ticker, {df['ticker'].nunique()} Ticker, "
        f"{df['date'].min().date()}–{df['date'].max().date()}",
        flush=True,
    )
    df.to_parquet(DATA / "oddlot_events.parquet", index=False)
    return df


def px_window(tkr: str, d0: pd.Timestamp) -> pd.Series | None:
    frm = (d0 - pd.Timedelta(days=10)).date()
    to = (d0 + pd.Timedelta(days=80)).date()
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
        if not rows:
            return None
        s = pd.Series(
            {pd.Timestamp(r["date"]): float(r["adjusted_close"]) for r in rows}
        ).sort_index()
        return s
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ev = harvest()
    spy = px_window("SPY", pd.Timestamp("2015-01-01"))  # not used; SPY separat unten
    # SPY-Gesamtreihe für Excess
    url = f"https://eodhd.com/api/eod/SPY.US?api_token={TOK}&fmt=json&from=2014-12-01"
    rows = json.loads(
        urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "research"}), timeout=60
        )
        .read()
        .decode()
    )
    spy = pd.Series(
        {pd.Timestamp(r["date"]): float(r["adjusted_close"]) for r in rows}
    ).sort_index()

    res = {h: [] for h in (10, 30, 45)}
    n_ok = n_fail = 0
    for _, row in ev.iterrows():
        s = px_window(row["ticker"], row["date"])
        if s is None or len(s) < 15:
            n_fail += 1
            continue
        after = s[s.index >= row["date"]]
        if len(after) < 12:
            n_fail += 1
            continue
        p0 = float(after.iloc[0])
        sp_after = spy[spy.index >= after.index[0]]
        sp0 = float(sp_after.iloc[0])
        for h in (10, 30, 45):
            if len(after) > h and len(sp_after) > h:
                r = float(after.iloc[h]) / p0 - 1
                rs = float(sp_after.iloc[h]) / sp0 - 1
                res[h].append(r - rs)
        n_ok += 1
        if n_ok % 100 == 0:
            print(f"[..] {n_ok} Events verarbeitet", flush=True)
        time.sleep(0.03)

    out = {"n_events_priced": n_ok, "n_fail": n_fail}
    for h in (10, 30, 45):
        a = np.array(res[h])
        if len(a) < 20:
            continue
        t = float(a.mean() / (a.std() / np.sqrt(len(a))))
        out[f"+{h}BD"] = {
            "n": int(len(a)),
            "mean_excess_pct": round(float(a.mean()) * 100, 2),
            "median_pct": round(float(np.median(a)) * 100, 2),
            "hit_rate_pct": round(float((a > 0).mean()) * 100, 1),
            "t": round(t, 2),
        }
        print(
            f"[+{h}BD] n={len(a)} mean={a.mean() * 100:+.2f}% med={np.median(a) * 100:+.2f}% "
            f"hit={100 * (a > 0).mean():.0f}% t={t:.2f}",
            flush=True,
        )
    crit = out.get("+30BD", {})
    out["PASS_stage1"] = bool(
        crit.get("mean_excess_pct", 0) > 0
        and crit.get("t", 0) > 2
        and crit.get("hit_rate_pct", 0) > 60
    )
    (OUTD / "h084_oddlot.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    print("[VERDICT]", json.dumps(out, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
