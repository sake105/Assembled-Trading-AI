"""Small-Cap-Universum-Pull — ALLE NYSE/NASDAQ Common Stocks inkl. Delisted, ab 2000.

Neutrale Universumsbasis (kein Selektions-Bias). Tranchen-Resume ueber Teil-Parquets
in data/smallcap/. ~27k Ticker; EODHD-Limit 100k/Tag. KEIN Trial.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
TOK = os.environ["EODHD_API_TOKEN"]
DATA = Path(__file__).resolve().parent / "data"
OUT_DIR = DATA / "smallcap"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH = 4000
WORKERS = 12
TIMEOUT = 12  # dead delisted tickers must fail fast, not hang 45s


def fetch(sym: str):
    """Return (sym, DataFrame|None). None -> empty/fail (goes to marker)."""
    url = f"https://eodhd.com/api/eod/{sym}.US?api_token={TOK}&fmt=json&from=2000-01-01"
    try:
        rows = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "research"}),
                timeout=TIMEOUT,
            )
            .read()
            .decode()
        )
        if rows:
            df = pd.DataFrame(rows)[["date", "adjusted_close", "volume"]]
            df.columns = ["timestamp", "close", "volume"]
            df["symbol"] = sym
            return sym, df
    except Exception:  # noqa: BLE001
        pass
    return sym, None


def wanted() -> list[str]:
    lv = pd.read_parquet(DATA / "us_symbols_live.parquet")
    dd = pd.read_parquet(DATA / "us_symbols_delisted.parquet")
    f = pd.concat([lv, dd], ignore_index=True)
    f = f[(f["Type"] == "Common Stock") & (f["Exchange"].isin(["NYSE", "NASDAQ"]))]
    return sorted(set(f["Code"].astype(str)))


def main() -> int:
    syms = wanted()
    done: set[str] = set()
    for p in OUT_DIR.glob("part_*.parquet"):
        done.update(pd.read_parquet(p, columns=["symbol"])["symbol"].unique())
    marker = OUT_DIR / "_empty_symbols.txt"
    if marker.exists():
        done.update(marker.read_text().split())
    todo = [s for s in syms if s not in done]
    print(
        f"[START] {len(syms)} wanted, {len(done)} done, {len(todo)} todo (batch {BATCH})",
        flush=True,
    )
    if not todo:
        print("[COMPLETE] universe pull finished", flush=True)
        return 0
    batch = todo[:BATCH]
    frames, empty = [], []
    done_n = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch, s): s for s in batch}
        for fut in as_completed(futs):
            sym, df = fut.result()
            if df is not None:
                frames.append(df)
            else:
                empty.append(sym)
            done_n += 1
            if done_n % 1000 == 0:
                print(f"[OK] {done_n}/{len(batch)}", flush=True)
    n_part = len(list(OUT_DIR.glob("part_*.parquet"))) + 1
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        out.to_parquet(OUT_DIR / f"part_{n_part:03d}.parquet", index=False)
    with open(marker, "a", encoding="utf-8") as fh:
        fh.write("\n".join(empty) + "\n")
    print(
        f"[DONE] part {n_part}: {sum(len(f) for f in frames)} rows, {len(empty)} empty/fail | remaining ~{len(todo) - BATCH}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
