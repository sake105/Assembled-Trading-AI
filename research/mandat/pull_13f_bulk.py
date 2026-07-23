"""13F-Bulk-Pull — SEC structured Form-13F data sets (Quartals-ZIPs, ab 2013Q2).

Laedt alle ZIPs nach research/mandat/data/13f/ (Resume: vorhandene werden
uebersprungen). Parsing/Auswertung folgt separat. KEIN Trial.
"""

from __future__ import annotations

import re
import sys
import time
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Assembled-Trading-AI hans.oertel2@gmail.com"}
DATA = Path(__file__).resolve().parent / "data" / "13f"
DATA.mkdir(parents=True, exist_ok=True)
INDEX = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"


def main() -> int:
    req = urllib.request.Request(INDEX, headers=UA)
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
    links = sorted(set(re.findall(r'href="([^"]*13f[^"]*\.zip)"', html, re.I)))
    print(f"[START] {len(links)} quarterly zips", flush=True)
    done = fail = 0
    for rel in links:
        name = rel.split("/")[-1]
        dst = DATA / name
        if dst.exists() and dst.stat().st_size > 1e6:
            done += 1
            continue
        url = "https://www.sec.gov" + rel if rel.startswith("/") else rel
        try:
            r = urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=300
            )
            dst.write_bytes(r.read())
            done += 1
            print(f"[OK] {name} ({dst.stat().st_size / 1e6:.0f} MB)", flush=True)
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[WARN] {name}: {exc}", flush=True)
        time.sleep(1)
    total_mb = sum(p.stat().st_size for p in DATA.glob("*.zip")) / 1e6
    print(f"[DONE] {done} ok, {fail} failed, {total_mb:.0f} MB total", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
